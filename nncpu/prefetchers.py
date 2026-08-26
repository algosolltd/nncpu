"""Streaming cache prefetchers.

The CPU asks a prefetcher for the *next* memory address after every
LOAD/STORE and prefetches whatever address comes back.  ``None`` is a
valid answer meaning "don't prefetch now".

* :class:`StridePrefetcher` -- classic heuristic: predict the next access
  continues the most recent stride (delta between the last two addresses).
  Cheap and excellent on smooth streams, but it prefetches *every* access,
  so it pollutes the cache on unpredictable traffic.
* :class:`NNPrefetcher` -- a streaming :class:`~sklearn.neural_network.MLPRegressor`
  that learns the address stream incrementally (with a replay buffer) and
  predicts the *next delta*.  A confidence gate tracks recent prediction
  error and issues ``None`` when the stream is too noisy to predict, so
  the NN matches stride on smooth streams *and* stays pollution-free on
  random access -- the thing stride cannot do.
"""

import warnings
from collections import deque
from typing import Optional

import numpy as np

from .mlp import DenseMLP

try:  # scikit-learn is optional: the default numpy backend needs only numpy
    from sklearn.neural_network import MLPRegressor
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised when sklearn is absent
    MLPRegressor = None
    _HAS_SKLEARN = False

OPCODE_IDS = {"LOAD": 0, "STORE": 1, "ADD": 2, "MUL": 3, "DIV": 4, "NOP": 5}

# Features/targets are scaled by these constants so the regressor sees a
# well-conditioned, roughly unit-magnitude problem.
_ADDR_SCALE = 1024.0
_PC_SCALE = 8192.0
_DELTA_SCALE = 16.0

_DEFAULT_DELTA = 1  # conservative guess while the model warms up

# Confidence gate: if the mean |predicted - actual| delta over the recent
# window climbs above one line of slack, stop prefetching (it is noise).
# When no prediction was issued (gate closed / warm-up) the reference is
# the stride delta, so the window keeps sliding and confidence recovers.
_GATE_WINDOW = 48
_GATE_THRESHOLD = 8.0  # words
_REPLAY_SIZE = 64
_WARMUP_EPOCHS = 4     # epochs on the very first fit, so the first prediction
                       # isn't pure random-init noise

# Feature-ablation modes: which columns of the 6-D encoding the MLP sees.
FEATURE_MODES = {
    "full": (0, 1, 2, 3, 4, 5),
    "no_opcode": (1, 2, 3, 4, 5),
    "no_pc": (0, 2, 3, 4, 5),
    "no_abs_addr": (0, 1, 3, 4, 5),
    "delta_only": (3, 4, 5),
}
_FEATURE_FULL = np.arange(6)


class Prefetcher:
    """Base class: a prefetcher just needs ``predict_next``."""

    name = "base"

    def predict_next(self, addr: int, pc: int, opcode: str) -> Optional[int]:
        raise NotImplementedError


class StridePrefetcher(Prefetcher):
    name = "stride"

    def __init__(self):
        self._last_addr: Optional[int] = None
        self._last_delta: Optional[int] = None

    def predict_next(self, addr: int, pc: int, opcode: str) -> Optional[int]:
        del pc, opcode  # unused
        if self._last_delta is None:
            nxt = addr + _DEFAULT_DELTA
        else:
            nxt = addr + self._last_delta
        self._last_delta = addr - self._last_addr if self._last_addr is not None else None
        self._last_addr = addr
        return nxt if nxt >= 0 else addr


class NNPrefetcher(Prefetcher):
    """Online MLP predicting the *next address delta* with error gating."""

    name = "nn"

    def __init__(
        self,
        batch_size: int = 16,
        feature_size: int = 6,
        hidden_layers: tuple = (32,),
        learning_rate: float = 1e-2,
        random_state: int = 42,
        mlp_backend: str = "numpy",
        confidence_gate: bool = True,
        gate_threshold: float = _GATE_THRESHOLD,
        gate_scale: Optional[float] = 0.5,
        gate_aggregator: str = "median",
        delta_cap: int = 16,
        feature_mode: str = "full",
        warmup_epochs: int = _WARMUP_EPOCHS,
    ):
        if feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"Unknown feature_mode {feature_mode!r}; "
                f"expected one of {sorted(FEATURE_MODES)}"
            )
        if gate_aggregator not in ("mean", "median"):
            raise ValueError(
                f"Unknown gate_aggregator {gate_aggregator!r}; "
                "expected 'mean' or 'median'"
            )
        self.feature_mode = feature_mode
        self.warmup_epochs = warmup_epochs
        self._mask = np.asarray(FEATURE_MODES[feature_mode])
        feature_size = len(self._mask)

        self.batch_size = batch_size
        self.feature_size = feature_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.mlp_backend = mlp_backend
        self.confidence_gate = confidence_gate
        self.gate_threshold = gate_threshold
        self.gate_scale = gate_scale           # None = use absolute threshold
        self.gate_aggregator = gate_aggregator
        self.delta_cap = delta_cap
        self._cap_scaled = delta_cap / _DELTA_SCALE
        self._recent_deltas = deque(maxlen=_GATE_WINDOW)  # |actual delta| for the scale
        if mlp_backend not in ("numpy", "sklearn"):
            raise ValueError(f"Unknown MLP backend: {mlp_backend!r}")
        if mlp_backend == "sklearn" and not _HAS_SKLEARN:
            raise ImportError("scikit-learn is not installed; use mlp_backend='numpy'")

        self._prev_addr: Optional[int] = None
        self._last_addr: Optional[int] = None
        self._last_delta: Optional[int] = None
        self._pending_features: Optional[np.ndarray] = None  # context for an open sample
        self._last_predicted_delta: Optional[int] = None
        self._replay: "deque[tuple[np.ndarray, float]]" = deque(maxlen=_REPLAY_SIZE)
        self._recent_error: "deque[float]" = deque(maxlen=_GATE_WINDOW)
        self._model = None

    def _build_model(self):
        if self.mlp_backend == "sklearn":
            return MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                learning_rate_init=self.learning_rate,
                max_iter=1,
                early_stopping=False,
                random_state=self.random_state,
            )
        return DenseMLP(
            input_size=self.feature_size,
            hidden_size=self.hidden_layers[0],
            learning_rate=self.learning_rate,
            seed=self.random_state,
        )

    def _encode(self, addr: int, pc: int, opcode: str) -> np.ndarray:
        d_cur = (addr - self._last_addr) / _DELTA_SCALE if self._last_addr is not None else 0.0
        d_prev = self._last_delta / _DELTA_SCALE if self._last_delta is not None else 0.0
        opcode_id = OPCODE_IDS.get(opcode, 5) / max(len(OPCODE_IDS), 1)
        x = np.array(
            [
                opcode_id,
                min(pc / _PC_SCALE, 1.0),
                addr / _ADDR_SCALE,
                d_cur,
                d_prev,
                1.0,
            ]
        )
        return x[self._mask]

    def predict_next(self, addr: int, pc: int, opcode: str) -> Optional[int]:
        # 1) close the previous sample: (features @ t-1) -> (delta @ t)
        if self._pending_features is not None and self._prev_addr is not None:
            delta_now = addr - self._prev_addr
            self._recent_deltas.append(abs(delta_now))
            # clamp the training target so gather-style outliers do not pull
            # the regression away from the dominant (sequential) delta
            clamped = max(-self.delta_cap, min(self.delta_cap, delta_now))
            self._replay.append((self._pending_features, clamped / _DELTA_SCALE))
            # Error is measured against whatever we would have prefetched:
            # the issued prediction, or the stride heuristic while suppressed.
            ref = (
                self._last_predicted_delta
                if self._last_predicted_delta is not None
                else self._last_delta
            )
            if ref is not None:
                self._recent_error.append(abs(ref - delta_now))
            self._fit_if_ready()

        # 2) remember the current context as the target of the next sample
        features = self._encode(addr, pc, opcode)
        self._pending_features = features

        # 3) decide: is this stream predictable right now?
        if self.confidence_gate and not self._confident():
            self._last_delta = addr - self._last_addr if self._last_addr is not None else None
            self._last_addr = addr
            self._prev_addr = addr
            self._last_predicted_delta = None
            return None

        # 4) pick a predicted delta (stride fallback until the model fits)
        if self._last_addr is None:
            self._last_delta = None
            predicted_delta = _DEFAULT_DELTA
        else:
            self._last_delta = addr - self._last_addr
            if self._model is None:
                predicted_delta = self._last_delta or _DEFAULT_DELTA
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = self._model.predict(features.reshape(1, -1))
                    predicted_delta = int(round(float(np.asarray(pred).ravel()[0]) * _DELTA_SCALE))
                    predicted_delta = max(-self.delta_cap,
                                          min(self.delta_cap, predicted_delta))

        self._last_predicted_delta = predicted_delta
        self._last_addr = addr
        self._prev_addr = addr
        nxt = addr + predicted_delta
        return nxt if nxt >= 0 else addr + _DEFAULT_DELTA

    def _confident(self) -> bool:
        if len(self._recent_error) < 8:  # allow a warm-up runway
            return True
        # threshold: relative to the stream's characteristic delta magnitude
        # scales with large strides (matmul rows) yet stays strict on noise.
        if self.gate_scale is not None and self._recent_deltas:
            ds = sorted(self._recent_deltas)
            scale = max(ds[len(ds) // 2], 4.0)   # floor of half a line
            threshold = self.gate_scale * scale
        else:
            threshold = self.gate_threshold
        if self.gate_aggregator == "median":
            # Robust to occasional large outliers (e.g. an attention gather
            # among sequential KV writes): sparse far accesses must not make
            # the whole stream look unpredictable.
            errs = sorted(self._recent_error)
            recent = errs[len(errs) // 2]
        else:
            recent = sum(self._recent_error) / len(self._recent_error)
        return recent <= threshold

    @property
    def gate_open(self) -> bool:
        """Expose the confidence-gate state (for phase-switch analysis)."""
        return self._confident()

    def _fit_if_ready(self) -> None:
        if self._model is None and len(self._replay) < self.batch_size:
            return
        if len(self._replay) < 8:
            return
        x = np.asarray([s[0] for s in self._replay]).reshape(len(self._replay), self.feature_size)
        y = np.asarray([s[1] for s in self._replay])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._model is None:
                self._model = self._build_model()
                if self.mlp_backend == "numpy":
                    self._model.partial_fit(x, y, epochs=self.warmup_epochs)
                    return
            self._model.partial_fit(x, y)


def make_prefetcher(name: str, **kwargs) -> Optional[Prefetcher]:
    """Factory used by the benchmark harness.

    ``kwargs`` are forwarded to :class:`NNPrefetcher` (e.g. ``batch_size``,
    ``learning_rate``, ``random_state``).
    """
    if name is None or name in ("none", "baseline"):
        return None
    if name == "stride":
        return StridePrefetcher()
    if name == "nn":
        return NNPrefetcher(**kwargs)
    raise ValueError(f"Unknown prefetcher: {name!r}")