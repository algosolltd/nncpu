"""A tiny pure-NumPy fully-connected ReLU MLP (regression) with Adam.

Purpose: the per-access ``predict``/``partial_fit`` calls the prefetcher
makes are dominated by scikit-learn's per-call Python/Cython overhead, not
by the (tiny) math.  This hand-rolled MLP does the same 6 -> 32 -> 1
computation in ~microseconds and is fully deterministic for a given seed,
which is exactly what a paper's "multiple runs" methodology wants.

The API mirrors the sklearn surface the prefetcher uses::

    mlp.predict(X)      # (n,) or (n, 1) predictions
    mlp.partial_fit(X, y)
"""

import numpy as np

OUT_SIZE = 1


class DenseMLP:
    """Small ReLU MLP regressor trained with Adam (beta1=0.9, beta2=0.999)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        output_size: int = OUT_SIZE,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        he1 = np.sqrt(2.0 / input_size)
        he2 = np.sqrt(2.0 / hidden_size)
        self.w1 = rng.randn(hidden_size, input_size) * he1
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.randn(output_size, hidden_size) * he2
        self.b2 = np.zeros(output_size)

        # Adam moments, one per trainable parameter block.
        self._m = [np.zeros_like(p) for p in (self.w1, self.w2, self.b1, self.b2)]
        self._v = [np.zeros_like(p) for p in (self.w1, self.w2, self.b1, self.b2)]
        self._t = 0
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8

    # -- inference ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.w1.T + self.b1
        a1 = np.maximum(z1, 0.0)
        return a1 @ self.w2.T + self.b2

    # -- training ----------------------------------------------------------

    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        """One (or more) batch SGD epochs over ``(X, y)``."""
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n = X.shape[0]
        if n == 0:
            return

        for _ in range(epochs):
            z1 = X @ self.w1.T + self.b1
            a1 = np.maximum(z1, 0.0)
            yhat = a1 @ self.w2.T + self.b2
            dy = 2.0 * (yhat - y) / n

            grad_w2 = dy.T @ a1
            grad_b2 = dy.sum(axis=0)
            da1 = dy @ self.w2
            dz1 = da1 * (z1 > 0)
            grad_w1 = dz1.T @ X
            grad_b1 = dz1.sum(axis=0)

            self._adam(
                [grad_w1, grad_w2, grad_b1, grad_b2],
                [self.w1, self.w2, self.b1, self.b2],
            )

    # -- Adam update -------------------------------------------------------

    def _adam(self, grads: list, params: list) -> None:
        self._t += 1
        bc1 = 1.0 - self._beta1 ** self._t
        bc2 = 1.0 - self._beta2 ** self._t
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * g * g
            p -= self.lr * (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + self._eps)