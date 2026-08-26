"""Unit tests for the stride and NN prefetchers."""

import numpy as np
import pytest

from nncpu.prefetchers import StridePrefetcher, NNPrefetcher, make_prefetcher


def strided_addresses(n=40, base=0x1000, step=4):
    return [base + i * step for i in range(n)]


def feed(prefetcher, addresses):
    """Drive a prefetcher with a stream of accesses; return predictions."""
    predictions = []
    for pc, addr in enumerate(addresses):
        predictions.append(prefetcher.predict_next(addr, pc, "LOAD"))
    return predictions


def confuse(prefetcher, addresses):
    """Predictions that miss the resident set are cache misses; prefetching
    adds the predicted line to the resident set (bounded LRU proxy)."""
    resident = set()  # line ids; unbounded enough for these short tests
    hits, misses = 0, 0
    for pc, addr in enumerate(addresses):
        line = addr // 8
        if line in resident:
            hits += 1
        else:
            misses += 1
            resident.add(line)
        pred = prefetcher.predict_next(addr, pc, "LOAD")
        if pred is not None:
            resident.add(pred // 8)
    return hits, misses


def test_stride_warm_up_then_stride():
    """First two touches are guesses (+1); afterwards the stride is exact."""
    addrs = strided_addresses(50, step=4)
    preds = feed(StridePrefetcher(), addrs[:-1])
    assert preds[0] == addrs[0] + 1
    assert preds[1] == addrs[1] + 1
    for i in range(2, len(addrs) - 1):
        assert preds[i] == addrs[i] + 4


def test_make_prefetcher_factory():
    assert make_prefetcher("none") is None
    assert make_prefetcher("baseline") is None
    assert make_prefetcher(None) is None
    assert isinstance(make_prefetcher("stride"), StridePrefetcher)
    assert isinstance(make_prefetcher("nn"), NNPrefetcher)


def test_nn_learns_sequential_stream():
    """After a tiny warm-up the NN is near-perfect on a sequential stream."""
    addrs = [0x1000 + i for i in range(400)]
    _, misses = confuse(NNPrefetcher(), addrs)
    assert misses <= 5


def test_nn_learns_strided_stream():
    """The NN picks up a constant stride: very few misses vs. an unbounded
    baseline where nearly every access is a miss."""
    addrs = strided_addresses(400, step=8)
    _, misses = confuse(NNPrefetcher(), addrs)
    assert misses < 40  # ~90% of accesses end up prefetched


def test_nn_avoids_pollution_where_stride_thrashes():
    """The point of the confidence gate: on random access the stride
    predictor pollutes the cache (hit rate drops below baseline), while
    the NN backs off and stays at baseline hit rate."""
    from nncpu.cpu import CPU
    from nncpu.workloads import random_read

    def hit_rate(config):
        cpu = CPU(prefetcher=make_prefetcher(config))
        for inst in random_read(n=2000):
            cpu.execute(inst)
        return cpu.report().hit_rate

    baseline_rate = hit_rate("baseline")
    stride_rate = hit_rate("stride")
    nn_rate = hit_rate("nn")

    assert nn_rate >= baseline_rate       # the NN does not pollute
    assert stride_rate < nn_rate          # stride actively hurts


def test_nn_feature_vector_shape():
    p = NNPrefetcher()
    x = p._encode(0x1000, 0, "LOAD")
    assert x.shape == (p.feature_size,)
    assert np.isfinite(x).all()


def test_feature_modes_change_input_size():
    full = NNPrefetcher(feature_mode="full")
    delta = NNPrefetcher(feature_mode="delta_only")
    assert full.feature_size == 6
    assert delta.feature_size == 3
    assert delta._encode(0x1000, 0, "LOAD").shape == (3,)
    for mode in ("no_opcode", "no_pc", "no_abs_addr", "delta_only"):
        p = NNPrefetcher(feature_mode=mode)
        assert p._encode(0x1000, 0, "LOAD").shape == (p.feature_size,)


@pytest.mark.parametrize("mode", ["no_opcode", "no_pc", "no_abs_addr", "delta_only"])
def test_feature_modes_run_without_error(mode):
    from nncpu.workloads import build_workloads
    from nncpu.prefetchers import make_prefetcher

    p = make_prefetcher("nn", feature_mode=mode, random_state=0)
    for inst in build_workloads(60)["sequential_read"]:
        p.predict_next(inst["address"], 0, "LOAD")
    assert p._model is not None


def test_delta_cap_bounds_replay_targets():
    """Huge address jumps are clamped to +-delta_cap words for training."""
    p = NNPrefetcher(delta_cap=16)
    p.predict_next(0x1000, 0, "LOAD")
    p.predict_next(0x1000 + 8192, 1, "LOAD")  # a 8192-word jump
    targets = [s[1] for s in p._replay]
    assert all(abs(t) <= 16 / 16.0 + 1e-9 for t in targets)


def test_median_gate_is_robust_to_sparse_outliers():
    """A sequential stream with an occasional far access must NOT trip the
    gate: the NN keeps prefetching (median) where the mean version shuts off."""
    from nncpu.cpu import CPU
    from nncpu.prefetchers import make_prefetcher

    instrs = []
    for i in range(400):
        instrs.append({"type": "LOAD", "address": 0x2000 + i})
        if i % 15 == 0:
            instrs.append({"type": "LOAD", "address": 0x5000 + (i * 7) % 4096})

    def cycles(agg):
        cpu = CPU(prefetcher=NNPrefetcher(gate_aggregator=agg, delta_cap=16,
                                          random_state=3))
        for inst in instrs:
            cpu.execute(inst)
        return cpu.report().cycles

    med, mean_ = cycles("median"), cycles("mean")
    assert med < mean_, (med, mean_)


def test_make_prefetcher_rejects_bad_gate_aggregator():
    with pytest.raises(ValueError):
        NNPrefetcher(gate_aggregator="whatever")