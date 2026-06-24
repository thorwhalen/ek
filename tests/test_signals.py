"""Tests for the intrinsic confidence signals (#5, cost tier 2)."""

import math

from ek.base import Signal
from ek.qe.signals import (
    IntrinsicConfidenceSignal,
    LogprobSignal,
    geo_mean,
    length_normalized,
    mean_prob,
    min_prob,
)

LOGPS = [math.log(0.9), math.log(0.8), math.log(0.95)]


def test_aggregator_values_match_doctest_numbers():
    assert abs(geo_mean(LOGPS) - 0.8811) < 1e-3
    assert abs(min_prob(LOGPS) - 0.8) < 1e-9
    assert abs(mean_prob(LOGPS) - (0.9 + 0.8 + 0.95) / 3) < 1e-9


def test_aggregators_empty_input_is_one():
    assert geo_mean([]) == 1.0
    assert min_prob([]) == 1.0
    assert mean_prob([]) == 1.0
    assert length_normalized([]) == 1.0


def test_length_normalized_penalty_bites():
    # alpha=1 divides by T**1 -> identical to the geometric mean (exp of the mean)
    assert abs(length_normalized(LOGPS, alpha=1.0) - geo_mean(LOGPS)) < 1e-12
    # alpha=0 divides by T**0=1 -> exp(sum), strictly smaller for T>1 (more penalty)
    assert length_normalized(LOGPS, alpha=0.0) < geo_mean(LOGPS)


def test_logprob_signal_dispatch_and_protocol():
    assert abs(LogprobSignal()(LOGPS) - geo_mean(LOGPS)) < 1e-12
    sig = LogprobSignal(aggregator="length_normalized", alpha=0.6)
    assert abs(sig(LOGPS) - length_normalized(LOGPS, alpha=0.6)) < 1e-12
    assert LogprobSignal().cost_tier == 2
    assert isinstance(LogprobSignal(), Signal)


class _Block:
    def __init__(self, confidence):
        self.confidence = confidence


class _Ocr:
    def __init__(self, blocks=(), text="", mean_confidence=None):
        self.blocks = list(blocks)
        self.text = text
        self.mean_confidence = mean_confidence


def test_intrinsic_confidence_pooling():
    assert IntrinsicConfidenceSignal(pool="min")([0.99, 0.4, 0.95]) == 0.4
    assert abs(IntrinsicConfidenceSignal(pool="mean")([0.4, 0.6]) - 0.5) < 1e-9
    # OcrResult-shaped: prefer per-block confidences
    res = _Ocr(blocks=[_Block(0.9), _Block(0.3)], text="a b", mean_confidence=0.6)
    assert IntrinsicConfidenceSignal(pool="min")(res) == 0.3
    assert IntrinsicConfidenceSignal().cost_tier == 2


def test_intrinsic_confidence_null_safe():
    # No usable confidence anywhere (VLM/markdown case) -> 1.0, never a crash.
    res = _Ocr(blocks=[], text=None, mean_confidence=None)
    assert IntrinsicConfidenceSignal()(res) == 1.0


def test_min_prob_keeps_zero_probability_token():
    # Regression: -inf (log of a p=0 token) was dropped, so min_prob returned the
    # SECOND-weakest token and geo_mean of an all-zero field returned a maximal 1.0.
    ninf = float("-inf")
    assert min_prob([ninf, math.log(0.9)]) < 1e-6      # weakest (p=0) dominates
    assert geo_mean([ninf, ninf]) < 1e-6               # all-zero field -> ~0, not 1.0
    assert geo_mean([math.log(0.9), math.log(0.8)]) > 0.8  # finite path unchanged
    assert min_prob([]) == 1.0                          # no-evidence convention preserved
    assert geo_mean([float("inf"), math.log(0.5)]) == 0.5  # +inf still dropped


def test_intrinsic_confidence_rejects_unknown_pool():
    import pytest

    with pytest.raises(ValueError, match="pool must be one of"):
        IntrinsicConfidenceSignal(pool="bogus")([0.5])
