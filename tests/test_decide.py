"""Tests for the selective-prediction decision stage (#5)."""

import math
import random

import pytest

from ek.base import Decision
from ek.qe.decide import (
    ConformalGate,
    CostSensitiveGate,
    GroupConformalGate,
    RiskControlGate,
    risk_coverage_curve,
    split_conformal_quantile,
)
from ek.registry import get


def test_cost_sensitive_gate_threshold_from_rho():
    # Bayes-optimal cost-sensitive threshold: tau = rho/(1+rho), NOT 1 - 1/rho.
    gate = CostSensitiveGate(rho=9.0)        # tau = 9/10 = 0.9
    assert abs(gate.tau - 9 / 10) < 1e-12
    assert gate(0.95) is Decision.ACCEPT
    assert gate(0.5) is Decision.FLAG
    # rho == 1 (symmetric costs): threshold is the 0.5 MAP boundary, and a
    # certain-WRONG item (p=0) must be FLAGGED, never auto-accepted.
    assert abs(CostSensitiveGate(rho=1.0).tau - 0.5) < 1e-12
    assert CostSensitiveGate(rho=1.0)(0.0) is Decision.FLAG
    assert CostSensitiveGate(rho=1.0)(0.6) is Decision.ACCEPT
    # rho < 1 (reviews costlier than misses): bar drops below 0.5 but stays > 0.
    assert 0.0 < CostSensitiveGate(rho=0.25).tau < 0.5
    # explicit block band
    g2 = CostSensitiveGate(rho=9.0, block_threshold=0.1)
    assert g2(0.05) is Decision.BLOCK


def test_split_conformal_quantile_formula():
    scores = [i / 10 for i in range(1, 11)]   # 0.1 .. 1.0, n=10
    # alpha=0.1: k = ceil(11*0.9) = 10 -> s[9] = 1.0
    assert split_conformal_quantile(scores, 0.1) == 1.0
    # alpha=0.2: k = ceil(11*0.8) = 9 -> s[8] = 0.9
    assert abs(split_conformal_quantile(scores, 0.2) - 0.9) < 1e-12
    # n too small to guarantee 1-alpha -> +inf (flag nothing)
    assert split_conformal_quantile([0.5], 0.1) == float("inf")


def test_conformal_gate_marginal_coverage():
    # Correct items have higher confidence than wrong ones, with overlap. The
    # split-conformal guarantee: at most alpha of truly-correct items are flagged.
    rng = random.Random(0)
    alpha = 0.1

    def confidence(is_correct):
        # bounded to [0,1]; correct items centered higher
        base = 0.7 if is_correct else 0.4
        return min(1.0, max(0.0, rng.gauss(base, 0.15)))

    cal_correct = [rng.random() < 0.6 for _ in range(800)]
    cal_probs = [confidence(c) for c in cal_correct]
    gate = ConformalGate(alpha=alpha).fit(cal_probs, cal_correct)

    # Fresh correct test items, exchangeable with the correct calibration items.
    test_probs = [confidence(True) for _ in range(2000)]
    flagged = sum(1 for p in test_probs if gate(p) is Decision.FLAG)
    flag_rate = flagged / len(test_probs)
    assert flag_rate <= alpha + 0.04   # marginal guarantee + finite-sample slack


def test_risk_control_gate_bounds_accepted_error_on_holdout():
    # Conformal risk control bounds E[accepted AND wrong] over the calibration draw,
    # so fit on one half and measure on a DISJOINT half, averaged over many seeds
    # (asserting on the fit data would be a tautology -- the bound holds by
    # construction there).
    target = 0.05

    def gen(rng, n):
        correct = [rng.random() < 0.7 for _ in range(n)]
        probs = [min(1.0, max(0.0, rng.gauss(0.80 if c else 0.45, 0.13))) for c in correct]
        return probs, correct

    rates = []
    for seed in range(200):
        rng = random.Random(seed)
        cp, cc = gen(rng, 400)
        tp, tc = gen(rng, 400)
        gate = RiskControlGate(target=target).fit(cp, cc)
        acc_err = sum(1 for p, c in zip(tp, tc) if p >= gate.lam and not c)
        rates.append(acc_err / len(tp))
    # the bound is in expectation over the calibration draw
    assert sum(rates) / len(rates) <= target + 0.01


def test_risk_coverage_curve_coverage_is_monotone():
    rng = random.Random(2)
    correct = [rng.random() < 0.7 for _ in range(300)]
    probs = [min(1.0, max(0.0, rng.gauss(0.7 if c else 0.4, 0.15))) for c in correct]
    curve = risk_coverage_curve(probs, correct)
    coverages = [row["coverage"] for row in curve]
    assert coverages == sorted(coverages, reverse=True)  # higher threshold -> less coverage
    assert all(0.0 <= row["selective_risk"] <= 1.0 for row in curve)


def test_group_conformal_gate_thresholds_differ_and_route():
    # 'tight' group: correct items are very high-confidence; 'loose' group: lower.
    # Class-conditional calibration must produce DIFFERENT per-group thresholds, so
    # the same confidence routes differently by group.
    rng = random.Random(3)
    probs, correct, groups = [], [], []
    for g, mu in (("tight", 0.9), ("loose", 0.6)):
        for _ in range(400):
            c = rng.random() < 0.7
            probs.append(min(1.0, max(0.0, rng.gauss(mu if c else mu - 0.3, 0.05))))
            correct.append(c)
            groups.append(g)
    gate = GroupConformalGate(alpha=0.1).fit(probs, correct, groups=groups)
    q_tight = gate.by_group["tight"].q
    q_loose = gate.by_group["loose"].q
    assert q_tight < q_loose                              # tighter group -> stricter threshold
    # a confidence of 0.7 (nonconformity 0.3) lands between the two thresholds
    assert gate(0.7, group="tight") is Decision.FLAG     # atypical of tight-correct
    assert gate(0.7, group="loose") is Decision.ACCEPT   # typical of loose-correct
    assert gate(0.95, group="unseen") in (Decision.ACCEPT, Decision.FLAG)  # pooled fallback


def test_conformal_p_value_monotone_and_known_value():
    gate = ConformalGate(alpha=0.1)
    assert gate.p_value(0.5) == 1.0                       # unfit -> vacuous p-value
    gate.fit([0.5, 0.6, 0.7, 0.8, 0.9], [True] * 5)
    # nonconformity = 1 - conf, so higher confidence -> higher p-value
    assert gate.p_value(0.95) >= gate.p_value(0.5)
    # cal nonconformities = [0.1,0.2,0.3,0.4,0.5]; conf 0.7 -> s=0.3 -> (1+3)/(5+1)
    assert abs(gate.p_value(0.7) - 4 / 6) < 1e-9


def test_policies_resolve_from_registry():
    assert get("policies", "cost_sensitive") is CostSensitiveGate
    assert get("policies", "conformal") is ConformalGate
    assert get("policies", "risk_control") is RiskControlGate
    assert get("policies", "group_conformal") is GroupConformalGate


def test_split_conformal_quantile_validates_alpha():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    for bad in (-0.1, 1.2, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            split_conformal_quantile(scores, bad)
    # alpha -> 1 collapses below every score (flag everything)
    assert split_conformal_quantile(scores, 1.0) == -math.inf


def test_conformal_gate_ignores_nan_in_calibration():
    # A NaN confidence among correct items must not corrupt the threshold.
    clean = ConformalGate(alpha=0.1).fit([0.6, 0.7, 0.8, 0.9], [True, True, True, True])
    with_nan = ConformalGate(alpha=0.1).fit(
        [0.6, 0.7, float("nan"), 0.8, 0.9], [True, True, True, True, True]
    )
    assert with_nan.q == clean.q
    assert with_nan(0.95) is Decision.ACCEPT


def test_gates_reject_non_finite_confidence():
    for gate in (
        CostSensitiveGate(rho=9.0),
        ConformalGate().fit([0.8, 0.9], [True, True]),
        RiskControlGate().fit([0.8, 0.9], [True, True]),
    ):
        with pytest.raises(ValueError, match="finite"):
            gate(float("nan"))


def test_risk_control_loss_bound_makes_gate_more_conservative():
    # A loss with range > 1 needs loss_bound to keep the CRC slack correct; a larger
    # bound -> a larger slack -> a stricter (higher) accept threshold on the same data.
    rng = random.Random(5)
    correct = [rng.random() < 0.7 for _ in range(500)]
    probs = [min(1.0, max(0.0, rng.gauss(0.78 if c else 0.45, 0.13))) for c in correct]
    loss = lambda c: 0.0 if c else 5.0  # noqa: E731
    g1 = RiskControlGate(target=0.05, loss=loss, loss_bound=1.0).fit(probs, correct)
    g5 = RiskControlGate(target=0.05, loss=loss, loss_bound=5.0).fit(probs, correct)
    assert g5.lam >= g1.lam
