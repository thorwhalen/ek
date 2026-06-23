"""Tests for the calibration stage (#5): Platt/isotonic, ECE, Mondrian, persistence."""

import tempfile

import pytest

from ek.base import Calibrator
from ek.qe.calibrate import (
    GroupCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    load_calibrator,
    reliability_curve,
    save_calibrator,
)


def _overconfident_dataset():
    """Monotone but overconfident: at score s, true accuracy is s - 0.2."""
    scores, correct = [], []
    for s, n_correct in [(0.5, 3), (0.6, 4), (0.7, 5), (0.8, 6), (0.9, 7)]:
        for i in range(10):
            scores.append(s)
            correct.append(i < n_correct)
    return scores, correct


def test_platt_is_monotonic_and_reduces_overconfidence():
    scores, correct = _overconfident_dataset()
    cal = PlattCalibrator().fit(scores, correct)
    assert cal(0.9) > cal(0.5)              # monotone increasing
    assert cal(0.9) < 0.9                   # overconfidence pulled down (true acc 0.7)
    # calibration improves ECE
    raw_ece = expected_calibration_error(scores, correct, n_bins=5)
    cal_ece = expected_calibration_error([cal(s) for s in scores], correct, n_bins=5)
    assert cal_ece < raw_ece


def test_isotonic_is_monotonic():
    scores, correct = _overconfident_dataset()
    cal = IsotonicCalibrator().fit(scores, correct)
    assert cal(0.5) <= cal(0.7) <= cal(0.9)
    assert 0.0 <= cal(0.6) <= 1.0


def test_isotonic_pools_tied_x_values():
    # Duplicate scores with mixed labels must pool to the empirical rate, not 0/1.
    cal = IsotonicCalibrator().fit([5, 5, 5, 5], [True, False, False, False])
    assert cal(5.0) == 0.25
    assert cal.x == [5.0]  # strictly-increasing breakpoints, no duplicate x
    # discrete multi-x case: per-x means 1/3, 2/3, 1.0 (already monotone)
    xs = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    ys = [True, False, False, True, True, False, True, True, True]
    cal2 = IsotonicCalibrator().fit(xs, ys)
    assert abs(cal2(1.0) - 1 / 3) < 1e-9
    assert abs(cal2(2.0) - 2 / 3) < 1e-9
    assert abs(cal2(3.0) - 1.0) < 1e-9


def test_isotonic_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.isotonic")
    xs = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    ys = [1, 0, 0, 1, 1, 0, 1, 1, 1]
    model = sk.IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
    cal = IsotonicCalibrator().fit(xs, [bool(y) for y in ys])
    for v in (1.0, 2.0, 3.0):
        assert abs(cal(v) - float(model.predict([v])[0])) < 1e-9


def test_ece_zero_when_perfectly_calibrated():
    # confidence 0.5 with exactly 50% correct -> bin gap 0.
    probs = [0.5] * 100
    correct = [True, False] * 50
    assert expected_calibration_error(probs, correct) == 0.0


def test_ece_maximal_when_confidently_wrong():
    probs = [0.95] * 50
    correct = [False] * 50
    assert abs(expected_calibration_error(probs, correct) - 0.95) < 1e-9


def test_reliability_curve_shape():
    scores, correct = _overconfident_dataset()
    curve = reliability_curve(scores, correct, n_bins=10)
    assert all({"confidence", "accuracy", "count"} <= set(row) for row in curve)
    assert sum(row["count"] for row in curve) == len(scores)


def test_calibrators_satisfy_protocol():
    assert isinstance(PlattCalibrator(), Calibrator)
    assert isinstance(IsotonicCalibrator(), Calibrator)


@pytest.mark.parametrize(
    "make, fit_x",
    [
        (PlattCalibrator, [0.2, 0.4, 0.6, 0.8, 0.9] * 4),
        (IsotonicCalibrator, [0.2, 0.4, 0.6, 0.8, 0.9] * 4),
        (TemperatureCalibrator, [-2.0, -1.0, 0.0, 1.0, 2.0] * 4),  # logits
    ],
)
def test_persistence_round_trip_all_kinds(make, fit_x):
    correct = [i % 2 == 0 for i in range(len(fit_x))]
    cal = make().fit(fit_x, correct)
    with tempfile.TemporaryDirectory() as root:
        save_calibrator(cal, "cal-v1", rootdir=root)
        loaded = load_calibrator("cal-v1", rootdir=root)
    assert type(loaded) is type(cal)  # the persisted `kind` dispatched correctly
    for s in fit_x[:5]:
        assert abs(loaded(s) - cal(s)) < 1e-12


def test_group_calibrator_routes_per_group_with_pooled_fallback():
    # Group 'a' overconfident, group 'b' underconfident; calibrate separately.
    scores = [0.9] * 20 + [0.2] * 20
    correct = [i < 10 for i in range(20)] + [i < 16 for i in range(20)]  # a: 0.5, b: 0.8
    groups = ["a"] * 20 + ["b"] * 20
    gc = GroupCalibrator().fit(scores, correct, groups=groups)
    # group a: raw 0.9 -> ~0.5 ; group b: raw 0.2 -> ~0.8
    assert gc(0.9, group="a") < 0.75
    assert gc(0.2, group="b") > 0.5
    # unseen group falls back to the pooled calibrator (does not raise)
    assert 0.0 <= gc(0.5, group="unseen") <= 1.0
