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


def test_calibrators_drop_non_finite_rows_in_fit():
    scores, correct = _overconfident_dataset()
    # inject a NaN and an inf score; the fit must ignore them, not crash/diverge.
    dirty_s = scores + [float("nan"), float("inf")]
    dirty_c = correct + [True, False]
    clean = PlattCalibrator().fit(scores, correct)
    dirty = PlattCalibrator().fit(dirty_s, dirty_c)
    assert abs(dirty(0.8) - clean(0.8)) < 1e-9
    iso = IsotonicCalibrator().fit(dirty_s, dirty_c)
    assert 0.0 <= iso(0.8) <= 1.0


def test_calibrator_raises_on_nan_input():
    cal = PlattCalibrator().fit(*_overconfident_dataset())
    with pytest.raises(ValueError, match="NaN"):
        cal(float("nan"))


def test_ece_validates_n_bins_and_skips_non_finite():
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0.5, 0.5], [True, False], n_bins=0)
    # a NaN prob is skipped, not binned
    assert expected_calibration_error([0.5, 0.5, float("nan")], [True, False, True]) == 0.0


def test_isotonic_matches_sklearn_off_breakpoints():
    sk = pytest.importorskip("sklearn.isotonic")
    xs = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    ys = [1, 0, 0, 1, 1, 0, 1, 1, 1]
    model = sk.IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
    cal = IsotonicCalibrator().fit(xs, [bool(y) for y in ys])
    for v in (1.0, 1.5, 2.0, 2.5, 3.0):  # includes off-breakpoint interpolation
        assert abs(cal(v) - float(model.predict([v])[0])) < 1e-9


def test_load_calibrator_validates_record():
    from ek.stores import json_store

    with tempfile.TemporaryDirectory() as root:
        store = json_store("calibrators", rootdir=root)
        store["no-kind"] = {"a": 1.0, "b": 0.0}
        store["bad-kind"] = {"kind": "nope"}
        with pytest.raises(ValueError, match="malformed"):
            load_calibrator("no-kind", rootdir=root)
        with pytest.raises(ValueError, match="unknown calibrator kind"):
            load_calibrator("bad-kind", rootdir=root)


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


def test_netcal_ece_runs_without_crashing():
    # Regression: netcal_ece passed Python lists; netcal>=1.4 requires ndarrays, so
    # it raised for every caller. (netcal bins by predicted-class confidence, a
    # different convention from the pure-Python positive-class ECE, so they need not
    # match numerically -- this just pins that it runs and returns a valid score.)
    pytest.importorskip("netcal")
    from ek.qe.calibrate import netcal_ece

    probs = [0.95, 0.9, 0.2, 0.6, 0.8, 0.3, 0.55, 0.99]
    correct = [True, True, False, True, True, False, False, True]
    nc = netcal_ece(probs, correct, bins=5)
    assert isinstance(nc, float) and 0.0 <= nc <= 1.0


def test_group_calibrator_round_trips():
    # Regression: the module told users to persist Mondrian calibrators, but
    # GroupCalibrator had no to_dict/from_dict and was unregistered.
    import tempfile

    gc = GroupCalibrator().fit(
        [0.2, 0.8, 0.3, 0.9], [False, True, False, True], groups=["a", "a", "b", "b"]
    )
    with tempfile.TemporaryDirectory() as d:
        from ek.qe.calibrate import load_calibrator, save_calibrator

        save_calibrator(gc, "g1", rootdir=d)
        gc2 = load_calibrator("g1", rootdir=d)
    assert isinstance(gc2, GroupCalibrator)
    assert abs(gc(0.7, group="a") - gc2(0.7, group="a")) < 1e-9
    assert abs(gc(0.7, group="b") - gc2(0.7, group="b")) < 1e-9
