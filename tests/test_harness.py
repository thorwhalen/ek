import math

from ek.harness import (
    cohen_kappa,
    evaluate_store,
    load_baseline,
    percent_agreement,
    regression_gate,
    save_baseline,
)


def _gold():
    return {
        "d1": {"input": "cat", "reference": "cat", "slice": "easy"},
        "d2": {"input": "dog", "reference": "dog", "slice": "easy"},
        "d3": {"input": "xxxx", "reference": "bird", "slice": "hard"},
    }


def test_evaluate_store_runs_predictor_per_slice():
    report = evaluate_store(lambda x: x, _gold(), metric="cer")
    assert report.n == 3
    assert set(report.per_slice) == {"easy", "hard"}
    assert report.per_slice["easy"] == 0.0  # echo predictor is perfect on d1,d2
    assert report.per_slice["hard"] > 0
    assert report.detail["per_item"]["d1"]["score"] == 0.0


def test_evaluate_store_persists(tmp_path):
    evaluate_store(lambda x: x, _gold(), metric="cer", persist=True, run_id="r1", rootdir=str(tmp_path))
    from ek.stores import json_store

    assert "r1" in json_store("runs", rootdir=str(tmp_path))


def test_regression_gate_passes_when_no_baseline(tmp_path):
    report = evaluate_store(lambda x: x, _gold(), metric="cer")
    gate = regression_gate(report, "never-saved", rootdir=str(tmp_path))
    assert gate.passed  # first run: nothing to regress against


def test_regression_gate_detects_per_slice_regression(tmp_path):
    # Baseline: a perfect echo predictor.
    base_report = evaluate_store(lambda x: x, _gold(), metric="cer")
    save_baseline(base_report, "v1", rootdir=str(tmp_path))
    assert load_baseline("v1", rootdir=str(tmp_path))["aggregate"] == base_report.aggregate

    # New run: predictor that mangles the 'easy' slice (returns empty for cat/dog).
    def worse(x):
        return "" if x in ("cat", "dog") else x

    new_report = evaluate_store(worse, _gold(), metric="cer")
    gate = regression_gate(new_report, "v1", rootdir=str(tmp_path))
    assert not gate.passed
    assert "easy" in gate.regressions  # CER rose on the easy slice (lower-is-better)
    assert gate.higher_is_better is False  # cer is an error rate


def test_regression_gate_higher_is_better_for_f1(tmp_path):
    gold = {
        "a": {"input": {"x": "1"}, "reference": {"x": "1"}},
        "b": {"input": {"x": "1"}, "reference": {"x": "1"}},
    }
    base = evaluate_store(lambda d: d, gold, metric="fields")  # perfect F1=1.0
    save_baseline(base, "f1", rootdir=str(tmp_path))
    worse = evaluate_store(lambda d: {"x": "WRONG"}, gold, metric="fields")  # F1 drops
    gate = regression_gate(worse, "f1", rootdir=str(tmp_path))
    assert gate.higher_is_better is True
    assert not gate.passed


def test_regression_gate_tolerance(tmp_path):
    base = evaluate_store(lambda x: x, _gold(), metric="cer")
    save_baseline(base, "tol", rootdir=str(tmp_path))

    def slightly_worse(x):
        return x[:-1] if x == "cat" else x  # drop a char from one easy item

    rep = evaluate_store(slightly_worse, _gold(), metric="cer")
    # A generous tolerance absorbs the small drift...
    assert regression_gate(rep, "tol", tolerance=1.0, rootdir=str(tmp_path)).passed
    # ...but zero tolerance flags it.
    assert not regression_gate(rep, "tol", tolerance=0.0, rootdir=str(tmp_path)).passed


def test_cohen_kappa_and_percent_agreement():
    a = ["yes", "yes", "no", "no"]
    b = ["yes", "yes", "no", "no"]
    assert percent_agreement(a, b) == 1.0
    assert cohen_kappa(a, b) == 1.0
    # total disagreement on a balanced 2-class problem -> kappa < 0
    c = ["yes", "no", "yes", "no"]
    d = ["no", "yes", "no", "yes"]
    assert percent_agreement(c, d) == 0.0
    assert cohen_kappa(c, d) < 0
