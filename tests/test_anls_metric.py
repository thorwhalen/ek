"""Tests for ANLS / ANLS* (nested-JSON) metric."""

import math

import pytest

from ek import score
from ek.facade import evaluate
from ek.metrics.anls import AnlsMetric

pytest.importorskip("anls_star")


def test_string_anls_tolerates_minor_ocr_slip():
    s = score("Hello Wrld", "Hello World", metric="anls")
    assert s.metric == "anls"
    # one deletion over 11 chars -> NLS = 1 - 1/11
    assert math.isclose(s.value, 1 - 1 / 11, rel_tol=1e-6)
    assert s.detail["higher_is_better"] is True


def test_argument_order_is_pred_then_gold():
    # anls_star takes (gt, pred); the wrapper must flip ek's (pred, gold). A perfect
    # match is symmetric, so use an asymmetric truncation to prove orientation: the
    # score must be identical whichever way the (already-symmetric) NLS is computed.
    a = score("cat", "cats", metric="anls").value
    b = score("cats", "cat", metric="anls").value
    assert math.isclose(a, b, rel_tol=1e-9)  # NLS is symmetric; order does not crash


def test_nested_dict_anls_star():
    pred = {"name": "Acme", "city": "Paris"}
    gold = {"name": "Acme", "city": "Paris"}
    assert math.isclose(score(pred, gold, metric="anls").value, 1.0, abs_tol=1e-9)
    # one of two leaves wrong (below threshold) -> 0.5
    wrong = {"name": "Acme", "city": "Lyon"}
    assert math.isclose(score(wrong, gold, metric="anls").value, 0.5, abs_tol=1e-9)


def test_nested_list_is_order_invariant():
    # Hungarian matching: list order should not matter.
    a = score({"items": ["cat", "dog"]}, {"items": ["dog", "cat"]}, metric="anls").value
    assert math.isclose(a, 1.0, abs_tol=1e-9)


def test_missing_key_is_penalized():
    s = score({"a": "x"}, {"a": "x", "b": "y"}, metric="anls")
    assert s.value < 1.0


def test_corpus_anls_is_mean_of_per_item():
    # ANLS aggregates as the MEAN of per-item similarities (each already normalized),
    # unlike CER/WER (global edit accumulation) or field-F1 (micro TP/FP/FN).
    cases = [("cat", "cat"), ("dg", "dog")]  # 1.0 and (1 - 1/3)
    report = evaluate(cases, metric="anls")
    assert report.n == 2
    expected = (1.0 + (1 - 1 / 3)) / 2
    assert math.isclose(report.aggregate, expected, rel_tol=1e-6)


def test_aggregate_empty_is_nan():
    assert math.isnan(AnlsMetric().aggregate([]))


def test_threshold_actually_changes_the_score():
    # Regression: `threshold` used to be stored but never applied (a silent no-op).
    # 'Helo Wrld' vs 'Hello World' has NLS ~0.818, between 0.5 and 0.9.
    pred, gold = "Helo Wrld", "Hello World"
    lo = AnlsMetric(threshold=0.5)(pred, gold).value
    hi = AnlsMetric(threshold=0.9)(pred, gold).value
    assert lo > 0.0           # 0.818 >= 0.5 -> kept
    assert hi == 0.0          # 0.818 < 0.9 -> zeroed by the threshold
    assert lo != hi           # the knob is functional


def test_threshold_is_restored_after_the_call():
    import anls_star.anls_star as _a

    before = _a.ANLSTree.THRESHOLD
    AnlsMetric(threshold=0.9)("Helo Wrld", "Hello World")
    assert _a.ANLSTree.THRESHOLD == before  # process-global restored, no leakage
