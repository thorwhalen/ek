"""Tests for span/slot F1 over seqeval AND nervaluate, with the explicit scheme."""

import math

import pytest

from ek import score
from ek.facade import evaluate
from ek.metrics.spans import MatchScheme, SpanF1Metric

pytest.importorskip("seqeval")
pytest.importorskip("nervaluate")


# --- the headline contract: the match scheme is REQUIRED, never defaulted --------


def test_scheme_is_required():
    with pytest.raises(TypeError) as exc:
        SpanF1Metric()
    assert "requires an explicit scheme" in str(exc.value)


def test_bad_tagging_scheme_rejected():
    with pytest.raises(ValueError):
        SpanF1Metric(scheme=MatchScheme.SEQEVAL_STRICT, tagging_scheme="NONSENSE")


# --- seqeval (tag-sequence) backend ----------------------------------------------


def test_seqeval_conll_perfect():
    m = SpanF1Metric(scheme=MatchScheme.SEQEVAL_CONLL)
    s = m(["B-PER", "I-PER", "O", "B-LOC"], ["B-PER", "I-PER", "O", "B-LOC"])
    assert s.f1 == 1.0
    assert s.detail["tp"] == 2 and s.detail["fp"] == 0 and s.detail["fn"] == 0


def test_seqeval_partial_span_is_a_full_miss_under_strict_span():
    # gold LOC is a 1-token span at index 3; pred misses it -> fn for LOC. PER matches.
    m = SpanF1Metric(scheme=MatchScheme.SEQEVAL_CONLL)
    s = m(["B-PER", "I-PER", "O", "O"], ["B-PER", "I-PER", "O", "B-LOC"])
    assert s.detail == {
        "scheme": "seqeval_conll",
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "higher_is_better": True,
        "backend": "seqeval",
    }
    assert math.isclose(s.f1, 2 / 3, rel_tol=1e-6)


def test_seqeval_micro_aggregation_not_mean_of_f1():
    m = SpanF1Metric(scheme=MatchScheme.SEQEVAL_CONLL)
    cases = [
        (["B-PER"], ["B-PER"]),  # tp=1
        (["B-LOC", "O"], ["B-PER", "B-LOC"]),  # pred LOC wrong span/type vs gold
    ]
    report = evaluate(cases, metric=m)
    # Verify it pools counts globally rather than averaging per-doc F1.
    total_tp = sum(s.detail["tp"] for s in report.scores)
    total_fp = sum(s.detail["fp"] for s in report.scores)
    total_fn = sum(s.detail["fn"] for s in report.scores)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro = 2 * p * r / (p + r) if (p + r) else 0.0
    assert math.isclose(report.aggregate, micro, rel_tol=1e-9)


# --- nervaluate (span-list) backend ----------------------------------------------


def _spans(*triples):
    return [{"label": lab, "start": s, "end": e} for lab, s, e in triples]


def test_nervaluate_partial_gives_half_credit():
    m = SpanF1Metric(scheme=MatchScheme.PARTIAL)
    gold = _spans(("PER", 0, 5))
    pred = _spans(("PER", 0, 3))  # overlapping but not exact boundary
    s = m(pred, gold)
    assert s.detail["backend"] == "nervaluate"
    assert s.detail["partial"] == 1
    # P = (0 + 0.5*1)/1 = 0.5, R = 0.5 -> F1 = 0.5
    assert math.isclose(s.value, 0.5, rel_tol=1e-9)


def test_strict_vs_partial_disagree_on_same_pair():
    gold = _spans(("ORG", 0, 16))  # "General Electric"
    pred = _spans(("ORG", 8, 16))  # "Electric" -- overlapping, wrong boundary
    strict = SpanF1Metric(scheme=MatchScheme.STRICT)(pred, gold).value
    partial = SpanF1Metric(scheme=MatchScheme.PARTIAL)(pred, gold).value
    assert strict == 0.0  # exact span + type: a miss
    assert partial > 0.0  # boundary overlap earns half credit
    assert partial != strict  # the schemes legitimately disagree


def test_nervaluate_micro_aggregation():
    m = SpanF1Metric(scheme=MatchScheme.STRICT)
    cases = [
        (_spans(("PER", 0, 5)), _spans(("PER", 0, 5))),  # correct
        (_spans(("LOC", 0, 3)), _spans(("LOC", 0, 5))),  # wrong boundary -> incorrect
    ]
    report = evaluate(cases, metric=m)
    cor = sum(s.detail["correct"] for s in report.scores)
    possible = sum(s.detail["possible"] for s in report.scores)
    actual = sum(s.detail["actual"] for s in report.scores)
    p = cor / actual
    r = cor / possible
    micro = 2 * p * r / (p + r) if (p + r) else 0.0
    assert math.isclose(report.aggregate, micro, rel_tol=1e-9)


def test_registered_names_cover_every_scheme():
    from ek import score as _score

    # Each scheme is registered under span_f1.<value> and reachable by name.
    s = _score(
        ["B-PER", "O"], ["B-PER", "O"], metric="span_f1.seqeval_conll"
    )
    assert s.f1 == 1.0
