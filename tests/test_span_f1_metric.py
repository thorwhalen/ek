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


def test_nervaluate_micro_aggregation_is_not_mean_of_f1():
    # Asymmetric docs so pooled micro-F1 genuinely diverges from the mean of per-doc
    # F1s (otherwise the test would pass even for a buggy mean-of-F1 aggregator).
    m = SpanF1Metric(scheme=MatchScheme.STRICT)
    cases = [
        # doc1: 4 gold, 2 predicted+correct -> P=1.0, R=0.5, f1=0.667
        (_spans(("A", 0, 1), ("B", 1, 2)),
         _spans(("A", 0, 1), ("B", 1, 2), ("C", 2, 3), ("D", 3, 4))),
        # doc2: 1 gold, 4 predicted (1 correct, 3 spurious) -> P=0.25, R=1.0, f1=0.4
        (_spans(("A", 0, 1), ("X", 5, 6), ("Y", 6, 7), ("Z", 7, 8)),
         _spans(("A", 0, 1))),
    ]
    report = evaluate(cases, metric=m)
    cor = sum(s.detail["correct"] for s in report.scores)
    possible = sum(s.detail["possible"] for s in report.scores)
    actual = sum(s.detail["actual"] for s in report.scores)
    p, r = cor / actual, cor / possible
    micro = 2 * p * r / (p + r) if (p + r) else 0.0
    mean_f1 = sum(s.f1 for s in report.scores) / len(report.scores)
    assert math.isclose(report.aggregate, micro, rel_tol=1e-9)
    assert not math.isclose(report.aggregate, mean_f1, rel_tol=1e-6)  # micro != mean


def test_seqeval_strict_differs_from_conll():
    # Regression: SEQEVAL_STRICT was a silent no-op alias of CONLL. A sequence valid
    # under lenient conll decoding but MALFORMED under strict IOB2 (an I- with no B-)
    # must score differently.
    pred, gold = ["I-PER", "I-PER", "O"], ["B-PER", "I-PER", "O"]
    conll = SpanF1Metric(MatchScheme.SEQEVAL_CONLL)(pred, gold).f1
    strict = SpanF1Metric(MatchScheme.SEQEVAL_STRICT)(pred, gold).f1
    assert conll == 1.0    # lenient: I-PER I-PER decodes as a PER span
    assert strict == 0.0   # strict IOB2: an I- with no preceding B- is not an entity
    assert conll != strict


def test_tagging_scheme_is_used_in_strict():
    # A valid IOBES span (B...E). Strict IOBES decodes it as one PER entity; the
    # tagging_scheme is genuinely consulted (it used to be dead code).
    pred = gold = ["B-PER", "E-PER", "O"]
    iobes = SpanF1Metric(MatchScheme.SEQEVAL_STRICT, tagging_scheme="IOBES")(pred, gold)
    assert iobes.f1 == 1.0 and iobes.detail["tp"] == 1
    # Under IOB2 the same tags are invalid (E- is not allowed), so strict decoding
    # finds NO entity -> a different result (tp collapses to 0).
    iob2 = SpanF1Metric(MatchScheme.SEQEVAL_STRICT, tagging_scheme="IOB2")(pred, gold)
    assert iob2.detail["tp"] == 0   # tagging_scheme changed the decoding


def test_type_scheme_does_not_crash():
    # Regression: MatchScheme.TYPE raised KeyError every call (key "type" vs nervaluate
    # "ent_type").
    gold = _spans(("PER", 0, 5))
    wrong_type = SpanF1Metric(MatchScheme.TYPE)(_spans(("LOC", 0, 5)), gold)
    assert wrong_type.detail["backend"] == "nervaluate"
    assert wrong_type.value == 0.0           # span overlaps but type wrong -> not credited
    right_type = SpanF1Metric(MatchScheme.TYPE)(_spans(("PER", 0, 3)), gold)
    assert right_type.value > 0.0            # right type + span overlap -> credited


def test_empty_span_lists_do_not_crash():
    # Regression: empty gold/pred raised IndexError inside nervaluate.
    assert SpanF1Metric(MatchScheme.STRICT)([], []).f1 == 1.0
    spurious = SpanF1Metric(MatchScheme.STRICT)(_spans(("PER", 0, 1)), [])
    assert spurious.f1 == 0.0 and spurious.detail["spurious"] == 1
    missed = SpanF1Metric(MatchScheme.STRICT)([], _spans(("PER", 0, 1)))
    assert missed.f1 == 0.0 and missed.detail["missed"] == 1


def test_mixed_backend_aggregation_raises():
    seq = SpanF1Metric(MatchScheme.SEQEVAL_CONLL)(["B-PER"], ["B-PER"])
    nerv = SpanF1Metric(MatchScheme.STRICT)(_spans(("PER", 0, 1)), _spans(("PER", 0, 1)))
    with pytest.raises(ValueError, match="mixed"):
        SpanF1Metric(MatchScheme.STRICT).aggregate([seq, nerv])


def test_registered_names_cover_every_scheme():
    from ek import score as _score

    # EVERY scheme is registered under span_f1.<value> and dispatches without error
    # (a spot-check of one scheme is what let span_f1.type's KeyError ship).
    tags = (["B-PER", "O"], ["B-PER", "O"])
    spans = (_spans(("PER", 0, 1)), _spans(("PER", 0, 1)))
    for scheme in MatchScheme:
        args = tags if scheme.value.startswith("seqeval") else spans
        s = _score(args[0], args[1], metric=f"span_f1.{scheme.value}")
        assert s.f1 == 1.0, f"span_f1.{scheme.value} did not dispatch cleanly"
