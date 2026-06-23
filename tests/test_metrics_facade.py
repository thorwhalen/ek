import math

import pytest

from ek.facade import evaluate, score
from ek.metrics.strings import StringMetric


def test_cer_default_for_strings():
    s = score("hello wrld", "hello world")
    assert s.metric == "cer"
    assert math.isclose(s.value, 1 / 11, rel_tol=1e-6)
    assert s.detail["edits"] == 1
    assert s.detail["ref_len"] == 11
    assert s.detail["higher_is_better"] is False


def test_wer_word_level():
    s = score("hello wrld", "hello world", metric="wer")
    assert math.isclose(s.value, 0.5, rel_tol=1e-6)  # 1 sub / 2 ref words


def test_corpus_cer_is_globally_accumulated_not_averaged():
    # Per-item CERs are 1/3 and 1/3; global = (1+1)/(3+3) = 1/3. Here they agree,
    # so use an asymmetric case to prove global != mean-of-rates.
    cases = [("xx", "x"), ("dg", "dog")]
    # item1: ref "x" (1 char), hyp "xx" -> 1 insertion, rate 1.0
    # item2: ref "dog" (3 chars), hyp "dg" -> 1 deletion, rate 1/3
    # mean of rates = (1.0 + 0.333)/2 = 0.667 ; global = (1+1)/(1+3) = 0.5
    report = evaluate(cases, metric="cer")
    assert report.n == 2
    assert math.isclose(report.aggregate, 0.5, rel_tol=1e-6)
    mean_of_rates = sum(s.value for s in report.scores) / 2
    assert not math.isclose(mean_of_rates, report.aggregate, rel_tol=1e-3)


def test_per_slice_reporting():
    cases = [
        ("cat", "cat", "easy"),
        ("xxx", "dog", "hard"),
        ("dog", "dog", "easy"),
    ]
    report = evaluate(cases, metric="cer")
    assert set(report.per_slice) == {"easy", "hard"}
    assert math.isclose(report.per_slice["easy"], 0.0, abs_tol=1e-9)
    assert report.per_slice["hard"] > 0


def test_normalize_canonicalizes_before_scoring():
    # Casefolding + whitespace collapse makes these identical.
    s = score("Hello   WORLD", "hello world", metric="cer", normalize="lower")
    # 'lower' alone leaves the double space; use a pipeline for full match
    assert s.value > 0
    s2 = score(
        "Hello   WORLD", "hello world", metric="cer",
        normalize=["lower", "collapse_whitespace"],
    )
    assert math.isclose(s2.value, 0.0, abs_tol=1e-9)


def test_field_metric_for_records():
    pred = {"name": "Acme", "city": "Paris", "amount": "100"}
    gold = {"name": "Acme", "city": "Paris", "amount": "200"}
    s = score(pred, gold)
    assert s.metric == "fields"
    # 2 correct (name, city), 1 wrong (amount) -> tp=2, fp=1, fn=1
    assert s.detail == {"tp": 2, "fp": 1, "fn": 1, "higher_is_better": True}
    assert math.isclose(s.precision, 2 / 3, rel_tol=1e-6)
    assert math.isclose(s.recall, 2 / 3, rel_tol=1e-6)


def test_field_metric_micro_aggregation():
    cases = [
        ({"a": "1", "b": "2"}, {"a": "1", "b": "2"}),  # 2 tp
        ({"a": "9"}, {"a": "1", "b": "2"}),  # a wrong (fp+fn), b missing (fn)
    ]
    report = evaluate(cases)
    # case2: 'a' wrong (fp+fn), 'b' missing (fn). totals: tp=2, fp=1, fn=2.
    # P = 2/3, R = 2/4, micro-F1 = 2PR/(P+R) -- not the mean of per-record F1s.
    p, r = 2 / 3, 2 / 4
    expected = 2 * p * r / (p + r)
    assert math.isclose(report.aggregate, expected, rel_tol=1e-6)


def test_empty_reference_is_pure_insertions():
    s = score("spurious", "", metric="cer")
    assert s.detail["ref_len"] == 0
    assert s.detail["edits"] == len("spurious")


def test_unknown_metric_raises_with_available_names():
    with pytest.raises(KeyError):
        score("a", "b", metric="no_such_metric")


def test_string_metric_aggregate_handles_empty():
    assert math.isnan(StringMetric(mode="cer").aggregate([]))


def test_no_default_metric_for_incompatible_types():
    with pytest.raises(TypeError):
        score(123, ["a", "b"])
