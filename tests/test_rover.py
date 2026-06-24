"""Tests for the ROVER multi-engine consensus engine (#3)."""

import math
from dataclasses import dataclass
from typing import Optional

from ek.base import Signal
from ek.qe.rover import AgreementSignal, rover


@dataclass
class FakeOcr:
    """Minimal OcrResult-shaped object (duck-typed by ROVER)."""

    text: str
    mean_confidence: Optional[float] = None
    blocks: tuple = ()


def test_majority_consensus_and_agreement():
    c = rover(["the cat sat", "the cat sit", "the bat sat"])
    assert c.text == "the cat sat"
    assert c.n_engines == 3
    # position 0 unanimous; positions 1 and 2 split 2-1.
    assert c.agreement[0] == 1.0
    assert math.isclose(c.agreement[1], 2 / 3)
    assert math.isclose(c.agreement[2], 2 / 3)
    assert math.isclose(c.mean_agreement, (1 + 2 / 3 + 2 / 3) / 3)


def test_single_hypothesis_is_full_agreement():
    c = rover(["alpha beta gamma"])
    assert c.text == "alpha beta gamma"
    assert c.agreement == [1.0, 1.0, 1.0]
    assert c.mean_agreement == 1.0


def test_empty_hypotheses():
    c = rover([])
    assert c.n_engines == 0
    assert c.tokens == []
    assert c.mean_agreement == 1.0  # vacuous: no disagreement observed


def test_minority_insertion_is_dropped():
    # 'x' appears in only one of three engines -> its slot's winner is NULL.
    c = rover(["a b c", "a x b c", "a b c"])
    assert c.text == "a b c"
    # there is a slot whose winner is None (the dropped insertion)
    assert any(s.winner is None for s in c.slots)


def test_confidence_breaks_frequency_ties():
    # Two engines, one token each: equal frequency, confidence decides.
    hi_cat = rover([[("cat", 0.9)], [("dog", 0.2)]], use_confidence=True)
    assert hi_cat.text == "cat"
    assert math.isclose(hi_cat.agreement[0], 0.5)
    # Frequency-only: deterministic tie-break (first-seen real token).
    freq = rover([[("cat", 0.9)], [("dog", 0.2)]], use_confidence=False)
    assert freq.text == "cat"


def test_ocrresult_shape_is_duck_typed():
    hyps = [
        FakeOcr("the quick brown fox", mean_confidence=0.9),
        FakeOcr("the quick brown fox", mean_confidence=0.8),
        FakeOcr("the quick brawn fox", mean_confidence=0.4),
    ]
    c = rover(hyps)
    assert c.text == "the quick brown fox"
    assert all(0.0 <= a <= 1.0 for a in c.agreement)


def test_null_text_is_null_safe():
    # A VLM engine that returned nothing (text=None) must not crash.
    c = rover([FakeOcr(None), FakeOcr("hello world"), FakeOcr("hello world")])
    assert c.text == "hello world"


def test_agreement_signal_is_a_signal():
    sig = AgreementSignal()
    assert sig.cost_tier == 3
    assert isinstance(sig, Signal)  # structural: has cost_tier + __call__
    score = sig(["the cat sat", "the cat sit", "the bat sat"])
    assert math.isclose(score, (1 + 2 / 3 + 2 / 3) / 3)


def test_agreement_in_unit_interval_under_total_disagreement():
    c = rover(["alpha", "beta", "gamma"])
    # one slot, three different tokens -> winner share 1/3
    assert all(0.0 <= a <= 1.0 for a in c.agreement)
    assert math.isclose(c.mean_agreement, 1 / 3)


def test_null_conf_lever_controls_deletion():
    hyps = ["a b c", "a x b c", "a b c"]
    # high null_conf -> the NULL votes win the minority-insertion slot -> 'x' dropped
    assert rover(hyps, null_conf=0.7).text == "a b c"
    # null_conf=0 -> the lone real token 'x' outscores NULL -> kept
    assert "x" in rover(hyps, null_conf=0.0).text


def test_conf_weight_blends_frequency_and_confidence():
    # two low-confidence 'cat' votes vs one high-confidence 'dog' vote (one slot)
    hyps = [[("cat", 0.3)], [("cat", 0.3)], [("dog", 0.99)]]
    assert rover(hyps, conf_weight=0.0).text == "cat"   # frequency wins
    assert rover(hyps, conf_weight=1.0).text == "dog"   # confidence wins


def test_word_level_blocks_give_per_token_confidence():
    # An OcrResult whose blocks are word-level: ROVER should read per-block conf.
    @dataclass
    class Block:
        text: str
        confidence: float

    @dataclass
    class WordOcr:
        text: str
        blocks: tuple
        mean_confidence: float = None

    def words(s, confs):
        return WordOcr(text=s, blocks=tuple(Block(w, c) for w, c in zip(s.split(), confs)))

    # engine A and B agree on "the cat"; on the 3rd token A says 'sat' (conf .9),
    # B says 'sit' (conf .2); a low-confidence dissent.
    a = words("the cat sat", [0.95, 0.95, 0.9])
    b = words("the cat sit", [0.95, 0.95, 0.2])
    c = rover([a, b], use_confidence=True, conf_weight=1.0)
    assert c.tokens[:2] == ["the", "cat"]
    assert c.tokens[2] == "sat"   # higher per-token confidence wins the slot


def test_order_dependence_does_not_crash():
    import itertools

    base = ["a b c", "a c", "b c"]
    texts = {rover(list(p)).text for p in itertools.permutations(base)}
    # ROVER's incremental alignment is order-dependent (documented); assert only
    # that every ordering yields a valid consensus string without crashing.
    assert all(isinstance(t, str) for t in texts)


def test_max_tokens_guards_the_quadratic_aligner():
    import pytest

    long_hyp = " ".join(["a"] * 50)
    with pytest.raises(ValueError, match="max_tokens"):
        rover([long_hyp, long_hyp], max_tokens=10)
    # raising the cap (or disabling it) lets it through
    assert rover([long_hyp, long_hyp], max_tokens=100).n_engines == 2
    assert rover([long_hyp], max_tokens=None).n_engines == 1


def test_empty_consensus_is_zero_agreement_for_multiple_engines():
    # Regression: an empty consensus (engines agreed on nothing) reported 1.0
    # (maximal confidence). With 2+ engines it must be 0.0; a lone engine is 1.0.
    from ek.qe.rover import RoverConsensus

    assert RoverConsensus(tokens=[], agreement=[], n_engines=3).mean_agreement == 0.0
    assert RoverConsensus(tokens=[], agreement=[], n_engines=1).mean_agreement == 1.0
