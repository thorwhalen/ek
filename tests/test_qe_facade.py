"""Tests for the estimate_quality() facade composing the QE pipeline (#3 + #5)."""

import math

import pytest

from ek.base import (
    AnnotatedExtraction,
    Decision,
    FieldEstimate,
    FieldSpec,
    Finding,
    GraphGrammar,
    NodePath,
    NodeType,
    Severity,
)
from ek.facade import estimate_quality
from ek.qe.calibrate import PlattCalibrator
from ek.qe.decide import CostSensitiveGate
from ek.qe.verifiers import checksum_validator, schema_validator


def test_simple_value_just_works():
    report = estimate_quality("hello")
    assert report.decision is None
    assert report.calibrated_confidence is None


def test_sources_trigger_rover_agreement():
    report = estimate_quality("the cat sat", sources=["the cat sit", "the bat sat"])
    assert "agreement" in report.raw_signals
    assert math.isclose(report.raw_signals["agreement"], (1 + 2 / 3 + 2 / 3) / 3)


def test_fieldestimate_calibrate_then_decide():
    fe = FieldEstimate(value="x", confidence=0.95)
    cal = lambda p: p  # noqa: E731  (an already-fit identity calibrator for the test)
    report = estimate_quality(fe, calibrator=cal, policy=CostSensitiveGate(rho=9.0))
    assert report.calibrated_confidence == 0.95
    assert report.decision is Decision.ACCEPT


def test_gating_without_calibrator_warns():
    with pytest.warns(UserWarning, match="uncalibrated"):
        estimate_quality(FieldEstimate(value="x", confidence=0.9), policy=CostSensitiveGate(rho=2.0))


def test_assume_calibrated_silences_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        report = estimate_quality(
            FieldEstimate(value="x", confidence=0.9),
            policy=CostSensitiveGate(rho=2.0),
            assume_calibrated=True,
        )
    assert report.decision is Decision.ACCEPT


def test_validators_produce_findings():
    report = estimate_quality("79927398710", validators=[checksum_validator("luhn")])
    assert len(report.findings) == 1
    assert report.findings[0].layer == "checksum"


def _invoice_grammar():
    return GraphGrammar(
        node_types={
            "invoice": NodeType(
                "invoice",
                fields={
                    "card": FieldSpec("card", "string"),
                    "amount": FieldSpec("amount", "number", domain=(0.0, 1000.0)),
                },
            )
        }
    )


def test_annotated_extraction_scored_per_field():
    grammar = _invoice_grammar()
    estimates = {
        NodePath("inv1", "invoice", "card"): FieldEstimate(value="ok", confidence=0.95),
        NodePath("inv1", "invoice", "amount"): FieldEstimate(value="9999", confidence=0.9),
    }
    ext = AnnotatedExtraction(grammar=grammar, estimates=estimates)
    report = estimate_quality(
        ext,
        validators=[schema_validator],
        policy=CostSensitiveGate(rho=2.0),  # tau = 0.5
        assume_calibrated=True,
    )
    assert report.per_field is not None and len(report.per_field) == 2
    amount_fe = report.per_field[NodePath("inv1", "invoice", "amount")]
    # amount 9999 is out of the [0,1000] domain -> a finding, and not auto-accepted
    assert any(f.field == "amount" for f in amount_fe.findings)
    assert amount_fe.decision is Decision.FLAG
    # summary: worst decision is FLAG, weakest confidence is the min
    assert report.decision is Decision.FLAG
    assert report.calibrated_confidence == 0.9


def test_annotated_extraction_with_rover_sources():
    grammar = _invoice_grammar()
    estimates = {
        NodePath("inv1", "invoice", "card"): FieldEstimate(value="ok", confidence=0.9),
    }
    ext = AnnotatedExtraction(grammar=grammar, estimates=estimates)
    # sources are the alternative full hypotheses ROVER fuses (>= 2 needed).
    report = estimate_quality(
        ext,
        sources=["Invoice total 1240", "Involce total 1240", "Invoice total 1240"],
        assume_calibrated=True,
    )
    assert "agreement" in report.raw_signals
    assert 0.0 <= report.raw_signals["agreement"] <= 1.0
    # the extraction-level agreement is propagated onto each field's raw_signals
    card_fe = report.per_field[NodePath("inv1", "invoice", "card")]
    assert "agreement" in card_fe.raw_signals


def test_per_field_validator_scoping_via_mapping():
    grammar = _invoice_grammar()
    estimates = {
        NodePath("inv1", "invoice", "card"): FieldEstimate(value="79927398710", confidence=0.99),
        NodePath("inv1", "invoice", "amount"): FieldEstimate(value="42", confidence=0.99),
    }
    ext = AnnotatedExtraction(grammar=grammar, estimates=estimates)
    # The luhn check is scoped to 'card' only -> it must NOT flag 'amount'.
    report = estimate_quality(
        ext,
        validators={"card": [checksum_validator("luhn")]},
        policy=CostSensitiveGate(rho=2.0),
        assume_calibrated=True,
    )
    card_fe = report.per_field[NodePath("inv1", "invoice", "card")]
    amount_fe = report.per_field[NodePath("inv1", "invoice", "amount")]
    assert card_fe.decision is Decision.FLAG       # bad luhn on the card
    assert amount_fe.decision is Decision.ACCEPT    # the card check did not touch it
    assert amount_fe.findings == ()


def test_verifier_hard_fail_forces_flag_despite_high_confidence():
    grammar = _invoice_grammar()
    estimates = {
        NodePath("inv1", "invoice", "card"): FieldEstimate(value="79927398710", confidence=0.99),
    }
    ext = AnnotatedExtraction(grammar=grammar, estimates=estimates)
    report = estimate_quality(
        ext,
        validators=[checksum_validator("luhn")],
        policy=CostSensitiveGate(rho=2.0),  # 0.99 >= tau=0.5 would ACCEPT...
        assume_calibrated=True,
    )
    card_fe = report.per_field[NodePath("inv1", "invoice", "card")]
    assert card_fe.decision is Decision.FLAG  # ...but the failed checksum forces a flag


def _always_flag(value, *, spec=None):
    yield Finding(field="card", layer="test", severity=Severity.FLAG, message="x")


def test_estimate_quality_is_idempotent_on_annotated_extraction():
    grammar = _invoice_grammar()
    estimates = {
        NodePath("inv1", "invoice", "card"): FieldEstimate(value="x", confidence=0.9),
    }
    ext = AnnotatedExtraction(grammar=grammar, estimates=estimates)
    r1 = estimate_quality(ext, validators=[_always_flag], assume_calibrated=True)
    r2 = estimate_quality(ext, validators=[_always_flag], assume_calibrated=True)
    # repeated runs must not accumulate findings...
    assert len(r1.findings) == len(r2.findings) == 1
    # ...and the input FieldEstimate must not be mutated.
    assert estimates[NodePath("inv1", "invoice", "card")].findings == ()
    assert estimates[NodePath("inv1", "invoice", "card")].decision is None


def test_block_decision_propagates_to_summary():
    grammar = _invoice_grammar()
    estimates = {
        NodePath("inv1", "invoice", "card"): FieldEstimate(value="ok", confidence=0.95),
        NodePath("inv1", "invoice", "amount"): FieldEstimate(value="0", confidence=0.02),
    }
    ext = AnnotatedExtraction(grammar=grammar, estimates=estimates)
    report = estimate_quality(
        ext,
        policy=CostSensitiveGate(rho=9.0, block_threshold=0.1),
        assume_calibrated=True,
    )
    amount_fe = report.per_field[NodePath("inv1", "invoice", "amount")]
    assert amount_fe.decision is Decision.BLOCK          # 0.02 falls in the block band
    assert report.decision is Decision.BLOCK              # worst-wins includes BLOCK


def test_callable_sources_raise_clear_error():
    with pytest.raises(TypeError, match="signals="):
        estimate_quality("x", sources=[lambda e: 0.5])


def test_signal_failure_is_recorded_not_swallowed():
    def boom(_target):
        raise RuntimeError("kaboom")

    with pytest.warns(UserWarning, match="kaboom"):
        report = estimate_quality("x", signals=[boom])
    assert report.provenance["failures"][0]["signal"] == "boom"


def test_failing_validator_is_recorded_not_swallowed():
    def bad_validator(value, *, spec=None):
        raise ValueError("regex blew up")
        yield  # pragma: no cover (makes it a generator)

    with pytest.warns(UserWarning, match="regex blew up"):
        report = estimate_quality("x", validators=[bad_validator])
    assert report.provenance["failures"][0]["validator"] == "bad_validator"


def test_callable_ocrresult_source_is_not_rejected():
    class CallableOcr:
        text = "the cat sat"

        def __call__(self):  # an OcrResult-shaped object that also happens to be callable
            return self.text

    obj = CallableOcr()
    report = estimate_quality("the cat sat", sources=[obj, "the cat sat"])
    assert "agreement" in report.raw_signals  # not rejected as a stray signal callable


def test_calibrator_is_applied_in_facade():
    # A pre-fit calibrator changes the gated confidence.
    scores = [0.9] * 10 + [0.5] * 10
    correct = [i < 5 for i in range(10)] + [i < 5 for i in range(10)]
    cal = PlattCalibrator().fit(scores, correct)
    fe = FieldEstimate(value="x", confidence=0.9)
    report = estimate_quality(fe, calibrator=cal)
    assert report.provenance["calibrated"] is True
    assert report.calibrated_confidence == cal(0.9)
