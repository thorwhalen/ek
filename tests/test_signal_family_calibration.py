"""Per-signal-family calibration in estimate_quality() (#29).

Heterogeneous raw signals (a judge score, a logprob) must each be calibrated by
their own calibrator before being fused, not averaged raw and calibrated once.
"""

import pytest

from ek.base import (
    AnnotatedExtraction,
    FieldEstimate,
    FieldSpec,
    GraphGrammar,
    NodePath,
    NodeType,
)
from ek.facade import estimate_quality
from ek.qe.calibrate import (
    GroupCalibrator,
    PlattCalibrator,
    SignalFamilyCalibrator,
    load_calibrator,
    mean_probability,
    min_probability,
    save_calibrator,
)
from ek.qe.decide import CostSensitiveGate


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


def _const(p):
    """A 'calibrator' that ignores its input and returns ``p`` (to trace routing)."""
    return lambda raw: p


def _judge(_target):
    return 0.9


_judge.name = "judge"


def _logprob(_target):
    return -0.2


_logprob.name = "logprob"


def test_mapping_calibrates_each_family_then_fuses_by_mean():
    report = estimate_quality(
        "answer",
        signals=[_judge, _logprob],
        calibrator={"judge": _const(0.8), "logprob": _const(0.4)},
    )
    assert report.provenance["calibrated_signals"] == {"judge": 0.8, "logprob": 0.4}
    assert report.calibrated_confidence == pytest.approx(0.6)
    assert report.provenance["calibrated"] is True
    # the raw signals are still reported untouched
    assert report.raw_signals == {"judge": 0.9, "logprob": -0.2}


def test_each_family_sees_only_its_own_raw_value():
    seen = {}

    def recording(name):
        def cal(raw):
            seen[name] = raw
            return 0.5

        return cal

    estimate_quality(
        "x",
        signals=[_judge, _logprob],
        calibrator={"judge": recording("judge"), "logprob": recording("logprob")},
    )
    assert seen == {"judge": 0.9, "logprob": -0.2}


def test_min_combiner_is_conservative():
    cal = SignalFamilyCalibrator(
        calibrators={"judge": _const(0.8), "logprob": _const(0.4)},
        combine=min_probability,
    )
    report = estimate_quality("x", signals=[_judge, _logprob], calibrator=cal)
    assert report.calibrated_confidence == 0.4


def test_star_entry_is_the_fallback():
    report = estimate_quality(
        "x",
        signals=[_judge, _logprob],
        calibrator={"judge": _const(1.0), "*": _const(0.0)},
    )
    assert report.provenance["calibrated_signals"] == {"judge": 1.0, "logprob": 0.0}


def test_uncovered_signal_is_refused_not_dropped():
    with pytest.raises(ValueError, match="logprob"):
        estimate_quality(
            "x", signals=[_judge, _logprob], calibrator={"judge": _const(0.8)}
        )


def test_policy_gates_on_the_fused_calibrated_value():
    # Hard Rule 1 is satisfied by a family calibrator (no assume_calibrated needed).
    report = estimate_quality(
        "x",
        signals=[_judge, _logprob],
        calibrator={"judge": _const(0.9), "logprob": _const(0.1)},
        policy=CostSensitiveGate(rho=9.0),  # tau = 0.9 -> 0.5 is not accepted
    )
    assert report.calibrated_confidence == pytest.approx(0.5)
    assert report.decision is not None
    assert report.decision.value != "accept"


def test_own_confidence_is_the_confidence_family():
    fe = FieldEstimate(value="x", confidence=0.7, raw_signals={"judge": 0.2})
    report = estimate_quality(
        fe, calibrator={"confidence": _const(0.66), "judge": _const(0.0)}
    )
    assert report.calibrated_confidence == 0.66
    assert report.provenance["calibrated_signals"] == {"confidence": 0.66}


def test_no_signals_means_no_confidence():
    report = estimate_quality("x", calibrator={"judge": _const(0.8)})
    assert report.calibrated_confidence is None
    assert report.provenance["calibrated"] is False


def test_single_calibrator_behaviour_unchanged():
    report = estimate_quality(
        "x", signals=[_judge, _logprob], calibrator=lambda raw: raw
    )
    # one calibrator over the mean of raw signals, as before
    assert report.calibrated_confidence == pytest.approx((0.9 - 0.2) / 2)
    assert "calibrated_signals" not in report.provenance


def test_annotated_extraction_per_field_family_calibration():
    grammar = _invoice_grammar()
    card = NodePath("inv1", "invoice", "card")
    amount = NodePath("inv1", "invoice", "amount")
    ext = AnnotatedExtraction(
        grammar=grammar,
        estimates={
            card: FieldEstimate(
                value="ok", raw_signals={"judge": 0.9, "logprob": -0.1}
            ),
            amount: FieldEstimate(value="42", raw_signals={"judge": 0.3}),
        },
    )
    report = estimate_quality(
        ext, calibrator={"judge": lambda r: r, "logprob": _const(0.5)}
    )
    assert report.per_field[card].confidence == pytest.approx(0.7)
    assert report.per_field[amount].confidence == pytest.approx(0.3)
    assert report.provenance["calibrated_signals"][card] == {
        "judge": 0.9,
        "logprob": 0.5,
    }
    assert report.calibrated_confidence == pytest.approx(0.3)


def test_group_aware_family_calibrator_receives_the_node_type():
    groups = []

    def mondrian(raw, *, group=None):
        groups.append(group)
        return 0.5

    ext = AnnotatedExtraction(
        grammar=_invoice_grammar(),
        estimates={
            NodePath("inv1", "invoice", "card"): FieldEstimate(
                value="ok", raw_signals={"judge": 0.9}
            )
        },
    )
    estimate_quality(ext, calibrator={"judge": mondrian})
    assert groups == ["invoice"]


# --- fitting and persistence ------------------------------------------------

_ROWS = [
    {"judge": 0.9, "logprob": -0.1},
    {"judge": 0.8, "logprob": -0.2},
    {"judge": 0.85, "logprob": -2.0},
    {"judge": 0.3, "logprob": -0.1},
    {"judge": 0.2, "logprob": -2.5},
    {"judge": 0.7, "logprob": -1.8},
    {"judge": 0.95},  # a row missing a family only trains the families it has
]
_CORRECT = [True, True, False, True, False, False, True]


def test_fit_one_calibrator_per_family():
    cal = SignalFamilyCalibrator().fit(_ROWS, _CORRECT)
    assert set(cal.calibrators) == {"judge", "logprob"}
    assert all(isinstance(c, PlattCalibrator) for c in cal.calibrators.values())
    p = cal({"judge": 0.9, "logprob": -0.1})
    assert 0.0 <= p <= 1.0


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="labels"):
        SignalFamilyCalibrator().fit(_ROWS, _CORRECT[:-1])


def test_round_trip_through_the_store(tmp_path):
    cal = SignalFamilyCalibrator(combine=min_probability).fit(_ROWS, _CORRECT)
    save_calibrator(cal, "agent_qe", rootdir=str(tmp_path))
    loaded = load_calibrator("agent_qe", rootdir=str(tmp_path))
    assert isinstance(loaded, SignalFamilyCalibrator)
    assert loaded.combine is min_probability
    x = {"judge": 0.6, "logprob": -1.0}
    assert loaded(x) == pytest.approx(cal(x))


def test_round_trip_with_nested_group_calibrator():
    gc = GroupCalibrator().fit(
        [0.9, 0.2, 0.8, 0.1], [True, False, True, False], groups=["a", "a", "b", "b"]
    )
    cal = SignalFamilyCalibrator(calibrators={"judge": gc})
    loaded = SignalFamilyCalibrator.from_dict(cal.to_dict())
    assert isinstance(loaded.calibrators["judge"], GroupCalibrator)


def test_custom_combiner_works_but_cannot_be_persisted():
    stacker = lambda probs: max(probs.values())  # noqa: E731
    cal = SignalFamilyCalibrator(
        calibrators={"judge": _const(0.2), "logprob": _const(0.9)}, combine=stacker
    )
    assert cal({"judge": 0.0, "logprob": 0.0}) == 0.9
    with pytest.raises(ValueError, match="registered combiner"):
        cal.to_dict()


def test_combiners_reject_empty_input():
    with pytest.raises(ValueError):
        mean_probability({})
    with pytest.raises(ValueError):
        min_probability({})


def test_exported_from_top_level():
    import ek

    assert ek.SignalFamilyCalibrator is SignalFamilyCalibrator
    assert ek.mean_probability is mean_probability
