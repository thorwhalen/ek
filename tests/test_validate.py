"""Tests for the post-extraction validation & correction pipeline (#7)."""

import pytest

from ek.base import FieldSpec, Severity
from ek.validate import (
    ValidationResult,
    benford_findings,
    canonicalize_corrector,
    checksum_validator,
    cross_field_validator,
    lexicon_corrector,
    ordering_validator,
    stop_on_correction,
    stop_on_flag,
    validate,
    validation_pipeline,
)


# --- the FLAG vs CORRECT distinction ----------------------------------------------


def test_lexicon_corrects_closed_set_typo():
    fix = validation_pipeline(lexicon_corrector(["France", "Germany", "Spain"]))
    r = fix("Frnace")  # a transposition of a closed-enum member
    assert isinstance(r, ValidationResult)
    assert r.value == "France" and r.corrected
    assert r.findings[0].severity is Severity.CORRECT
    assert r.findings[0].suggestion == "France"


def test_lexicon_flags_out_of_vocabulary():
    fix = validation_pipeline(lexicon_corrector(["France", "Germany", "Spain"]))
    r = fix("Atlantis")
    assert not r.corrected and r.flagged
    assert r.findings[0].severity is Severity.FLAG


def test_lexicon_exact_member_is_clean():
    fix = validation_pipeline(lexicon_corrector(["France", "Germany"]))
    r = fix("France")
    assert r.clean and not r.corrected and not r.flagged


def test_lexicon_open_vocabulary_does_not_flag():
    fix = validation_pipeline(lexicon_corrector(["France"], flag_unmatched=False))
    r = fix("Atlantis")
    assert r.clean  # open set: no match, but not flagged either


# --- the noisy-channel chain (corrections feed the next layer) --------------------


def test_corrections_chain_canonicalize_then_lexicon():
    chain = validation_pipeline(
        canonicalize_corrector("lower"),
        lexicon_corrector(["france", "germany"]),
    )
    r = chain("FRNACE")  # canonicalized to 'frnace', then matched to 'france'
    assert r.value == "france"
    assert [f.layer for f in r.findings] == ["canonicalize", "lexicon"]
    assert all(f.severity is Severity.CORRECT for f in r.findings)


def test_apply_corrections_false_keeps_original_value():
    chain = validation_pipeline(
        lexicon_corrector(["France"]), apply_corrections=False
    )
    r = chain("Frnace")
    assert r.value == "Frnace" and not r.corrected  # findings recorded, value untouched
    assert r.findings[0].suggestion == "France"


# --- stop-early policies ----------------------------------------------------------


def test_stop_on_correction_short_circuits_later_layers():
    hits = []

    def spy(value, *, spec=None):
        hits.append(value)
        return []

    validation_pipeline(
        lexicon_corrector(["France"]), spy, stop_when=stop_on_correction
    )("Frnace")
    assert hits == []  # the later layer never ran


def test_stop_on_flag_fails_fast():
    hits = []

    def spy(value, *, spec=None):
        hits.append(value)
        return []

    validation_pipeline(
        lexicon_corrector(["France"]), spy, stop_when=stop_on_flag
    )("Atlantis")
    assert hits == []


def test_default_runs_every_layer():
    hits = []

    def spy(value, *, spec=None):
        hits.append(value)
        return []

    validation_pipeline(lexicon_corrector(["France"]), spy)("Atlantis")
    assert hits == ["Atlantis"]  # no stop policy -> later layer still runs


# --- composing with the deterministic verifier layer (re-exported) ----------------


def test_verifier_validator_composes_in_pipeline():
    pipe = validation_pipeline(checksum_validator("luhn"))
    assert pipe("79927398713").clean  # valid Luhn
    bad = pipe("79927398710")
    assert bad.flagged and bad.findings[0].layer == "checksum"


def test_field_name_comes_from_spec():
    pipe = validation_pipeline(lexicon_corrector(["France"]))
    r = pipe("Atlantis", spec=FieldSpec(name="country", type="string"))
    assert r.findings[0].field == "country"


# --- cross-field consistency ------------------------------------------------------


def test_ordering_validator_flags_descending():
    ov = ordering_validator(["start", "end"])
    assert list(ov({"start": 1, "end": 5})) == []
    findings = list(ov({"start": 5, "end": 1}))
    assert findings and findings[0].severity is Severity.FLAG


def test_ordering_validator_safe_on_partial_record():
    ov = ordering_validator(["start", "end"])
    assert list(ov({"start": 1})) == []  # only one key present -> nothing to order


def test_cross_field_validator_predicate():
    received_le_pledged = cross_field_validator(
        lambda r: r["received"] <= r["pledged"],
        message="received exceeds pledged",
        fields=("received", "pledged"),
    )
    assert list(received_le_pledged({"received": 5, "pledged": 10})) == []
    bad = list(received_le_pledged({"received": 20, "pledged": 10}))
    assert bad and "exceeds" in bad[0].message
    # missing field -> predicate raises -> skipped (not a cross-field violation)
    assert list(received_le_pledged({"received": 5})) == []


# --- statistical anomaly (Benford) ------------------------------------------------


def test_benford_flags_unnatural_distribution():
    # All values in [100, 300): leading digits 1 and 2 dominate -> not Benford.
    findings = benford_findings(list(range(100, 300)), field="amount")
    assert findings and findings[0].severity is Severity.FLAG
    assert findings[0].layer == "anomaly"


def test_benford_passes_benford_like_distribution():
    # Magnitudes spread log-uniformly across decades approximate Benford.
    import random

    rng = random.Random(0)
    values = [10 ** (rng.random() * 5) for _ in range(500)]
    assert benford_findings(values, field="amount") == []


def test_benford_skips_small_samples():
    assert benford_findings([100, 200, 300], field="x") == []  # below min_n


def test_benford_ignores_non_numeric_and_zero():
    vals = [None, "x", 0, float("inf")] + list(range(100, 300))
    findings = benford_findings(vals, field="amount")
    assert findings  # the junk is dropped, the real values still flag


# --- facade -----------------------------------------------------------------------


def test_validate_facade():
    r = validate("Frnace", layers=[lexicon_corrector(["France"])])
    assert r.value == "France"


def test_public_api_exported_from_top_level():
    import ek

    assert ek.validation_pipeline is validation_pipeline
    assert ek.lexicon_corrector is lexicon_corrector
    # the module is NOT shadowed by a function of the same name
    assert ek.validate.__name__ == "ek.validate"
