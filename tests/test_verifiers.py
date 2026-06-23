"""Tests for the deterministic verifier signal layer (#5, cost tier 1)."""

from ek.base import FieldSpec, Severity
from ek.qe.verifiers import (
    VerifierSignal,
    checksum_validator,
    enum_validator,
    iban_check,
    isbn10_check,
    isbn13_check,
    isbn_check,
    luhn_check,
    range_validator,
    regex_validator,
    schema_validator,
    totals_consistent,
)


def test_luhn_checksum():
    assert luhn_check("79927398713") is True   # classic Luhn-valid
    assert luhn_check("4111111111111111") is True  # Visa test number
    assert luhn_check("79927398710") is False
    assert luhn_check("") is False


def test_iban_checksum():
    assert iban_check("GB82 WEST 1234 5698 7654 32") is True
    assert iban_check("GB82WEST12345698765432") is True
    assert iban_check("GB82WEST12345698765433") is False  # tampered last digit
    assert iban_check("nope") is False


def test_isbn_checksums():
    assert isbn10_check("0-306-40615-2") is True
    assert isbn13_check("978-0-306-40615-7") is True
    assert isbn_check("0306406152") is True          # dispatches to isbn10
    assert isbn_check("9780306406157") is True        # dispatches to isbn13
    assert isbn_check("0306406153") is False
    assert isbn_check("123") is False


def test_checksum_validator_yields_finding_only_on_failure():
    v = checksum_validator("luhn", field_name="card")
    assert list(v("79927398713")) == []                       # valid -> no finding
    findings = list(v("79927398710"))
    assert len(findings) == 1
    assert findings[0].layer == "checksum"
    assert findings[0].field == "card"
    assert findings[0].severity is Severity.FLAG


def test_regex_range_enum_validators():
    assert list(regex_validator(r"\d{4}")("2026")) == []
    assert len(list(regex_validator(r"\d{4}")("20x6"))) == 1

    assert list(range_validator(0, 100)("42")) == []
    assert len(list(range_validator(0, 100)("999"))) == 1
    assert len(list(range_validator(0, 100)("abc"))) == 1  # non-numeric flagged

    assert list(enum_validator(["A", "B"])("A")) == []
    assert len(list(enum_validator(["A", "B"])("C"))) == 1


def test_schema_validator_reads_fieldspec_domain():
    # numeric range domain
    spec = FieldSpec("amount", "number", domain=(0.0, 100.0))
    assert list(schema_validator("50", spec=spec)) == []
    assert len(list(schema_validator("150", spec=spec))) == 1
    # enum domain
    enum_spec = FieldSpec("status", "enum", domain=("open", "closed"))
    assert list(schema_validator("open", spec=enum_spec)) == []
    assert len(list(schema_validator("pending", spec=enum_spec))) == 1
    # regex domain (single-string domain)
    rx_spec = FieldSpec("zip", "string", domain=(r"\d{5}",))
    assert list(schema_validator("12345", spec=rx_spec)) == []
    assert len(list(schema_validator("abcde", spec=rx_spec))) == 1
    # no spec -> no-op
    assert list(schema_validator("anything", spec=None)) == []


def test_cross_field_totals_consistency():
    ok = {"total": 30, "a": 10, "b": 20}
    bad = {"total": 99, "a": 10, "b": 20}
    assert list(totals_consistent(ok, total_key="total", item_keys=["a", "b"])) == []
    findings = list(totals_consistent(bad, total_key="total", item_keys=["a", "b"]))
    assert len(findings) == 1
    assert findings[0].layer == "cross_field"
    # missing parts are skipped, not raised
    assert list(totals_consistent({"total": 5}, total_key="total", item_keys=["a"])) == []


def test_verifier_signal_fraction_and_findings():
    v = VerifierSignal([checksum_validator("luhn"), regex_validator(r"\d+")])
    assert v("79927398713") == 1.0           # both checks pass
    assert v("79927398710") == 0.5           # luhn fails, regex passes
    assert v("abc") == 0.0                    # luhn fails (no digits) and regex fails
    assert v.cost_tier == 1
    findings = v.findings("79927398710")
    assert len(findings) == 1
    # empty validator list -> trivially passing
    assert VerifierSignal([])("anything") == 1.0
