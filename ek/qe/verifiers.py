"""Deterministic verifier signals -- the cheapest, first-to-run QE layer.

Reference-free quality estimation should try the **free, label-free, deterministic
checks first** (``misc/docs/ek_03`` §1d, §5; ``ek_04`` Layers 0-3): they cost
nothing, need no training data, and catch exactly the errors a confidence signal
misses -- a value the extractor was *confident* about but that is structurally
impossible (a checksum that does not validate, a total that does not reconcile, an
enum value out of range). Only escalate to intrinsic confidence, agreement (ROVER),
or LLM signals on the residual that verifiers cannot settle.

The check primitives here (Luhn / IBAN / ISBN checksums, type/range/enum/regex,
cross-field reconciliation) are pure-Python and dependency-free *by design*: they
are reused both as reference-free :class:`~ek.base.Signal` evidence (this module,
the :func:`ek.estimate_quality` side) and -- later -- as the L1-L2 layers of the
flag-vs-correct validation pipeline (``ek/validate.py``, issue #7). Implementing the
checksums directly (rather than depending on the LGPL ``python-stdnum``) keeps the
permissive-core promise intact.

A verifier emits raw evidence -- a pass/fail :class:`~ek.base.Finding` and a
fraction-passed score -- never a calibrated probability or a decision.

Example:
    >>> luhn_check("79927398713")          # a classic Luhn-valid number
    True
    >>> luhn_check("79927398710")
    False
    >>> isbn_check("0-306-40615-2")         # ISBN-10
    True
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ..base import Finding, FieldSpec, Severity
from ..registry import register

#: Default absolute tolerance for the cross-field totals reconciliation check.
DEFAULT_TOTALS_TOL = 0.01

#: ASCII digits only -- ``str.isdigit()`` also accepts Unicode digits (superscripts,
#: Devanagari, ...) that ``int()`` then rejects or that corrupt a checksum.
_ASCII_DIGITS = "0123456789"

# ---------------------------------------------------------------------------
# Checksum primitives (pure-Python; reusable across modules -> no underscore)
# ---------------------------------------------------------------------------


def luhn_check(number: Any) -> bool:
    """Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs."""
    digits = [int(c) for c in str(number) if c in _ASCII_DIGITS]
    if len(digits) < 2:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_check(iban: Any) -> bool:
    """IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1)."""
    s = "".join(str(iban).split()).upper()
    # ASCII alphanumerics only (ord(c)-55 is meaningful only for ASCII A-Z).
    if len(s) < 5 or not all(c.isascii() and c.isalnum() for c in s):
        return False
    if not ("A" <= s[0] <= "Z" and "A" <= s[1] <= "Z"):
        return False
    if s[2] not in _ASCII_DIGITS or s[3] not in _ASCII_DIGITS:
        return False
    rearranged = s[4:] + s[:4]
    converted = "".join(str(ord(c) - 55) if "A" <= c <= "Z" else c for c in rearranged)
    return int(converted) % 97 == 1


def isbn10_check(value: Any) -> bool:
    """ISBN-10 checksum (weighted mod-11; a trailing ``X`` counts as 10)."""
    s = re.sub(r"[\s-]", "", str(value)).upper()
    if len(s) != 10:
        return False
    total = 0
    for i, c in enumerate(s):
        if c == "X" and i == 9:
            v = 10
        elif c in _ASCII_DIGITS:
            v = int(c)
        else:
            return False
        total += (10 - i) * v
    return total % 11 == 0


def isbn13_check(value: Any) -> bool:
    """ISBN-13 / EAN-13 checksum (alternating 1/3 weights, mod-10)."""
    s = re.sub(r"[\s-]", "", str(value))
    if len(s) != 13 or not all(c in _ASCII_DIGITS for c in s):
        return False
    total = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(s))
    return total % 10 == 0


def isbn_check(value: Any) -> bool:
    """Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored)."""
    digits = re.sub(r"[\s-]", "", str(value))
    if len(digits) == 10:
        return isbn10_check(value)
    if len(digits) == 13:
        return isbn13_check(value)
    return False


#: Built-in checksum checks, by name (register more under the ``checks`` namespace).
CHECKSUMS: dict[str, Callable[[Any], bool]] = {
    "luhn": luhn_check,
    "iban": iban_check,
    "isbn": isbn_check,
    "isbn10": isbn10_check,
    "isbn13": isbn13_check,
}
for _name, _fn in CHECKSUMS.items():
    register("checks", _name, _fn)


# ---------------------------------------------------------------------------
# Validator factories (each returns a Validator: value -> Iterable[Finding])
# ---------------------------------------------------------------------------


def checksum_validator(
    kind: str = "luhn", *, field_name: str = "", layer: str = "checksum"
) -> Callable[..., Iterable[Finding]]:
    """A :class:`~ek.base.Validator` that flags a value failing a named checksum."""
    check = CHECKSUMS[kind]

    def validate(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        name = field_name or (spec.name if spec is not None else "")
        if value is not None and not check(value):
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.FLAG,
                message=f"{kind} checksum failed for {value!r}",
            )

    return validate


def regex_validator(
    pattern: str, *, field_name: str = "", layer: str = "format"
) -> Callable[..., Iterable[Finding]]:
    """A :class:`~ek.base.Validator` that flags a value not fully matching ``pattern``."""
    rx = re.compile(pattern)

    def validate(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        name = field_name or (spec.name if spec is not None else "")
        if value is not None and rx.fullmatch(str(value)) is None:
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.FLAG,
                message=f"{value!r} does not match /{pattern}/",
            )

    return validate


def range_validator(
    lo: float, hi: float, *, field_name: str = "", layer: str = "range"
) -> Callable[..., Iterable[Finding]]:
    """A :class:`~ek.base.Validator` that flags a number outside ``[lo, hi]``."""

    def validate(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        name = field_name or (spec.name if spec is not None else "")
        try:
            x = float(value)
        except (TypeError, ValueError):
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.FLAG,
                message=f"{value!r} is not numeric",
            )
            return
        if not (lo <= x <= hi):
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.FLAG,
                message=f"{x} outside [{lo}, {hi}]",
            )

    return validate


def enum_validator(
    members: Sequence[Any], *, field_name: str = "", layer: str = "enum"
) -> Callable[..., Iterable[Finding]]:
    """A :class:`~ek.base.Validator` that flags a value not in an allowed set."""
    # A list (not a set) so unhashable members/values never raise; membership is
    # by ``==`` and an enum is small, so the linear scan is fine.
    allowed = list(members)

    def validate(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        name = field_name or (spec.name if spec is not None else "")
        try:
            inside = value in allowed
        except TypeError:  # an unhashable/uncomparable value is simply not a member
            inside = False
        if not inside:
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.FLAG,
                message=f"{value!r} not in {sorted(map(str, allowed))}",
            )

    return validate


def schema_validator(
    value: Any, *, spec: Optional[FieldSpec] = None
) -> Iterable[Finding]:
    """A spec-driven :class:`~ek.base.Validator`: type and ``domain`` checks from a FieldSpec.

    Reads the :class:`~ek.base.FieldSpec` ``type`` and ``domain`` -- a ``(lo, hi)``
    numeric range, an enum tuple, or a regex string -- and applies the matching
    check. With no spec it is a no-op, so it composes safely in any validator list.
    """
    if spec is None:
        return
    name = spec.name
    if spec.type == "number":
        try:
            float(value)
        except (TypeError, ValueError):
            yield Finding(
                field=name,
                layer="type",
                severity=Severity.FLAG,
                message=f"{value!r} is not a number",
            )
            return
    domain = spec.domain
    if not domain:
        return
    # A (lo, hi) numeric range vs an enum/regex domain.
    if (
        spec.type == "number"
        and len(domain) == 2
        and all(_is_number(d) for d in domain)
    ):
        yield from range_validator(domain[0], domain[1], field_name=name)(
            value, spec=spec
        )
    elif len(domain) == 1 and isinstance(domain[0], str):
        yield from regex_validator(domain[0], field_name=name)(value, spec=spec)
    else:
        yield from enum_validator(domain, field_name=name)(value, spec=spec)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Cross-field consistency (record-level; FLAG-only, never auto-corrects)
# ---------------------------------------------------------------------------


def totals_consistent(
    record: Mapping,
    *,
    total_key: str,
    item_keys: Sequence[str],
    tol: float = DEFAULT_TOTALS_TOL,
    layer: str = "cross_field",
) -> Iterable[Finding]:
    """Flag when ``record[total_key]`` does not equal the sum of ``item_keys`` (±tol).

    The canonical cross-field check (invoice total = Σ line items). Skipped (rather
    than raising or flagging) when the total or *all* item keys are absent or
    non-numeric, so it is safe on partial records.
    """
    present = [k for k in item_keys if k in record]
    if total_key not in record or not present:
        return
    try:
        total = float(record[total_key])
        parts = [float(record[k]) for k in present]
    except (TypeError, ValueError):
        return
    if not all(math.isfinite(x) for x in (total, *parts)):
        yield Finding(
            field=total_key,
            layer=layer,
            severity=Severity.FLAG,
            message=f"{total_key} or its line items are non-finite (NaN/inf)",
        )
        return
    if abs(total - sum(parts)) > tol:
        yield Finding(
            field=total_key,
            layer=layer,
            severity=Severity.FLAG,
            message=f"{total_key}={total} != sum({item_keys})={sum(parts)}",
        )


# ---------------------------------------------------------------------------
# The verifier Signal (cost tier 1: runs first, always)
# ---------------------------------------------------------------------------


@dataclass
class VerifierSignal:
    """Deterministic verifier evidence as a reference-free :class:`~ek.base.Signal`.

    Cost tier 1 -- the free, first-to-run layer. Runs a list of
    :class:`~ek.base.Validator` s on a value and returns the **fraction that passed**
    in ``[0, 1]`` as the raw signal (``1.0`` = every check passed). It also exposes
    the :class:`~ek.base.Finding` s it produced via :meth:`findings`, so the same
    object feeds both the score path (-> calibrate) and the audit path (-> review).
    Like every signal it is uncalibrated: a verifier failing is strong evidence, not
    a probability, so a :class:`~ek.base.Calibrator` still runs before any gate.

    Example:
        >>> v = VerifierSignal([checksum_validator("luhn")])
        >>> v("79927398713")          # valid -> all checks pass
        1.0
        >>> v("79927398710")          # invalid -> the one check fails
        0.0
    """

    validators: List[Callable[..., Iterable[Finding]]] = field(default_factory=list)
    cost_tier: int = 1

    def _by_validator(
        self, value: Any, *, spec: Optional[FieldSpec] = None
    ) -> List[List[Finding]]:
        """Run each validator exactly once; return its findings list (shared pass so
        the raw score and the audit always come from the same evaluation)."""
        return [list(v(value, spec=spec)) for v in self.validators]

    def findings(
        self, value: Any, *, spec: Optional[FieldSpec] = None
    ) -> List[Finding]:
        """All findings the validators produce for ``value`` (empty == all clear)."""
        out: List[Finding] = []
        for fs in self._by_validator(value, spec=spec):
            out.extend(fs)
        return out

    def __call__(self, value: Any, *, spec: Optional[FieldSpec] = None) -> float:
        if not self.validators:
            return 1.0
        failed = sum(1 for fs in self._by_validator(value, spec=spec) if fs)
        return 1.0 - failed / len(self.validators)
