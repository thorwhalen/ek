"""Reference-free post-extraction validation & correction (issue #7).

A layered, **cheapest -> most-expensive** pipeline with an explicit
**FLAG-vs-CORRECT** distinction (``misc/docs/ek_04``). Every method estimates
``argmax_v P(v|o)`` without a gold reference: the prior ``P(v)`` strengthens in
layers -- schema/type/range (cheapest, strongest), then lexicon/gazetteer, then
language-model surprisal (softest) -- with cross-field consistency and statistical
anomaly checks on top.

The single most important architectural line is **FLAG vs CORRECT**:

- A **Validator** only *flags*: it yields :class:`~ek.base.Finding` s with
  ``severity=FLAG`` and never rewrites the value. The deterministic verifier layer
  (``ek.qe.verifiers``: checksums, regex, range, enum, schema, totals) is re-exported
  here so a caller composes the whole stack from one place.
- A **Corrector** can also *correct*: it yields a ``severity=CORRECT`` Finding whose
  ``suggestion`` is the proposed replacement. Only a narrow, safe, deterministic
  class corrects automatically -- canonicalization (L0) and closed-set lexicon
  resolution (L2); free invention is reserved for the gated L5 neural corrector.

Both share one callable shape -- ``(value, *, spec) -> Iterable[Finding]`` -- so
:func:`validation_pipeline` composes them uniformly. A CORRECT finding's suggestion
is applied to the value *before the next layer runs* (the noisy-channel chain: a
cheap deterministic fix short-circuits the expensive layers), and an injectable
``stop_when`` policy ends the pass early (e.g. once a value is corrected or clean).

The deterministic layers here need no extra dependencies (``rapidfuzz`` is core). The
softer/heavier layers are documented **extension points**, not shipped wiring -- plug
any ``Corrector``/``Validator`` of your own:

- **L3 LM surprisal** (FLAG): score substrings under an in-domain n-gram/masked LM
  (KenLM / ``minicons`` PLL) and flag low-probability spans.
- **L4 constrained generation** (generative extractors only): ``outlines``/XGrammar.
- **L5 neural/LLM correction** (CORRECT, gated): the only layer that invents content;
  send it only flagged spans, verify its output, keep an audit trail.

Example:
    >>> from ek.validate import validation_pipeline, lexicon_corrector, stop_on_correction
    >>> from ek.base import Severity
    >>> fix = validation_pipeline(
    ...     lexicon_corrector(["France", "Germany", "Spain"]),
    ...     stop_when=stop_on_correction,
    ... )
    >>> r = fix("Frnace")               # one transposition from a closed enum
    >>> r.value
    'France'
    >>> r.corrected
    True
    >>> r.findings[0].severity is Severity.CORRECT
    True
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .base import FieldSpec, Finding, Severity

# Re-export the deterministic verifier validators (L1) so the whole stack composes
# from one import. These FLAG (or coerce upstream); they never invent content.
from .qe.verifiers import (  # noqa: F401
    checksum_validator,
    enum_validator,
    range_validator,
    regex_validator,
    schema_validator,
    totals_consistent,
)

#: A Validator/Corrector callable: a value (and optional spec) -> zero+ Findings.
Layer = Callable[..., Iterable[Finding]]

#: Default minimum similarity (``rapidfuzz`` ratio in [0, 1]) for a closed-set lexicon
#: match to be auto-applied as a correction rather than flagged. 0.8 catches typical
#: single-edit and transposition typos ("Frnace"->"France" ~0.83) while the wide gap
#: to an unrelated word (~0.3) keeps false corrections rare.
DEFAULT_LEXICON_THRESHOLD = 0.8

#: Benford's law expected first-significant-digit frequencies, ``P(d)=log10(1+1/d)``.
BENFORD_EXPECTED = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

#: Default max absolute per-digit deviation from Benford before a field is flagged.
DEFAULT_BENFORD_TOL = 0.15

#: Default minimum sample size before applying Benford (the law is asymptotic).
DEFAULT_BENFORD_MIN_N = 30


@runtime_checkable
class Corrector(Protocol):
    """A :class:`~ek.base.Validator` that may also CORRECT.

    Same callable shape as a Validator, but it can emit a ``severity=CORRECT``
    :class:`~ek.base.Finding` whose ``suggestion`` is the proposed replacement value.
    The ``layer`` attribute names its place on the cheap->expensive spine.
    """

    layer: str

    def __call__(
        self, value: Any, *, spec: Optional[FieldSpec] = None
    ) -> Iterable[Finding]: ...


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of running a value through a :func:`validation_pipeline`.

    ``value`` is the final value after any applied corrections; ``original`` is the
    input; ``findings`` is the full audit trail (FLAG and CORRECT), in layer order.
    """

    original: Any
    value: Any
    findings: tuple[Finding, ...] = ()

    @property
    def corrected(self) -> bool:
        """Whether any correction changed the value."""
        return self.value != self.original

    @property
    def flagged(self) -> bool:
        """Whether any layer raised a FLAG (something a human should look at)."""
        return any(f.severity is Severity.FLAG for f in self.findings)

    @property
    def clean(self) -> bool:
        """No findings at all -- the value passed every layer untouched."""
        return not self.findings


# ---------------------------------------------------------------------------
# The pipeline combinator (cost order + stop-early + FLAG-vs-CORRECT chaining)
# ---------------------------------------------------------------------------


def _never_stop(_findings: Sequence[Finding]) -> bool:
    return False


def stop_on_correction(findings: Sequence[Finding]) -> bool:
    """Stop policy: end the pass as soon as any layer proposes a correction."""
    return any(f.severity is Severity.CORRECT for f in findings)


def stop_on_flag(findings: Sequence[Finding]) -> bool:
    """Stop policy: end the pass at the first FLAG (fail fast)."""
    return any(f.severity is Severity.FLAG for f in findings)


def validation_pipeline(
    *layers: Layer,
    apply_corrections: bool = True,
    stop_when: Callable[[Sequence[Finding]], bool] = _never_stop,
) -> Callable[..., ValidationResult]:
    """Compose validators/correctors into a cheapest -> most-expensive pipeline.

    ``layers`` run in the given order (the order encodes the cost spine). Each layer
    is a ``(value, *, spec) -> Iterable[Finding]`` callable. When ``apply_corrections``
    (default), the first ``CORRECT`` finding a layer emits is applied to the value
    before the next layer runs -- so a cheap deterministic fix is what the expensive
    layers then see (the noisy-channel chain). ``stop_when`` is an injectable
    early-exit policy over the findings accumulated so far (see
    :func:`stop_on_correction` / :func:`stop_on_flag`); the default runs every layer.
    """
    layers = tuple(layers)

    def run(value: Any, *, spec: Optional[FieldSpec] = None) -> ValidationResult:
        original = value
        collected: list[Finding] = []
        for layer in layers:
            layer_findings = list(layer(value, spec=spec))
            collected.extend(layer_findings)
            if apply_corrections:
                for finding in layer_findings:
                    if finding.severity is Severity.CORRECT and finding.suggestion is not None:
                        value = finding.suggestion
                        break  # at most one correction applied per layer
            if stop_when(collected):
                break
        return ValidationResult(
            original=original, value=value, findings=tuple(collected)
        )

    return run


def _field_name(field_name: str, spec: Optional[FieldSpec]) -> str:
    return field_name or (spec.name if spec is not None else "")


# ---------------------------------------------------------------------------
# Layer 0 -- canonicalization (CORRECT, narrow + deterministic)
# ---------------------------------------------------------------------------


def canonicalize_corrector(
    normalize: Any, *, field_name: str = "", layer: str = "canonicalize"
) -> Corrector:
    """L0: fold a value to canonical form, emitting a CORRECT finding when it changes.

    ``normalize`` is anything :func:`ek.canonicalize.resolve_canonicalizer` accepts
    (a registered name, a callable, a step list). The narrowest, safest corrector --
    run it first so later layers compare canonical forms.
    """
    from .canonicalize import resolve_canonicalizer

    canon = resolve_canonicalizer(normalize)

    def correct(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        if canon is None or not isinstance(value, str):
            return
        folded = canon(value)
        if folded != value:
            yield Finding(
                field=_field_name(field_name, spec),
                layer=layer,
                severity=Severity.CORRECT,
                suggestion=folded,
                message=f"canonicalized {value!r} -> {folded!r}",
            )

    correct.layer = layer
    return correct


# ---------------------------------------------------------------------------
# Layer 2 -- lexicon / gazetteer resolution (CORRECT on a closed set; rapidfuzz)
# ---------------------------------------------------------------------------


def lexicon_corrector(
    vocabulary: Iterable[str],
    *,
    threshold: float = DEFAULT_LEXICON_THRESHOLD,
    scorer: Any = None,
    field_name: str = "",
    layer: str = "lexicon",
    flag_unmatched: bool = True,
) -> Corrector:
    """L2: resolve a value against a **closed** vocabulary by fuzzy match (``rapidfuzz``).

    The safest corrector in the stack when the candidate set is closed (country codes,
    SKUs, enums): an exact member passes untouched; a single close match
    (similarity ``>= threshold``) is applied as a CORRECT finding; otherwise the value
    is FLAGged as out-of-vocabulary (when ``flag_unmatched``). Against an *open*
    vocabulary, set ``flag_unmatched=False`` and treat matches as suggestions only.

    Args:
        vocabulary: The closed set of valid values.
        threshold: Minimum ``rapidfuzz`` similarity in ``[0, 1]`` to auto-correct.
        scorer: A ``rapidfuzz`` scorer (default ``fuzz.ratio``).
        flag_unmatched: FLAG a value with no close match (default ``True``).
    """
    vocab = list(vocabulary)
    vocab_set = set(vocab)

    def correct(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        name = _field_name(field_name, spec)
        if not isinstance(value, str) or not vocab or value in vocab_set:
            return
        from rapidfuzz import fuzz, process

        match, raw_score, _ = process.extractOne(
            value, vocab, scorer=scorer or fuzz.ratio
        )
        similarity = raw_score / 100.0
        if similarity >= threshold:
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.CORRECT,
                suggestion=match,
                message=f"{value!r} -> {match!r} (lexicon match {similarity:.2f})",
            )
        elif flag_unmatched:
            yield Finding(
                field=name,
                layer=layer,
                severity=Severity.FLAG,
                message=f"{value!r} not in vocabulary (nearest {match!r} @ {similarity:.2f})",
            )

    correct.layer = layer
    return correct


# ---------------------------------------------------------------------------
# Cross-field consistency (FLAG; operate on a whole record)
# ---------------------------------------------------------------------------


def cross_field_validator(
    predicate: Callable[[Any], bool],
    *,
    message: str,
    fields: Sequence[str] = (),
    layer: str = "cross_field",
) -> Callable[..., Iterable[Finding]]:
    """FLAG a record when ``predicate(record)`` is False (a general cross-field check).

    Skips silently when ``predicate`` raises on a partial record (a missing field is
    not itself a cross-field violation)."""

    def validate(record: Any, *, grammar: Any = None) -> Iterable[Finding]:
        try:
            ok = predicate(record)
        except (KeyError, TypeError, ValueError):
            return
        if not ok:
            yield Finding(
                field=",".join(fields),
                layer=layer,
                severity=Severity.FLAG,
                message=message,
            )

    return validate


def ordering_validator(
    keys: Sequence[str], *, strict: bool = True, layer: str = "cross_field"
) -> Callable[..., Iterable[Finding]]:
    """FLAG a record whose comparable values at ``keys`` are not ascending.

    The canonical date/sequence check (``start <= end``, issue before due). Only the
    keys actually present and mutually comparable are checked, so it is safe on
    partial records.
    """

    def validate(record: Any, *, grammar: Any = None) -> Iterable[Finding]:
        values = [record[k] for k in keys if k in record]
        try:
            pairs = list(zip(values, values[1:]))
            ok = all((a < b) if strict else (a <= b) for a, b in pairs)
        except TypeError:
            return  # values not mutually comparable -> not an ordering violation
        if not ok:
            rel = "strictly ascending" if strict else "ascending"
            yield Finding(
                field=",".join(keys),
                layer=layer,
                severity=Severity.FLAG,
                message=f"values at {list(keys)} are not {rel}: {values}",
            )

    return validate


# ---------------------------------------------------------------------------
# Statistical anomaly detection (FLAG; corpus-level, never corrects)
# ---------------------------------------------------------------------------


def _leading_digit(x: Any) -> Optional[int]:
    """First significant decimal digit of ``|x|`` (1..9), or ``None`` if not usable."""
    try:
        v = abs(float(x))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v == 0:
        return None
    while v < 1:
        v *= 10
    while v >= 10:
        v /= 10
    return int(v)


def benford_findings(
    numbers: Iterable[Any],
    *,
    field: str = "",
    tol: float = DEFAULT_BENFORD_TOL,
    min_n: int = DEFAULT_BENFORD_MIN_N,
    layer: str = "anomaly",
) -> list:
    """FLAG a numeric field whose first-digit distribution deviates from Benford's law.

    A reference-free anomaly check for naturally-occurring magnitudes (amounts,
    populations, counts): the first significant digit should follow
    ``P(d)=log10(1+1/d)``. A large deviation flags fabricated or systematically-wrong
    data. **Corpus-level** (takes the whole column of values) and **FLAG-only** -- it
    routes a field to review, never auto-edits, and can false-positive on legitimately
    bounded/skewed fields. Skipped (returns ``[]``) below ``min_n`` usable values,
    since the law is asymptotic.
    """
    firsts = [d for d in (_leading_digit(x) for x in numbers) if d is not None]
    if len(firsts) < min_n:
        return []
    counts = Counter(firsts)
    n = len(firsts)
    worst = max(abs(counts.get(d, 0) / n - BENFORD_EXPECTED[d]) for d in range(1, 10))
    if worst > tol:
        return [
            Finding(
                field=field,
                layer=layer,
                severity=Severity.FLAG,
                message=(
                    f"first-digit distribution deviates from Benford's law "
                    f"(max deviation {worst:.3f} > {tol}) over {n} values"
                ),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


def validate(
    value: Any,
    *,
    layers: Sequence[Layer] = (),
    spec: Optional[FieldSpec] = None,
    apply_corrections: bool = True,
    stop_when: Callable[[Sequence[Finding]], bool] = _never_stop,
) -> ValidationResult:
    """Run one value through a validation/correction pipeline (progressive disclosure).

    The simple call ``validate(v, layers=[...])`` Just Works; tune the correction
    behaviour with ``apply_corrections``/``stop_when``. Compose ``layers`` from the
    factories in this module (``canonicalize_corrector``, ``lexicon_corrector``, the
    re-exported verifier validators) in cheapest-first order.
    """
    pipe = validation_pipeline(
        *layers, apply_corrections=apply_corrections, stop_when=stop_when
    )
    return pipe(value, spec=spec)
