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
softer/heavier layers ship as **dependency-injected** layers: the layer is dep-free
and you inject the model/LLM call (your in-domain LM scorer, your LLM client), so the
expensive backend is your deployment choice, not a default dependency.

- **L3 LM surprisal** (FLAG): :func:`lm_surprisal_validator` takes an injected
  ``(str) -> float`` surprisal scorer (n-gram / masked-LM PLL, e.g. ``minicons``) and
  flags low-probability spans. Statistical anomaly without an LM:
  :func:`benford_findings` (leading-digit) and :func:`zscore_anomaly_findings`
  (robust median/MAD magnitude outliers), both dep-free.
- **L5 neural/LLM correction** (CORRECT, gated): :func:`llm_corrector` takes an
  injected ``(str) -> Optional[str]`` correction call; it is the only layer that
  invents content, so it fires only on already-flagged values (``only_flagged``),
  reads the pipeline's findings, and keeps the original for audit.
- **L4 constrained generation** is a *generation*-time concern (it constrains a
  generative extractor, not post-hoc output -- ek_04 separates "constrain-on-generate"
  from "post-hoc validation"), so it lives on the extraction side
  (``ek[constrained]``: ``outlines``/XGrammar/``instructor``), not in this pipeline.

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
            # A layer that sets ``wants_findings = True`` (e.g. a gated L5 corrector
            # that should fire only on already-flagged values) receives the findings
            # accumulated by the cheaper layers before it.
            if getattr(layer, "wants_findings", False):
                layer_findings = list(
                    layer(value, spec=spec, findings=tuple(collected))
                )
            else:
                layer_findings = list(layer(value, spec=spec))
            collected.extend(layer_findings)
            if apply_corrections:
                for finding in layer_findings:
                    if (
                        finding.severity is Severity.CORRECT
                        and finding.suggestion is not None
                    ):
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


# ---------------------------------------------------------------------------
# Statistical anomaly detection beyond Benford (FLAG; corpus-level, dep-free)
# ---------------------------------------------------------------------------

#: Modified (median/MAD) z-score above which a value is flagged an outlier
#: (Iglewicz & Hoaglin recommend 3.5).
DEFAULT_ZSCORE_THRESHOLD = 3.5

#: 0.6745 = Phi^-1(0.75); scales the MAD to a standard-deviation estimate.
_MAD_TO_SIGMA = 0.6745


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def zscore_anomaly_findings(
    numbers: Iterable[Any],
    *,
    field: str = "",
    threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    min_n: int = DEFAULT_BENFORD_MIN_N,
    layer: str = "anomaly",
) -> list:
    """FLAG numeric values that are robust-z-score outliers (median + MAD based).

    A reference-free, **dependency-free** anomaly check that complements
    :func:`benford_findings` (which checks the leading-digit *distribution*; this
    checks individual *magnitudes*). For a numeric column it computes each value's
    **modified z-score** -- ``0.6745 * (x - median) / MAD`` -- which is robust to the
    very outliers it is looking for, and FLAGs those whose absolute score exceeds
    ``threshold``. One FLAG per outlying value (carrying its index); **FLAG-only** --
    it routes a value to review, never edits. Skipped below ``min_n`` (robust
    statistics on a handful of points are noise). When the MAD is 0 (a *near*-constant
    column, e.g. many identical values plus one outlier) it falls back to the
    mean-absolute-deviation scale (Iglewicz & Hoaglin), so a lone outlier is still
    caught; a *truly* constant column (no spread at all) yields no findings. For
    *multivariate* outliers, plug an isolation forest via ``pyod`` (already in
    ``ek[validation]``) as a custom validator.
    """
    pairs = [(i, _as_float(x)) for i, x in enumerate(numbers)]
    pairs = [(i, v) for i, v in pairs if v is not None]
    if len(pairs) < min_n:
        return []
    xs = [v for _, v in pairs]
    med = _median(xs)
    devs = [abs(v - med) for v in xs]
    mad = _median(devs)
    if mad > 0:
        scale = _MAD_TO_SIGMA / mad
    else:
        # MAD collapses to 0 when >half the values are identical; fall back to the
        # mean absolute deviation so a lone outlier in a near-constant column is still
        # scored (1.253314 = sqrt(pi/2), the MeanAD->sigma factor).
        mean_ad = sum(devs) / len(devs)
        if mean_ad == 0:
            return []  # a truly constant column has no outliers
        scale = 1.0 / (1.253314 * mean_ad)
    findings = []
    for i, v in pairs:
        score = scale * (v - med)
        if abs(score) > threshold:
            findings.append(
                Finding(
                    field=field,
                    layer=layer,
                    severity=Severity.FLAG,
                    message=(
                        f"value {v} at index {i} is a robust-z outlier "
                        f"(|z|={abs(score):.1f} > {threshold})"
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Layer 3 -- language-model surprisal (FLAG; dependency-injected scorer)
# ---------------------------------------------------------------------------


def lm_surprisal_validator(
    scorer: Callable[[str], float],
    *,
    threshold: float,
    higher_is_worse: bool = True,
    field_name: str = "",
    layer: str = "lm_prior",
) -> Callable[..., Iterable[Finding]]:
    """L3: FLAG a value whose language-model surprisal crosses ``threshold``.

    ``scorer`` is an **injected** ``(str) -> float`` returning a surprisal /
    perplexity / negative-log-likelihood (dependency injection: bring your own
    in-domain n-gram or masked-LM scorer -- an *in-domain* prior is what makes this
    useful). FLAG-only on its own; pair it with a candidate generator
    (``lexicon_corrector`` / ``llm_corrector``) to actually correct. Set
    ``higher_is_worse=False`` if your scorer returns a probability (higher = better).

    Recipe (masked-LM pseudo-log-likelihood): wrap ``minicons`` --
    ``from minicons import scorer; m = scorer.MaskedLMScorer("bert-base-uncased", "cpu")``
    -- and pass ``scorer=lambda s: -m.sequence_score([s])[0]`` (surprisal = negative
    PLL). ``minicons`` is MIT; install your own LM backend (it pulls torch).
    """

    def validate(value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        if not isinstance(value, str) or not value:
            return
        score = float(scorer(value))
        anomalous = score > threshold if higher_is_worse else score < threshold
        if anomalous:
            rel = ">" if higher_is_worse else "<"
            yield Finding(
                field=_field_name(field_name, spec),
                layer=layer,
                severity=Severity.FLAG,
                message=f"language-model surprisal {score:.3f} {rel} {threshold}",
            )

    validate.layer = layer
    return validate


# ---------------------------------------------------------------------------
# Layer 5 -- gated neural / LLM correction (CORRECT; dependency-injected call)
# ---------------------------------------------------------------------------


def llm_corrector(
    correct_fn: Callable[[str], Optional[str]],
    *,
    only_flagged: bool = True,
    field_name: str = "",
    layer: str = "llm_correct",
) -> Corrector:
    """L5: a **gated** neural/LLM corrector -- the only layer that invents content.

    ``correct_fn`` is an **injected** ``(value) -> Optional[str]`` (bring your own
    LLM/seq2seq call; return a corrected string, or ``None`` to leave the value
    unchanged). It is the most expensive and stochastic layer, so by default
    (``only_flagged``) it fires **only on a value that a cheaper layer already
    FLAGged** -- gate it to the residual, never let it silently rewrite clean fields
    (this layer reads the pipeline's accumulated findings via ``wants_findings``).
    Emits a CORRECT finding; the pipeline applies the rewrite and keeps the original
    for audit. Always verify its output downstream -- it can make text worse.

    Recipe (Anthropic): ``correct_fn=lambda v: client.messages.create(model=...,
    messages=[{"role": "user", "content": prompt(v)}]).content[0].text.strip()`` with
    ``client = anthropic.Anthropic()`` (MIT). Constrain/verify the result and keep an
    audit trail.
    """

    def correct(
        value: Any,
        *,
        spec: Optional[FieldSpec] = None,
        findings: Sequence[Finding] = (),
    ) -> Iterable[Finding]:
        if not isinstance(value, str):
            return
        if only_flagged and not any(f.severity is Severity.FLAG for f in findings):
            return  # gated: nothing cheaper flagged this value, so do not pay for an LLM
        proposed = correct_fn(value)
        if isinstance(proposed, str) and proposed != value:
            yield Finding(
                field=_field_name(field_name, spec),
                layer=layer,
                severity=Severity.CORRECT,
                suggestion=proposed,
                message=f"LLM correction {value!r} -> {proposed!r}",
            )

    correct.layer = layer
    correct.wants_findings = True  # the pipeline passes accumulated findings (gating)
    return correct
