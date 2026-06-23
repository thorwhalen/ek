"""The two top-level facades: :func:`score` (offline) and :func:`estimate_quality` (online).

Both operate on the same Layer-A object and follow progressive disclosure -- the
simple call Just Works, every strategy is replaceable by keyword:

- :func:`score` -- reference-based: compare one prediction to one gold reference,
  the metric chosen by output type (string -> CER, record -> field-F1) unless you
  name or pass one.
- :func:`evaluate` -- reference-based at corpus scale: aggregate many comparisons
  *correctly* (global error-rate accumulation, micro-F1), with optional per-slice
  cuts. This is what the OCR benchmark and the regression harness build on.
- :func:`estimate_quality` -- reference-free: gather signals -> calibrate ->
  validate -> decide, returning a :class:`~ek.base.QualityReport`. The heavy signal
  families (ROVER, conformal gates) are injected; this facade composes whatever you
  give it and ships sensible no-op defaults.

Example:
    >>> score("hello wrld", "hello world").metric
    'cer'
    >>> round(score("hello wrld", "hello world", metric="wer").value, 3)
    0.5
    >>> r = evaluate([("ct", "cat"), ("dg", "dog")], metric="cer")
    >>> r.n, round(r.aggregate, 3)
    (2, 0.333)
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Iterable, Optional, Union

from .base import (
    AnnotatedExtraction,
    Decision,
    FieldEstimate,
    GraphGrammar,
    Metric,
    QualityReport,
    Report,
    Score,
    Severity,
)
from .canonicalize import resolve_canonicalizer
from .metrics.fields import FieldMetric
from .metrics.graphs import TypedGraph, TypedGraphMetric
from .metrics.strings import StringMetric
from .registry import get


def _is_text_like(x: Any) -> bool:
    return isinstance(x, str) or hasattr(x, "text")


def _default_metric_name(pred: Any, gold: Any) -> str:
    if isinstance(pred, TypedGraph) and isinstance(gold, TypedGraph):
        return "graph"
    if _is_text_like(pred) and _is_text_like(gold):
        return "cer"
    if isinstance(pred, Mapping) and isinstance(gold, Mapping):
        return "fields"
    raise TypeError(
        f"No default metric for {type(pred).__name__} vs {type(gold).__name__}; "
        "pass metric=<name|callable> explicitly (e.g. metric='wer')."
    )


def _resolve_metric(
    metric: Union[None, str, Metric], pred: Any, gold: Any, canonicalizer, weights=None
) -> Metric:
    """Coerce ``metric`` to a callable, injecting canonicalizer/weights into built-ins."""
    if metric is not None and not isinstance(metric, str):
        return metric  # already a callable Metric
    name = metric or _default_metric_name(pred, gold)
    if name in ("cer", "wer"):
        return StringMetric(mode=name, canonicalizer=canonicalizer)
    if name == "fields":
        return FieldMetric(canonicalizer=canonicalizer)
    if name in ("graph", "typed_graph"):
        return TypedGraphMetric(weights=weights)
    return get("metrics", name)  # registered custom metric


def _metric_label(m: Any) -> str:
    return getattr(m, "name", getattr(m, "mode", getattr(m, "__name__", "")))


def _aggregate(metric: Any, scores: list) -> Optional[float]:
    if not scores:
        return None
    if hasattr(metric, "aggregate"):
        return metric.aggregate(scores)
    return sum(float(s) for s in scores) / len(scores)


def score(
    pred: Any,
    gold: Any,
    *,
    grammar: Optional[GraphGrammar] = None,
    metric: Union[None, str, Metric] = None,
    normalize: Any = None,
    weights: Any = None,
) -> Score:
    """Score one prediction against one gold reference (reference-based).

    Args:
        pred: The predicted output (string, record dict, or anything with ``.text``).
        gold: The gold reference, same shape as ``pred``.
        grammar: Optional Layer-A :class:`~ek.base.GraphGrammar` (carries cost weights).
        metric: A registered metric name (``"cer"``, ``"wer"``, ``"fields"``, ...), a
            callable :class:`~ek.base.Metric`, or ``None`` to dispatch by type.
        normalize: Optional canonicalizer (name, callable, step list, or
            :class:`~ek.canonicalize.Canonicalizer`) applied before comparison.
        weights: A :data:`~ek.base.CostWeight` for cost-weighted metrics (the
            typed-graph distance); overrides the schema's importance weights.

    Returns:
        A :class:`~ek.base.Score`.
    """
    canon = resolve_canonicalizer(normalize)
    m = _resolve_metric(metric, pred, gold, canon, weights)
    return m(pred, gold, grammar=grammar)


def evaluate(
    cases: Iterable,
    *,
    metric: Union[None, str, Metric] = None,
    grammar: Optional[GraphGrammar] = None,
    normalize: Any = None,
    weights: Any = None,
) -> Report:
    """Aggregate many comparisons into a :class:`~ek.base.Report` (corpus level).

    Args:
        cases: Iterable of ``(pred, gold)`` or ``(pred, gold, slice_label)`` tuples.
        metric: As in :func:`score` (resolved once from the first case's types).
        grammar: Optional Layer-A grammar passed to every comparison.
        normalize: Optional canonicalizer applied before every comparison.

    Returns:
        A Report whose ``aggregate`` is computed by the metric's own aggregator
        (e.g. globally accumulated CER/WER, micro-F1) -- never a naive mean.
    """
    canon = resolve_canonicalizer(normalize)
    cases = list(cases)
    if not cases:
        return Report(metric=metric if isinstance(metric, str) else "", n=0)
    p0, g0 = cases[0][0], cases[0][1]
    m = _resolve_metric(metric, p0, g0, canon, weights)

    scores: list = []
    by_slice: dict = {}
    for case in cases:
        pred, gold = case[0], case[1]
        s = m(pred, gold, grammar=grammar)
        scores.append(s)
        if len(case) > 2 and case[2] is not None:
            by_slice.setdefault(case[2], []).append(s)

    return Report(
        metric=_metric_label(m),
        aggregate=_aggregate(m, scores),
        n=len(scores),
        scores=scores,
        per_slice={label: _aggregate(m, ss) for label, ss in by_slice.items()},
    )


def _accepts_group(fn: Any) -> bool:
    """Whether a calibrator/policy accepts a keyword-only ``group`` (Mondrian-aware)."""
    try:
        return "group" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _calibrate(calibrator, raw: Optional[float], group: Any):
    """Apply a calibrator (passing ``group`` to a Mondrian one), reporting if it ran.

    Capability is detected by signature inspection rather than catching ``TypeError``,
    so a genuine ``TypeError`` raised *inside* the calibrator is not silently masked.
    """
    if calibrator is None or raw is None:
        return raw, False
    if group is not None and _accepts_group(calibrator):
        return calibrator(raw, group=group), True
    return calibrator(raw), True


def _decide(policy, confidence: Optional[float], group: Any):
    """Apply a decision policy (passing ``group`` to a Mondrian one) -> a Decision."""
    if policy is None or confidence is None:
        return None
    if group is not None and _accepts_group(policy):
        return policy(confidence, group=group)
    return policy(confidence)


def _run_signals(
    signals: Iterable, target: Any, raw_signals: dict, failures: list
) -> None:
    """Call each injected signal on ``target``; merge output, *recording* any failure.

    A failing signal must not abort the pipeline (the others still run), but it is
    not swallowed silently: its name and error are appended to ``failures`` (surfaced
    in the report's provenance) and a warning is emitted.
    """
    for i, sig in enumerate(signals):
        name = getattr(sig, "name", getattr(sig, "__name__", f"signal_{i}"))
        try:
            out = sig(target)
            if isinstance(out, Mapping):
                raw_signals.update(out)
            else:
                raw_signals[name] = float(out)
        except Exception as exc:  # keep the pipeline alive, but leave a trace
            failures.append({"signal": name, "error": repr(exc)})
            warnings.warn(
                f"estimate_quality: signal {name!r} failed: {exc!r}", stacklevel=2
            )


def _run_validators(
    validators: Iterable, value: Any, spec: Any, findings: list, failures: list
) -> None:
    """Run each validator on ``value``, recording (not swallowing) any that errors.

    A throwing validator (e.g. a bad regex) is isolated like a failing signal -- the
    others still run, and the failure is surfaced in provenance + a warning -- so one
    misbehaving check cannot take down the whole estimate.
    """
    for i, v in enumerate(validators):
        name = getattr(v, "__name__", f"validator_{i}")
        try:
            findings.extend(v(value, spec=spec))
        except Exception as exc:
            failures.append({"validator": name, "error": repr(exc)})
            warnings.warn(
                f"estimate_quality: validator {name!r} failed: {exc!r}", stacklevel=2
            )


def _base_confidence(estimate: Any, raw_signals: dict) -> Optional[float]:
    """A field's pre-calibration confidence: its own, else the mean of raw signals."""
    base = getattr(estimate, "confidence", None)
    if base is None and raw_signals:
        base = sum(raw_signals.values()) / len(raw_signals)
    return base


def _agreement_score(hypotheses: list, **rover_kwargs) -> float:
    """Mean ROVER agreement over a list of hypotheses (the AgreementSignal score)."""
    from .qe.rover import AgreementSignal

    return AgreementSignal(**rover_kwargs)(hypotheses)


def _flat_validators(validators) -> list:
    """All validators as a flat list (flattening a per-field Mapping if given)."""
    if isinstance(validators, Mapping):
        return [v for vs in validators.values() for v in vs]
    return list(validators)


def _field_validators(validators, path) -> list:
    """Validators to run on one field: global (``"*"``) plus those scoped to its
    node type or field name, when ``validators`` is a per-field :class:`Mapping`."""
    if not isinstance(validators, Mapping):
        return list(validators)
    out = list(validators.get("*", ()))
    out += list(validators.get(path.node_type, ()))
    if path.field is not None:
        out += list(validators.get(path.field, ()))
    return out


def estimate_quality(
    extraction: Any,
    *,
    sources: Iterable = (),
    signals: Iterable = (),
    calibrator=None,
    validators: Iterable = (),
    policy=None,
    agreement: bool = True,
    assume_calibrated: bool = False,
) -> QualityReport:
    """Estimate the quality of an extraction with no gold reference.

    Composes the strict ``signal -> calibrate -> validate -> decide`` pipeline
    (``misc/docs/ek_03``) over a single value, a :class:`~ek.base.FieldEstimate`, or a
    whole :class:`~ek.base.AnnotatedExtraction` (scored per field, with per-field
    specs feeding the validators and the node type as the Mondrian group key). The
    simple call ``estimate_quality(value)`` Just Works; every stage is injectable.

    Args:
        extraction: A raw value, a :class:`~ek.base.FieldEstimate`, or an
            :class:`~ek.base.AnnotatedExtraction`.
        sources: Additional hypotheses of the *same* content (strings or
            ``OcrResult``-shaped objects) to fuse with ROVER -- their mean agreement
            becomes a ``raw_signals["agreement"]`` entry (the flagship online signal).
        signals: Explicit :class:`~ek.base.Signal` callables ``target -> float |
            Mapping`` producing further raw signals.
        calibrator: A :class:`~ek.base.Calibrator` mapping raw score -> probability.
            **Calibration is non-optional before gating**: with a ``policy`` but no
            calibrator (and ``assume_calibrated`` false), a warning is issued.
        validators: :class:`~ek.base.Validator` callables yielding findings. A flat
            iterable runs on every field (use spec-driven validators like
            :func:`~ek.qe.verifiers.schema_validator`). For per-field scoping pass a
            ``Mapping`` keyed by field name or node type (``"*"`` runs on all).
        policy: A :class:`~ek.base.DecisionPolicy` producing accept/flag/block from
            the *calibrated* confidence.
        agreement: Auto-run ROVER over ``sources`` when any are given (default true).
        assume_calibrated: Treat the incoming confidence as already calibrated
            (silences the uncalibrated-gating warning).
    """
    if policy is not None and calibrator is None and not assume_calibrated:
        warnings.warn(
            "estimate_quality: gating without a Calibrator. Raw signals are "
            "uncalibrated (Hard Rule 1); fit a Calibrator first or pass "
            "assume_calibrated=True if the input confidence is already calibrated.",
            stacklevel=2,
        )

    sources = list(sources)
    signals = list(signals)
    # A bare callable in `sources` is almost certainly a Signal passed to the wrong
    # slot; a legitimate hypothesis is a str or OcrResult-shaped object (which may
    # itself be callable), so exempt those.
    if any(
        callable(s) and not (isinstance(s, str) or hasattr(s, "text")) for s in sources
    ):
        raise TypeError(
            "estimate_quality: `sources` are alternative hypotheses "
            "(strings/OcrResult-shaped objects) to fuse with ROVER, not signal "
            "callables. Pass Signal callables via `signals=` instead."
        )

    if isinstance(extraction, AnnotatedExtraction):
        return _estimate_annotated(
            extraction,
            sources=sources,
            signals=signals,
            calibrator=calibrator,
            validators=validators,
            policy=policy,
            agreement=agreement,
        )

    value = extraction.value if isinstance(extraction, FieldEstimate) else extraction
    raw_signals = dict(getattr(extraction, "raw_signals", {}) or {})
    failures: list = []
    _run_signals(signals, extraction, raw_signals, failures)
    if agreement and sources:
        # The value plus the alternative hypotheses are the engines ROVER fuses.
        raw_signals["agreement"] = _agreement_score([value, *sources])

    findings = list(getattr(extraction, "findings", ()) or ())
    _run_validators(_flat_validators(validators), value, None, findings, failures)

    base = _base_confidence(extraction, raw_signals)
    calibrated, did_calibrate = _calibrate(calibrator, base, None)
    decision = _decide(policy, calibrated, None)

    provenance = {"n_signals": len(raw_signals), "calibrated": did_calibrate}
    if failures:
        provenance["failures"] = failures
    return QualityReport(
        calibrated_confidence=calibrated,
        decision=decision,
        findings=tuple(findings),
        raw_signals=raw_signals,
        provenance=provenance,
    )


# Decision severity ordering for summarizing a whole extraction (worst wins).
_DECISION_RANK = {Decision.ACCEPT: 0, Decision.FLAG: 1, Decision.BLOCK: 2}


def _estimate_annotated(
    extraction: AnnotatedExtraction,
    *,
    sources: list,
    signals: list,
    calibrator,
    validators: Iterable,
    policy,
    agreement: bool,
) -> QualityReport:
    """Score each field of an :class:`~ek.base.AnnotatedExtraction` and summarize."""
    grammar = extraction.grammar
    per_field: dict = {}
    all_findings: list = []
    confidences: list = []
    failures: list = []
    any_calibrated = False
    worst: Optional[Decision] = None

    # At the extraction level there is no single primary text, so the provided
    # sources are themselves the alternative hypotheses to fuse (needs >= 2).
    extraction_agreement = (
        _agreement_score(list(sources)) if (agreement and len(sources) >= 2) else None
    )

    for path, fe in extraction.estimates.items():
        spec = _field_spec(grammar, path)
        raw_signals = dict(fe.raw_signals or {})
        _run_signals(signals, fe, raw_signals, failures)
        if extraction_agreement is not None:
            raw_signals.setdefault("agreement", extraction_agreement)

        findings = list(fe.findings or ())
        _run_validators(
            _field_validators(validators, path), fe.value, spec, findings, failures
        )

        base = _base_confidence(fe, raw_signals)
        group = path.node_type
        calibrated, did_calibrate = _calibrate(calibrator, base, group)
        any_calibrated = any_calibrated or did_calibrate
        decision = _decide(policy, calibrated, group)
        # A hard verifier failure forces at least a flag even if confidence is high.
        if decision is Decision.ACCEPT and any(
            f.severity is Severity.FLAG for f in findings
        ):
            decision = Decision.FLAG

        # Build a fresh FieldEstimate rather than mutate the input, so re-running
        # estimate_quality on the same extraction is idempotent (no accumulation).
        per_field[path] = replace(
            fe,
            raw_signals=raw_signals,
            confidence=calibrated,
            findings=tuple(findings),
            decision=decision,
        )

        all_findings.extend(findings)
        if calibrated is not None:
            confidences.append(calibrated)
        if decision is not None and (
            worst is None or _DECISION_RANK[decision] > _DECISION_RANK[worst]
        ):
            worst = decision

    provenance = {"n_fields": len(per_field), "calibrated": any_calibrated}
    if failures:
        provenance["failures"] = failures
    return QualityReport(
        calibrated_confidence=min(confidences) if confidences else None,
        decision=worst,
        findings=tuple(all_findings),
        raw_signals={"agreement": extraction_agreement}
        if extraction_agreement is not None
        else {},
        provenance=provenance,
        per_field=per_field,
    )


def _field_spec(grammar: GraphGrammar, path):
    """The :class:`~ek.base.FieldSpec` a NodePath addresses, or ``None``."""
    if path.field is None:
        return None
    nt = grammar.node_types.get(path.node_type)
    if nt is None:
        return None
    return nt.fields.get(path.field)
