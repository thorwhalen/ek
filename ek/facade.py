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

from collections.abc import Mapping
from typing import Any, Iterable, Optional, Union

from .base import (
    FieldEstimate,
    GraphGrammar,
    Metric,
    QualityReport,
    Report,
    Score,
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


def estimate_quality(
    extraction: Any,
    *,
    sources: Iterable = (),
    calibrator=None,
    validators: Iterable = (),
    policy=None,
) -> QualityReport:
    """Estimate the quality of one extraction with no gold reference.

    Composes the signal -> calibrate -> validate -> decide pipeline from whatever
    strategies you inject (defaults are no-ops, so the call always returns a
    report). The rich signal families (ROVER agreement, conformal gates) plug in
    here as ``sources``/``calibrator``/``policy`` -- see ``misc/docs/ek_03``.

    Args:
        extraction: A :class:`~ek.base.FieldEstimate` or a raw value.
        sources: Callables ``extraction -> float | Mapping[str, float]`` producing
            raw quality signals.
        calibrator: Optional :class:`~ek.base.Calibrator` mapping a raw score to a
            probability.
        validators: Optional :class:`~ek.base.Validator` callables yielding findings.
        policy: Optional :class:`~ek.base.DecisionPolicy` producing accept/flag/block.
    """
    value = extraction.value if isinstance(extraction, FieldEstimate) else extraction
    raw_signals = dict(getattr(extraction, "raw_signals", {}) or {})

    for i, src in enumerate(sources):
        try:
            out = src(extraction)
        except Exception:
            continue
        if isinstance(out, Mapping):
            raw_signals.update(out)
        else:
            raw_signals[getattr(src, "__name__", f"signal_{i}")] = float(out)

    findings = []
    for v in validators:
        findings.extend(v(value, spec=None))

    base = getattr(extraction, "confidence", None)
    if base is None and raw_signals:
        base = sum(raw_signals.values()) / len(raw_signals)

    did_calibrate = calibrator is not None and base is not None
    calibrated = calibrator(base) if did_calibrate else base
    decision = policy(calibrated) if (policy is not None and calibrated is not None) else None

    return QualityReport(
        calibrated_confidence=calibrated,
        decision=decision,
        findings=tuple(findings),
        raw_signals=raw_signals,
        provenance={"n_signals": len(raw_signals), "calibrated": did_calibrate},
    )
