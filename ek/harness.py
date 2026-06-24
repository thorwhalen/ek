"""The offline evaluation harness: run a predictor over gold, gate regressions.

Reference-based benchmarking (Need #1) is only useful if it is *repeatable* and
*regression-safe*: a "better" system must not silently get worse on some slice. The
harness adds that discipline on top of :func:`ek.score`/:func:`ek.evaluate`:

- :func:`evaluate_store` -- run any ``input -> prediction`` predictor over a gold
  store (a dict or a ``ek`` gold store), scored per slice (grouped by each record's
  slice label), with results persisted. (The OCR benchmark in :mod:`ek.ocr` is the
  OCR-specific specialization of this.)
- :func:`save_baseline` / :func:`regression_gate` -- freeze a baseline and fail when
  a later run regresses beyond a tolerance, **per slice**, not just on the aggregate.
  This is the golden-set CI gate.
- :func:`cohen_kappa` / :func:`percent_agreement` / :func:`krippendorff_alpha` --
  inter-annotator agreement (IAA), to know the ceiling of your gold standard before
  trusting any model score (all pure-Python, no optional dependency).

See ``misc/docs/ek_02`` for the harness design (canonicalizer versioning, slicing,
IAA conventions, golden-set CI).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .facade import evaluate
from .stores import json_store

# Last-resort fallback only: metrics whose Score.value is an ERROR/distance.
# The authoritative source is each Score's detail['higher_is_better'].
_LOWER_IS_BETTER = {"cer", "wer", "graph", "typed_graph"}


def _higher_is_better(metric: str) -> bool:
    return metric not in _LOWER_IS_BETTER


def _direction_from_report(report: Any, metric: str) -> bool:
    """Resolve higher-is-better from the report's own Scores (authoritative; every
    metric stamps ``detail['higher_is_better']``), falling back to the metric-name set
    only when no Score carries the flag -- so custom/unlisted metrics gate correctly."""
    for s in getattr(report, "scores", None) or ():
        flag = (getattr(s, "detail", None) or {}).get("higher_is_better")
        if flag is not None:
            return bool(flag)
    return _higher_is_better(metric)


# ---------------------------------------------------------------------------
# Run a predictor over a gold store
# ---------------------------------------------------------------------------


def evaluate_store(
    predict: Callable[[Any], Any],
    gold: Mapping,
    *,
    metric: Optional[str] = None,
    grammar: Any = None,
    normalize: Any = None,
    weights: Any = None,
    input_key: str = "input",
    reference_key: str = "reference",
    slice_key: str = "slice",
    persist: bool = False,
    run_id: Optional[str] = None,
    rootdir: Optional[str] = None,
):
    """Run ``predict`` over a gold store and score it per slice.

    Args:
        predict: ``input -> prediction`` callable (the system under test).
        gold: Mapping ``key -> {input, reference, [slice], ...}`` (a dict or a
            ``ek`` gold store).
        metric: Metric name/callable (defaults to type-dispatch on the first case).
        grammar/normalize/weights: forwarded to scoring.
        input_key/reference_key/slice_key: field names within each gold record.
        persist/run_id/rootdir: persist the run to the ``results``/``runs`` stores.

    Returns:
        A :class:`~ek.base.Report`; ``detail['per_item']`` maps each key to its
        prediction, reference, slice, and score.
    """
    items = list(gold.items()) if isinstance(gold, Mapping) else list(gold)
    cases = []
    per_item: dict = {}
    for key, rec in items:
        prediction = predict(rec[input_key])
        reference = rec[reference_key]
        slice_label = rec.get(slice_key)
        cases.append((prediction, reference, slice_label))
        per_item[key] = {
            "prediction": prediction,
            "reference": reference,
            "slice": slice_label,
        }

    report = evaluate(
        cases, metric=metric, grammar=grammar, normalize=normalize, weights=weights
    )
    for key, sc in zip(per_item, report.scores):
        per_item[key]["score"] = sc.value
    report.detail["per_item"] = per_item

    if persist:
        if run_id is None:
            from datetime import datetime, timezone

            run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")
        summary = {
            "metric": report.metric,
            "aggregate": report.aggregate,
            "n": report.n,
            "per_slice": report.per_slice,
        }
        json_store("results", rootdir=rootdir)[run_id] = {
            **summary,
            "per_item": per_item,
        }
        json_store("runs", rootdir=rootdir)[run_id] = summary
    return report


# ---------------------------------------------------------------------------
# Baselines + regression gate
# ---------------------------------------------------------------------------


def save_baseline(report, name: str, *, rootdir: Optional[str] = None) -> dict:
    """Freeze a report's aggregate + per-slice scores as a named baseline."""
    record = {
        "metric": report.metric,
        "aggregate": report.aggregate,
        "per_slice": dict(report.per_slice),
        "n": report.n,
    }
    json_store("baselines", rootdir=rootdir)[name] = record
    return record


def load_baseline(name: str, *, rootdir: Optional[str] = None) -> Optional[dict]:
    """Load a named baseline (or ``None`` if it does not exist)."""
    store = json_store("baselines", rootdir=rootdir)
    return store[name] if name in store else None


@dataclass
class GateResult:
    """Outcome of a :func:`regression_gate` check."""

    passed: bool
    metric: str
    higher_is_better: bool
    tolerance: float
    aggregate_current: Optional[float] = None
    aggregate_baseline: Optional[float] = None
    regressions: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def _is_regression(
    current: float, baseline: float, *, higher_is_better: bool, tol: float
) -> bool:
    if current is None or baseline is None:
        return False
    return (
        (baseline - current) > tol if higher_is_better else (current - baseline) > tol
    )


def regression_gate(
    report,
    baseline: Any,
    *,
    tolerance: float = 0.0,
    higher_is_better: Optional[bool] = None,
    rootdir: Optional[str] = None,
) -> GateResult:
    """Fail if ``report`` regresses beyond ``tolerance`` vs a baseline, per slice.

    Args:
        report: The current :class:`~ek.base.Report`.
        baseline: A baseline name (loaded from the ``baselines`` store) or a baseline
            dict from :func:`save_baseline`.
        tolerance: Allowed drift before a change counts as a regression.
        higher_is_better: Override metric-direction inference (CER/WER/graph are
            lower-is-better; F1/similarity are higher-is-better).
        rootdir: data root (when ``baseline`` is a name).

    Returns:
        A :class:`GateResult` (falsy if any regression was found).
    """
    base = (
        load_baseline(baseline, rootdir=rootdir)
        if isinstance(baseline, str)
        else baseline
    )
    metric = report.metric
    hib = (
        _direction_from_report(report, metric)
        if higher_is_better is None
        else higher_is_better
    )
    if base is None:
        # No baseline yet: nothing to regress against -> pass (first run).
        return GateResult(
            passed=True,
            metric=metric,
            higher_is_better=hib,
            tolerance=tolerance,
            aggregate_current=report.aggregate,
        )

    base_metric = base.get("metric") if isinstance(base, dict) else None
    if base_metric and base_metric != metric:
        raise ValueError(
            f"baseline metric {base_metric!r} != report metric {metric!r}; refusing "
            "to compare incomparable scores (re-baseline against the new metric)."
        )

    regressions: dict = {}
    if _is_regression(
        report.aggregate, base.get("aggregate"), higher_is_better=hib, tol=tolerance
    ):
        regressions["__aggregate__"] = {
            "current": report.aggregate,
            "baseline": base.get("aggregate"),
        }
    base_slices = base.get("per_slice", {})
    for slice_label, cur in report.per_slice.items():
        if slice_label in base_slices and _is_regression(
            cur, base_slices[slice_label], higher_is_better=hib, tol=tolerance
        ):
            regressions[slice_label] = {
                "current": cur,
                "baseline": base_slices[slice_label],
            }

    return GateResult(
        passed=not regressions,
        metric=metric,
        higher_is_better=hib,
        tolerance=tolerance,
        aggregate_current=report.aggregate,
        aggregate_baseline=base.get("aggregate"),
        regressions=regressions,
    )


# ---------------------------------------------------------------------------
# Inter-annotator agreement (gold-standard ceiling)
# ---------------------------------------------------------------------------


def percent_agreement(rater_a: Sequence, rater_b: Sequence) -> float:
    """Raw fraction of items two raters labelled identically."""
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    if not rater_a:
        return 1.0
    return sum(1 for x, y in zip(rater_a, rater_b) if x == y) / len(rater_a)


def cohen_kappa(rater_a: Sequence, rater_b: Sequence) -> float:
    """Cohen's kappa for two raters on nominal labels (chance-corrected agreement).

    Use only for exactly two raters, nominal labels, complete data; prefer
    :func:`krippendorff_alpha` for the general case (missing data, any measurement
    level).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    n = len(rater_a)
    if n == 0:
        return 1.0
    labels = set(rater_a) | set(rater_b)
    p_observed = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n
    p_expected = sum(
        (rater_a.count(label) / n) * (rater_b.count(label) / n) for label in labels
    )
    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1.0 - p_expected)


def _is_missing(x: Any) -> bool:
    """Whether a cell is unlabelled (``None`` or a float ``NaN``)."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def _difference_metric(
    level: str, values: Sequence, n_v: Mapping
) -> Callable[[Any, Any], float]:
    """The squared difference function ``delta^2(c, k)`` for a measurement level.

    ``nominal`` (categorical), ``interval`` (numeric distance), ``ratio`` (relative
    distance with a true zero), and ``ordinal`` (rank distance via the cumulative
    marginal counts ``n_v``, Krippendorff's ordinal metric).
    """
    if level == "nominal":
        return lambda c, k: 1.0
    if level == "interval":
        return lambda c, k: (float(c) - float(k)) ** 2
    if level == "ratio":

        def ratio(c, k):
            s = float(c) + float(k)
            return 0.0 if s == 0 else ((float(c) - float(k)) / s) ** 2

        return ratio
    if level == "ordinal":
        ordered = sorted(values)

        def ordinal(c, k):
            lo, hi = (c, k) if c <= k else (k, c)
            between = sum(n_v[g] for g in ordered if lo <= g <= hi)
            return (between - (n_v[c] + n_v[k]) / 2.0) ** 2

        return ordinal
    raise ValueError(f"unknown level {level!r}; use nominal/ordinal/interval/ratio")


def krippendorff_alpha(
    reliability_data: Sequence[Sequence], *, level: str = "nominal"
) -> float:
    """Krippendorff's alpha -- the general inter-annotator agreement coefficient.

    Pure-Python (no dependency, permissive core): any number of raters, any
    measurement ``level`` (``"nominal"``, ``"ordinal"``, ``"interval"``, ``"ratio"``),
    and missing data (use ``None`` or ``float('nan')`` for an unlabelled cell).
    ``reliability_data`` is one row per rater, one column per item. Alpha is ``1.0``
    for perfect agreement, ``0.0`` at chance, and goes negative for systematic
    disagreement. Computed via the coincidence-matrix method (Krippendorff 2011):
    observed vs expected disagreement under the level-specific difference metric.

    Example:
        >>> data = [[1, 1, 1, 1], [1, 1, 1, 1]]   # two raters, perfect agreement
        >>> krippendorff_alpha(data)
        1.0
    """
    raters = [list(row) for row in reliability_data]
    n_units = max((len(r) for r in raters), default=0)

    # Per-unit value lists; only units rated by >= 2 raters are "pairable".
    units = []
    for u in range(n_units):
        vals = [r[u] for r in raters if u < len(r) and not _is_missing(r[u])]
        if len(vals) >= 2:
            units.append(vals)
    if not units:
        return 1.0

    # Coincidence matrix: o[(c, k)] over ordered value pairs within each unit,
    # each contribution weighted by 1 / (m_u - 1) for a unit with m_u ratings.
    coincidence: dict = {}
    for vals in units:
        m = len(vals)
        counts: dict = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        for c, cc in counts.items():
            for k, kc in counts.items():
                pairs = cc * (cc - 1) if c == k else cc * kc
                if pairs:
                    coincidence[(c, k)] = coincidence.get((c, k), 0.0) + pairs / (m - 1)

    values = list({v for vals in units for v in vals})
    n_v = {v: sum(coincidence.get((v, w), 0.0) for w in values) for v in values}
    n = sum(n_v.values())
    if n == 0:
        return 1.0

    delta2 = _difference_metric(level, values, n_v)
    observed = expected = 0.0
    for c in values:
        for k in values:
            if c == k:
                continue
            d2 = delta2(c, k)
            observed += coincidence.get((c, k), 0.0) * d2
            expected += n_v[c] * n_v[k] * d2
    if expected == 0:
        return 1.0  # no expected disagreement -> perfect agreement by convention
    return 1.0 - (n - 1) * observed / expected
