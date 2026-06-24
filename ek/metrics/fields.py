"""Field-level metric for record / dict-shaped extractions.

Compares two flat records key by key into precision / recall / F1, with a wrong
value counting as both a spurious prediction and a missed gold (the conventional
slot-error accounting). It is dependency-free, so it ships in core for the common
"compare these two extracted records" case; the tag-sequence span metrics
(``seqeval`` / ``nervaluate``, with their *explicit* match schemes) and nested
metrics (ANLS*) arrive behind the ``[metrics]`` extra -- see ``misc/docs/ek_02``.

Aggregation is micro-averaged over the corpus (sum TP/FP/FN, then divide), carried
in each :class:`~ek.base.Score`'s ``detail`` so :func:`~ek.facade.evaluate` does it
correctly rather than averaging per-record F1s.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

from ..base import GraphGrammar, Score

_MISSING = object()


def _f1(precision: float, recall: float) -> float:
    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )


def _field_normalizers(grammar: Optional[GraphGrammar]) -> dict:
    """Map ``field name -> resolved canonicalizer`` from each ``FieldSpec.normalizer``.

    A schema can declare a per-field canonicalizer (e.g. a date or currency folder)
    that must be applied to that field alone before comparison; without this the
    grammar's ``normalizer`` contract is silently ignored. Field names are taken
    across all node types (unique enough for a flat record)."""
    if grammar is None:
        return {}
    from ..canonicalize import resolve_canonicalizer

    out: dict = {}
    for node_type in grammar.node_types.values():
        for fname, spec in node_type.fields.items():
            name = getattr(spec, "normalizer", None)
            if name:
                out[fname] = resolve_canonicalizer(name)
    return out


def _apply(norm, value: Any) -> Any:
    """Apply a canonicalizer to a string value (pass non-strings through unchanged)."""
    return norm(value) if (norm is not None and isinstance(value, str)) else value


class FieldMetric:
    """Per-field precision/recall/F1 for two dict records.

    Args:
        canonicalizer: Optional ``str -> str`` applied to string values before
            comparison (e.g. casefold/whitespace folding).
    """

    def __init__(self, *, canonicalizer=None):
        self.canonicalizer = canonicalizer
        self.name = "fields"

    def __call__(
        self, pred: Mapping, gold: Mapping, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        if not isinstance(pred, Mapping) or not isinstance(gold, Mapping):
            raise TypeError(
                "FieldMetric compares Mapping records; got "
                f"{type(pred).__name__} vs {type(gold).__name__}"
            )
        # Per-field normalizers (from the schema) take precedence over the
        # facade-level canonicalizer for the fields they name.
        field_norms = _field_normalizers(grammar)
        tp = fp = fn = 0
        for key in set(gold) | set(pred):
            g = gold.get(key, _MISSING)
            p = pred.get(key, _MISSING)
            norm = field_norms.get(key, self.canonicalizer)
            if g is _MISSING:  # predicted a field that gold does not have
                fp += 1
            elif p is _MISSING:  # gold has a field the prediction missed
                fn += 1
            elif _apply(norm, p) == _apply(norm, g):
                tp += 1
            else:  # both present but disagree
                fp += 1
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
        f1 = _f1(precision, recall)
        return Score(
            value=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            metric="fields",
            detail={"tp": tp, "fp": fp, "fn": fn, "higher_is_better": True},
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide)."""
        tp = sum(s.detail.get("tp", 0) for s in scores)
        fp = sum(s.detail.get("fp", 0) for s in scores)
        fn = sum(s.detail.get("fn", 0) for s in scores)
        if tp + fp + fn == 0:
            return 1.0  # nothing to extract and nothing extracted -> perfect (matches per-item)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return _f1(precision, recall)
