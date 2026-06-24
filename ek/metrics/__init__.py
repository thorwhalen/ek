"""Reference-based (offline) metrics: the ``score()`` side of ``ek``.

A :class:`~ek.base.Metric` compares one prediction to one gold reference and
returns a :class:`~ek.base.Score`. The *right* metric is a function of the output
**object type** -- a string wants CER/WER, a record wants field-level F1, a table
wants TEDS/GriTS, a typed graph wants a cost-weighted edit distance. This package
holds the concrete metrics and registers each under the ``"metrics"`` namespace so
the :func:`~ek.facade.score` facade can dispatch by type or by name.

Shipping in core (zero/low-dependency): :class:`StringMetric` (CER/WER, via
``jiwer`` when present) and :class:`FieldMetric` (record/dict field F1). The richer
metrics ship behind the ``[metrics]`` extra and import their backends lazily, so
constructing or registering them never pulls a heavy dependency:

- :class:`~ek.metrics.anls.AnlsMetric` -- ANLS / ANLS* (nested-JSON), via ``anls_star``.
- :class:`~ek.metrics.spans.SpanF1Metric` -- span/slot P/R/F1 via ``seqeval`` AND
  ``nervaluate``, under an **explicit, required** :class:`~ek.metrics.spans.MatchScheme`.
- :class:`~ek.metrics.tables.TedsMetric` / :class:`~ek.metrics.tables.GritsMetric` --
  table-structure metrics (TEDS/TEDS-Struct, GriTS-Top/GriTS-Con) with a
  ``structure_only`` toggle.
- :class:`~ek.metrics.graphs.TypedGraphMetric` -- the flagship cost-weighted typed
  graph edit distance.

See ``misc/docs/ek_02`` (the reference-based decision table and gotchas).
"""

from __future__ import annotations

from ..registry import register
from .anls import AnlsMetric
from .fields import FieldMetric
from .graphs import TypedEdge, TypedGraph, TypedGraphMetric, TypedNode
from .spans import MatchScheme, SpanF1Metric
from .strings import StringMetric
from .tables import Cell, GritsMetric, Table, TedsMetric

# Register the built-in metrics (idempotent; import side effect). Constructing these
# does NOT import any [metrics] backend (networkx/anls_star/seqeval/nervaluate/apted)
# -- that happens lazily on __call__, so importing ek stays light. SpanF1Metric needs
# an explicit scheme, so it is registered per-scheme under a stable name.
register("metrics", "cer", StringMetric(mode="cer"))
register("metrics", "wer", StringMetric(mode="wer"))
register("metrics", "fields", FieldMetric())
register("metrics", "graph", TypedGraphMetric())
register("metrics", "typed_graph", TypedGraphMetric())
register("metrics", "anls", AnlsMetric())
for _scheme in MatchScheme:
    register("metrics", f"span_f1.{_scheme.value}", SpanF1Metric(scheme=_scheme))
register("metrics", "teds", TedsMetric())
register("metrics", "teds_struct", TedsMetric(structure_only=True))
register("metrics", "grits", GritsMetric())
register("metrics", "grits_con", GritsMetric())
register("metrics", "grits_top", GritsMetric(structure_only=True))

__all__ = [
    "StringMetric",
    "FieldMetric",
    "TypedGraphMetric",
    "TypedGraph",
    "TypedNode",
    "TypedEdge",
    "AnlsMetric",
    "SpanF1Metric",
    "MatchScheme",
    "TedsMetric",
    "GritsMetric",
    "Table",
    "Cell",
]
