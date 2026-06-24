"""The flagship metric: cost-weighted, type-aware distance over a typed graph.

No off-the-shelf library takes a typed schema + per-field/per-type cost weights and
returns a weighted distance over the extracted graph -- it is *the* must-build of
this project (see ``misc/docs/ek_02`` and ``misc/docs/ek_06``). The mechanism is a
graph edit distance (GED) whose per-edit costs are supplied by ``ek``: relabelling
a node, deleting it, or changing a field each costs an amount read from the Layer-A
:class:`~ek.base.GraphGrammar` importance weights. That is how "two extra digits on
a monetary amount" outweighs "a misspelled city" -- no hardcoded costs, open-closed
via the schema.

GED is NP-hard; this uses ``networkx.graph_edit_distance`` with a configurable
``timeout`` (it returns the best edit cost found within the budget -- an
*approximation* on large graphs), normalizes the distance by the maximum possible
edit cost (delete-all-gold + insert-all-pred), and reports both the normalized
distance (``Score.value``; lower is better) and the derived similarity.

``networkx`` ships in the ``[metrics]`` extra and is imported lazily, so importing
``ek`` never pulls it in.

Example:
    >>> from ek import GraphGrammar, NodeType, FieldSpec, TypedGraph, TypedNode
    >>> g = GraphGrammar(node_types={"donation": NodeType("donation", fields={
    ...     "amount": FieldSpec("amount", "number", importance=10.0),
    ...     "city": FieldSpec("city", "string", importance=1.0)})})
    >>> gold = TypedGraph([TypedNode("d1", "donation", {"amount": "100", "city": "Paris"})])
    >>> # a wrong amount costs far more than a wrong city, per the schema weights:
    >>> wrong_amount = TypedGraph([TypedNode("d1", "donation", {"amount": "900", "city": "Paris"})])
    >>> wrong_city = TypedGraph([TypedNode("d1", "donation", {"amount": "100", "city": "Lyon"})])
    >>> m = TypedGraphMetric(grammar=g)
    >>> m(wrong_amount, gold).detail["raw_distance"]
    10.0
    >>> m(wrong_city, gold).detail["raw_distance"]
    1.0
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from ..base import CostWeight, GraphGrammar, Score, TypeRef
from ..registry import MissingExtraError

_MISSING = object()

#: Default ceiling on per-graph node count: graph edit distance is NP-hard, so a
#: larger pair is rejected (raise) rather than silently spinning. A keyword override
#: on :class:`TypedGraphMetric` lifts it for callers who accept the cost.
DEFAULT_MAX_NODES = 60


@dataclass(frozen=True)
class TypedNode:
    """A node in a typed graph: an id, a node-type tag, and its field values."""

    node_id: str
    node_type: str
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TypedEdge:
    """A directed, typed relation between two node ids."""

    src: str
    dst: str
    edge_type: str = ""


@dataclass(frozen=True)
class TypedGraph:
    """A typed graph: typed nodes plus directed typed edges between them."""

    nodes: Sequence[TypedNode] = ()
    edges: Sequence[TypedEdge] = ()

    def to_networkx(self):
        """Build a ``networkx.DiGraph`` carrying type/fields as node/edge attrs."""
        import networkx as nx

        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.node_id, type=n.node_type, fields=dict(n.fields))
        for e in self.edges:
            g.add_edge(e.src, e.dst, type=e.edge_type)
        return g


def _cost_resolvers(grammar: Optional[GraphGrammar], weights: Optional[CostWeight]):
    """Return ``(node_cost, edge_cost, field_cost)`` closures over schema/weights.

    With neither a grammar nor a weights callable, all costs are ``1.0`` (a plain
    structural GED). A ``weights`` :data:`~ek.base.CostWeight` overrides the schema.
    """
    g = grammar if grammar is not None else GraphGrammar()

    def node_cost(t: str) -> float:
        if weights is not None:
            return weights(g, TypeRef("node", t))
        return g.node_cost(t)

    def edge_cost(t: str) -> float:
        if weights is not None:
            return weights(g, TypeRef("edge", t))
        return g.edge_cost(t)

    def field_cost(t: str, f: str) -> float:
        if weights is not None:
            return weights(g, TypeRef("field", t, f))
        return g.field_cost(t, f)

    return node_cost, edge_cost, field_cost


def _graph_field_norms(grammar: Optional[GraphGrammar]) -> dict:
    """``(node-type, field) -> resolved canonicalizer`` from each ``FieldSpec.normalizer``."""
    if grammar is None:
        return {}
    from ..canonicalize import resolve_canonicalizer

    out: dict = {}
    for tname, node_type in grammar.node_types.items():
        for fname, spec in node_type.fields.items():
            name = getattr(spec, "normalizer", None)
            if name:
                out[(tname, fname)] = resolve_canonicalizer(name)
    return out


def _apply(norm, value: Any) -> Any:
    """Apply a canonicalizer to a string value (pass non-strings/sentinels through)."""
    return norm(value) if (norm is not None and isinstance(value, str)) else value


class TypedGraphMetric:
    """Cost-weighted, type-aware typed-graph edit distance as a :class:`~ek.base.Metric`.

    Args:
        grammar: Layer-A :class:`~ek.base.GraphGrammar` supplying cost weights (may
            also be passed per-call to ``__call__``).
        weights: Optional :data:`~ek.base.CostWeight` overriding the schema weights.
        canonicalizer: Optional ``str -> str`` applied to field values before the
            equality check (the facade passes ``normalize=`` through here); a schema's
            per-field ``FieldSpec.normalizer`` takes precedence for the fields it names.
        timeout: Seconds budget for the (NP-hard) GED search; on timeout the best
            cost found so far is used (an approximation). Default 10.0.
    """

    name = "graph"

    def __init__(
        self,
        grammar: Optional[GraphGrammar] = None,
        weights: Optional[CostWeight] = None,
        *,
        canonicalizer=None,
        timeout: float = 10.0,
        max_nodes: int = DEFAULT_MAX_NODES,
    ):
        self.grammar = grammar
        self.weights = weights
        self.canonicalizer = canonicalizer
        self.timeout = timeout
        self.max_nodes = max_nodes

    def __call__(
        self,
        pred: TypedGraph,
        gold: TypedGraph,
        *,
        grammar: Optional[GraphGrammar] = None,
    ) -> Score:
        try:
            import networkx as nx
        except Exception as exc:  # pragma: no cover - exercised only without networkx
            raise MissingExtraError(
                "TypedGraphMetric needs networkx -- install it with:  pip install ek[metrics]"
            ) from exc

        active_grammar = grammar if grammar is not None else self.grammar
        node_cost, edge_cost, field_cost = _cost_resolvers(active_grammar, self.weights)
        # Per-field schema normalizers (precedence) over the facade canonicalizer.
        field_norms = _graph_field_norms(active_grammar)

        def _field_eq(t: str, f: str, va: Any, vb: Any) -> bool:
            norm = field_norms.get((t, f), self.canonicalizer)
            return _apply(norm, va) == _apply(norm, vb)

        def node_mass(attrs: Mapping) -> float:
            t = attrs["type"]
            return node_cost(t) + sum(field_cost(t, f) for f in attrs["fields"])

        def node_subst(a: Mapping, b: Mapping) -> float:
            ta, tb = a["type"], b["type"]
            if ta != tb:  # incompatible types: a full relabel (delete + insert)
                return node_mass(a) + node_mass(b)
            fa, fb = a["fields"], b["fields"]
            return sum(
                field_cost(ta, f)
                for f in set(fa) | set(fb)
                if not _field_eq(ta, f, fa.get(f, _MISSING), fb.get(f, _MISSING))
            )

        def edge_subst(a: Mapping, b: Mapping) -> float:
            return (
                0.0
                if a.get("type") == b.get("type")
                else max(edge_cost(a.get("type", "")), edge_cost(b.get("type", "")))
            )

        gp, gg = pred.to_networkx(), gold.to_networkx()

        denom = (
            sum(node_mass(gg.nodes[n]) for n in gg.nodes)
            + sum(edge_cost(gg.edges[e].get("type", "")) for e in gg.edges)
            + sum(node_mass(gp.nodes[n]) for n in gp.nodes)
            + sum(edge_cost(gp.edges[e].get("type", "")) for e in gp.edges)
        )

        if denom == 0:  # both graphs empty -> identical
            return Score(
                value=0.0,
                metric="graph",
                detail={
                    "raw_distance": 0.0,
                    "denom": 0.0,
                    "similarity": 1.0,
                    "higher_is_better": False,
                    "exact": True,
                },
            )

        n_nodes = max(gp.number_of_nodes(), gg.number_of_nodes())
        if self.max_nodes is not None and n_nodes > self.max_nodes:
            raise ValueError(
                f"TypedGraphMetric: {n_nodes} nodes exceeds max_nodes={self.max_nodes}. "
                "Graph edit distance is NP-hard; raise max_nodes only if you accept the "
                "cost, or pre-reduce the graphs."
            )

        # Time the (budgeted) NP-hard search so the result's exactness is recorded:
        # a run that consumes its whole timeout was truncated to a best-so-far bound
        # and is only approximate (otherwise the score is wall-clock dependent and the
        # approximation is invisible).
        started = time.perf_counter()
        raw = nx.graph_edit_distance(
            gp,
            gg,
            node_subst_cost=lambda a, b: node_subst(a, b),
            node_del_cost=lambda a: node_mass(a),
            node_ins_cost=lambda a: node_mass(a),
            edge_subst_cost=lambda a, b: edge_subst(a, b),
            edge_del_cost=lambda a: edge_cost(a.get("type", "")),
            edge_ins_cost=lambda a: edge_cost(a.get("type", "")),
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - started
        if raw is None:  # no path within budget -> fall back to the max (delete+insert)
            raw = denom
        exact = self.timeout is None or elapsed < 0.95 * self.timeout

        normalized = min(1.0, raw / denom)
        return Score(
            value=normalized,
            metric="graph",
            detail={
                "raw_distance": raw,
                "denom": denom,
                "similarity": 1.0 - normalized,
                "higher_is_better": False,
                "exact": exact,
                "timeout_s": self.timeout,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Corpus normalized distance = total raw distance / total max-distance."""
        total_raw = sum(s.detail.get("raw_distance", 0.0) for s in scores)
        total_denom = sum(s.detail.get("denom", 0.0) for s in scores)
        return (total_raw / total_denom) if total_denom else 0.0
