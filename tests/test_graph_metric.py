import math

import pytest

from ek import score
from ek.base import FieldSpec, GraphGrammar, NodeType, EdgeType
from ek.facade import evaluate
from ek.metrics.graphs import TypedEdge, TypedGraph, TypedGraphMetric, TypedNode

networkx = pytest.importorskip("networkx")


def _grammar():
    return GraphGrammar(
        node_types={
            "donation": NodeType(
                "donation",
                fields={
                    "amount": FieldSpec("amount", "number", importance=10.0),
                    "city": FieldSpec("city", "string", importance=1.0),
                },
                importance=2.0,
            ),
            "donor": NodeType("donor", fields={"name": FieldSpec("name", "string")}),
        },
        edge_types={"made_by": EdgeType("made_by", "donation", "donor", importance=3.0)},
    )


def _gold():
    return TypedGraph([TypedNode("d1", "donation", {"amount": "100", "city": "Paris"})])


def test_identical_graphs_have_zero_distance():
    g = _grammar()
    s = score(_gold(), _gold(), grammar=g)  # auto-dispatches to the graph metric
    assert s.metric == "graph"
    assert s.value == 0.0
    assert s.detail["similarity"] == 1.0


def test_cost_weighting_prefers_cheap_field_errors():
    g = _grammar()
    wrong_amount = TypedGraph([TypedNode("d1", "donation", {"amount": "900", "city": "Paris"})])
    wrong_city = TypedGraph([TypedNode("d1", "donation", {"amount": "100", "city": "Lyon"})])
    amt = score(wrong_amount, _gold(), grammar=g)
    city = score(wrong_city, _gold(), grammar=g)
    # a wrong high-importance amount (10.0) costs 10x a wrong low-importance city (1.0)
    assert amt.detail["raw_distance"] == 10.0
    assert city.detail["raw_distance"] == 1.0
    assert amt.value > city.value


def test_structural_distance_without_grammar_uses_unit_costs():
    wrong = TypedGraph([TypedNode("d1", "donation", {"amount": "900", "city": "Paris"})])
    s = score(wrong, _gold())  # no grammar -> all weights 1.0
    assert s.detail["raw_distance"] == 1.0  # exactly one field differs, unit cost


def test_missing_node_costs_its_full_mass():
    g = _grammar()
    empty = TypedGraph([])
    s = score(empty, _gold(), grammar=g)
    # node mass = node_cost(2.0) + field_cost(amount=10) + field_cost(city=1) = 13
    assert s.detail["raw_distance"] == 13.0
    assert math.isclose(s.value, 1.0)  # deleting everything == max distance


def test_node_type_mismatch_is_a_full_relabel():
    g = _grammar()
    pred = TypedGraph([TypedNode("d1", "donor", {"name": "Acme"})])
    s = score(pred, _gold(), grammar=g)
    # incompatible types -> delete gold node + insert pred node (both masses)
    gold_mass = 2.0 + 10.0 + 1.0
    pred_mass = 1.0 + 1.0  # donor node_cost default 1.0 + name field 1.0
    assert s.detail["raw_distance"] == gold_mass + pred_mass


def test_edge_weight_counts():
    g = _grammar()
    gold = TypedGraph(
        nodes=[TypedNode("d1", "donation", {"amount": "100"}), TypedNode("p1", "donor", {})],
        edges=[TypedEdge("d1", "p1", "made_by")],
    )
    no_edge = TypedGraph(
        nodes=[TypedNode("d1", "donation", {"amount": "100"}), TypedNode("p1", "donor", {})],
    )
    s = score(no_edge, gold, grammar=g)
    assert s.detail["raw_distance"] == 3.0  # the missing made_by edge weight


def test_corpus_aggregate_is_globally_normalized():
    g = _grammar()
    wrong_amount = TypedGraph([TypedNode("d1", "donation", {"amount": "900", "city": "Paris"})])
    cases = [(_gold(), _gold()), (wrong_amount, _gold())]
    report = evaluate(cases, grammar=g, metric="graph")
    assert report.n == 2
    # per-case denom counts BOTH sides (delete-all-gold + insert-all-pred) = 13+13=26.
    # total raw = 0 + 10 ; total denom = 26 + 26 = 52.
    assert math.isclose(report.aggregate, 10 / 52, rel_tol=1e-9)


def test_weights_override_callable():
    # A CostWeight that makes every node/edge/field cost 5.0, ignoring the schema.
    def flat5(grammar, ref):
        return 5.0

    m = TypedGraphMetric(weights=flat5)
    wrong = TypedGraph([TypedNode("d1", "donation", {"amount": "900", "city": "Paris"})])
    s = m(wrong, _gold())
    assert s.detail["raw_distance"] == 5.0  # one differing field at flat cost 5.0


def test_graph_metric_applies_canonicalizer_and_field_normalizer():
    # Regression: facade normalize= was silently dropped for graph inputs, and
    # FieldSpec.normalizer was never applied by the graph metric.
    from ek.canonicalize import resolve_canonicalizer

    g_norm = GraphGrammar(
        node_types={"rec": NodeType("rec", fields={
            "name": FieldSpec("name", "string", normalizer="lower"),
        })},
        edge_types={},
    )
    a = TypedGraph(nodes=[TypedNode("n1", "rec", {"name": "ACME"})])
    b = TypedGraph(nodes=[TypedNode("n1", "rec", {"name": "acme"})])
    # FieldSpec.normalizer=lower -> the casing difference is folded away (distance 0).
    assert TypedGraphMetric(grammar=g_norm)(a, b).detail["raw_distance"] == 0.0
    # Without the normalizer, the facade-level canonicalizer must reach the metric.
    assert TypedGraphMetric(canonicalizer=resolve_canonicalizer("lower"))(a, b).detail["raw_distance"] == 0.0
    # And with neither, the casing difference is a real edit.
    assert TypedGraphMetric()(a, b).detail["raw_distance"] > 0.0


def test_graph_metric_node_count_guard():
    # GED is NP-hard: a pair above max_nodes is rejected, not silently spun.
    big = TypedGraph(nodes=[TypedNode(f"n{i}", "t", {}) for i in range(5)])
    with pytest.raises(ValueError, match="max_nodes"):
        TypedGraphMetric(max_nodes=3)(big, big)
    # within the bound, the score records its exactness
    small = TypedGraph(nodes=[TypedNode("n", "t", {"x": "a"})])
    assert TypedGraphMetric()(small, small).detail["exact"] is True
