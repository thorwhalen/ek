import json

from ek.base import (
    AnnotatedExtraction,
    Decision,
    EdgeType,
    FieldEstimate,
    FieldSpec,
    GraphGrammar,
    NodePath,
    NodeType,
    Provenance,
    Score,
    TypeRef,
    default_cost_weight,
)


def _grammar():
    return GraphGrammar(
        node_types={
            "donation": NodeType(
                "donation",
                fields={
                    "amount": FieldSpec("amount", "number", importance=10.0),
                    "city": FieldSpec("city", "string", importance=1.0),
                },
                importance=3.0,
            )
        },
        edge_types={"made_by": EdgeType("made_by", "donation", "donor", importance=2.0)},
    )


def test_cost_weights_live_on_layer_a():
    g = _grammar()
    assert g.field_cost("donation", "amount") == 10.0
    assert g.field_cost("donation", "city") == 1.0
    assert g.node_cost("donation") == 3.0
    assert g.edge_cost("made_by") == 2.0
    # undeclared things default to 1.0, so a partial grammar is always usable
    assert g.field_cost("donation", "nope") == 1.0
    assert g.node_cost("nope") == 1.0


def test_default_cost_weight_reads_importance():
    g = _grammar()
    assert default_cost_weight(g, TypeRef("field", "donation", "amount")) == 10.0
    assert default_cost_weight(g, TypeRef("node", "donation")) == 3.0
    assert default_cost_weight(g, TypeRef("edge", "made_by")) == 2.0


def test_layer_b_references_grammar_without_mutating():
    g = _grammar()
    est = FieldEstimate(value="100", confidence=0.9, decision=Decision.ACCEPT)
    ax = AnnotatedExtraction(
        grammar=g,
        estimates={NodePath("d1", "donation", "amount"): est},
    )
    assert ax.grammar is g  # held by reference
    assert ax.estimates[NodePath("d1", "donation", "amount")].value == "100"


def test_decision_enum_is_json_serializable():
    # str-subclassed enums serialize transparently (important for dol JSON stores)
    assert json.dumps({"decision": Decision.FLAG}) == '{"decision": "flag"}'


def test_score_behaves_like_a_float():
    s = Score(value=0.25, metric="cer")
    assert float(s) == 0.25
    assert s.metric == "cer"


def test_provenance_defaults_are_safe():
    p = Provenance(engine="tesseract")
    assert p.bbox is None
    assert tuple(p.raw_transcripts) == ()
