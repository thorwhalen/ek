# ek.metrics.graphs

The flagship metric: cost-weighted, type-aware distance over a typed graph.

No off-the-shelf library takes a typed schema + per-field/per-type cost weights and
returns a weighted distance over the extracted graph – it is *the* must-build of
this project (see `misc/docs/ek_02` and `misc/docs/ek_06`). The mechanism is a
graph edit distance (GED) whose per-edit costs are supplied by `ek`: relabelling
a node, deleting it, or changing a field each costs an amount read from the Layer-A
[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) importance weights. That is how “two extra digits on
a monetary amount” outweighs “a misspelled city” – no hardcoded costs, open-closed
via the schema.

GED is NP-hard; this uses `networkx.graph_edit_distance` with a configurable
`timeout` (it returns the best edit cost found within the budget – an
*approximation* on large graphs), normalizes the distance by the maximum possible
edit cost (delete-all-gold + insert-all-pred), and reports both the normalized
distance (`Score.value`; lower is better) and the derived similarity.

`networkx` ships in the `[metrics]` extra and is imported lazily, so importing
`ek` never pulls it in.

### Example

```pycon
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
```

### Module Attributes

| [`DEFAULT_MAX_NODES`](#ek.metrics.graphs.DEFAULT_MAX_NODES)   | graph edit distance is NP-hard, so a larger pair is rejected (raise) rather than silently spinning.   |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|

### Classes

| [`TypedEdge`](#ek.metrics.graphs.TypedEdge)(src, dst[, edge_type])          | A directed, typed relation between two node ids.                                                                                 |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`TypedGraph`](#ek.metrics.graphs.TypedGraph)([nodes, edges])                | A typed graph: typed nodes plus directed typed edges between them.                                                               |
| [`TypedGraphMetric`](#ek.metrics.graphs.TypedGraphMetric)([grammar, weights, ...]) | Cost-weighted, type-aware typed-graph edit distance as a [`Metric`](ek.base.html.md#ek.base.Metric). |
| [`TypedNode`](#ek.metrics.graphs.TypedNode)(node_id, node_type[, fields])   | A node in a typed graph: an id, a node-type tag, and its field values.                                                           |

### ek.metrics.graphs.DEFAULT_MAX_NODES *= 60*

graph edit distance is NP-hard, so a
larger pair is rejected (raise) rather than silently spinning. A keyword override
on [`TypedGraphMetric`](#ek.metrics.graphs.TypedGraphMetric) lifts it for callers who accept the cost.

* **Type:**
  Default ceiling on per-graph node count

### *class* ek.metrics.graphs.TypedEdge(src, dst, edge_type='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A directed, typed relation between two node ids.

### *class* ek.metrics.graphs.TypedGraph(nodes=(), edges=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A typed graph: typed nodes plus directed typed edges between them.

#### to_networkx()

Build a `networkx.DiGraph` carrying type/fields as node/edge attrs.

### *class* ek.metrics.graphs.TypedGraphMetric(grammar=None, weights=None, , canonicalizer=None, timeout=10.0, max_nodes=60)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cost-weighted, type-aware typed-graph edit distance as a [`Metric`](ek.base.html.md#ek.base.Metric).

* **Parameters:**
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) supplying cost weights (may
    also be passed per-call to `__call__`).
  * **weights** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar), [`TypeRef`](ek.base.html.md#ek.base.TypeRef)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Optional [`CostWeight`](ek.base.html.md#ek.base.CostWeight) overriding the schema weights.
  * **canonicalizer** – Optional `str -> str` applied to field values before the
    equality check (the facade passes `normalize=` through here); a schema’s
    per-field `FieldSpec.normalizer` takes precedence for the fields it names.
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds budget for the (NP-hard) GED search; on timeout the best
    cost found so far is used (an approximation). Default 10.0.

#### aggregate(scores)

Corpus normalized distance = total raw distance / total max-distance.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.graphs.TypedNode(node_id, node_type, fields=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A node in a typed graph: an id, a node-type tag, and its field values.
