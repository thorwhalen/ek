# ek.base

Foundational data structures and strategy protocols for `ek`.

This module is the single source of truth (SSOT) for `ek`’s *two-layer* data
model and for the small set of `typing.Protocol` strategy interfaces that every
swappable behaviour in the framework implements. It contains \*\*data structures
and contracts only\*\* – no business logic, no heavy imports. Importing it is
cheap and pulls in nothing beyond the standard library.

## The two layers

Evaluating an information-extraction (IE) output splits into two halves – scoring
against a gold reference (*offline*) and estimating quality with no reference
(*online*) – and both operate on the **same** object:

- **Layer A – the schema** ([`GraphGrammar`](#ek.base.GraphGrammar) and friends): a *frozen*
  description of the typed graph you extract, carrying both the **types** and the
  **importance/cost weights** per field/node/edge. It is the SSOT consulted by
  cost-sensitive metrics, by validators, and (eventually) by constrained decoders.
- **Layer B – the extraction metadata** ([`AnnotatedExtraction`](#ek.base.AnnotatedExtraction) and
  friends): per-extraction runtime metadata – provenance, raw signals,
  (calibrated) confidence, validator findings, and the accept/flag/block decision
  – keyed by a [`NodePath`](#ek.base.NodePath) into the Layer-A schema, riding *alongside* the
  grammar without ever mutating it.

## The strategy protocols

[`Metric`](#ek.base.Metric), [`Validator`](#ek.base.Validator), [`Calibrator`](#ek.base.Calibrator), [`DecisionPolicy`](#ek.base.DecisionPolicy)
and [`Signal`](#ek.base.Signal) are structural (duck-typed) contracts. Concrete
implementations are resolved from [`ek.registry`](ek.registry.html.md#module-ek.registry) by name and injected as
keyword-only arguments with smart defaults, so the simple path Just Works while
every layer stays replaceable (open-closed).

### Example

```pycon
>>> g = GraphGrammar(
...     node_types={"donation": NodeType(
...         "donation",
...         fields={
...             "amount": FieldSpec("amount", "number", importance=10.0),
...             "city": FieldSpec("city", "string", importance=1.0),
...         },
...     )},
... )
>>> g.field_cost("donation", "amount")
10.0
>>> g.field_cost("donation", "city")
1.0
```

### Module Attributes

| [`Normalizer`](#ek.base.Normalizer)       | Canonicalize a string before comparison (lowercase, fold, normalize, ...).                                          |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`ConfidenceSource`](#ek.base.ConfidenceSource) | Read named raw signals off a [`FieldEstimate`](#ek.base.FieldEstimate) (e.g. for calibration). |
| [`CostWeight`](#ek.base.CostWeight)       | Cost of an edit on a typed element; defaults to reading `*.importance`.                                             |
| [`OcrBackend`](#ek.base.OcrBackend)       | Anything that turns an image (path/bytes/array) into an `OcrResult`-shaped object (with `.text` and `.blocks`).     |

### Functions

| [`default_cost_weight`](#ek.base.default_cost_weight)(grammar, ref)   | The default [`CostWeight`](#ek.base.CostWeight): read the importance weight off the schema.   |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|

### Classes

| [`AnnotatedExtraction`](#ek.base.AnnotatedExtraction)(grammar[, estimates])        | Layer B: the verification metadata, referencing a frozen Layer-A grammar.                                               |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [`Calibrator`](#ek.base.Calibrator)(\*args, \*\*kwargs)                   | Maps a raw scalar signal to a probability after being `fit` on labels.                                                  |
| [`Decision`](#ek.base.Decision)(\*values)                               | Terminal action a selective-prediction gate emits for an extracted value.                                               |
| [`DecisionPolicy`](#ek.base.DecisionPolicy)(\*args, \*\*kwargs)               | Turns a (calibrated) confidence into an accept/flag/block [`Decision`](#ek.base.Decision).    |
| [`EdgeType`](#ek.base.EdgeType)(name, src, dst[, importance])           | A directed relation kind between two [`NodeType`](#ek.base.NodeType) names.                   |
| [`FieldEstimate`](#ek.base.FieldEstimate)(value[, raw_signals, ...])         | One extracted value plus all of its reference-free verification metadata.                                               |
| [`FieldSpec`](#ek.base.FieldSpec)(name[, type, importance, domain, ...]) | A leaf attribute of a node: its type, validation domain and cost weight.                                                |
| [`Finding`](#ek.base.Finding)(field, layer[, severity, message, ...])  | One validator observation about a value: a flag, optionally a correction.                                               |
| [`GraphGrammar`](#ek.base.GraphGrammar)([node_types, edge_types])           | The frozen schema SSOT: typed nodes/edges plus their importance weights.                                                |
| [`Metric`](#ek.base.Metric)(\*args, \*\*kwargs)                       | Compares one prediction to one gold reference, returning a [`Score`](#ek.base.Score).      |
| [`NodePath`](#ek.base.NodePath)(node_id, node_type[, field])            | Addresses a node (or one of its fields) in an extracted graph; keys Layer B.                                            |
| [`NodeType`](#ek.base.NodeType)(name[, fields, importance])             | A node kind in the typed graph (e.g. an "invoice" or a "line item").                                                    |
| [`Provenance`](#ek.base.Provenance)([engine, source_span, bbox, ...])     | Where an extracted value came from, for audit and click-to-source review.                                               |
| [`QualityReport`](#ek.base.QualityReport)([calibrated_confidence, ...])      | The result of a reference-free quality estimate for one extraction.                                                     |
| [`Report`](#ek.base.Report)([metric, aggregate, n, scores, ...])      | Aggregate of many [`Score`](#ek.base.Score) s over a corpus, with optional per-slice cuts. |
| [`Score`](#ek.base.Score)(value[, precision, recall, f1, ...])       | The result of comparing one prediction against one reference.                                                           |
| [`Severity`](#ek.base.Severity)(\*values)                               | Whether a [`Finding`](#ek.base.Finding) merely flags or can also correct a value.            |
| [`Signal`](#ek.base.Signal)(\*args, \*\*kwargs)                       | A reference-free quality signal: extractor output -> raw scalar score.                                                  |
| [`TypeRef`](#ek.base.TypeRef)(kind, name[, field])                     | A reference into the schema, so an injected cost function can read weights.                                             |
| [`Validator`](#ek.base.Validator)(\*args, \*\*kwargs)                    | Reference-free check on a value, yielding zero or more [`Finding`](#ek.base.Finding) s.      |

### *class* ek.base.AnnotatedExtraction(grammar, estimates=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Layer B: the verification metadata, referencing a frozen Layer-A grammar.

The grammar is held *by reference* and never mutated; `estimates` maps a
[`NodePath`](#ek.base.NodePath) to the [`FieldEstimate`](#ek.base.FieldEstimate) for that node/field.

### *class* ek.base.Calibrator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Maps a raw scalar signal to a probability after being `fit` on labels.

### ek.base.ConfidenceSource

Read named raw signals off a [`FieldEstimate`](#ek.base.FieldEstimate) (e.g. for calibration).

alias of `Callable`[[[`FieldEstimate`](#ek.base.FieldEstimate)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### ek.base.CostWeight

Cost of an edit on a typed element; defaults to reading `*.importance`.

alias of `Callable`[[[`GraphGrammar`](#ek.base.GraphGrammar), [`TypeRef`](#ek.base.TypeRef)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### *class* ek.base.Decision(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Terminal action a selective-prediction gate emits for an extracted value.

### *class* ek.base.DecisionPolicy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Turns a (calibrated) confidence into an accept/flag/block [`Decision`](#ek.base.Decision).

### *class* ek.base.EdgeType(name, src, dst, importance=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A directed relation kind between two [`NodeType`](#ek.base.NodeType) names.

### *class* ek.base.FieldEstimate(value, raw_signals=<factory>, confidence=None, findings=(), provenance=None, decision=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One extracted value plus all of its reference-free verification metadata.

### *class* ek.base.FieldSpec(name, type='string', importance=1.0, domain=(), normalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A leaf attribute of a node: its type, validation domain and cost weight.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Attribute name.
  * **type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A free-form type tag (e.g. `"string"`, `"number"`, `"date"`);
    `ek` does not impose a type system – the tag drives metric/validator
    dispatch and is yours to define.
  * **importance** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-field cost weight; the lever that makes “two extra digits
    on a donation amount” outweigh “a misspelled city”. Defaults to `1.0`.
  * **domain** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Optional allowed values: an enum tuple, a `(lo, hi)` range, or a
    regex string – consumed by validators, never by metrics.
  * **normalizer** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional key naming a canonicalizer (resolved from the registry)
    to apply to this field before comparison.

### *class* ek.base.Finding(field, layer, severity=Severity.FLAG, message='', suggestion=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One validator observation about a value: a flag, optionally a correction.

A `suggestion` is present iff the producing layer can *correct* (only the
deterministic canonicalizer and the gated LLM corrector freely do so). The
original value is always retained elsewhere; a Finding never mutates.

### *class* ek.base.GraphGrammar(node_types=<factory>, edge_types=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The frozen schema SSOT: typed nodes/edges plus their importance weights.

Methods return the cost weight for a referenced type, defaulting to `1.0` for
anything not declared, so a partial grammar is always usable.

#### cost(ref)

Importance weight for any [`TypeRef`](#ek.base.TypeRef) (the default `CostWeight`).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### edge_cost(name)

Importance weight of an edge type (`1.0` if undeclared).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### field_cost(node, field_name)

Importance weight of a field on a node type (`1.0` if undeclared).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### node_cost(name)

Importance weight of a node type (`1.0` if undeclared).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.base.Metric(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Compares one prediction to one gold reference, returning a [`Score`](#ek.base.Score).

### *class* ek.base.NodePath(node_id, node_type, field=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Addresses a node (or one of its fields) in an extracted graph; keys Layer B.

### *class* ek.base.NodeType(name, fields=<factory>, importance=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A node kind in the typed graph (e.g. an “invoice” or a “line item”).

### ek.base.Normalizer

Canonicalize a string before comparison (lowercase, fold, normalize, …).

alias of `Callable`[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ek.base.OcrBackend

Anything that turns an image (path/bytes/array) into an `OcrResult`-shaped
object (with `.text` and `.blocks`). Typed loosely so `ek` core never
imports an OCR package.

alias of `Callable`[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### *class* ek.base.Provenance(engine='', source_span=None, bbox=None, raw_transcripts=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Where an extracted value came from, for audit and click-to-source review.

`bbox` is intentionally typed `Any` so `ek` core need not import any OCR
package; the OCR instance stores normalized `[0, 1]` top-left-origin geometry.

### *class* ek.base.QualityReport(calibrated_confidence=None, decision=None, findings=(), raw_signals=<factory>, provenance=<factory>, per_field=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of a reference-free quality estimate for one extraction.

For a single value the scalar fields carry the verdict. For a whole
[`AnnotatedExtraction`](#ek.base.AnnotatedExtraction), `per_field` maps each [`NodePath`](#ek.base.NodePath) to its
updated [`FieldEstimate`](#ek.base.FieldEstimate) (confidence/findings/decision filled in), and the
scalar fields summarize: `calibrated_confidence` is the weakest field, and
`decision` the most severe (`block` > `flag` > `accept`).

### *class* ek.base.Report(metric='', aggregate=None, n=0, scores=<factory>, per_slice=<factory>, detail=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate of many [`Score`](#ek.base.Score) s over a corpus, with optional per-slice cuts.

`aggregate` is the corpus-level headline (computed by the metric’s own
aggregator – e.g. globally accumulated WER – not a naive mean of per-item
values). `per_slice` maps a slice label to its own aggregate.

### *class* ek.base.Score(value, precision=None, recall=None, f1=None, metric='', detail=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of comparing one prediction against one reference.

`value` is the headline number (higher is better unless a metric documents
otherwise). The optional decomposition fields let structured metrics carry
precision/recall/F1, an alignment, and raw counts (e.g. edit operations) so
that corpus aggregation can be done *correctly* (e.g. global WER accumulation)
rather than by averaging per-item scores. `float(score)` returns `value`.

### *class* ek.base.Severity(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Whether a [`Finding`](#ek.base.Finding) merely flags or can also correct a value.

### *class* ek.base.Signal(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A reference-free quality signal: extractor output -> raw scalar score.

The `cost_tier` attribute (lower is cheaper) lets an escalation policy run
cheap signals first and pay for expensive ones only on residual uncertainty.

### *class* ek.base.TypeRef(kind, name, field=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reference into the schema, so an injected cost function can read weights.

`kind` is one of `"node"`, `"edge"` or `"field"`; for a field, `name`
is the node-type name and `field` the field name.

### *class* ek.base.Validator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Reference-free check on a value, yielding zero or more [`Finding`](#ek.base.Finding) s.

### ek.base.default_cost_weight(grammar, ref)

The default [`CostWeight`](#ek.base.CostWeight): read the importance weight off the schema.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
