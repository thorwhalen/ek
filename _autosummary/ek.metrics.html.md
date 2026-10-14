# ek.metrics

Reference-based (offline) metrics: the `score()` side of `ek`.

A [`Metric`](ek.base.html.md#ek.base.Metric) compares one prediction to one gold reference and
returns a [`Score`](ek.base.html.md#ek.base.Score). The *right* metric is a function of the output
**object type** – a string wants CER/WER, a record wants field-level F1, a table
wants TEDS/GriTS, a typed graph wants a cost-weighted edit distance. This package
holds the concrete metrics and registers each under the `"metrics"` namespace so
the [`score()`](ek.facade.html.md#ek.facade.score) facade can dispatch by type or by name.

Shipping in core (zero/low-dependency): [`StringMetric`](#ek.metrics.StringMetric) (CER/WER, via
`jiwer` when present) and [`FieldMetric`](#ek.metrics.FieldMetric) (record/dict field F1). The richer
metrics ship behind the `[metrics]` extra and import their backends lazily, so
constructing or registering them never pulls a heavy dependency:

- [`AnlsMetric`](ek.metrics.anls.html.md#ek.metrics.anls.AnlsMetric) – ANLS / ANLS\* (nested-JSON), via `anls_star`.
- [`SpanF1Metric`](ek.metrics.spans.html.md#ek.metrics.spans.SpanF1Metric) – span/slot P/R/F1 via `seqeval` AND
  `nervaluate`, under an **explicit, required** [`MatchScheme`](ek.metrics.spans.html.md#ek.metrics.spans.MatchScheme).
- [`TedsMetric`](ek.metrics.tables.html.md#ek.metrics.tables.TedsMetric) / [`GritsMetric`](ek.metrics.tables.html.md#ek.metrics.tables.GritsMetric) –
  table-structure metrics (TEDS/TEDS-Struct, GriTS-Top/GriTS-Con) with a
  `structure_only` toggle.
- [`TypedGraphMetric`](ek.metrics.graphs.html.md#ek.metrics.graphs.TypedGraphMetric) – the flagship cost-weighted typed
  graph edit distance.

See `misc/docs/ek_02` (the reference-based decision table and gotchas).

### Classes

| [`StringMetric`](#ek.metrics.StringMetric)([mode, canonicalizer])       | CER (`mode='cer'`) or WER (`mode='wer'`) as a [`Metric`](ek.base.html.md#ek.base.Metric).            |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`FieldMetric`](#ek.metrics.FieldMetric)(\*[, canonicalizer])          | Per-field precision/recall/F1 for two dict records.                                                                              |
| [`TypedGraphMetric`](#ek.metrics.TypedGraphMetric)([grammar, weights, ...]) | Cost-weighted, type-aware typed-graph edit distance as a [`Metric`](ek.base.html.md#ek.base.Metric). |
| [`TypedGraph`](#ek.metrics.TypedGraph)([nodes, edges])                | A typed graph: typed nodes plus directed typed edges between them.                                                               |
| [`TypedNode`](#ek.metrics.TypedNode)(node_id, node_type[, fields])   | A node in a typed graph: an id, a node-type tag, and its field values.                                                           |
| [`TypedEdge`](#ek.metrics.TypedEdge)(src, dst[, edge_type])          | A directed, typed relation between two node ids.                                                                                 |
| [`AnlsMetric`](#ek.metrics.AnlsMetric)(\*[, threshold])               | ANLS / ANLS\* (nested-JSON) similarity as a [`Metric`](ek.base.html.md#ek.base.Metric).              |
| [`SpanF1Metric`](#ek.metrics.SpanF1Metric)([scheme, tagging_scheme])    | Entity/slot precision/recall/F1 under an **explicit** [`MatchScheme`](#ek.metrics.MatchScheme).              |
| [`MatchScheme`](#ek.metrics.MatchScheme)(\*values)                     | The (backend, scheme) contract a span/slot F1 is computed under.                                                                 |
| [`TedsMetric`](#ek.metrics.TedsMetric)(\*[, structure_only])          | TEDS / TEDS-Struct as a [`Metric`](ek.base.html.md#ek.base.Metric) (clean-room on `apted`).          |
| [`GritsMetric`](#ek.metrics.GritsMetric)(\*[, structure_only])         | GriTS-Top / GriTS-Con as a [`Metric`](ek.base.html.md#ek.base.Metric) (pure-Python, P/R aware).      |
| [`Table`](#ek.metrics.Table)([rows])                             | A table as an ordered list of rows of [`Cell`](#ek.metrics.Cell) s (the logical cells).               |
| [`Cell`](#ek.metrics.Cell)([text, rowspan, colspan])            | One logical table cell: its text plus its row/column span (>=1 each).                                                            |

### *class* ek.metrics.AnlsMetric(, threshold=0.5)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

ANLS / ANLS\* (nested-JSON) similarity as a [`Metric`](ek.base.html.md#ek.base.Metric).

Handles a bare string (classic ANLS) or an arbitrarily nested
dict/list/tuple/`None` structure (ANLS\*); the backend dispatches on type.

* **Parameters:**
  **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – ANLS zeroes a per-leaf similarity below this value before
  averaging (the classic ANLS 0.5 cut tolerates minor OCR/spelling
  noise). `anls_star` exposes no parameter for this – it reads a
  class constant `ANLSTree.THRESHOLD` (default `0.5`) – so the metric
  applies the threshold by setting that constant around the call and
  restoring it. **Caveat:** this mutates a process-global, so it is not
  thread-safe; concurrent `AnlsMetric` calls with different thresholds in
  the same process can interfere. Defaults to `0.5`.

#### aggregate(scores)

Corpus ANLS = mean of per-item ANLS (each is already length-normalized).

Unlike CER/WER (which accumulate global edit counts) or field-F1 (micro-TP/
FP/FN), ANLS is defined as the *average* normalized similarity over the
evaluation set – the “A” in ANLS – so the corpus statistic is the plain
mean of the per-item scores.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.Cell(text='', rowspan=1, colspan=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One logical table cell: its text plus its row/column span (>=1 each).

### *class* ek.metrics.FieldMetric(, canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-field precision/recall/F1 for two dict records.

* **Parameters:**
  **canonicalizer** – Optional `str -> str` applied to string values before
  comparison (e.g. casefold/whitespace folding).

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.GritsMetric(, structure_only=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

GriTS-Top / GriTS-Con as a [`Metric`](ek.base.html.md#ek.base.Metric) (pure-Python, P/R aware).

* **Parameters:**
  **structure_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True`, score cell topology only (GriTS-Top); when
  `False` (default), average topology with cell-content similarity
  (GriTS-Con).

#### aggregate(scores)

Corpus GriTS F1: pool overlap and per-table cell counts, then divide once.

GriTS precision = overlap / 

```
|pred cells|
```

 and recall = overlap / 

```
|gold cells|
```

;
the corpus statistic micro-averages by summing the numerators and
denominators globally – never a mean of per-table F1s.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.MatchScheme(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

The (backend, scheme) contract a span/slot F1 is computed under.

The two backends legitimately disagree (seqeval ignores other-type tags at the
tag level; nervaluate includes them), so the scheme is always explicit. The
`SEQEVAL_*` members consume BIO/IOBES *tag sequences*; the others consume
SemEval-2013 *span lists* via nervaluate.

#### EXACT *= 'exact'*

exact span boundaries, type ignored.

* **Type:**
  nervaluate exact

#### PARTIAL *= 'partial'*

boundary *overlap* (half credit), type ignored.

* **Type:**
  nervaluate partial

#### SEQEVAL_CONLL *= 'seqeval_conll'*

seqeval, CoNLL `conlleval`-compatible entity F1 (the default seqeval mode).

#### SEQEVAL_STRICT *= 'seqeval_strict'*

span boundaries AND type must match, keyed to a scheme.

* **Type:**
  seqeval strict mode

#### STRICT *= 'strict'*

exact span boundaries AND correct type.

* **Type:**
  nervaluate SemEval-2013 strict

#### TYPE *= 'type'*

correct type with at least some span overlap.

* **Type:**
  nervaluate type

### *class* ek.metrics.SpanF1Metric(scheme=None, , tagging_scheme='IOB2')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Entity/slot precision/recall/F1 under an **explicit** [`MatchScheme`](#ek.metrics.MatchScheme).

* **Parameters:**
  * **scheme** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`MatchScheme`](ek.metrics.spans.html.md#ek.metrics.spans.MatchScheme)]) – REQUIRED. The (backend, scheme) contract – there is deliberately no
    default, because seqeval and nervaluate disagree and a bare F1 is
    ambiguous (`misc/docs/ek_02` §2.2). Pass a [`MatchScheme`](#ek.metrics.MatchScheme).
  * **tagging_scheme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – For [`MatchScheme.SEQEVAL_STRICT`](#ek.metrics.MatchScheme.SEQEVAL_STRICT) only: the BIO/IOBES
    tagging scheme name (e.g. `"IOB2"`, `"IOBES"`, `"BILOU"`) that
    strict-mode entity decoding is keyed to. Defaults to `"IOB2"`.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `scheme` is omitted (the whole point of this metric).

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum the raw counts, then divide once).

seqeval schemes carry TP/FP/FN; nervaluate schemes carry COR/INC/PAR/MIS/SPU
(PAR worth half). Both micro-average by summing counts globally – never a
mean of per-document F1s. The two backends’ counts are not commensurable, so
a corpus mixing them is rejected rather than silently dropping one.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.StringMetric(mode='cer', , canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

CER (`mode='cer'`) or WER (`mode='wer'`) as a [`Metric`](ek.base.html.md#ek.base.Metric).

* **Parameters:**
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"cer"` (character) or `"wer"` (word) error rate.
  * **canonicalizer** – Optional `str -> str` applied to both sides before scoring
    (see [`ek.canonicalize`](ek.canonicalize.html.md#module-ek.canonicalize)). Usually supplied by the facade instead.

#### aggregate(scores)

Corpus error rate = total edits / total reference length (the correct way).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.Table(rows=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A table as an ordered list of rows of [`Cell`](#ek.metrics.Cell) s (the logical cells).

`rows[i]` is the i-th row’s logical cells in reading order; spans are carried
on each [`Cell`](#ek.metrics.Cell). Use [`from_html()`](#ek.metrics.Table.from_html) / [`from_grid()`](#ek.metrics.Table.from_grid) to build one,
and [`as_grid()`](#ek.metrics.Table.as_grid) for the dense 2-D occupancy view GriTS scores.

#### as_grid()

Expand spans into a dense 2-D grid; a spanned position repeats its cell.

Each physical grid position holds the (shared) [`Cell`](#ek.metrics.Cell) covering it, so
two tables with the same topology have grids of the same shape – the view
GriTS aligns position-by-position.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Cell`](ek.metrics.tables.html.md#ek.metrics.tables.Cell)]]]

#### *classmethod* coerce(obj)

Coerce a [`Table`](#ek.metrics.Table), an HTML string, or a 2-D grid into a Table.

* **Return type:**
  [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)

#### *classmethod* from_grid(grid)

Build a Table from a 2-D grid of cell texts (`None` = spanned-over).

* **Return type:**
  [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)

#### *classmethod* from_html(html)

Parse an HTML `<table>` (with `colspan`/`rowspan`) into a Table.

* **Return type:**
  [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)

### *class* ek.metrics.TedsMetric(, structure_only=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

TEDS / TEDS-Struct as a [`Metric`](ek.base.html.md#ek.base.Metric) (clean-room on `apted`).

* **Parameters:**
  **structure_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True`, ignore cell text and score structure alone
  (TEDS-Struct), isolating table-structure accuracy from OCR noise.
  Defaults to `False` (full TEDS: structure + content).

#### aggregate(scores)

Corpus TEDS = **mean** of per-table TEDS (the PubTabNet/OmniDocBench
convention). Each per-table TEDS is already node-count-normalized, so the
reported corpus statistic is the average, not a pooled edit-rate. The pooled
`1 - sum(dist)/sum(nodes)` is available from the per-item `detail` if a
size-weighted variant is wanted instead.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.metrics.TypedEdge(src, dst, edge_type='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A directed, typed relation between two node ids.

### *class* ek.metrics.TypedGraph(nodes=(), edges=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A typed graph: typed nodes plus directed typed edges between them.

#### to_networkx()

Build a `networkx.DiGraph` carrying type/fields as node/edge attrs.

### *class* ek.metrics.TypedGraphMetric(grammar=None, weights=None, , canonicalizer=None, timeout=10.0, max_nodes=60)

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

### *class* ek.metrics.TypedNode(node_id, node_type, fields=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A node in a typed graph: an id, a node-type tag, and its field values.

### Modules

| [`anls`](ek.metrics.anls.html.md#module-ek.metrics.anls)       | ANLS / ANLS\* (nested-JSON) similarity, wrapping `anls_star`.                  |
|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`fields`](ek.metrics.fields.html.md#module-ek.metrics.fields)   | Field-level metric for record / dict-shaped extractions.                       |
| [`graphs`](ek.metrics.graphs.html.md#module-ek.metrics.graphs)   | The flagship metric: cost-weighted, type-aware distance over a typed graph.    |
| [`spans`](ek.metrics.spans.html.md#module-ek.metrics.spans)     | Span / slot precision-recall-F1 over two backends that legitimately disagree.  |
| [`strings`](ek.metrics.strings.html.md#module-ek.metrics.strings) | String metrics: Character/Word Error Rate (CER/WER).                           |
| [`tables`](ek.metrics.tables.html.md#module-ek.metrics.tables)   | Table-structure metrics: TEDS / TEDS-Struct and GriTS (structure-only toggle). |
