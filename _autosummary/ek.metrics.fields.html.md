# ek.metrics.fields

Field-level metric for record / dict-shaped extractions.

Compares two flat records key by key into precision / recall / F1, with a wrong
value counting as both a spurious prediction and a missed gold (the conventional
slot-error accounting). It is dependency-free, so it ships in core for the common
“compare these two extracted records” case; the tag-sequence span metrics
(`seqeval` / `nervaluate`, with their *explicit* match schemes) and nested
metrics (ANLS\*) arrive behind the `[metrics]` extra – see `misc/docs/ek_02`.

Aggregation is micro-averaged over the corpus (sum TP/FP/FN, then divide), carried
in each [`Score`](ek.base.html.md#ek.base.Score)’s `detail` so [`evaluate()`](ek.facade.html.md#ek.facade.evaluate) does it
correctly rather than averaging per-record F1s.

### Classes

| [`FieldMetric`](#ek.metrics.fields.FieldMetric)(\*[, canonicalizer])   | Per-field precision/recall/F1 for two dict records.   |
|-------------------------------------------------------------------------------------|-------------------------------------------------------|

### *class* ek.metrics.fields.FieldMetric(, canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-field precision/recall/F1 for two dict records.

* **Parameters:**
  **canonicalizer** – Optional `str -> str` applied to string values before
  comparison (e.g. casefold/whitespace folding).

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
