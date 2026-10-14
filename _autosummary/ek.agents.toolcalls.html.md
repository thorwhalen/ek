# ek.agents.toolcalls

Tool-call correctness: BFCL-style AST matching over a *multiset* of calls.

The Berkeley Function Calling Leaderboard defines the reference algorithm: parse the emitted
call and – **without executing it** – check that the function name matches, that every
expected argument is present with the right value (case-insensitive, whitespace-folded string
comparison), and that no argument outside the schema was hallucinated. Arguments are matched
**by name**, so argument *order* is irrelevant; element order *within a list value* is not
(`misc/docs/ek_08`, `misc/docs/ek_12`).

**Why this cannot just reuse** [`FieldMetric`](ek.metrics.fields.html.md#ek.metrics.fields.FieldMetric). That metric compares two
flat records *keyed by field name* – the key **is** the alignment. A trajectory’s tool calls are
a **keyless multiset**: two `search(q=...)` calls collide onto one key, and reordering or
repeating a call silently miscounts. So this module adds the missing layer – an **assignment**
between predicted and gold calls – and then reuses `FieldMetric`’s exact TP/FP/FN accounting
and micro-averaged aggregation on top of it.

Cost-sensitivity is the reason to build this rather than borrow one: the per-argument weights
come from the Layer-A grammar (`FieldSpec.importance`), so a wrong argument to a *destructive*
tool is not one unit of error. No off-the-shelf tool-call metric models that.

### Example

```pycon
>>> from ek.agents.base import Step
>>> pred = [Step("search", {"q": "cats"}), Step("answer", {"a": "meow"})]
>>> gold = [("search", {"q": "cats"}), ("answer", {"a": "meow"})]
>>> ToolCallMetric()(pred, gold).f1
1.0
>>> hallucinated = [Step("search", {"q": "cats"}), Step("delete_all", {})]
>>> round(ToolCallMetric()(hallucinated, gold).f1, 3)     # 1 TP, 1 FP, 1 FN
0.5
```

### Functions

| [`match_calls`](#ek.agents.toolcalls.match_calls)(pred, gold, \*[, canon])   | Assign predicted calls to gold calls -- the layer `FieldMetric` does not provide.   |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|

### Classes

| [`ToolCallMetric`](#ek.agents.toolcalls.ToolCallMetric)([level, canonicalizer, grammar])   | Tool-call correctness (BFCL AST match) as a cost-weighted [`Metric`](ek.base.html.md#ek.base.Metric).   |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|

### *class* ek.agents.toolcalls.ToolCallMetric(level='call', , canonicalizer=None, grammar=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Tool-call correctness (BFCL AST match) as a cost-weighted [`Metric`](ek.base.html.md#ek.base.Metric).

* **Parameters:**
  * **level** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"call"` (default) scores whole calls – a call is a TP only on an exact AST
    match; `"arg"` gives partial credit at argument granularity.
  * **canonicalizer** – `str -> str` applied to string arguments before comparison
    (BFCL’s case/whitespace folding). Defaults to `ek`’s default canonicalizer.
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Layer-A grammar supplying per-tool and per-argument cost weights (may also be
    passed per call).

The `Score` carries `tp`/`fp`/`fn` in `detail` so [`ek.evaluate()`](ek.html.md#ek.evaluate) aggregates a
correct **micro**-averaged F1 over the corpus rather than averaging per-episode F1s.

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide) – never a mean.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.toolcalls.match_calls(pred, gold, , canon=None)

Assign predicted calls to gold calls – the layer `FieldMetric` does not provide.

Calls are grouped by tool name (a call to the *wrong tool* is never a match). Within a tool
name, candidate pairs are scored and taken **globally best-first** (exact-argument matches
before partial ones, more agreeing arguments before fewer), so identical repeated calls are
interchangeable and a reordered pair is still matched – while a strong pairing can never be
stolen by a weaker one that merely happened to come first in the prediction.

This is a deterministic greedy approximation to the optimal assignment. It is exact for the
common cases (distinct calls, repeats, reorderings); a full min-cost bipartite matching
(Hungarian) could differ only on pathological many-to-many partial-overlap sets, and is not
worth a dependency here.

Returns `(pairs, unmatched_pred, unmatched_gold)`, `pairs` being `(pred_call, gold_call)`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> pairs, up, ug = match_calls([("s", {"q": "b"}), ("s", {"q": "a"})],
...                             [("s", {"q": "a"}), ("s", {"q": "b"})])
>>> len(pairs), up, ug          # order-insensitive within a tool name
(2, [], [])
```
