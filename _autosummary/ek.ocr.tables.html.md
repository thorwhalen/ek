# ek.ocr.tables

Recover a normalized table from an `OcrResult` so ek’s table metrics can score it.

`ocracy` exposes **no** normalized table/cell type: `TextBlock.level` stops at
`block`, with no cell/row/column model, so OCR table structure survives only inside
the engine-specific `OcrResult.raw` (Textract `Block` objects, PaddleOCR
`pred_html`, Azure `tables[]`) or in the markdown/HTML an engine emits
(`res.markdown` for VLM engines). ek therefore can’t uniformly score table
structure (TEDS / GriTS) off the `OcrResult` shape alone.

This module is the **ek-side seam** that closes that gap *without touching ocracy*
(the dependency arrow stays `ek -> ocracy`, never the reverse). It turns an
`OcrResult`-shaped object into the existing [`ek.metrics.tables.Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) (the
one TEDS/GriTS already score) – it does **not** define a new table type.

Because every engine stores tables differently, the per-engine extractor is an
**injected** callable (open-closed / dependency-injection): pass `parser=` a
`(raw) -> Table | grid | html | None` function, or register one under the
`"table_parsers"` namespace and pass its name. With no parser,
[`table_from_ocr()`](#ek.ocr.tables.table_from_ocr) falls back to a few **safe, documented** heuristics over the
result’s `raw`/`markdown`/`meta` and returns `None` when no table structure is
present – it never guesses wildly.

Nothing here imports `ocracy` or any OCR engine: ek depends only on the
`OcrResult` *shape* (`.text` / `.raw` / `.markdown` / `.meta`), so this works
on any `image -> OcrResult` callable’s output.

### Example

```pycon
>>> from ek.metrics.tables import Table
>>> class R:  # a minimal OcrResult-shaped object carrying a 2-D grid in .raw
...     raw = {"tables": [[["a", "b"], ["c", "d"]]]}
>>> t = table_from_ocr(R())
>>> [[c.text for c in row] for row in t.rows]
[['a', 'b'], ['c', 'd']]
>>> has_table_structure(R())
True
>>> class Empty:  # no table anywhere in the result
...     raw = {"text": "just prose"}
>>> table_from_ocr(Empty()) is None
True
```

A registered third-party engine parser, used by name:

```pycon
>>> from ek.registry import register
>>> @register("table_parsers", "demo")
... def _demo(raw):
...     return raw.get("cells")  # this engine stores a grid under raw['cells']
>>> class D:
...     raw = {"cells": [["x"]]}
>>> table_from_ocr(D(), parser="demo").rows[0][0].text
'x'
```

### Module Attributes

| [`TABLE_PARSERS_NAMESPACE`](#ek.ocr.tables.TABLE_PARSERS_NAMESPACE)   | Registry namespace under which per-engine table parsers live.   |
|----------------------------------------------------------------------------|-----------------------------------------------------------------|

### Functions

| [`engine_yields_tables`](#ek.ocr.tables.engine_yields_tables)(engine)                  | Whether `engine`'s capability profile claims scoreable table structure.                                                  |
|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`has_table_structure`](#ek.ocr.tables.has_table_structure)(ocr_result, \*[, parser]) | Whether [`table_from_ocr()`](#ek.ocr.tables.table_from_ocr) recovers a non-empty table from `ocr_result`.  |
| [`resolve_table_parser`](#ek.ocr.tables.resolve_table_parser)(ref)                     | Resolve a table parser reference (a registered name or a callable) to a callable.                                        |
| [`table_from_ocr`](#ek.ocr.tables.table_from_ocr)(ocr_result, \*[, parser])      | Extract a normalized [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult`. |

### ek.ocr.tables.TABLE_PARSERS_NAMESPACE *= 'table_parsers'*

Registry namespace under which per-engine table parsers live. Register a parser
with `ek.registry.register("table_parsers", "<engine>", fn)` (or an
`ek.table_parsers` entry point) and resolve it by name via the `parser=` arg.

### ek.ocr.tables.engine_yields_tables(engine)

Whether `engine`’s capability profile claims scoreable table structure.

A convenience over [`ek.ocr.profile()`](ek.ocr.html.md#ek.ocr.profile): returns the engine profile’s `tables`
flag. ek treats the profile as a *prior* (Azure DI, AWS Textract, PaddleOCR and the
markdown engines are expected to yield scoreable tables; Google Vision is not), to
be verified empirically by actually running [`table_from_ocr()`](#ek.ocr.tables.table_from_ocr). Use it to skip
table scoring for engines that cannot emit cells in the first place.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.ocr.tables.has_table_structure(ocr_result, , parser=None)

Whether [`table_from_ocr()`](#ek.ocr.tables.table_from_ocr) recovers a non-empty table from `ocr_result`.

A thin predicate over [`table_from_ocr()`](#ek.ocr.tables.table_from_ocr) (same `parser` semantics): true iff
a table with at least one cell can be recovered. Use it to route an OCR result to
the table metrics vs the text-only CER/WER fallback.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.ocr.tables.resolve_table_parser(ref)

Resolve a table parser reference (a registered name or a callable) to a callable.

Mirrors how other ek strategies resolve from the registry (see
[`ek.registry.resolve()`](ek.registry.html.md#ek.registry.resolve)). A `str` is looked up under the `"table_parsers"`
namespace; an already-callable parser is returned unchanged.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ek.ocr.tables.table_from_ocr(ocr_result, , parser=None)

Extract a normalized [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult`.

The single seam ek’s table metrics (TEDS / GriTS) use to score OCR table output
uniformly. Returns the recovered Table, or `None` when the result carries no
recoverable table structure.

* **Parameters:**
  * **ocr_result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Any `OcrResult`-shaped object. ek reads only its `.raw` (the
    engine-specific payload), `.markdown` (VLM/markdown engines), and
    `.meta` – never ocracy types, so any `image -> OcrResult` callable’s
    output works.
  * **parser** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – 

    How to turn the raw payload into a table (dependency injection /
    open-closed). One of:
    - `None` (default): try the built-in **safe heuristics** – `.raw` is
      already a [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table), a 2-D grid, or a
      `<table>...</table>` HTML string; or `.raw`/`.meta` is a mapping
      carrying one of `{tables, table, cells, grid, html}`; or `.markdown`
      contains an HTML `<table>`. Returns `None` if none match – it never
      guesses wildly.
    - a `str`: the name of a parser registered under `"table_parsers"`
      (per-engine extractor), resolved from [`ek.registry`](ek.registry.html.md#module-ek.registry).
    - a callable `(raw) -> Table | grid | html | None`: an inline per-engine
      extractor. It receives the result’s `.raw` (falling back to the result
      itself when there is no `.raw`).
  * **through** (*Any non-None value an injected parser returns is fed*)
* **Return type:**
  [*Table*](ek.metrics.tables.html.md#ek.metrics.tables.Table) | *None*

:param `Table.coerce()`:
:param so a parser may yield a Table:
:param a 2-D grid:
:param or HTML.:

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)]
* **Returns:**
  A non-empty [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table), or `None` when no table
  structure is recoverable. An empty Table (no rows/cells) is normalized to
  `None` so the predicate [`has_table_structure()`](#ek.ocr.tables.has_table_structure) and downstream metrics
  see “no table”, not a degenerate one.
