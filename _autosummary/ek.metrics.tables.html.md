# ek.metrics.tables

Table-structure metrics: TEDS / TEDS-Struct and GriTS (structure-only toggle).

Two complementary table-recognition metrics from `misc/docs/ek_02` §3.1:

- **TEDS** (Tree-Edit-Distance-based Similarity; Zhong et al., PubTabNet, IBM,
  [arXiv 1911.10683](https://arxiv.org/abs/1911.10683)) renders a table as an HTML tree and scores
  `1 - EditDist(Ta, Tb) / max(|Ta|, |Tb|)`. It captures both *structure* and
  *cell content*. **TEDS-Struct** (the `structure_only=True` toggle) ignores cell
  text to isolate structure, since “taking OCR errors into account may lead to an
  unfair comparison due to the different OCR models used by various TSR methods”.
- **GriTS** (Grid Table Similarity; Smock et al., Microsoft, [arXiv 2203.12555](https://arxiv.org/abs/2203.12555))
  scores the predicted table directly as a 2-D matrix and yields **precision/recall**
  over the most-similar common sub-grid. Its **GriTS-Top** (topology, structure-only)
  and **GriTS-Con** (content) variants share the `structure_only` toggle.

Licensing note (`skills/ek-dev-licensing`): the obvious wrapper target,
`table-recognition-metric`, pulls in the **GPL** `Levenshtein` package as a hard
runtime dependency (`from Levenshtein import distance` in its source) – a
scanner-invisible copyleft landmine that would fail the CI license gate. So TEDS is
implemented **clean-room** here on the MIT `apted` engine (already in
`[metrics]`), and GriTS is a dependency-free pure-Python implementation of the
2D-MSS heuristic. Both follow the precedent of reimplementing Krippendorff’s alpha in
pure Python rather than quarantining a copyleft dependency.

## Input shapes

`pred` and `gold` are [`Table`](#ek.metrics.tables.Table) objects, or anything the constructors
accept: an HTML `str` (`<table>...</table>`; via [`Table.from_html()`](#ek.metrics.tables.Table.from_html)) or a
2-D grid – a list of rows of cell strings, where `None` marks a cell spanned over
by a neighbour (via [`Table.from_grid()`](#ek.metrics.tables.Table.from_grid)). The facade and registry coerce HTML
strings automatically.

### Example

```pycon
>>> from ek.metrics.tables import TedsMetric, GritsMetric, Table
>>> a = "<table><tr><td>a</td><td>b</td></tr></table>"
>>> b = "<table><tr><td>a</td><td>c</td></tr></table>"
>>> round(TedsMetric()(a, b).value, 3)            # one cell-content edit
0.75
>>> TedsMetric(structure_only=True)(a, b).value   # same structure -> 1.0
1.0
>>> GritsMetric(structure_only=True)(a, b).value  # identical grid topology
1.0
```

### Classes

| [`Cell`](#ek.metrics.tables.Cell)([text, rowspan, colspan])    | One logical table cell: its text plus its row/column span (>=1 each).                                                       |
|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`GritsMetric`](#ek.metrics.tables.GritsMetric)(\*[, structure_only]) | GriTS-Top / GriTS-Con as a [`Metric`](ek.base.html.md#ek.base.Metric) (pure-Python, P/R aware). |
| [`Table`](#ek.metrics.tables.Table)([rows])                     | A table as an ordered list of rows of [`Cell`](#ek.metrics.tables.Cell) s (the logical cells).          |
| [`TedsMetric`](#ek.metrics.tables.TedsMetric)(\*[, structure_only])  | TEDS / TEDS-Struct as a [`Metric`](ek.base.html.md#ek.base.Metric) (clean-room on `apted`).     |

### *class* ek.metrics.tables.Cell(text='', rowspan=1, colspan=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One logical table cell: its text plus its row/column span (>=1 each).

### *class* ek.metrics.tables.GritsMetric(, structure_only=False)

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

### *class* ek.metrics.tables.Table(rows=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A table as an ordered list of rows of [`Cell`](#ek.metrics.tables.Cell) s (the logical cells).

`rows[i]` is the i-th row’s logical cells in reading order; spans are carried
on each [`Cell`](#ek.metrics.tables.Cell). Use [`from_html()`](#ek.metrics.tables.Table.from_html) / [`from_grid()`](#ek.metrics.tables.Table.from_grid) to build one,
and [`as_grid()`](#ek.metrics.tables.Table.as_grid) for the dense 2-D occupancy view GriTS scores.

#### as_grid()

Expand spans into a dense 2-D grid; a spanned position repeats its cell.

Each physical grid position holds the (shared) [`Cell`](#ek.metrics.tables.Cell) covering it, so
two tables with the same topology have grids of the same shape – the view
GriTS aligns position-by-position.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Cell`](#ek.metrics.tables.Cell)]]]

#### *classmethod* coerce(obj)

Coerce a [`Table`](#ek.metrics.tables.Table), an HTML string, or a 2-D grid into a Table.

* **Return type:**
  [`Table`](#ek.metrics.tables.Table)

#### *classmethod* from_grid(grid)

Build a Table from a 2-D grid of cell texts (`None` = spanned-over).

* **Return type:**
  [`Table`](#ek.metrics.tables.Table)

#### *classmethod* from_html(html)

Parse an HTML `<table>` (with `colspan`/`rowspan`) into a Table.

* **Return type:**
  [`Table`](#ek.metrics.tables.Table)

### *class* ek.metrics.tables.TedsMetric(, structure_only=False)

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
