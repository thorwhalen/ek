# ek.harness

The offline evaluation harness: run a predictor over gold, gate regressions.

Reference-based benchmarking (Need #1) is only useful if it is *repeatable* and
*regression-safe*: a “better” system must not silently get worse on some slice. The
harness adds that discipline on top of [`ek.score()`](ek.html.md#ek.score)/[`ek.evaluate()`](ek.html.md#ek.evaluate):

- [`evaluate_store()`](#ek.harness.evaluate_store) – run any `input -> prediction` predictor over a gold
  store (a dict or a `ek` gold store), scored per slice (grouped by each record’s
  slice label), with results persisted. (The OCR benchmark in [`ek.ocr`](ek.ocr.html.md#module-ek.ocr) is the
  OCR-specific specialization of this.)
- [`save_baseline()`](#ek.harness.save_baseline) / [`regression_gate()`](#ek.harness.regression_gate) – freeze a baseline and fail when
  a later run regresses beyond a tolerance, **per slice**, not just on the aggregate.
  This is the golden-set CI gate.
- [`cohen_kappa()`](#ek.harness.cohen_kappa) / [`percent_agreement()`](#ek.harness.percent_agreement) / [`krippendorff_alpha()`](#ek.harness.krippendorff_alpha) –
  inter-annotator agreement (IAA), to know the ceiling of your gold standard before
  trusting any model score (all pure-Python, no optional dependency).

See `misc/docs/ek_02` for the harness design (canonicalizer versioning, slicing,
IAA conventions, golden-set CI).

### Functions

| [`cohen_kappa`](#ek.harness.cohen_kappa)(rater_a, rater_b)                     | Cohen's kappa for two raters on nominal labels (chance-corrected agreement).   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`evaluate_store`](#ek.harness.evaluate_store)(predict, gold, \*[, metric, ...])  | Run `predict` over a gold store and score it per slice.                        |
| [`krippendorff_alpha`](#ek.harness.krippendorff_alpha)(reliability_data, \*[, level]) | Krippendorff's alpha -- the general inter-annotator agreement coefficient.     |
| [`load_baseline`](#ek.harness.load_baseline)(name, \*[, rootdir])                | Load a named baseline (or `None` if it does not exist).                        |
| [`percent_agreement`](#ek.harness.percent_agreement)(rater_a, rater_b)               | Raw fraction of items two raters labelled identically.                         |
| [`regression_gate`](#ek.harness.regression_gate)(report, baseline, \*[, ...])      | Fail if `report` regresses beyond `tolerance` vs a baseline, per slice.        |
| [`save_baseline`](#ek.harness.save_baseline)(report, name, \*[, rootdir])        | Freeze a report's aggregate + per-slice scores as a named baseline.            |

### Classes

| [`GateResult`](#ek.harness.GateResult)(passed, metric, higher_is_better, ...)   | Outcome of a [`regression_gate()`](#ek.harness.regression_gate) check.   |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|

### *class* ek.harness.GateResult(passed, metric, higher_is_better, tolerance, aggregate_current=None, aggregate_baseline=None, regressions=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Outcome of a [`regression_gate()`](#ek.harness.regression_gate) check.

### ek.harness.cohen_kappa(rater_a, rater_b)

Cohen’s kappa for two raters on nominal labels (chance-corrected agreement).

Use only for exactly two raters, nominal labels, complete data; prefer
[`krippendorff_alpha()`](#ek.harness.krippendorff_alpha) for the general case (missing data, any measurement
level).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.harness.evaluate_store(predict, gold, , metric=None, grammar=None, normalize=None, weights=None, input_key='input', reference_key='reference', slice_key='slice', persist=False, run_id=None, rootdir=None)

Run `predict` over a gold store and score it per slice.

* **Parameters:**
  * **predict** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `input -> prediction` callable (the system under test).
  * **gold** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Mapping `key -> {input, reference, [slice], ...}` (a dict or a
    `ek` gold store).
  * **metric** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Metric name/callable (defaults to type-dispatch on the first case).
  * **grammar/normalize/weights** – forwarded to scoring.
  * **input_key/reference_key/slice_key** – field names within each gold record.
  * **persist/run_id/rootdir** – persist the run to the `results`/`runs` stores.
* **Returns:**
  A [`Report`](ek.base.html.md#ek.base.Report); `detail['per_item']` maps each key to its
  prediction, reference, slice, and score.

### ek.harness.krippendorff_alpha(reliability_data, , level='nominal')

Krippendorff’s alpha – the general inter-annotator agreement coefficient.

Pure-Python (no dependency, permissive core): any number of raters, any
measurement `level` (`"nominal"`, `"ordinal"`, `"interval"`, `"ratio"`),
and missing data (use `None` or `float('nan')` for an unlabelled cell).
`reliability_data` is one row per rater, one column per item. Alpha is `1.0`
for perfect agreement, `0.0` at chance, and goes negative for systematic
disagreement. Computed via the coincidence-matrix method (Krippendorff 2011):
observed vs expected disagreement under the level-specific difference metric.

### Example

```pycon
>>> data = [[1, 1, 1, 1], [1, 1, 1, 1]]   # two raters, perfect agreement
>>> krippendorff_alpha(data)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.harness.load_baseline(name, , rootdir=None)

Load a named baseline (or `None` if it does not exist).

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.harness.percent_agreement(rater_a, rater_b)

Raw fraction of items two raters labelled identically.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.harness.regression_gate(report, baseline, , tolerance=0.0, higher_is_better=None, rootdir=None)

Fail if `report` regresses beyond `tolerance` vs a baseline, per slice.

* **Parameters:**
  * **report** – The current [`Report`](ek.base.html.md#ek.base.Report).
  * **baseline** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A baseline name (loaded from the `baselines` store) or a baseline
    dict from [`save_baseline()`](#ek.harness.save_baseline).
  * **tolerance** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Allowed drift before a change counts as a regression.
  * **higher_is_better** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Override metric-direction inference (CER/WER/graph are
    lower-is-better; F1/similarity are higher-is-better).
  * **rootdir** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – data root (when `baseline` is a name).
* **Return type:**
  [`GateResult`](#ek.harness.GateResult)
* **Returns:**
  A [`GateResult`](#ek.harness.GateResult) (falsy if any regression was found).

### ek.harness.save_baseline(report, name, , rootdir=None)

Freeze a report’s aggregate + per-slice scores as a named baseline.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
