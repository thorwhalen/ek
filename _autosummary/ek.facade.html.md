# ek.facade

The two top-level facades: [`score()`](#ek.facade.score) (offline) and [`estimate_quality()`](#ek.facade.estimate_quality) (online).

Both operate on the same Layer-A object and follow progressive disclosure – the
simple call Just Works, every strategy is replaceable by keyword:

- [`score()`](#ek.facade.score) – reference-based: compare one prediction to one gold reference,
  the metric chosen by output type (string -> CER, record -> field-F1) unless you
  name or pass one.
- [`evaluate()`](#ek.facade.evaluate) – reference-based at corpus scale: aggregate many comparisons
  *correctly* (global error-rate accumulation, micro-F1), with optional per-slice
  cuts. This is what the OCR benchmark and the regression harness build on.
- [`estimate_quality()`](#ek.facade.estimate_quality) – reference-free: gather signals -> calibrate ->
  validate -> decide, returning a [`QualityReport`](ek.base.html.md#ek.base.QualityReport). The heavy signal
  families (ROVER, conformal gates) are injected; this facade composes whatever you
  give it and ships sensible no-op defaults.

### Example

```pycon
>>> score("hello wrld", "hello world").metric
'cer'
>>> round(score("hello wrld", "hello world", metric="wer").value, 3)
0.5
>>> r = evaluate([("ct", "cat"), ("dg", "dog")], metric="cer")
>>> r.n, round(r.aggregate, 3)
(2, 0.333)
```

### Functions

| [`estimate_quality`](#ek.facade.estimate_quality)(extraction, \*[, sources, ...])   | Estimate the quality of an extraction with no gold reference.                                                            |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`evaluate`](#ek.facade.evaluate)(cases, \*[, metric, grammar, ...])        | Aggregate many comparisons into a [`Report`](ek.base.html.md#ek.base.Report) (corpus level). |
| [`score`](#ek.facade.score)(pred, gold, \*[, grammar, metric, ...])      | Score one prediction against one gold reference (reference-based).                                                       |

### ek.facade.estimate_quality(extraction, , sources=(), signals=(), calibrator=None, validators=(), policy=None, agreement=True, assume_calibrated=False)

Estimate the quality of an extraction with no gold reference.

Composes the strict `signal -> calibrate -> validate -> decide` pipeline
(`misc/docs/ek_03`) over a single value, a [`FieldEstimate`](ek.base.html.md#ek.base.FieldEstimate), or a
whole [`AnnotatedExtraction`](ek.base.html.md#ek.base.AnnotatedExtraction) (scored per field, with per-field
specs feeding the validators and the node type as the Mondrian group key). The
simple call `estimate_quality(value)` Just Works; every stage is injectable.

* **Parameters:**
  * **extraction** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A raw value, a [`FieldEstimate`](ek.base.html.md#ek.base.FieldEstimate), or an
    [`AnnotatedExtraction`](ek.base.html.md#ek.base.AnnotatedExtraction).
  * **sources** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – Additional hypotheses of the *same* content (strings or
    `OcrResult`-shaped objects) to fuse with ROVER – their mean agreement
    becomes a `raw_signals["agreement"]` entry (the flagship online signal).
  * **signals** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – Explicit [`Signal`](ek.base.html.md#ek.base.Signal) callables `target -> float |
    Mapping` producing further raw signals.
  * **calibrator** – A [`Calibrator`](ek.base.html.md#ek.base.Calibrator) mapping raw score -> probability,
    applied to the field’s confidence (else the mean of its raw signals).
    When the signals are *heterogeneous* (e.g. an LLM-judge score next to a
    logprob), pass a `Mapping[str, Calibrator]` keyed by signal name (a
    `"*"` entry is the fallback) or a
    [`SignalFamilyCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.SignalFamilyCalibrator): each signal family is
    then calibrated on its own and the calibrated probabilities are fused by
    its `combine` rule (mean by default) followed by its learned
    `fusion` calibrator. A field’s own `confidence` is the
    `"confidence"` family. The per-family values are reported in
    `provenance["calibrated_signals"]`.
    **Calibration is non-optional before gating** (Hard Rule 1): passing a
    `policy` with no calibrator and `assume_calibrated` false **raises** –
    a raw, uncalibrated score must never reach a DecisionPolicy.
  * **validators** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – [`Validator`](ek.base.html.md#ek.base.Validator) callables yielding findings. A flat
    iterable runs on every field (use spec-driven validators like
    [`schema_validator()`](ek.qe.verifiers.html.md#ek.qe.verifiers.schema_validator)). For per-field scoping pass a
    `Mapping` keyed by field name or node type (`"*"` runs on all).
  * **policy** – A [`DecisionPolicy`](ek.base.html.md#ek.base.DecisionPolicy) producing accept/flag/block from
    the *calibrated* confidence.
  * **agreement** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Auto-run ROVER over `sources` when any are given (default true).
  * **assume_calibrated** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Treat the incoming confidence as already calibrated
    (silences the uncalibrated-gating warning). With a per-family
    calibrator it also waives the check that a value fused from two or more
    families went through a learned `fusion` calibrator.
* **Return type:**
  [`QualityReport`](ek.base.html.md#ek.base.QualityReport)

### ek.facade.evaluate(cases, , metric=None, grammar=None, normalize=None, weights=None)

Aggregate many comparisons into a [`Report`](ek.base.html.md#ek.base.Report) (corpus level).

* **Parameters:**
  * **cases** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – Iterable of `(pred, gold)` or `(pred, gold, slice_label)` tuples.
  * **metric** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Metric`](ek.base.html.md#ek.base.Metric)]) – As in [`score()`](#ek.facade.score) (resolved once from the first case’s types).
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Optional Layer-A grammar passed to every comparison.
  * **normalize** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional canonicalizer applied before every comparison.
* **Return type:**
  [`Report`](ek.base.html.md#ek.base.Report)
* **Returns:**
  A Report whose `aggregate` is computed by the metric’s own aggregator
  (e.g. globally accumulated CER/WER, micro-F1) – never a naive mean.

### ek.facade.score(pred, gold, , grammar=None, metric=None, normalize=None, weights=None)

Score one prediction against one gold reference (reference-based).

* **Parameters:**
  * **pred** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The predicted output (string, record dict, or anything with `.text`).
  * **gold** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The gold reference, same shape as `pred`.
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Optional Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) (carries cost weights).
  * **metric** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Metric`](ek.base.html.md#ek.base.Metric)]) – A registered metric name (`"cer"`, `"wer"`, `"fields"`, …), a
    callable [`Metric`](ek.base.html.md#ek.base.Metric), or `None` to dispatch by type.
  * **normalize** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional canonicalizer (name, callable, step list, or
    [`Canonicalizer`](ek.canonicalize.html.md#ek.canonicalize.Canonicalizer)) applied before comparison.
  * **weights** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A [`CostWeight`](ek.base.html.md#ek.base.CostWeight) for cost-weighted metrics (the
    typed-graph distance); overrides the schema’s importance weights.
* **Return type:**
  [`Score`](ek.base.html.md#ek.base.Score)
* **Returns:**
  A [`Score`](ek.base.html.md#ek.base.Score).
