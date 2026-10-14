# ek.metrics.strings

String metrics: Character/Word Error Rate (CER/WER).

CER and WER are graded edit-distance metrics – the right baseline for OCR/ASR and
free-text fields, far less brittle than exact match. Two correctness rules from the
research drive the design:

1. **Accumulate globally.** A corpus error rate is `sum(edits) / sum(reference
   length)` over the whole corpus – never the mean of per-item rates (variable
   lengths make that mathematically wrong). So each [`Score`](ek.base.html.md#ek.base.Score) carries
   raw `edits` and `ref_len` in its `detail`, and [`StringMetric.aggregate()`](#ek.metrics.strings.StringMetric.aggregate)
   sums them. [`evaluate()`](ek.facade.html.md#ek.facade.evaluate) uses this.
2. **CER/WER are error rates, not bounded similarities.** A hypothesis longer than
   the reference pushes the rate above 1.0 – itself a useful hallucination signal.
   `Score.value` is the error rate (lower is better; `detail["higher_is_better"]
   is False`).

The canonical backend is `jiwer` (Apache-2.0, `rapidfuzz` C++ core), which also
yields the substitution/deletion/insertion/hit breakdown. `ek` prefers it,
falls back to `rapidfuzz` directly, then to a pure-Python edit distance, so the
metric works even in a bare environment.

### Classes

| [`StringMetric`](#ek.metrics.strings.StringMetric)([mode, canonicalizer])   | CER (`mode='cer'`) or WER (`mode='wer'`) as a [`Metric`](ek.base.html.md#ek.base.Metric).   |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|

### *class* ek.metrics.strings.StringMetric(mode='cer', , canonicalizer=None)

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
