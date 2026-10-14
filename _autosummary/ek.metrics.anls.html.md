# ek.metrics.anls

ANLS / ANLS\* (nested-JSON) similarity, wrapping `anls_star`.

ANLS (Average Normalized Levenshtein Similarity) is the DocVQA-family metric for a
short answer against acceptable variants: a normalized edit similarity thresholded
at ~0.5 so minor OCR/spelling differences do not zero a correct answer. \*\*ANLS\*\*\*
(Peer et al., 2024, [arXiv 2402.03848](https://arxiv.org/abs/2402.03848)) generalizes it to *arbitrarily nested*
JSON – strings, tuples (best match), lists (Hungarian matching, penalizing
missing/hallucinated items), and dicts (key-value with penalties for missing or
hallucinated keys) – by mapping both sides to a tree and comparing them. It is the
closest thing to an off-the-shelf nested-JSON IE metric (`misc/docs/ek_02` §3.2).

This wraps the `anls_star` package (Apache-2.0; its only dependency is `munkres`
for the Hungarian matching, and it carries its own pure-Python Levenshtein – no GPL
`Levenshtein` is pulled in). It ships behind the `[metrics]` extra and is
imported lazily, so importing `ek` never requires it.

Because `anls_star` already returns a `[0, 1]` higher-is-better similarity, the
wrapper is thin: it flips `ek`’s `(pred, gold)` argument order to the library’s
`(gt, pred)` order, records the raw similarity in `detail`,
and – for a corpus – [`AnlsMetric.aggregate()`](#ek.metrics.anls.AnlsMetric.aggregate) returns the *mean* ANLS (the “A”
in ANLS: the score is already length-normalized per item, so the corpus statistic is
the average of per-item similarities, unlike CER/WER which accumulate edit counts).

### Example

```pycon
>>> from ek.metrics.anls import AnlsMetric
>>> m = AnlsMetric()
>>> round(m("Hello Wrld", "Hello World").value, 3)   # one-char OCR slip
0.909
>>> m({"name": "Acme", "city": "Paris"},
...   {"name": "Acme", "city": "Paris"}).value         # nested, exact
1.0
```

### Classes

| [`AnlsMetric`](#ek.metrics.anls.AnlsMetric)(\*[, threshold])   | ANLS / ANLS\* (nested-JSON) similarity as a [`Metric`](ek.base.html.md#ek.base.Metric).   |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|

### *class* ek.metrics.anls.AnlsMetric(, threshold=0.5)

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
