# ek.qe.signals

Intrinsic confidence signals: token logprobs and per-unit posteriors (cost tier 2).

When the extractor already emits something usable – token log-probabilities
(generative locals like TrOCR/pix2tex, GPT-4o, Mistral OCR) or per-unit posteriors
(Google Cloud Vision, Azure, Tesseract) – the cheapest reliable signal is *free*:
it was computed during inference. This module turns those raw numbers into a single
field-level score (`misc/docs/ek_03` §1a). Two modelling facts drive the design:

1. **Raw summed logprob is length-biased** – longer fields accumulate more negative
   terms and look worse. The fix is length normalisation; the aggregation is a
   *modelling choice to validate on held-out data*, so it stays **pluggable** (no
   hardcoded pooling). Use `min` to catch the single weakest token (a transposed
   digit in an amount); use the geometric mean for overall field plausibility (the
   family Mistral OCR reports as `average_page_confidence_score`).
2. **It is uncalibrated** – and for RLHF’d LLMs *systematically overconfident*.
   Every signal here is a raw score; a [`Calibrator`](ek.base.html.md#ek.base.Calibrator) must run before
   any gate reads it.

Aggregators are registered under the `aggregators` namespace so a caller (or a
third party) can swap pooling by name.

### Example

```pycon
>>> import math
>>> logps = [math.log(0.9), math.log(0.8), math.log(0.95)]
>>> round(geo_mean(logps), 4)            # exp(mean log p)
0.8811
>>> round(min_prob(logps), 4)            # weakest token
0.8
```

### Module Attributes

| [`DEFAULT_LENGTH_ALPHA`](#ek.qe.signals.DEFAULT_LENGTH_ALPHA)   | Length-penalty exponent for [`length_normalized()`](#ek.qe.signals.length_normalized) (Wu et al. 2016 found `alpha` in `[0.6, 0.7]` best); a keyword default, never a magic constant.   |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DEFAULT_PROB_FLOOR`](#ek.qe.signals.DEFAULT_PROB_FLOOR)     | Smallest probability treated as non-zero.                                                                                                                                                          |

### Functions

| [`geo_mean`](#ek.qe.signals.geo_mean)(logps)                       | Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).   |
|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`length_normalized`](#ek.qe.signals.length_normalized)(logps, \*[, alpha]) | Length-normalised score `exp(Σ log p / T**alpha)` (tunable length penalty).      |
| [`mean_prob`](#ek.qe.signals.mean_prob)(logps)                      | Arithmetic mean of token probabilities `mean(exp(log p))`.                       |
| [`min_prob`](#ek.qe.signals.min_prob)(logps)                       | Weakest-token probability `exp(min log p)` -- catches one bad character.         |

### Classes

| [`IntrinsicConfidenceSignal`](#ek.qe.signals.IntrinsicConfidenceSignal)([pool, cost_tier])   | Aggregate an extractor's per-unit confidences into a field score.                                                                  |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| [`LogprobSignal`](#ek.qe.signals.LogprobSignal)([aggregator, alpha, cost_tier])  | Aggregate token log-probabilities into one field score (a [`Signal`](ek.base.html.md#ek.base.Signal)). |

### ek.qe.signals.DEFAULT_LENGTH_ALPHA *= 0.6*

Length-penalty exponent for [`length_normalized()`](#ek.qe.signals.length_normalized) (Wu et al. 2016 found
`alpha` in `[0.6, 0.7]` best); a keyword default, never a magic constant.

### ek.qe.signals.DEFAULT_PROB_FLOOR *= 1e-12*

Smallest probability treated as non-zero. `log(p=0) = -inf` clamps to
`log(DEFAULT_PROB_FLOOR)` (mirrors the confidence floor) so a p=0 token still
drives min_prob/geo_mean toward 0 instead of being silently discarded.

### *class* ek.qe.signals.IntrinsicConfidenceSignal(pool='min', cost_tier=2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate an extractor’s per-unit confidences into a field score.

* **Parameters:**
  * **pool** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"min"` (weakest unit), `"mean"`, or `"geo_mean"` over the
    per-unit confidences (already probabilities in `[0, 1]`).
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `2` – free (the posteriors are emitted at inference).

### Example

```pycon
>>> IntrinsicConfidenceSignal(pool="min")([0.99, 0.4, 0.95])
0.4
```

### *class* ek.qe.signals.LogprobSignal(aggregator=<function geo_mean>, alpha=0.6, cost_tier=2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate token log-probabilities into one field score (a [`Signal`](ek.base.html.md#ek.base.Signal)).

* **Parameters:**
  * **aggregator** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A logprob aggregator (name registered under `aggregators` or a
    callable). Defaults to [`geo_mean()`](#ek.qe.signals.geo_mean).
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Length penalty passed to [`length_normalized()`](#ek.qe.signals.length_normalized) only.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `2` – free intrinsic signal (already computed at inference).

#### aggregator()

Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.signals.geo_mean(logps)

Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.signals.length_normalized(logps, , alpha=0.6)

Length-normalised score `exp(Σ log p / T**alpha)` (tunable length penalty).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.signals.mean_prob(logps)

Arithmetic mean of token probabilities `mean(exp(log p))`.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.signals.min_prob(logps)

Weakest-token probability `exp(min log p)` – catches one bad character.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
