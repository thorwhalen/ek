# ek.qe

Reference-free quality estimation: the `estimate_quality()` side of `ek`.

This package implements the online (no-gold) half of evaluation as the strict
pipeline `signal -> calibrate -> validate -> decide` (`misc/docs/ek_03`):

- **signals** (cheapest first): the deterministic [`VerifierSignal`](ek.qe.verifiers.html.md#ek.qe.verifiers.VerifierSignal)
  (schema/checksum/cross-field, cost tier 1, runs always), free intrinsic
  [`IntrinsicConfidenceSignal`](ek.qe.signals.html.md#ek.qe.signals.IntrinsicConfidenceSignal) / [`LogprobSignal`](ek.qe.signals.html.md#ek.qe.signals.LogprobSignal)
  > (tier 2), and the flagship ROVER [`AgreementSignal`](#ek.qe.AgreementSignal) multi-engine
  > vote (tier 3);
- **calibrate** – [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator) (default) / isotonic /
  temperature, with Mondrian grouping and ECE diagnostics: \*non-optional before any
  gate\*;
- **decide** – [`CostSensitiveGate`](ek.qe.decide.html.md#ek.qe.decide.CostSensitiveGate) (default),
  [`ConformalGate`](ek.qe.decide.html.md#ek.qe.decide.ConformalGate), and [`RiskControlGate`](ek.qe.decide.html.md#ek.qe.decide.RiskControlGate),
  > reading only calibrated probabilities.

Everything is pure-Python and dependency-free by default; the `ek[calibration]`
(netcal/sklearn/MAPIE/crepes) and `ek[agreement]` (uqlm) extras supply
library-backed equivalents, imported lazily so `import ek` stays light.

Importing this package registers the built-in strategies (signals, aggregators,
checks, calibrators, policies) by name in [`ek.registry`](ek.registry.html.md#module-ek.registry).

### Functions

| [`rover`](#ek.qe.rover)(hypotheses, \*[, use_confidence, ...])      | Align N hypotheses, vote per slot, and emit consensus + per-position agreement.                                                        |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`geo_mean`](#ek.qe.geo_mean)(logps)                                   | Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).                                                         |
| [`length_normalized`](#ek.qe.length_normalized)(logps, \*[, alpha])             | Length-normalised score `exp(Σ log p / T**alpha)` (tunable length penalty).                                                            |
| [`min_prob`](#ek.qe.min_prob)(logps)                                   | Weakest-token probability `exp(min log p)` -- catches one bad character.                                                               |
| [`mean_prob`](#ek.qe.mean_prob)(logps)                                  | Arithmetic mean of token probabilities `mean(exp(log p))`.                                                                             |
| [`mean_probability`](#ek.qe.mean_probability)(probs)                           | Fuse calibrated per-family probabilities by their mean (the default combiner).                                                         |
| [`min_probability`](#ek.qe.min_probability)(probs)                            | Fuse calibrated per-family probabilities by their minimum (conservative).                                                              |
| [`expected_calibration_error`](#ek.qe.expected_calibration_error)(probs, correct, \*)    | Expected Calibration Error: weighted mean gap between confidence and accuracy.                                                         |
| [`reliability_curve`](#ek.qe.reliability_curve)(probs, correct, \*[, n_bins])   | Per-bin `{confidence, accuracy, count}` for a reliability diagram.                                                                     |
| [`save_calibrator`](#ek.qe.save_calibrator)(calibrator, name, \*[, rootdir])  | Persist a fitted calibrator's parameters to the `calibrators` store.                                                                   |
| [`load_calibrator`](#ek.qe.load_calibrator)(name, \*[, rootdir])              | Reconstruct a persisted calibrator by name (dispatched on its `kind`).                                                                 |
| [`sklearn_calibrator`](#ek.qe.sklearn_calibrator)([method])                      | A calibrator backed by scikit-learn (`method='sigmoid'` Platt or `'isotonic'`).                                                        |
| [`risk_coverage_curve`](#ek.qe.risk_coverage_curve)(probs, correct, \*[, ...])    | The risk-coverage trade-off: `{threshold, coverage, selective_risk}` per point.                                                        |
| [`split_conformal_quantile`](#ek.qe.split_conformal_quantile)(scores, alpha)           | The split-conformal threshold: the `ceil((n+1)(1-alpha)) / n` empirical quantile.                                                      |
| [`checksum_validator`](#ek.qe.checksum_validator)([kind, field_name, layer])     | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value failing a named checksum.            |
| [`regex_validator`](#ek.qe.regex_validator)(pattern, \*[, field_name, layer]) | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not fully matching `pattern`.        |
| [`range_validator`](#ek.qe.range_validator)(lo, hi, \*[, field_name, layer])  | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a number outside `[lo, hi]`.                 |
| [`enum_validator`](#ek.qe.enum_validator)(members, \*[, field_name, layer])  | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not in an allowed set.               |
| [`schema_validator`](#ek.qe.schema_validator)(value, \*[, spec])               | A spec-driven [`Validator`](ek.base.html.md#ek.base.Validator): type and `domain` checks from a FieldSpec. |
| [`totals_consistent`](#ek.qe.totals_consistent)(record, \*, total_key, ...)     | Flag when `record[total_key]` does not equal the sum of `item_keys` (±tol).                                                            |
| [`luhn_check`](#ek.qe.luhn_check)(number)                                | Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs.                                                                        |
| [`iban_check`](#ek.qe.iban_check)(iban)                                  | IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1).                                                                 |
| [`isbn_check`](#ek.qe.isbn_check)(value)                                 | Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored).                                                              |

### Classes

| [`RoverConsensus`](#ek.qe.RoverConsensus)([tokens, slots, agreement, ...])   | The result of a ROVER pass over N hypotheses.                                                                                      |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| [`RoverSlot`](#ek.qe.RoverSlot)([entries, winner, score, vote_share])   | One branch point of the transition network: each engine's token (or NULL).                                                         |
| [`AgreementSignal`](#ek.qe.AgreementSignal)([cost_tier, use_confidence, ...]) | ROVER multi-engine agreement as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).           |
| [`VerifierSignal`](#ek.qe.VerifierSignal)([validators, cost_tier])           | Deterministic verifier evidence as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).        |
| [`LogprobSignal`](#ek.qe.LogprobSignal)([aggregator, alpha, cost_tier])     | Aggregate token log-probabilities into one field score (a [`Signal`](ek.base.html.md#ek.base.Signal)). |
| [`IntrinsicConfidenceSignal`](#ek.qe.IntrinsicConfidenceSignal)([pool, cost_tier])      | Aggregate an extractor's per-unit confidences into a field score.                                                                  |
| [`PlattCalibrator`](#ek.qe.PlattCalibrator)([a, b, max_iter, kind])           | Platt scaling: `sigmoid(a * score + b)`, fit by Newton/IRLS on labels.                                                             |
| [`IsotonicCalibrator`](#ek.qe.IsotonicCalibrator)([x, y, kind])                  | Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.                                                       |
| [`TemperatureCalibrator`](#ek.qe.TemperatureCalibrator)([T, kind, t_min, ...])      | Temperature scaling: `sigmoid(logit / T)` with one `T` fit on a holdout.                                                           |
| [`GroupCalibrator`](#ek.qe.GroupCalibrator)([factory, by_group, pooled, ...]) | Per-group (Mondrian) calibration: one calibrator per `NodeType`/`FieldSpec`.                                                       |
| [`SignalFamilyCalibrator`](#ek.qe.SignalFamilyCalibrator)([calibrators, ...])        | One calibrator per signal family, fused by a documented combination rule.                                                          |
| [`CostSensitiveGate`](#ek.qe.CostSensitiveGate)([rho, block_threshold, ...])    | Accept/flag/block from the cost ratio `rho = c_FN / c_FP` on calibrated `p`.                                                       |
| [`ConformalGate`](#ek.qe.ConformalGate)([alpha, nonconformity, q, \_cal])   | Split-conformal accept/flag with a finite-sample **marginal** guarantee.                                                           |
| [`GroupConformalGate`](#ek.qe.GroupConformalGate)([alpha, nonconformity, ...])   | Mondrian (class-conditional) conformal: one [`ConformalGate`](#ek.qe.ConformalGate) per group.              |
| [`RiskControlGate`](#ek.qe.RiskControlGate)([target, loss, loss_bound, lam])  | Conformal Risk Control: accept `p >= lambda` bounding the accepted-error rate.                                                     |

### *class* ek.qe.AgreementSignal(cost_tier=3, use_confidence=True, conf_weight=0.5, null_conf=0.7, tokenize=None, max_tokens=5000)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

ROVER multi-engine agreement as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).

Cost tier 3 (`N` engine runs): try the deterministic verifier layer and any
free intrinsic confidence first. Called on a collection of hypotheses, it runs
[`rover()`](#ek.qe.rover) and returns the mean per-position agreement as the raw signal –
*uncalibrated*, like every signal, so it must pass through a
[`Calibrator`](ek.base.html.md#ek.base.Calibrator) before any gate reads it.

### Example

```pycon
>>> sig = AgreementSignal()
>>> round(sig(["the cat sat", "the cat sit", "the bat sat"]), 3)
0.778
```

### *class* ek.qe.ConformalGate(alpha=0.1, nonconformity=<function ConformalGate.<lambda>>, q=inf, \_cal=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Split-conformal accept/flag with a finite-sample **marginal** guarantee.

Fit on `(calibrated_prob, correct)`: the nonconformity scores of the *correct*
calibration items set a threshold `q` such that, among truly-correct test items
exchangeable with them, at most `alpha` are flagged. The guarantee is
*marginal*, not per-field-type – use [`GroupConformalGate`](#ek.qe.GroupConformalGate) for that.

* **Parameters:**
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Target false-flag rate on correct items (e.g. `0.1`).
  * **nonconformity** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Maps calibrated confidence -> nonconformity (default
    `1 - confidence`: low confidence is nonconforming).

#### fit(probs, correct)

Fit `q` from the nonconformity scores of the *correct* calibration items.

Non-finite confidences are dropped (a single `NaN` would otherwise sort to
the end and corrupt the quantile, silently breaking the coverage guarantee).

* **Return type:**
  [`ConformalGate`](ek.qe.decide.html.md#ek.qe.decide.ConformalGate)

#### p_value(confidence)

Per-instance conformal p-value (small = atypical of correct items).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.qe.CostSensitiveGate(rho=1.0, block_threshold=0.0, accept_threshold=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Accept/flag/block from the cost ratio `rho = c_FN / c_FP` on calibrated `p`.

Accept when the expected cost of accepting is below that of routing to a human.
Accepting risks an undetected error: cost `(1 - p) * c_FN`. Flagging a value
that was actually *correct* wastes a review: cost `p * c_FP` (the false-positive
cost is incurred only with probability `p`, not unconditionally). So accept iff
`(1 - p) * c_FN <= p * c_FP`, i.e. the Bayes-optimal threshold
`p >= c_FN / (c_FN + c_FP) = rho / (1 + rho)`. A larger `rho` (misses much
costlier than reviews) raises the bar to auto-accept (`tau -> 1` as
`rho -> inf`); `rho = 1` gives the symmetric `tau = 0.5`.
`block_threshold` adds an optional hard-fail band at the bottom.

* **Parameters:**
  * **rho** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Cost ratio `c_FN / c_FP` (>= 0). The single lever; no magic numbers.
  * **block_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Calibrated probability at or below which to `block` rather
    than `flag` (default `0` -> never block on probability alone; a
    failed verifier is what forces a block, upstream).
  * **accept_threshold** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Override the derived accept threshold explicitly.

#### *property* tau *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The accept threshold on calibrated probability (derived from `rho`).

### *class* ek.qe.GroupCalibrator(factory=<class 'ek.qe.calibrate.PlattCalibrator'>, by_group=<factory>, pooled=None, kind='group')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-group (Mondrian) calibration: one calibrator per `NodeType`/`FieldSpec`.

Distribution-free *conditional* (per-field-type) coverage is impossible
in general (Barber et al. 2019); calibrating separately per group restores it
approximately. `fit` takes a parallel `groups` sequence; `__call__` routes
by group key, falling back to a pooled calibrator for unseen groups.

* **Parameters:**
  **factory** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Zero-arg callable producing a fresh per-group calibrator (default
  [`PlattCalibrator`](#ek.qe.PlattCalibrator)).

#### factory

alias of [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator)

#### fit(scores, correct, , groups)

Fit one calibrator per distinct group key, plus a pooled fallback.

* **Return type:**
  [`GroupCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.GroupCalibrator)

#### to_dict()

Serialize each per-group calibrator (keyed by stringified group) plus the
pooled fallback, so a fitted Mondrian calibrator round-trips like the others.
Group keys are stringified (they are `NodeType`/`FieldSpec` names).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ek.qe.GroupConformalGate(alpha=0.1, nonconformity=<function GroupConformalGate.<lambda>>, by_group=<factory>, pooled=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Mondrian (class-conditional) conformal: one [`ConformalGate`](#ek.qe.ConformalGate) per group.

Restores approximate per-field-type validity by calibrating separately per group
key (e.g. `NodeType`/`FieldSpec`), with a pooled fallback for unseen groups.

### *class* ek.qe.IntrinsicConfidenceSignal(pool='min', cost_tier=2)

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

### *class* ek.qe.IsotonicCalibrator(x=<factory>, y=<factory>, kind='isotonic')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.

More flexible than Platt; needs more calibration data and can overfit small
sets. Predicts by linear interpolation between fitted points, clipped at the
ends.

#### fit(scores, correct)

Fit the monotonic step function to `(score, correct)` pairs.

* **Return type:**
  [`IsotonicCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.IsotonicCalibrator)

### *class* ek.qe.LogprobSignal(aggregator=<function geo_mean>, alpha=0.6, cost_tier=2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate token log-probabilities into one field score (a [`Signal`](ek.base.html.md#ek.base.Signal)).

* **Parameters:**
  * **aggregator** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A logprob aggregator (name registered under `aggregators` or a
    callable). Defaults to [`geo_mean()`](#ek.qe.geo_mean).
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Length penalty passed to [`length_normalized()`](#ek.qe.length_normalized) only.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `2` – free intrinsic signal (already computed at inference).

#### aggregator()

Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.qe.PlattCalibrator(a=1.0, b=0.0, max_iter=100, kind='platt')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Platt scaling: `sigmoid(a * score + b)`, fit by Newton/IRLS on labels.

The default calibrator: works on any scalar (aggregated confidence/logprob), no
logits required. Uses Platt’s target smoothing so it does not overfit small
calibration sets.

* **Parameters:**
  **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Newton iterations (converges in a handful for a 2-parameter model).

#### fit(scores, correct)

Fit `a, b` to maximise the likelihood of `correct` given `scores`.

* **Return type:**
  [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator)

### *class* ek.qe.RiskControlGate(target=0.05, loss=<function RiskControlGate.<lambda>>, loss_bound=1.0, lam=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Conformal Risk Control: accept `p >= lambda` bounding the accepted-error rate.

Picks the *smallest* threshold `lambda` (the most coverage) whose finite-sample
risk bound is at or below `target`. The controlled quantity is
`E[ loss(item) * 1(accepted) ]` – with the default 0/1 loss, the population
rate of accepted-yet-wrong items.

* **Parameters:**
  * **target** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Upper bound on the accepted-error rate (e.g. `0.05`).
  * **loss** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – `correct -> loss` in `[0, loss_bound]`; default `0` if correct
    else `1`. Any monotone loss bounded by `loss_bound` works.
  * **loss_bound** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The loss’s upper bound `B` – the CRC finite-sample slack is
    `+B/(n+1)`. Must match `loss`’s range or the guarantee is invalid;
    keep the default `1.0` for the 0/1 loss.

#### fit(probs, correct)

Find the smallest accept threshold whose risk bound is <= `target`.

* **Return type:**
  [`RiskControlGate`](ek.qe.decide.html.md#ek.qe.decide.RiskControlGate)

### *class* ek.qe.RoverConsensus(tokens=<factory>, slots=<factory>, agreement=<factory>, n_engines=0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of a ROVER pass over N hypotheses.

#### *property* mean_agreement *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Mean per-position agreement over consensus tokens, a reference-free
confidence in `[0, 1]`: `1.0` means every engine agreed at every emitted
position; lower means at least one engine dissented.

An **empty** consensus (no token survived voting – NULL won every slot) means
the engines agreed on *nothing*, so this returns `0.0` when 2+ engines were
fused, NOT a misleading `1.0` (which would feed a maximal raw confidence to
the calibrator/gate). A single engine – or none – is vacuously `1.0`.

#### *property* text *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The consensus transcription (space-joined winning tokens).

### *class* ek.qe.RoverSlot(entries=<factory>, winner=None, score=0.0, vote_share=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One branch point of the transition network: each engine’s token (or NULL).

### *class* ek.qe.SignalFamilyCalibrator(calibrators=<factory>, combine=<function mean_probability>, factory=<class 'ek.qe.calibrate.PlattCalibrator'>, fusion=None, kind='signal_family')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One calibrator per signal family, fused by a documented combination rule.

Averaging heterogeneous raw signals (an LLM-judge score, a token logprob, a ROVER
agreement) and calibrating the mean discards exactly what a calibrator needs:
each family has its own scale, reliability and direction of miscalibration. This
calibrates every signal *by name* with its own calibrator, then fuses the
calibrated probabilities with `combine`.

Pass it (or simply a `Mapping[str, Calibrator]`) as
`estimate_quality(..., calibrator=...)`.

**The fused value must itself be calibrated before it is gated.** Fusing
calibrated probabilities (by mean, min, …) does not yield a calibrated
probability: agreeing families make the mean under-confident, a weak family
drags it toward the base rate. So [`fit()`](#ek.qe.SignalFamilyCalibrator.fit) also learns a second-stage
`fusion` calibrator on `combine`’s output, and `__call__` applies it.
Without a `fusion` calibrator (e.g. a hand-built mapping),
[`ek.estimate_quality()`](ek.html.md#ek.estimate_quality) refuses to gate a value fused from two or more
families unless told `assume_calibrated=True` (Hard Rule 1). A value from a
single family is that family’s calibrated probability and gates as is.

A field’s own upstream `confidence` is just one more family here, named
`"confidence"`: include it in the training rows under that name.

* **Parameters:**
  * **calibrators** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – `signal name -> Calibrator`. A `"*"` entry is the fallback
    for any signal without its own entry. A signal with neither is refused
    (`ValueError`) rather than silently dropped or passed through raw
    (Hard Rule 1).
  * **combine** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – `Mapping[str, float] -> float` fusing the calibrated values;
    [`mean_probability()`](#ek.qe.mean_probability) (default) or [`min_probability()`](#ek.qe.min_probability), or any
    callable of that shape (e.g. a learned stacker).
  * **factory** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Zero-arg callable producing a fresh calibrator, used by [`fit()`](#ek.qe.SignalFamilyCalibrator.fit).
  * **fusion** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Second-stage calibrator applied to `combine`’s output (learned by
    [`fit()`](#ek.qe.SignalFamilyCalibrator.fit)); `None` leaves the fused value uncalibrated.

### Example

```pycon
>>> rows = [{"judge": j, "logprob": l} for j, l in
...         [(0.9, -0.1), (0.8, -0.2), (0.85, -2.0), (0.3, -0.1), (0.2, -2.5), (0.7, -1.8)]]
>>> correct = [True, True, False, True, False, False]
>>> cal = SignalFamilyCalibrator().fit(rows, correct)
>>> sorted(cal.calibrators)
['judge', 'logprob']
>>> 0.0 <= cal({"judge": 0.9, "logprob": -0.1}) <= 1.0
True
```

#### calibrate_signals(raw_signals, , group=None)

`{name: calibrated probability}` for each raw signal (before fusion).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### combine()

Fuse calibrated per-family probabilities by their mean (the default combiner).

Each family’s calibrated value estimates the *same* event (“this field is
correct”), so the mean is a defensible, assumption-light pooling rule.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> mean_probability({"judge": 0.9, "logprob": 0.7})
0.8
```

#### factory

alias of [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator)

#### fit(signals, correct)

Fit one calibrator per signal name, then the `fusion` calibrator.

Each family is fit over the rows where it appears with a finite value. A
family with fewer than two such rows, or only one label class, cannot be
calibrated and is refused (`ValueError`) rather than turned into a
constant that would then dilute the fusion.

* **Return type:**
  [`SignalFamilyCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.SignalFamilyCalibrator)

#### fuse(calibrated)

Fuse per-family calibrated probabilities: `combine`, then `fusion`.

A single family’s value is returned as is: it is already calibrated, and
`fusion` was learned on fused values.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### to_dict()

Serialize every family’s calibrator and the (registered) combiner’s name.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ek.qe.TemperatureCalibrator(T=1.0, kind='temperature', t_min=0.05, t_max=10.0, max_iter=60)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Temperature scaling: `sigmoid(logit / T)` with one `T` fit on a holdout.

Use **only when you have logits**: `__call__` expects a *logit*, not a
probability. `T > 1` softens overconfidence; the argmax is unchanged.

#### fit(logits, correct)

Fit `T` by minimising NLL with a bounded 1-D search.

* **Return type:**
  [`TemperatureCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.TemperatureCalibrator)

#### t_min *: [float](https://docs.python.org/3/builtins/functions.html#float)* *= 0.05*

Golden-section search bounds and iteration count (config, not magic numbers).

### *class* ek.qe.VerifierSignal(validators=<factory>, cost_tier=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Deterministic verifier evidence as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).

Cost tier 1 – the free, first-to-run layer. Runs a list of
[`Validator`](ek.base.html.md#ek.base.Validator) s on a value and returns the **fraction that passed**
in `[0, 1]` as the raw signal (`1.0` = every check passed). It also exposes
the [`Finding`](ek.base.html.md#ek.base.Finding) s it produced via [`findings()`](#ek.qe.VerifierSignal.findings), so the same
object feeds both the score path (-> calibrate) and the audit path (-> review).
Like every signal it is uncalibrated: a verifier failing is strong evidence, not
a probability, so a [`Calibrator`](ek.base.html.md#ek.base.Calibrator) still runs before any gate.

### Example

```pycon
>>> v = VerifierSignal([checksum_validator("luhn")])
>>> v("79927398713")          # valid -> all checks pass
1.0
>>> v("79927398710")          # invalid -> the one check fails
0.0
```

#### findings(value, , spec=None)

All findings the validators produce for `value` (empty == all clear).

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### ek.qe.checksum_validator(kind='luhn', , field_name='', layer='checksum')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value failing a named checksum.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.enum_validator(members, , field_name='', layer='enum')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not in an allowed set.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.expected_calibration_error(probs, correct, , n_bins=10)

Expected Calibration Error: weighted mean gap between confidence and accuracy.

Bins predictions by confidence into `n_bins` equal-width bins and averages
`|mean_confidence - accuracy|` weighted by bin population. `0` is perfect.
Non-finite probs are skipped; out-of-range probs are clamped into `[0, 1]`.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.geo_mean(logps)

Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.iban_check(iban)

IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.isbn_check(value)

Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.length_normalized(logps, , alpha=0.6)

Length-normalised score `exp(Σ log p / T**alpha)` (tunable length penalty).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.load_calibrator(name, , rootdir=None)

Reconstruct a persisted calibrator by name (dispatched on its `kind`).

Validates the stored record so a malformed or unknown `kind` fails with an
actionable error rather than a raw `KeyError` or a load-then-crash-later.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ek.qe.luhn_check(number)

Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.mean_prob(logps)

Arithmetic mean of token probabilities `mean(exp(log p))`.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.mean_probability(probs)

Fuse calibrated per-family probabilities by their mean (the default combiner).

Each family’s calibrated value estimates the *same* event (“this field is
correct”), so the mean is a defensible, assumption-light pooling rule.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> mean_probability({"judge": 0.9, "logprob": 0.7})
0.8
```

### ek.qe.min_prob(logps)

Weakest-token probability `exp(min log p)` – catches one bad character.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.min_probability(probs)

Fuse calibrated per-family probabilities by their minimum (conservative).

Use when any one family flagging a problem should be enough to lower the gate.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> min_probability({"judge": 0.9, "logprob": 0.7})
0.7
```

### ek.qe.range_validator(lo, hi, , field_name='', layer='range')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a number outside `[lo, hi]`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.regex_validator(pattern, , field_name='', layer='format')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not fully matching `pattern`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.reliability_curve(probs, correct, , n_bins=10)

Per-bin `{confidence, accuracy, count}` for a reliability diagram.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.qe.risk_coverage_curve(probs, correct, , thresholds=None)

The risk-coverage trade-off: `{threshold, coverage, selective_risk}` per point.

Coverage is the fraction auto-accepted at each threshold; selective risk is the
error rate *among the accepted*. Lower-left is better; pick the operating point
against a target risk or coverage.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.qe.rover(hypotheses, , use_confidence=True, conf_weight=0.5, null_conf=0.7, tokenize=None, max_tokens=5000)

Align N hypotheses, vote per slot, and emit consensus + per-position agreement.

* **Parameters:**
  * **hypotheses** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The recognizer/extractor outputs to fuse. Each may be a string,
    an `OcrResult`-shaped object (`.text`/`.mean_confidence`), a list of
    token strings, or a list of `(token, confidence)` pairs.
  * **use_confidence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Blend confidence into the vote (`True`) or vote purely by
    frequency (`False`, which forces `conf_weight` to `0`).
  * **conf_weight** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Weight on average confidence vs vote frequency in the slot score
    (Fiscus’s `1 - alpha`); `0` is frequency-only, `1` is
    confidence-only. Ignored when `use_confidence` is `False`.
  * **null_conf** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Confidence credited to a NULL vote (an engine that produced no
    token at a slot) – the lever for how readily a deletion wins.
  * **tokenize** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]]) – Optional `hypothesis -> [(token, confidence), ...]` override; by
    default `_as_units()` handles strings/OcrResults/token lists.
  * **max_tokens** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Reject any hypothesis longer than this (the aligner is
    `O(N*l*L*L')`; an unbounded input is a quadratic time/memory DoS).
    `None` disables the guard.
* **Return type:**
  [`RoverConsensus`](#ek.qe.RoverConsensus)
* **Returns:**
  A [`RoverConsensus`](#ek.qe.RoverConsensus) with the consensus `tokens`/`text`, the per-slot
  breakdown, and the per-consensus-token `agreement` usable as a raw signal.

### ek.qe.save_calibrator(calibrator, name, , rootdir=None)

Persist a fitted calibrator’s parameters to the `calibrators` store.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.qe.schema_validator(value, , spec=None)

A spec-driven [`Validator`](ek.base.html.md#ek.base.Validator): type and `domain` checks from a FieldSpec.

Reads the [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec) `type` and `domain` – a `(lo, hi)`
numeric range, an enum tuple, or a regex string – and applies the matching
check. With no spec it is a no-op, so it composes safely in any validator list.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### ek.qe.sklearn_calibrator(method='sigmoid')

A calibrator backed by scikit-learn (`method='sigmoid'` Platt or `'isotonic'`).

Behind `ek[calibration]`. Returns an object satisfying the
[`Calibrator`](ek.base.html.md#ek.base.Calibrator) protocol that wraps sklearn’s calibration. The
pure-Python [`PlattCalibrator`](#ek.qe.PlattCalibrator)/[`IsotonicCalibrator`](#ek.qe.IsotonicCalibrator) are the
dependency-free defaults; use this for parity with an sklearn-centric stack.

### ek.qe.split_conformal_quantile(scores, alpha)

The split-conformal threshold: the `ceil((n+1)(1-alpha)) / n` empirical quantile.

`scores` are nonconformity scores on an exchangeable calibration set. Returns
`+inf` when `n` is too small to guarantee `1 - alpha` (so nothing is
flagged – the honest behaviour at that sample size), and `-inf` at the
`alpha -> 1` limit (flag everything).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.totals_consistent(record, , total_key, item_keys, tol=0.01, layer='cross_field')

Flag when `record[total_key]` does not equal the sum of `item_keys` (±tol).

The canonical cross-field check (invoice total = Σ line items). Skipped (rather
than raising or flagging) when the total or *all* item keys are absent or
non-numeric, so it is safe on partial records.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### Modules

| [`calibrate`](ek.qe.calibrate.html.md#module-ek.qe.calibrate)   | Calibration: make a raw score *mean* a probability (the non-optional stage).        |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`decide`](ek.qe.decide.html.md#module-ek.qe.decide)         | Selective prediction: turn a calibrated probability into accept / flag / block.     |
| [`signals`](ek.qe.signals.html.md#module-ek.qe.signals)       | Intrinsic confidence signals: token logprobs and per-unit posteriors (cost tier 2). |
| [`verifiers`](ek.qe.verifiers.html.md#module-ek.qe.verifiers)   | Deterministic verifier signals -- the cheapest, first-to-run QE layer.              |
