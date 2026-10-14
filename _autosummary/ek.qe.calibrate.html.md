# ek.qe.calibrate

Calibration: make a raw score *mean* a probability (the non-optional stage).

A confidence of 0.9 should mean “correct 90% of the time.” Modern models violate
this badly – they are systematically **overconfident** (Guo et al. 2017), and RLHF
makes LLMs worse. So \*\*calibration is non-optional: never gate a raw posterior or a
raw logprob\*\* (`misc/docs/ek_03` §2, Hard Rule 1). A [`Calibrator`](ek.base.html.md#ek.base.Calibrator)
is fit on a labelled holdout of `(raw_score, field_correct?)` pairs and maps any
later raw score to a calibrated probability.

Three methods, by what input you have:

- [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator) – logistic fit on any scalar score (no logits needed):
  **the default** for aggregated OCR confidence or aggregated logprobs.
- [`IsotonicCalibrator`](#ek.qe.calibrate.IsotonicCalibrator) – non-parametric monotonic fit; more flexible, needs
  more data, can overfit small sets.
- [`TemperatureCalibrator`](#ek.qe.calibrate.TemperatureCalibrator) – a single scalar `T` on **logits**; use only
  when you have logits (it does not change the argmax).

All three are pure-Python (stdlib only) so the calibration stage works with zero
extra dependencies; [`sklearn_calibrator()`](#ek.qe.calibrate.sklearn_calibrator) / `netcal_calibrator()` offer the
library-backed equivalents behind the `ek[calibration]` extra. Measure calibration
with [`expected_calibration_error()`](#ek.qe.calibrate.expected_calibration_error) (+ a reliability curve). For per-field-type
validity, wrap per group with [`GroupCalibrator`](#ek.qe.calibrate.GroupCalibrator) (Mondrian / class-conditional)
– distribution-free *conditional* coverage is otherwise impossible (Hard Rule 2).
When the raw signals are *heterogeneous* (a judge score next to a logprob next to a
ROVER agreement), calibrate each signal family on its own with
[`SignalFamilyCalibrator`](#ek.qe.calibrate.SignalFamilyCalibrator) and fuse the calibrated probabilities with a
documented combiner ([`mean_probability()`](#ek.qe.calibrate.mean_probability) by default, [`min_probability()`](#ek.qe.calibrate.min_probability)
for a conservative gate), rather than calibrating the mean of incomparable scores.

Calibrate at the **granularity of the decision** (gate on fields -> calibrate a
“field-correct?” target), and persist the fit ([`save_calibrator()`](#ek.qe.calibrate.save_calibrator)); calibration
is dataset-specific and decays, so re-fit on drift.

### Example

```pycon
>>> # An overconfident raw signal, calibrated against ground truth.
>>> raw =     [0.95, 0.93, 0.92, 0.90, 0.55, 0.52, 0.51, 0.50]
>>> correct = [True, True, False, False, True, False, False, False]
>>> cal = PlattCalibrator().fit(raw, correct)
>>> cal(0.95) < 0.95            # overconfidence pulled down
True
>>> 0.0 <= cal(0.5) <= 1.0
True
```

### Module Attributes

| [`DEFAULT_N_BINS`](#ek.qe.calibrate.DEFAULT_N_BINS)   | Default number of bins for [`expected_calibration_error()`](#ek.qe.calibrate.expected_calibration_error) / reliability curves.   |
|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`expected_calibration_error`](#ek.qe.calibrate.expected_calibration_error)(probs, correct, \*)   | Expected Calibration Error: weighted mean gap between confidence and accuracy.            |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`load_calibrator`](#ek.qe.calibrate.load_calibrator)(name, \*[, rootdir])             | Reconstruct a persisted calibrator by name (dispatched on its `kind`).                    |
| [`mean_probability`](#ek.qe.calibrate.mean_probability)(probs)                          | Fuse calibrated per-family probabilities by their mean (the default combiner).            |
| [`min_probability`](#ek.qe.calibrate.min_probability)(probs)                           | Fuse calibrated per-family probabilities by their minimum (conservative).                 |
| [`netcal_ece`](#ek.qe.calibrate.netcal_ece)(probs, correct, \*[, bins])           | ECE via `netcal` (behind `ek[calibration]`); D-ECE for localized outputs lives there too. |
| [`reliability_curve`](#ek.qe.calibrate.reliability_curve)(probs, correct, \*[, n_bins])  | Per-bin `{confidence, accuracy, count}` for a reliability diagram.                        |
| [`save_calibrator`](#ek.qe.calibrate.save_calibrator)(calibrator, name, \*[, rootdir]) | Persist a fitted calibrator's parameters to the `calibrators` store.                      |
| [`sklearn_calibrator`](#ek.qe.calibrate.sklearn_calibrator)([method])                     | A calibrator backed by scikit-learn (`method='sigmoid'` Platt or `'isotonic'`).           |

### Classes

| [`GroupCalibrator`](#ek.qe.calibrate.GroupCalibrator)([factory, by_group, pooled, ...])   | Per-group (Mondrian) calibration: one calibrator per `NodeType`/`FieldSpec`.   |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`IsotonicCalibrator`](#ek.qe.calibrate.IsotonicCalibrator)([x, y, kind])                    | Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.   |
| [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator)([a, b, max_iter, kind])             | Platt scaling: `sigmoid(a * score + b)`, fit by Newton/IRLS on labels.         |
| [`SignalFamilyCalibrator`](#ek.qe.calibrate.SignalFamilyCalibrator)([calibrators, ...])          | One calibrator per signal family, fused by a documented combination rule.      |
| [`TemperatureCalibrator`](#ek.qe.calibrate.TemperatureCalibrator)([T, kind, t_min, ...])        | Temperature scaling: `sigmoid(logit / T)` with one `T` fit on a holdout.       |

### ek.qe.calibrate.DEFAULT_N_BINS *= 10*

Default number of bins for [`expected_calibration_error()`](#ek.qe.calibrate.expected_calibration_error) / reliability curves.

### *class* ek.qe.calibrate.GroupCalibrator(factory=<class 'ek.qe.calibrate.PlattCalibrator'>, by_group=<factory>, pooled=None, kind='group')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-group (Mondrian) calibration: one calibrator per `NodeType`/`FieldSpec`.

Distribution-free *conditional* (per-field-type) coverage is impossible
in general (Barber et al. 2019); calibrating separately per group restores it
approximately. `fit` takes a parallel `groups` sequence; `__call__` routes
by group key, falling back to a pooled calibrator for unseen groups.

* **Parameters:**
  **factory** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Zero-arg callable producing a fresh per-group calibrator (default
  [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator)).

#### factory

alias of [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator)

#### fit(scores, correct, , groups)

Fit one calibrator per distinct group key, plus a pooled fallback.

* **Return type:**
  [`GroupCalibrator`](#ek.qe.calibrate.GroupCalibrator)

#### to_dict()

Serialize each per-group calibrator (keyed by stringified group) plus the
pooled fallback, so a fitted Mondrian calibrator round-trips like the others.
Group keys are stringified (they are `NodeType`/`FieldSpec` names).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ek.qe.calibrate.IsotonicCalibrator(x=<factory>, y=<factory>, kind='isotonic')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.

More flexible than Platt; needs more calibration data and can overfit small
sets. Predicts by linear interpolation between fitted points, clipped at the
ends.

#### fit(scores, correct)

Fit the monotonic step function to `(score, correct)` pairs.

* **Return type:**
  [`IsotonicCalibrator`](#ek.qe.calibrate.IsotonicCalibrator)

### *class* ek.qe.calibrate.PlattCalibrator(a=1.0, b=0.0, max_iter=100, kind='platt')

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
  [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator)

### *class* ek.qe.calibrate.SignalFamilyCalibrator(calibrators=<factory>, combine=<function mean_probability>, factory=<class 'ek.qe.calibrate.PlattCalibrator'>, fusion=None, kind='signal_family')

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
drags it toward the base rate. So [`fit()`](#ek.qe.calibrate.SignalFamilyCalibrator.fit) also learns a second-stage
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
    [`mean_probability()`](#ek.qe.calibrate.mean_probability) (default) or [`min_probability()`](#ek.qe.calibrate.min_probability), or any
    callable of that shape (e.g. a learned stacker).
  * **factory** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Zero-arg callable producing a fresh calibrator, used by [`fit()`](#ek.qe.calibrate.SignalFamilyCalibrator.fit).
  * **fusion** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Second-stage calibrator applied to `combine`’s output (learned by
    [`fit()`](#ek.qe.calibrate.SignalFamilyCalibrator.fit)); `None` leaves the fused value uncalibrated.

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

alias of [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator)

#### fit(signals, correct)

Fit one calibrator per signal name, then the `fusion` calibrator.

Each family is fit over the rows where it appears with a finite value. A
family with fewer than two such rows, or only one label class, cannot be
calibrated and is refused (`ValueError`) rather than turned into a
constant that would then dilute the fusion.

* **Return type:**
  [`SignalFamilyCalibrator`](#ek.qe.calibrate.SignalFamilyCalibrator)

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

### *class* ek.qe.calibrate.TemperatureCalibrator(T=1.0, kind='temperature', t_min=0.05, t_max=10.0, max_iter=60)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Temperature scaling: `sigmoid(logit / T)` with one `T` fit on a holdout.

Use **only when you have logits**: `__call__` expects a *logit*, not a
probability. `T > 1` softens overconfidence; the argmax is unchanged.

#### fit(logits, correct)

Fit `T` by minimising NLL with a bounded 1-D search.

* **Return type:**
  [`TemperatureCalibrator`](#ek.qe.calibrate.TemperatureCalibrator)

#### t_min *: [float](https://docs.python.org/3/builtins/functions.html#float)* *= 0.05*

Golden-section search bounds and iteration count (config, not magic numbers).

### ek.qe.calibrate.expected_calibration_error(probs, correct, , n_bins=10)

Expected Calibration Error: weighted mean gap between confidence and accuracy.

Bins predictions by confidence into `n_bins` equal-width bins and averages
`|mean_confidence - accuracy|` weighted by bin population. `0` is perfect.
Non-finite probs are skipped; out-of-range probs are clamped into `[0, 1]`.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.calibrate.load_calibrator(name, , rootdir=None)

Reconstruct a persisted calibrator by name (dispatched on its `kind`).

Validates the stored record so a malformed or unknown `kind` fails with an
actionable error rather than a raw `KeyError` or a load-then-crash-later.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ek.qe.calibrate.mean_probability(probs)

Fuse calibrated per-family probabilities by their mean (the default combiner).

Each family’s calibrated value estimates the *same* event (“this field is
correct”), so the mean is a defensible, assumption-light pooling rule.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> mean_probability({"judge": 0.9, "logprob": 0.7})
0.8
```

### ek.qe.calibrate.min_probability(probs)

Fuse calibrated per-family probabilities by their minimum (conservative).

Use when any one family flagging a problem should be enough to lower the gate.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> min_probability({"judge": 0.9, "logprob": 0.7})
0.7
```

### ek.qe.calibrate.netcal_ece(probs, correct, , bins=10)

ECE via `netcal` (behind `ek[calibration]`); D-ECE for localized outputs lives there too.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.qe.calibrate.reliability_curve(probs, correct, , n_bins=10)

Per-bin `{confidence, accuracy, count}` for a reliability diagram.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.qe.calibrate.save_calibrator(calibrator, name, , rootdir=None)

Persist a fitted calibrator’s parameters to the `calibrators` store.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.qe.calibrate.sklearn_calibrator(method='sigmoid')

A calibrator backed by scikit-learn (`method='sigmoid'` Platt or `'isotonic'`).

Behind `ek[calibration]`. Returns an object satisfying the
[`Calibrator`](ek.base.html.md#ek.base.Calibrator) protocol that wraps sklearn’s calibration. The
pure-Python [`PlattCalibrator`](#ek.qe.calibrate.PlattCalibrator)/[`IsotonicCalibrator`](#ek.qe.calibrate.IsotonicCalibrator) are the
dependency-free defaults; use this for parity with an sklearn-centric stack.
