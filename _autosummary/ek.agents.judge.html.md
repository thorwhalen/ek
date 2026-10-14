# ek.agents.judge

LLM-as-judge: a reference-free [`Signal`](ek.base.html.md#ek.base.Signal), a reference-based
[`Metric`](ek.base.html.md#ek.base.Metric), and the validation that must precede either.

When a task has no checkable outcome, an LLM judge is the fallback. It is also a \*\*liability
unless validated and calibrated\*\*, because its known failure modes have double-digit effect
sizes: *position* bias (which answer came first), *verbosity* bias (longer looks better),
*self-preference* (a model prefers its own text). `misc/docs/ek_09` has the numbers.

Three seams, deliberately kept apart (they are not interchangeable):

- [`JudgeSignal`](#ek.agents.judge.JudgeSignal) – **reference-free, criteria-only**. This is the only judge that is a
  `Signal`: it scores an output against *criteria*, with no gold, so it plugs into
  [`ek.estimate_quality()`](ek.html.md#ek.estimate_quality)’s `signal -> calibrate -> validate -> decide` cascade. Its
  > `cost_tier` is the highest in the framework, so an escalation policy runs the cheap signals
  > first and pays for the judge only on residual uncertainty.
- [`JudgeMetric`](#ek.agents.judge.JudgeMetric) – **reference-based**. A judge that needs the gold answer is not a
  Signal at all; it is a Metric, and it belongs on the [`ek.score()`](ek.html.md#ek.score) side.
- [`pairwise_judge()`](#ek.agents.judge.pairwise_judge) – pairwise comparison needs *two* outputs and cannot fit
  `Signal.__call__(one_output)`. It is a helper, and it **swaps the order and averages** –
  the standard, necessary correction for position bias.

**Hard Rule 1 applies to judges.** A raw judge score is not a probability. Calibrate it before
any [`DecisionPolicy`](ek.base.html.md#ek.base.DecisionPolicy) reads it – [`ek.estimate_quality()`](ek.html.md#ek.estimate_quality) already refuses to
gate an uncalibrated signal, and a judge is no exception (-> `misc/docs/ek_03`).

**Calibrate a judge per signal family.** With a single calibrator, [`ek.estimate_quality()`](ek.html.md#ek.estimate_quality)
averages *all* raw signals into one confidence and calibrates that mean. Mixing a judge score with,
say, a logprob – different scales, different reliabilities – that way is not statistically sound.
Pass `calibrator={"judge": judge_cal, "logprob": logprob_cal}` (or a
[`SignalFamilyCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.SignalFamilyCalibrator), which can `fit` one calibrator per signal from
labelled rows) so each family is calibrated on its own before the calibrated probabilities are fused.

`ek` never imports an LLM SDK: `judge` is an **injected callable**.

### Example

```pycon
>>> stub = lambda output, *, criteria="", reference=None: 0.9 if "meow" in output else 0.1
>>> sig = JudgeSignal(stub, criteria="Is the answer helpful?")
>>> sig("the cat says meow")
0.9
>>> sig.cost_tier                       # the most expensive signal family
5
```

### Module Attributes

| [`JUDGE_COST_TIER`](#ek.agents.judge.JUDGE_COST_TIER)   | Judges are the most expensive signal family -- run cheap signals first and escalate.   |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`ALPHA_GOOD`](#ek.agents.judge.ALPHA_GOOD)        | >= 0.8 is good agreement, 0.67-0.8 only tentative.                                     |

### Functions

| [`judge_validation`](#ek.agents.judge.judge_validation)(judge_labels, human_labels, \*)   | Certify a judge against human labels **before** trusting it.           |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`pairwise_judge`](#ek.agents.judge.pairwise_judge)(judge, a, b, \*[, criteria, swap])  | Pairwise preference for `a` over `b`, **corrected for position bias**. |

### Classes

| [`JudgeMetric`](#ek.agents.judge.JudgeMetric)(judge[, criteria])                  | A **reference-based** LLM judge as a [`Metric`](ek.base.html.md#ek.base.Metric) (the `score()` side).   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| [`JudgeSignal`](#ek.agents.judge.JudgeSignal)(judge[, criteria, cost_tier, name]) | A reference-free, criteria-only LLM judge as a [`Signal`](ek.base.html.md#ek.base.Signal).              |

### ek.agents.judge.ALPHA_GOOD *= 0.8*

>= 0.8 is good agreement, 0.67-0.8 only tentative.

* **Type:**
  Krippendorff conventions

### ek.agents.judge.JUDGE_COST_TIER *= 5*

Judges are the most expensive signal family – run cheap signals first and escalate.

### *class* ek.agents.judge.JudgeMetric(judge, criteria='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A **reference-based** LLM judge as a [`Metric`](ek.base.html.md#ek.base.Metric) (the `score()` side).

A judge that reads the gold answer is not a reference-free signal, so it does not belong in
[`ek.estimate_quality()`](ek.html.md#ek.estimate_quality). It lives here.

* **Parameters:**
  * **judge** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Injected `(output, *, criteria, reference) -> float` in `[0, 1]`.
  * **criteria** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The rubric.

#### aggregate(scores)

Mean judge score (a judge score *is* per-item; there is no global accumulation).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.agents.judge.JudgeSignal(judge, criteria='', cost_tier=5, name='judge')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reference-free, criteria-only LLM judge as a [`Signal`](ek.base.html.md#ek.base.Signal).

* **Parameters:**
  * **judge** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – An **injected** callable `(output, *, criteria) -> float`. `ek` core never
    imports an LLM client; you pass your own (or an `ek[judge]` adapter).
  * **criteria** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The rubric the judge scores against, bound at construction (exactly as
    [`LogprobSignal`](ek.qe.signals.html.md#ek.qe.signals.LogprobSignal) binds its aggregator), so the object still
    satisfies `Signal.__call__(output) -> float`.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – [`JUDGE_COST_TIER`](#ek.agents.judge.JUDGE_COST_TIER) – the most expensive tier.

#### finding(target, rationale)

Surface a judge’s *rationale* as a [`Finding`](ek.base.html.md#ek.base.Finding) (not just a scalar).

A judge’s reasoning is auditable evidence; a bare number throws it away.

* **Return type:**
  [`Finding`](ek.base.html.md#ek.base.Finding)

### ek.agents.judge.judge_validation(judge_labels, human_labels, , level='nominal')

Certify a judge against human labels **before** trusting it. No new code – reuse the IAA.

A judge is just another annotator, so `ek`’s existing inter-annotator-agreement machinery
(which already tells you the *ceiling* of a gold standard) is exactly the right instrument:
Krippendorff’s alpha is the default (it alone handles any number of raters, any measurement
level, and missing data); Cohen’s kappa applies for exactly two raters on complete nominal
data. This reproduces the MT-Bench judge-vs-human protocol.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  `alpha`, `kappa`, `percent_agreement`, `n`, and a `verdict` against the
  conventional thresholds (alpha >= 0.8 good, 0.67-0.8 tentative, below that unreliable).

### Example

```pycon
>>> r = judge_validation([1, 1, 0, 0], [1, 1, 0, 0])
>>> r["alpha"], r["verdict"]
(1.0, 'good')
```

### ek.agents.judge.pairwise_judge(judge, a, b, , criteria='', swap=True)

Pairwise preference for `a` over `b`, **corrected for position bias**.

Position bias is not a rounding error – judges materially prefer whichever answer they see
first. The standard correction is to ask twice with the order swapped and average, which is
why this cannot be a plain `Signal` (it needs two outputs).

* **Parameters:**
  * **judge** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Injected `(a, b, *, criteria) -> float` in `[0, 1]`, the preference for `a`.
  * **swap** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Ask again with the arguments swapped and average (default; disable only if your
    judge is known to be order-invariant).
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Preference for `a` in `[0, 1]` (`0.5` = a tie).

### Example

```pycon
>>> first_wins = lambda x, y, *, criteria="": 1.0     # a maximally position-biased judge
>>> pairwise_judge(first_wins, "A", "B")              # swapping exposes it as a tie
0.5
```
