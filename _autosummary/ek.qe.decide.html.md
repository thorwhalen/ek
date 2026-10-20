# ek.qe.decide

Selective prediction: turn a calibrated probability into accept / flag / block.

Calibration makes a score *mean* something; a **decision policy** turns it into an
action with a guarantee (`misc/docs/ek_03` §3-4). This is the formal home of
“accept / flag / block” – a selective predictor that abstains (routes to a human)
rather than emit a low-confidence answer. Every policy here reads \*\*only calibrated
probabilities\*\* (Hard Rule 1): a raw score must already have passed through a
[`Calibrator`](ek.base.html.md#ek.base.Calibrator).

Three policies, by what you want to guarantee:

- [`CostSensitiveGate`](#ek.qe.decide.CostSensitiveGate) – **the default**. The accept threshold falls out of the
  cost ratio `rho = c_FN / c_FP` on the calibrated probability; no labelled set
  needed once you know your costs.
- [`ConformalGate`](#ek.qe.decide.ConformalGate) – split (inductive) conformal: a finite-sample,
  distribution-free **marginal** guarantee that at most `alpha` of truly-correct
  items are flagged. Per-field-type validity needs the Mondrian
  ([`GroupConformalGate`](#ek.qe.decide.GroupConformalGate)) variant – conditional coverage is otherwise
  impossible (Barber et al. 2019).
- [`RiskControlGate`](#ek.qe.decide.RiskControlGate) – conformal risk control: bound a monotone loss (the rate
  of accepted-yet-wrong items) at a target level.

[`risk_coverage_curve()`](#ek.qe.decide.risk_coverage_curve) exposes the trade-off so stakeholders pick the operating
point explicitly. The pure-Python gates need no extra dependencies; the
`ek[calibration]` extra (MAPIE/crepes) offers library-backed conformal for an
sklearn-centric stack. Exchangeability is the load-bearing assumption – track
realised coverage and re-fit on drift.

### Example

```pycon
>>> from ek.base import Decision
>>> gate = CostSensitiveGate(rho=9.0)          # a false-negative costs 9x a review
>>> gate(0.95) is Decision.ACCEPT              # tau = rho/(1+rho) = 9/10 = 0.9
True
>>> gate(0.5) is Decision.FLAG
True
```

### Module Attributes

| [`DEFAULT_ALPHA`](#ek.qe.decide.DEFAULT_ALPHA)       | Default target miscoverage for conformal gates (flag at most ~alpha of correct items).                                 |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [`DEFAULT_RISK_TARGET`](#ek.qe.decide.DEFAULT_RISK_TARGET) | Default risk bound for [`RiskControlGate`](#ek.qe.decide.RiskControlGate) (accepted-error rate ceiling). |

### Functions

| [`crepes_mondrian_gate`](#ek.qe.decide.crepes_mondrian_gate)(\*args, \*\*kwargs)       | A Mondrian (class-conditional) conformal gate backed by `crepes` (`ek[calibration]`).   |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`risk_coverage_curve`](#ek.qe.decide.risk_coverage_curve)(probs, correct, \*[, ...]) | The risk-coverage trade-off: `{threshold, coverage, selective_risk}` per point.         |
| [`split_conformal_quantile`](#ek.qe.decide.split_conformal_quantile)(scores, alpha)        | The split-conformal threshold: the `ceil((n+1)(1-alpha)) / n` empirical quantile.       |

### Classes

| [`ConformalGate`](#ek.qe.decide.ConformalGate)([alpha, nonconformity, q, \_cal])   | Split-conformal accept/flag with a finite-sample **marginal** guarantee.                                              |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`CostSensitiveGate`](#ek.qe.decide.CostSensitiveGate)([rho, block_threshold, ...])    | Accept/flag/block from the cost ratio `rho = c_FN / c_FP` on calibrated `p`.                                          |
| [`GroupConformalGate`](#ek.qe.decide.GroupConformalGate)([alpha, nonconformity, ...])   | Mondrian (class-conditional) conformal: one [`ConformalGate`](#ek.qe.decide.ConformalGate) per group. |
| [`RiskControlGate`](#ek.qe.decide.RiskControlGate)([target, loss, loss_bound, lam])  | Conformal Risk Control: accept `p >= lambda` bounding the accepted-error rate.                                        |

### *class* ek.qe.decide.ConformalGate(alpha=0.1, nonconformity=<function ConformalGate.<lambda>>, q=inf, \_cal=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Split-conformal accept/flag with a finite-sample **marginal** guarantee.

Fit on `(calibrated_prob, correct)`: the nonconformity scores of the *correct*
calibration items set a threshold `q` such that, among truly-correct test items
exchangeable with them, at most `alpha` are flagged. The guarantee is
*marginal*, not per-field-type – use [`GroupConformalGate`](#ek.qe.decide.GroupConformalGate) for that.

* **Parameters:**
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Target false-flag rate on correct items (e.g. `0.1`).
  * **nonconformity** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Maps calibrated confidence -> nonconformity (default
    `1 - confidence`: low confidence is nonconforming).

#### fit(probs, correct)

Fit `q` from the nonconformity scores of the *correct* calibration items.

Non-finite confidences are dropped (a single `NaN` would otherwise sort to
the end and corrupt the quantile, silently breaking the coverage guarantee).

* **Return type:**
  [`ConformalGate`](#ek.qe.decide.ConformalGate)

#### p_value(confidence)

Per-instance conformal p-value (small = atypical of correct items).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.qe.decide.CostSensitiveGate(rho=1.0, block_threshold=0.0, accept_threshold=None)

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

### ek.qe.decide.DEFAULT_ALPHA *= 0.1*

Default target miscoverage for conformal gates (flag at most ~alpha of correct items).

### ek.qe.decide.DEFAULT_RISK_TARGET *= 0.05*

Default risk bound for [`RiskControlGate`](#ek.qe.decide.RiskControlGate) (accepted-error rate ceiling).

### *class* ek.qe.decide.GroupConformalGate(alpha=0.1, nonconformity=<function GroupConformalGate.<lambda>>, by_group=dict(), pooled=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Mondrian (class-conditional) conformal: one [`ConformalGate`](#ek.qe.decide.ConformalGate) per group.

Restores approximate per-field-type validity by calibrating separately per group
key (e.g. `NodeType`/`FieldSpec`), with a pooled fallback for unseen groups.

### *class* ek.qe.decide.RiskControlGate(target=0.05, loss=<function RiskControlGate.<lambda>>, loss_bound=1.0, lam=1.0)

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
  [`RiskControlGate`](#ek.qe.decide.RiskControlGate)

### ek.qe.decide.crepes_mondrian_gate(\*args, \*\*kwargs)

A Mondrian (class-conditional) conformal gate backed by `crepes` (`ek[calibration]`).

The pure-Python [`GroupConformalGate`](#ek.qe.decide.GroupConformalGate) is the dependency-free default; use
this for parity with a crepes/sklearn conformal stack (full CDFs, normalised
variants). Raises an actionable install hint if `crepes` is absent.

### ek.qe.decide.risk_coverage_curve(probs, correct, , thresholds=None)

The risk-coverage trade-off: `{threshold, coverage, selective_risk}` per point.

Coverage is the fraction auto-accepted at each threshold; selective risk is the
error rate *among the accepted*. Lower-left is better; pick the operating point
against a target risk or coverage.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.qe.decide.split_conformal_quantile(scores, alpha)

The split-conformal threshold: the `ceil((n+1)(1-alpha)) / n` empirical quantile.

`scores` are nonconformity scores on an exchangeable calibration set. Returns
`+inf` when `n` is too small to guarantee `1 - alpha` (so nothing is
flagged – the honest behaviour at that sample size), and `-inf` at the
`alpha -> 1` limit (flag everything).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
