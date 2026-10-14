# ek.agents.metrics

Task-success and cost-per-successful-task: the two metrics that drive the whole thing.

Two reference-based metrics over an [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode), both of which fit
[`ek.evaluate()`](ek.html.md#ek.evaluate) exactly because their corpus aggregate is a **ratio of two sums**, never a
mean of per-item scores – the same discipline that makes `ek`’s WER accumulate globally and
its F1 micro-average (`misc/docs/ek_08`, `misc/docs/ek_11`):

- [`TaskSuccessMetric`](#ek.agents.metrics.TaskSuccessMetric) – grade the **final state**, not the surface text. The oracle is
  *injected*: `ek` never owns an LLM or a sandbox, so a checker is a plugged-in strategy
  (a database-state comparison, a hidden-test run, a programmatic predicate, a judge).
  Aggregate = the success **rate**.
- [`CostPerSuccessMetric`](#ek.agents.metrics.CostPerSuccessMetric) – **Cost-of-Pass**: `E[cost] / P(success)`. Aggregate =
  `sum(dollars) / sum(successes)`, diverging to `inf` when nothing succeeded. A benchmark
  that scores accuracy *without* cost lets an agent chase tiny gains with unbounded API calls,
  so this metric is the point, not a nicety.

**A safety property, not a nicety:** the checker must run somewhere the agent’s trajectory
cannot reach. An audited agent-benchmark red-team produced a ~10-line `conftest.py` that
pytest auto-loads and that rewrote *every* test outcome to “passed” – scoring near-perfect
without solving a single task. Verifier isolation is a security boundary.

### Example

```pycon
>>> from ek.agents.base import Cost, Episode
>>> from ek.agents.cost import per_million
>>> ep = Episode(task_id="t1", output="42", cost=Cost(input_tokens=1_000_000))
>>> TaskSuccessMetric()(ep, "42").value
1.0
>>> m = CostPerSuccessMetric(price=per_million(2.0, 6.0))
>>> round(m(ep, "42").value, 4)          # this episode's dollars
2.0
```

### Functions

| [`final_state_match`](#ek.agents.metrics.final_state_match)(episode, gold)   | Compare the episode's **final state** with the goal state (the tau-bench oracle).   |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`output_match`](#ek.agents.metrics.output_match)(episode, gold)        | Canonicalized exact match on the final answer (the GAIA-style oracle).              |
| [`recorded_success`](#ek.agents.metrics.recorded_success)(episode, gold)    | Trust a `success` already recorded on the episode (e.g. by an external harness).    |

### Classes

| [`Checker`](#ek.agents.metrics.Checker)(\*args, \*\*kwargs)                  | The success oracle: did this episode leave the world in the required state?      |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`CostPerSuccessMetric`](#ek.agents.metrics.CostPerSuccessMetric)([check, price, prices]) | Cost-of-Pass: dollars per **successfully completed** task (**lower is better**). |
| [`TaskSuccessMetric`](#ek.agents.metrics.TaskSuccessMetric)([check])                   | Did the episode complete the task? (`1.0`/`0.0`; aggregate = success rate).      |

### *class* ek.agents.metrics.Checker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The success oracle: did this episode leave the world in the required state?

Injected, never owned by `ek` – a real suite plugs in a database-state comparison, a
hidden-test runner, or a programmatic predicate. It must run **isolated** from the agent.

### *class* ek.agents.metrics.CostPerSuccessMetric(check=None, , price=None, prices=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cost-of-Pass: dollars per **successfully completed** task (**lower is better**).

The per-episode `Score.value` is that episode’s dollar cost – the real number is the
corpus `aggregate`, `sum(dollars) / sum(successes)`, which is `inf` when nothing
succeeded (infeasibility is the honest answer; a per-token metric reporting “$0.003/call”
for an agent that never succeeds is actively misleading).

* **Parameters:**
  * **check** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The success oracle (as in [`TaskSuccessMetric`](#ek.agents.metrics.TaskSuccessMetric)).
  * **price** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)]) – A [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice) applied to every episode, or `None` to
    resolve each episode’s model against `prices`.
  * **prices** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)]]) – A `{model: ModelPrice}` catalog (see [`ek.agents.load_prices()`](ek.agents.html.md#ek.agents.load_prices)).

#### aggregate(scores)

`sum(dollars) / sum(successes)` – a ratio of two sums, never a mean of ratios.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.agents.metrics.TaskSuccessMetric(check=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Did the episode complete the task? (`1.0`/`0.0`; aggregate = success rate).

* **Parameters:**
  **check** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A [`Checker`](#ek.agents.metrics.Checker) – a registered name (`"output"`, `"final_state"`,
  `"recorded"`) or any `(episode, gold) -> bool` callable. Defaults to
  `"output"`; **inject a state-based or executable oracle for a real suite.**

#### aggregate(scores)

The success **rate** over the corpus.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.metrics.final_state_match(episode, gold)

Compare the episode’s **final state** with the goal state (the tau-bench oracle).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.metrics.output_match(episode, gold)

Canonicalized exact match on the final answer (the GAIA-style oracle).

The weakest useful oracle, and the default only because it needs nothing. Prefer a
state-based or executable checker whenever the task has one.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.metrics.recorded_success(episode, gold)

Trust a `success` already recorded on the episode (e.g. by an external harness).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
