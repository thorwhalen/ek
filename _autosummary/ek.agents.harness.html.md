# ek.agents.harness

The agent task-suite harness: k trials, per-slice cuts, and a gate that survives noise.

[`run_suite()`](#ek.agents.harness.run_suite) is [`ek.evaluate_store()`](ek.html.md#ek.evaluate_store) specialized for agents: it runs the system under
test over a suite **k times per task** (reliability needs repeats), grades each episode with an
*injected* checker, and reports `pass^k` + cost + latency per slice.

**Why a separate gate.** `ek`’s offline [`ek.regression_gate()`](ek.html.md#ek.regression_gate) is a scalar point
comparison with `tolerance=0.0` – correct for a deterministic OCR benchmark, **unsound** for
an agent. Agent metrics are random variables: re-running the same agent moves the number. A
point gate therefore (a) flags sampling noise as a regression and (b) misses a real regression
that hides inside the tolerance band.

[`agent_regression_gate()`](#ek.agents.harness.agent_regression_gate) instead tests the **difference** `current - baseline` with both
runs’ uncertainty folded in (a Newcombe score interval on the success rate; a combined-SE bound
on `pass^k`), failing only when the upper bound of that change is confidently negative. Two
tempting shortcuts are **both wrong**, and this module has shipped and fixed each of them:
comparing our interval to the baseline’s *point* (the baseline is itself noisy, so one flake
against a lucky baseline reads as a regression), and asking whether two 95% intervals *overlap*
(that is a ~0.5% test, not a 5% one – it goes blind to a 15-point drop on a 60-task suite).
An empty run, and an empty **baseline**, are both refused: with no trials every bound is the
maximally-uncertain default, so nothing could ever be shown to fall below it.

**Hidden eval variables.** In a tau-bench-style suite the *user simulator is itself an LLM* –
change its model or prompt and the scores move, silently. So a baseline records its
[`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance) and the gate **refuses** to compare across a changed
simulator, suite version, or scaffold, exactly as the offline gate already refuses to compare
two different metrics. Comparing across them is not a regression check, it is a category error.

**Verifier isolation is a security property.** The checker must run where the agent’s trajectory
cannot reach it (a red-team produced a ~10-line `conftest.py` that rewrote every test outcome
to “passed”). `ek` cannot enforce your sandbox – it only refuses to pretend the problem does
not exist.

### Example

```pycon
>>> from ek.agents.base import Cost, Episode, TaskSpec
>>> tasks = [TaskSpec("t1", input="2+2", gold="4"), TaskSpec("t2", input="3+3", gold="6")]
>>> def agent(task):
...     answers = {"2+2": "4", "3+3": "7"}      # gets t2 wrong, every time
...     return Episode(output=answers[task.input], cost=Cost(input_tokens=10))
>>> report = run_suite(agent, tasks, k=1)
>>> report.success_rate, report.pass_hat_k
(0.5, 0.5)
```

### Functions

| [`agent_regression_gate`](#ek.agents.harness.agent_regression_gate)(report, baseline, \*[, ...])   | Fail only when the evidence says the agent **really** got worse -- not on noise.       |
|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`load_agent_baseline`](#ek.agents.harness.load_agent_baseline)(name, \*[, rootdir])             | Load a named agent baseline (or `None`).                                               |
| [`run_suite`](#ek.agents.harness.run_suite)(agent, tasks, \*[, k, check, ...])         | Run `agent` over a task suite, `k` trials per task, and report reliability + cost.     |
| [`save_agent_baseline`](#ek.agents.harness.save_agent_baseline)(report, name, \*[, rootdir])     | Freeze a run as a named baseline (including its provenance, so the gate can guard it). |

### Classes

| [`AgentGateResult`](#ek.agents.harness.AgentGateResult)([passed, reasons, ...])   | Outcome of a variance-aware agent regression check.   |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------|

### *class* ek.agents.harness.AgentGateResult(passed=True, reasons=list(), pass_hat_k=None, pass_hat_k_baseline=None, success_rate=None, success_rate_baseline=None, cost_per_success=None, cost_per_success_baseline=None, underpowered=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Outcome of a variance-aware agent regression check.

### ek.agents.harness.agent_regression_gate(report, baseline, , tolerance=0.0, cost_tolerance=0.0, rootdir=None)

Fail only when the evidence says the agent **really** got worse – not on noise.

Tests the **difference** `current - baseline`, with *both* runs’ uncertainty folded in:
a Newcombe score interval on the success rate, and (for `k > 1`) a combined-SE bound on
`pass^k`. A regression is declared only when the upper bound of that change is below
`-tolerance` – i.e. we are confident the true change is negative.

Two tempting shortcuts are **both wrong**, and this module has shipped and fixed each:

- Comparing our interval to the baseline’s **point**. The baseline is *itself* a noisy
  estimate; a Wilson upper bound only reaches 1.0 when every trial passed, so a single flake
  against a lucky baseline reads as a “confident regression”.
- Asking whether the two 95% intervals **overlap**. That is not a 5% test but roughly a 0.5%
  one: it buys a calm gate by going blind (a 15-point drop on a 60-task suite becomes
  invisible). A gate that cannot see a real regression is as broken as one that cries wolf.

A result that is merely *worse but not significantly so* is not a clean pass either – it is
an underpowered experiment, and the result says so (`underpowered`, plus a warning).

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the baseline was produced with a different **user simulator**, suite
      version, or scaffold – those runs are not comparable, and silently comparing them
      would be a category error (the same reason the offline gate refuses to compare two
      different metrics); if the baseline recorded zero trials (a permanent free pass); or
      if it is malformed.
* **Return type:**
  [`AgentGateResult`](#ek.agents.harness.AgentGateResult)

### ek.agents.harness.load_agent_baseline(name, , rootdir=None)

Load a named agent baseline (or `None`).

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.agents.harness.run_suite(agent, tasks, , k=1, check=None, metrics=None, price=None, prices=None, run=None, seed=None, persist=False, run_id=None, rootdir=None)

Run `agent` over a task suite, `k` trials per task, and report reliability + cost.

* **Parameters:**
  * **agent** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`TaskSpec`](ek.agents.base.html.md#ek.agents.base.TaskSpec)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The system under test: `TaskSpec -> Episode` (a bare answer is wrapped). It
    receives the whole spec – `.input`, `.tools`, `.gold` – not just the input.
    If it accepts a keyword `seed`, the per-trial seed is passed to it.
  * **tasks** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – [`TaskSpec`](ek.agents.base.html.md#ek.agents.base.TaskSpec) s (or a `{task_id: spec}` mapping).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Trials per task. `k > 1` is what makes `pass^k` meaningful – and it only means
    anything if the agent is genuinely stochastic (the report warns if it is not).
  * **check** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The success oracle – a registered checker name or an `(episode, gold) -> bool`
    callable. Defaults to `"output"`; **inject a state-based/executable oracle** for a
    real suite, and run it isolated from the agent.
  * **metrics** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – Optional `{name: Metric}` scored on every episode against `task.gold`, with
    the **suite’s Layer-A grammar injected** – this is how the tool/argument cost
    weights reach a `tool_call`/`trajectory` metric. Results land in
    `report.detail["metrics"]`.
  * **price/prices** – Rates for costing the episodes (see [`ek.agents.cost`](ek.agents.cost.html.md#module-ek.agents.cost)).
  * **run** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance)]) – [`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance) for this run (model, simulator, suite
    version, scaffold). Recorded on every episode and on any baseline saved from it.
  * **seed** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Base RNG seed. Trial *i* of each task uses `seed + i`: it is recorded on the
    episode’s provenance **and passed to the agent** when the agent accepts a `seed`
    keyword.
  * **persist/run_id/rootdir** – Persist the run to the `runs`/`results` stores.
* **Return type:**
  [`ReliabilityReport`](ek.agents.html.md#ek.agents.ReliabilityReport)
* **Returns:**
  A [`ReliabilityReport`](ek.agents.html.md#ek.agents.ReliabilityReport) carrying `pass@k`, `pass^k` (with
  a bootstrap CI), the success rate (with a Wilson CI), the per-slice cuts, and the cost
  report – because reliability without cost is only half the answer.

### ek.agents.harness.save_agent_baseline(report, name, , rootdir=None)

Freeze a run as a named baseline (including its provenance, so the gate can guard it).

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the run evaluated nothing. An empty baseline is worse than no baseline –
      it is a *permanent free pass*, because every bound in it is the maximally-uncertain
      default and no later run can be shown to fall below it.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
