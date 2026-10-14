# ek.agents

`ek.agents` – evaluating AI agents and assistants, in cost per successful task.

The agent instance of `ek`, mirroring [`ek.ocr`](ek.ocr.html.md#module-ek.ocr). Agent evaluation is the **same 2x2** the
framework already implements – (reference availability) x (granularity) – but the evaluated
object is an **episode** (tool calls + observations ending in a final state) and the objective is
**cost per successfully completed task**, not cost per token.

corner | agent meaning | facade |
<br/>

```
|---|
```

—

```
|---|
```


| reference-based, one item | grade an episode against a gold outcome/trajectory | [`ek.score()`](ek.html.md#ek.score) |
| reference-based, corpus | task-suite benchmark: `pass^k` + cost, per slice | [`run_suite()`](#ek.agents.run_suite) |
| reference-free, one item | judge / verifier / the cascade’s confidence gate | [`ek.estimate_quality()`](ek.html.md#ek.estimate_quality) |

What it reuses (rather than rebuilds): the Layer-A `GraphGrammar` becomes the \*\*task/tool
grammar\*\* (`FieldSpec.importance` = the cost of a wrong argument, so a wrong argument to a
*destructive* tool is not one unit of error); `FieldMetric`’s TP/FP/FN accounting powers
tool-call scoring; the ROVER agreement seam is self-consistency; the IAA helpers validate a
judge; and the whole `signal -> calibrate -> validate -> decide` cascade is the confidence gate.

Quickstart:

```pycon
>>> from ek.agents import TaskSpec, run_suite, as_agent
>>> tasks = [TaskSpec("t1", input="hi", gold="HI"), TaskSpec("t2", input="yo", gold="YO")]
>>> report = run_suite(as_agent(str.upper), tasks)
>>> report.success_rate
1.0
```

Research: `misc/docs/ek_07` (map) - `ek_08` (task success) - `ek_09` (judge/QE) -
`ek_10` (trajectory) - `ek_11` (cost economics) - `ek_12` (libraries + integration).

### Functions

| [`tool_grammar`](#ek.agents.tool_grammar)(\*tools)                              | Build the Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) from [`ToolSpec`](#ek.agents.ToolSpec) s.   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`suite_grammar`](#ek.agents.suite_grammar)(tasks)                               | The union grammar over a whole task suite (later tools win on name collision).                                                                                           |
| [`as_call`](#ek.agents.as_call)(x)                                         | Coerce a [`Step`](#ek.agents.Step), `(tool, args)` tuple, or `{"tool":…, "args":…}` dict.                                                    |
| [`is_episode`](#ek.agents.is_episode)(x)                                      | Duck-typed episode check (core never imports this module -- it checks the *shape*).                                                                                      |
| [`match_calls`](#ek.agents.match_calls)(pred, gold, \*[, canon])               | Assign predicted calls to gold calls -- the layer `FieldMetric` does not provide.                                                                                        |
| [`output_match`](#ek.agents.output_match)(episode, gold)                        | Canonicalized exact match on the final answer (the GAIA-style oracle).                                                                                                   |
| [`final_state_match`](#ek.agents.final_state_match)(episode, gold)                   | Compare the episode's **final state** with the goal state (the tau-bench oracle).                                                                                        |
| [`recorded_success`](#ek.agents.recorded_success)(episode, gold)                    | Trust a `success` already recorded on the episode (e.g. by an external harness).                                                                                         |
| [`pass_at_k`](#ek.agents.pass_at_k)(\*, n, c, k)                             | Unbiased `pass@k`: probability that **at least one** of k samples succeeds.                                                                                              |
| [`pass_hat_k`](#ek.agents.pass_hat_k)(\*, n, c, k)                            | Unbiased `pass^k`: probability that **all k** independent trials succeed.                                                                                                |
| [`reliability`](#ek.agents.reliability)(source, \*[, k, slices, cost, ...])    | Compute `pass@k` / `pass^k` / success-rate over a k-trial run.                                                                                                           |
| [`wilson_interval`](#ek.agents.wilson_interval)(successes, n, \*[, z])             | Wilson score interval for a proportion -- correct in the tails, unlike Wald/CLT.                                                                                         |
| [`bootstrap_ci`](#ek.agents.bootstrap_ci)(values, statistic, \*[, ...])         | Percentile bootstrap CI for any `statistic` over `values` (deterministic by seed).                                                                                       |
| [`newcombe_difference`](#ek.agents.newcombe_difference)(successes_a, n_a, ...[, z])    | Newcombe score interval for the **difference** of two proportions `p_a - p_b`.                                                                                           |
| [`difference_upper_bound`](#ek.agents.difference_upper_bound)(value_a, ci_a, ...[, z])    | Upper bound of `a - b` for two statistics known only through their CIs.                                                                                                  |
| [`per_million`](#ek.agents.per_million)(input, output, \*[, ...])              | Build a [`ModelPrice`](#ek.agents.ModelPrice) from the human-facing **USD per 1M tokens** figures.                                                 |
| [`dollars`](#ek.agents.dollars)(cost, price, \*[, prices, batch])          | Monetary cost of one [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) tally, in USD.                                                            |
| [`episode_dollars`](#ek.agents.episode_dollars)(episode, \*[, prices, price, ...]) | Monetary cost of one episode (0.0 if it carries no [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)).                                           |
| [`cost_of_pass`](#ek.agents.cost_of_pass)(total_cost, n_success)                | Expected monetary cost of one *successful* task: `total_cost / n_success`.                                                                                               |
| [`cost_report`](#ek.agents.cost_report)(episodes, \*[, prices, price])         | The quality x cost x latency triple, reported **together**.                                                                                                              |
| [`load_prices`](#ek.agents.load_prices)(source)                                | Parse a LiteLLM-format price catalog into a `{model: ModelPrice}` table.                                                                                                 |
| [`price_of`](#ek.agents.price_of)(model, \*[, prices])                      | Resolve a model's rates from an injected catalog, or fail *actionably*.                                                                                                  |
| [`judge_validation`](#ek.agents.judge_validation)(judge_labels, human_labels, \*)   | Certify a judge against human labels **before** trusting it.                                                                                                             |
| [`pairwise_judge`](#ek.agents.pairwise_judge)(judge, a, b, \*[, criteria, swap])  | Pairwise preference for `a` over `b`, **corrected for position bias**.                                                                                                   |
| [`run_suite`](#ek.agents.run_suite)(agent, tasks, \*[, k, check, ...])       | Run `agent` over a task suite, `k` trials per task, and report reliability + cost.                                                                                       |
| [`save_agent_baseline`](#ek.agents.save_agent_baseline)(report, name, \*[, rootdir])   | Freeze a run as a named baseline (including its provenance, so the gate can guard it).                                                                                   |
| [`load_agent_baseline`](#ek.agents.load_agent_baseline)(name, \*[, rootdir])           | Load a named agent baseline (or `None`).                                                                                                                                 |
| [`agent_regression_gate`](#ek.agents.agent_regression_gate)(report, baseline, \*[, ...]) | Fail only when the evidence says the agent **really** got worse -- not on noise.                                                                                         |
| [`as_agent`](#ek.agents.as_agent)(fn)                                       | Wrap a plain `input -> answer` function into the harness's `TaskSpec -> Episode` shape.                                                                                  |
| [`trajectory_from_messages`](#ek.agents.trajectory_from_messages)(messages)                 | Parse a chat transcript into a [`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory).                                                    |
| [`episode_from_messages`](#ek.agents.episode_from_messages)(messages, \*[, ...])         | Build a full [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode) from a transcript (+ optional usage/state).                                 |
| [`cost_from_usage`](#ek.agents.cost_from_usage)(usage, \*[, latency_s])            | Build a [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) from a provider `usage` object or dict.                                                |
| [`from_inspect_sample`](#ek.agents.from_inspect_sample)(sample, \*[, task_id])         | Adapt an Inspect AI `EvalSample`-shaped object into an [`Episode`](#ek.agents.Episode) (duck-typed).                                            |
| [`from_deepeval_test_case`](#ek.agents.from_deepeval_test_case)(case, \*[, task_id])       | Adapt a DeepEval `LLMTestCase`-shaped object into an [`Episode`](#ek.agents.Episode) (duck-typed).                                              |

### Classes

| [`TaskSpec`](#ek.agents.TaskSpec)(task_id[, input, gold, value, ...])      | One task in a suite: its input, its oracle, its value, and its allowed tools.                                                        |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [`ToolSpec`](#ek.agents.ToolSpec)(name[, params, importance, destructive]) | One tool the agent may call: its argument schema and its error costs.                                                                |
| [`Episode`](#ek.agents.Episode)([task_id, trajectory, output, ...])       | Layer B: one agent run over one task -- the object agent evaluation scores.                                                          |
| [`Trajectory`](#ek.agents.Trajectory)([steps])                               | The ordered sequence of [`Step`](#ek.agents.Step) s an agent took.                                       |
| [`Step`](#ek.agents.Step)(tool[, args, observation, error])            | One step of a trajectory: a tool call and what came back.                                                                            |
| [`Cost`](#ek.agents.Cost)([input_tokens, output_tokens, ...])          | What one episode consumed: tokens (by kind), retries, and wall-clock.                                                                |
| [`RunProvenance`](#ek.agents.RunProvenance)([seed, temperature, model, ...])    | The *hidden eval variables* that decide whether two runs are even comparable.                                                        |
| [`TaskSuccessMetric`](#ek.agents.TaskSuccessMetric)([check])                        | Did the episode complete the task? (`1.0`/`0.0`; aggregate = success rate).                                                          |
| [`CostPerSuccessMetric`](#ek.agents.CostPerSuccessMetric)([check, price, prices])      | Cost-of-Pass: dollars per **successfully completed** task (**lower is better**).                                                     |
| [`ToolCallMetric`](#ek.agents.ToolCallMetric)([level, canonicalizer, grammar])   | Tool-call correctness (BFCL AST match) as a cost-weighted [`Metric`](ek.base.html.md#ek.base.Metric).    |
| [`TrajectoryMetric`](#ek.agents.TrajectoryMetric)([scheme, grammar, ...])          | Cost-weighted trajectory distance as a [`Metric`](ek.base.html.md#ek.base.Metric) (**lower is better**). |
| [`JudgeMetric`](#ek.agents.JudgeMetric)(judge[, criteria])                    | A **reference-based** LLM judge as a [`Metric`](ek.base.html.md#ek.base.Metric) (the `score()` side).    |
| [`Checker`](#ek.agents.Checker)(\*args, \*\*kwargs)                       | The success oracle: did this episode leave the world in the required state?                                                          |
| [`ReliabilityReport`](#ek.agents.ReliabilityReport)([k, n_tasks, n_trials, ...])    | Corpus-level reliability over a k-trial run, with per-task and per-slice cuts.                                                       |
| [`ModelPrice`](#ek.agents.ModelPrice)([input, output, cached_input, ...])    | Per-token rates for one model, in **USD per token**.                                                                                 |
| [`JudgeSignal`](#ek.agents.JudgeSignal)(judge[, criteria, cost_tier, name])   | A reference-free, criteria-only LLM judge as a [`Signal`](ek.base.html.md#ek.base.Signal).               |
| [`AgentGateResult`](#ek.agents.AgentGateResult)([passed, reasons, ...])           | Outcome of a variance-aware agent regression check.                                                                                  |

### Exceptions

| [`UnknownModelPrice`](#ek.agents.UnknownModelPrice)   | Raised when a model's rates are not in the injected price catalog.   |
|----------------------------------------------------------------------|----------------------------------------------------------------------|

### *class* ek.agents.AgentGateResult(passed=True, reasons=<factory>, pass_hat_k=None, pass_hat_k_baseline=None, success_rate=None, success_rate_baseline=None, cost_per_success=None, cost_per_success_baseline=None, underpowered=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Outcome of a variance-aware agent regression check.

### *class* ek.agents.Checker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The success oracle: did this episode leave the world in the required state?

Injected, never owned by `ek` – a real suite plugs in a database-state comparison, a
hidden-test runner, or a programmatic predicate. It must run **isolated** from the agent.

### *class* ek.agents.Cost(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0, retries=0, latency_s=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What one episode consumed: tokens (by kind), retries, and wall-clock.

Token *kinds* are tracked separately because they are **priced asymmetrically** –
output typically costs ~5x input, cached input bills at a fraction, and reasoning
tokens are billed. This is precisely why the objective is *dollars* per successful
task and never a raw token count (`misc/docs/ek_11`). Pricing lives in
[`ek.agents.cost`](ek.agents.cost.html.md#module-ek.agents.cost); this dataclass is the provider-neutral tally.

### Example

```pycon
>>> a = Cost(input_tokens=100, output_tokens=50)
>>> b = Cost(input_tokens=10, output_tokens=5, retries=1)
>>> c = a + b
>>> c.input_tokens, c.output_tokens, c.retries
(110, 55, 1)
```

#### *property* total_tokens *: [int](https://docs.python.org/3/builtins/functions.html#int)*

All billed tokens (a rough size proxy – *not* a cost; see `dollars()`).

### *class* ek.agents.CostPerSuccessMetric(check=None, , price=None, prices=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cost-of-Pass: dollars per **successfully completed** task (**lower is better**).

The per-episode `Score.value` is that episode’s dollar cost – the real number is the
corpus `aggregate`, `sum(dollars) / sum(successes)`, which is `inf` when nothing
succeeded (infeasibility is the honest answer; a per-token metric reporting “$0.003/call”
for an agent that never succeeds is actively misleading).

* **Parameters:**
  * **check** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The success oracle (as in [`TaskSuccessMetric`](#ek.agents.TaskSuccessMetric)).
  * **price** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)]) – A [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice) applied to every episode, or `None` to
    resolve each episode’s model against `prices`.
  * **prices** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)]]) – A `{model: ModelPrice}` catalog (see [`ek.agents.load_prices()`](#ek.agents.load_prices)).

#### aggregate(scores)

`sum(dollars) / sum(successes)` – a ratio of two sums, never a mean of ratios.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.agents.Episode(task_id='', trajectory=<factory>, output=None, final_state=None, cost=None, success=None, run=None, provenance=None, meta=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Layer B: one agent run over one task – the object agent evaluation scores.

The direct analog of `AnnotatedExtraction`: it rides alongside the frozen Layer-A
grammar and carries the run’s metadata. It is scored *offline* against gold
([`ek.score()`](ek.html.md#ek.score) / [`ek.evaluate()`](ek.html.md#ek.evaluate)) and estimated *online* against a judge or a
consensus ([`ek.estimate_quality()`](ek.html.md#ek.estimate_quality)) – the same object, both halves.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Which task this episode answers (the k-trial grouping key).
  * **trajectory** ([`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory)) – The tool calls taken.
  * **output** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The agent’s final user-facing answer.
  * **final_state** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The end state of the world (the DB, the filesystem) – what a
    state-based oracle actually grades. Success is a *final-state* check, not a
    surface-text check.
  * **cost** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)]) – The [`Cost`](#ek.agents.Cost) consumed.
  * **success** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Filled in by a checker; `None` until graded.
  * **run** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance)]) – The [`RunProvenance`](#ek.agents.RunProvenance) (seed, model, simulator, suite version).
  * **provenance** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Provenance`](ek.base.html.md#ek.base.Provenance)]) – Optional core [`Provenance`](ek.base.html.md#ek.base.Provenance) for click-to-source audit.
  * **meta** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Anything else the bridge captured.

#### *property* calls *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pairs taken, in order.

#### graded(success)

A copy with `success` set (never mutates – grading stays idempotent).

`meta` is copied too: `dataclasses.replace` is a *shallow* copy, so without this
the “copy” would share the original’s mutable `meta` dict.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### *class* ek.agents.JudgeMetric(judge, criteria='')

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

### *class* ek.agents.JudgeSignal(judge, criteria='', cost_tier=5, name='judge')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reference-free, criteria-only LLM judge as a [`Signal`](ek.base.html.md#ek.base.Signal).

* **Parameters:**
  * **judge** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – An **injected** callable `(output, *, criteria) -> float`. `ek` core never
    imports an LLM client; you pass your own (or an `ek[judge]` adapter).
  * **criteria** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The rubric the judge scores against, bound at construction (exactly as
    [`LogprobSignal`](ek.qe.signals.html.md#ek.qe.signals.LogprobSignal) binds its aggregator), so the object still
    satisfies `Signal.__call__(output) -> float`.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `JUDGE_COST_TIER` – the most expensive tier.

#### finding(target, rationale)

Surface a judge’s *rationale* as a [`Finding`](ek.base.html.md#ek.base.Finding) (not just a scalar).

A judge’s reasoning is auditable evidence; a bare number throws it away.

* **Return type:**
  [`Finding`](ek.base.html.md#ek.base.Finding)

### *class* ek.agents.ModelPrice(input=0.0, output=0.0, cached_input=None, reasoning=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-token rates for one model, in **USD per token**.

* **Parameters:**
  * **input** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – USD per input (prompt) token.
  * **output** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – USD per output (completion) token.
  * **cached_input** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – USD per cached-input token; defaults to
    `DEFAULT_CACHE_DISCOUNT * input` when not declared.
  * **reasoning** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – USD per reasoning token; defaults to the `output` rate (reasoning
    tokens are billed as output by the major providers).

#### *property* cached_rate *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The cached-input rate (explicit, else the discounted input rate).

#### *property* reasoning_rate *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The reasoning-token rate (explicit, else the output rate).

### *class* ek.agents.ReliabilityReport(k=1, n_tasks=0, n_trials=0, n_success=0, success_rate=0.0, success_ci=(0.0, 1.0), pass_at_k=None, pass_hat_k=None, pass_hat_k_ci=(0.0, 1.0), stochastic=True, per_task=<factory>, per_slice=<factory>, cost=<factory>, detail=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Corpus-level reliability over a k-trial run, with per-task and per-slice cuts.

`success_rate` is micro (over all trials); `pass_at_k`/`pass_hat_k` are averaged
over *tasks* (each task’s own unbiased estimator), never over trials.

### *class* ek.agents.RunProvenance(seed=None, temperature=None, model='', simulator='', suite_version='', scaffold='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The *hidden eval variables* that decide whether two runs are even comparable.

Agent evaluation is stochastic, and several of its inputs are themselves models. In a
tau-bench-style suite the **user simulator is an LLM agent**, so its model/prompt/version
silently changes results run to run. Recording these is what lets the harness *refuse*
to compare a baseline against a run that changed the simulator or the suite – the same
discipline `ek`’s offline harness already applies when it refuses to compare across
two different metrics (`misc/docs/ek_08`, `misc/docs/ek_11`).

* **Parameters:**
  * **seed** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – RNG seed for the trial (`None` = unseeded/nondeterministic).
  * **temperature** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Sampling temperature of the agent under test.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The agent’s model id.
  * **simulator** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The **user-simulator** id/version – an eval variable, not a constant.
  * **suite_version** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Version of the task suite (contamination and task fixes move scores).
  * **scaffold** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The agent scaffold/harness id (the same model moves points on scaffold alone).

#### comparability_key()

The fields that must match for two runs to be comparable (seed excluded).

Seed and temperature are *expected* to vary across trials; the simulator, the suite
version and the scaffold are not – changing them invalidates a baseline.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### *class* ek.agents.Step(tool, args=<factory>, observation=None, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One step of a trajectory: a tool call and what came back.

`error` is set when the call itself failed (a bad argument, a tool exception) –
error *recovery* across subsequent steps is itself an evaluable signal.

#### *property* call *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pair – what tool-call correctness actually compares.

### *class* ek.agents.TaskSpec(task_id, input=None, gold=None, value=1.0, tools=(), slice=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One task in a suite: its input, its oracle, its value, and its allowed tools.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Stable identifier (the grouping key for k-trial reliability).
  * **input** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Whatever the agent under test is called with.
  * **gold** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The reference the checker grades against (a goal state, an expected answer,
    a gold trajectory – the checker decides how to read it).
  * **value** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Task-level cost weight: how much a completed task is worth. The task-level
    analog of `FieldSpec.importance`; feeds value-weighted cost accounting.
  * **tools** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`ToolSpec`](ek.agents.base.html.md#ek.agents.base.ToolSpec)]) – The tools this task permits (the Layer-A grammar for the task).
  * **slice** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional slice label (domain, difficulty, language) – per-slice cuts are
    mandatory, not optional (a low pass^k concentrated in one hard slice is a
    different problem from uniform flakiness).

#### grammar()

The Layer-A grammar for this task’s tool set.

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### *class* ek.agents.TaskSuccessMetric(check=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Did the episode complete the task? (`1.0`/`0.0`; aggregate = success rate).

* **Parameters:**
  **check** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A [`Checker`](#ek.agents.Checker) – a registered name (`"output"`, `"final_state"`,
  `"recorded"`) or any `(episode, gold) -> bool` callable. Defaults to
  `"output"`; **inject a state-based or executable oracle for a real suite.**

#### aggregate(scores)

The success **rate** over the corpus.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.agents.ToolCallMetric(level='call', , canonicalizer=None, grammar=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Tool-call correctness (BFCL AST match) as a cost-weighted [`Metric`](ek.base.html.md#ek.base.Metric).

* **Parameters:**
  * **level** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"call"` (default) scores whole calls – a call is a TP only on an exact AST
    match; `"arg"` gives partial credit at argument granularity.
  * **canonicalizer** – `str -> str` applied to string arguments before comparison
    (BFCL’s case/whitespace folding). Defaults to `ek`’s default canonicalizer.
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Layer-A grammar supplying per-tool and per-argument cost weights (may also be
    passed per call).

The `Score` carries `tp`/`fp`/`fn` in `detail` so [`ek.evaluate()`](ek.html.md#ek.evaluate) aggregates a
correct **micro**-averaged F1 over the corpus rather than averaging per-episode F1s.

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide) – never a mean.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.agents.ToolSpec(name, params=<factory>, importance=None, destructive=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One tool the agent may call: its argument schema and its error costs.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The tool/function name (the AST-match ground truth).
  * **params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec)]) – Argument name -> [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec). Each spec’s `importance`
    is the cost of getting *that argument* wrong, and its `domain` feeds
    validators – the same contract as an extracted field.
  * **importance** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Cost weight of the *call itself* (a missing or spurious call).
    Defaults to `DESTRUCTIVE_WEIGHT` when `destructive` is set, else 1.0.
  * **destructive** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether calling this tool mutates the world irreversibly (a refund,
    a delete, a send). Drives the default weight and the safety validators.

#### to_node_type()

Render this tool as a Layer-A [`NodeType`](ek.base.html.md#ek.base.NodeType) (args as fields).

* **Return type:**
  [`NodeType`](ek.base.html.md#ek.base.NodeType)

#### *property* weight *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The call-level cost weight (explicit, else destructive-aware default).

### *class* ek.agents.Trajectory(steps=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The ordered sequence of [`Step`](#ek.agents.Step) s an agent took.

A trajectory is *linear* – which is why it is scored with a sequence edit distance,
not a graph edit distance (see [`ek.agents.trajectory`](ek.agents.trajectory.html.md#module-ek.agents.trajectory)).

### Example

```pycon
>>> t = Trajectory([Step("search", {"q": "cat"}), Step("answer", {"a": "meow"})])
>>> len(t), t.tools
(2, ('search', 'answer'))
```

#### *property* calls *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pairs in call order.

#### *property* tools *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The tool names in call order.

### *class* ek.agents.TrajectoryMetric(scheme='in_order', , grammar=None, canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cost-weighted trajectory distance as a [`Metric`](ek.base.html.md#ek.base.Metric) (**lower is better**).

* **Parameters:**
  * **scheme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – One of `SCHEMES`.
    `in_order` (default) – Needleman-Wunsch edit distance; order matters, gaps allowed.
    `exact` – 0 iff the call sequences are identical, else the full mass.
    `any_order` – multiset distance; order ignored (the calls, not the path).
    `superset` – 0 iff every gold call occurs in the prediction; extra steps are free
    (use when the reference lists *required* calls rather than the whole path).
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Layer-A grammar supplying step/argument cost weights (may be passed per call).
  * **canonicalizer** – `str -> str` applied to string arguments before comparison.

#### aggregate(scores)

Corpus distance = total raw distance / total maximum distance (never a mean).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *exception* ek.agents.UnknownModelPrice

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

Raised when a model’s rates are not in the injected price catalog.

### ek.agents.agent_regression_gate(report, baseline, , tolerance=0.0, cost_tolerance=0.0, rootdir=None)

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
  [`AgentGateResult`](ek.agents.harness.html.md#ek.agents.harness.AgentGateResult)

### ek.agents.as_agent(fn)

Wrap a plain `input -> answer` function into the harness’s `TaskSpec -> Episode` shape.

Progressive disclosure: the trivial agent should not have to learn the Episode type.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### Example

```pycon
>>> from ek.agents.base import TaskSpec
>>> agent = as_agent(str.upper)
>>> agent(TaskSpec("t1", input="hi")).output
'HI'
```

### ek.agents.as_call(x)

Coerce a [`Step`](#ek.agents.Step), `(tool, args)` tuple, or `{"tool":…, "args":…}` dict.

Gold tool calls are usually written as bare pairs or dicts (no observation), while a
prediction arrives as a [`Step`](#ek.agents.Step); both must compare. Progressive disclosure: the
simple literal works, the rich object works.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> as_call(Step("search", {"q": "cat"}))
('search', {'q': 'cat'})
>>> as_call(("search", {"q": "cat"}))
('search', {'q': 'cat'})
>>> as_call({"tool": "search", "args": {"q": "cat"}})
('search', {'q': 'cat'})
>>> as_call("search")
('search', {})
```

### ek.agents.bootstrap_ci(values, statistic, , resamples=2000, alpha=0.05, seed=0)

Percentile bootstrap CI for any `statistic` over `values` (deterministic by seed).

Used for the **cost-per-success ratio**, whose distribution is skewed and heavy-tailed
(a normal interval on it is meaningless).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> lo, hi = bootstrap_ci([1.0] * 20, lambda v: sum(v) / len(v), resamples=50)
>>> lo == hi == 1.0
True
```

### ek.agents.cost_from_usage(usage, , latency_s=None)

Build a [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) from a provider `usage` object or dict.

Understands the OpenAI (`prompt_tokens` / `completion_tokens`) and Anthropic
(`input_tokens` / `output_tokens` / `cache_read_input_tokens`) spellings.

* **Return type:**
  [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)

### Example

```pycon
>>> c = cost_from_usage({"prompt_tokens": 100, "completion_tokens": 20})
>>> c.input_tokens, c.output_tokens
(100, 20)
```

### ek.agents.cost_of_pass(total_cost, n_success)

Expected monetary cost of one *successful* task: `total_cost / n_success`.

Returns `inf` when nothing succeeded – infeasibility is the honest answer, not a
cheap-looking zero (this divergence is the point of the metric).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Example

```pycon
>>> cost_of_pass(10.0, 4)
2.5
>>> cost_of_pass(10.0, 0)
inf
```

### ek.agents.cost_report(episodes, , prices=None, price=None)

The quality x cost x latency triple, reported **together**.

Scoring accuracy *without* cost lets an agent chase tiny gains with unbounded API calls, so
a report that omits cost is not a report. Returns `n`, `n_success`, `success_rate`,
`total_dollars`, `cost_per_success` (Cost-of-Pass), `mean_latency_s` and `total_tokens`.

With **no** `price`/`prices`, dollar figures are `None` rather than a fabricated zero:
`ek` will not invent rates, and a silent 0.0 would make a costly agent look free. Tokens
and latency are still reported.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Example

```pycon
>>> from ek.agents.base import Cost, Episode
>>> p = per_million(1.0, 1.0)
>>> eps = [Episode(task_id="a", cost=Cost(input_tokens=1_000_000), success=True),
...        Episode(task_id="b", cost=Cost(input_tokens=1_000_000), success=False)]
>>> r = cost_report(eps, price=p)
>>> r["success_rate"], round(r["cost_per_success"], 2)
(0.5, 2.0)
>>> cost_report(eps)["cost_per_success"] is None      # no rates -> no dollars invented
True
```

### ek.agents.difference_upper_bound(value_a, ci_a, value_b, ci_b, , z=1.96)

Upper bound of `a - b` for two statistics known only through their CIs.

Used for `pass^k` (a mean of per-task estimators, not a raw proportion), where a Newcombe
interval does not apply but the same principle does: test the **difference**, combining both
runs’ uncertainty, rather than asking whether two intervals happen to overlap.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### ek.agents.dollars(cost, price, , prices=None, batch=False)

Monetary cost of one [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) tally, in USD.

* **Parameters:**
  * **cost** ([`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)) – The token/retry tally.
  * **price** (`Union`[[`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A [`ModelPrice`](#ek.agents.ModelPrice), or a model name to resolve against `prices`.
  * **prices** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)]]) – The injected price catalog (required when `price` is a name).
  * **batch** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Apply the batch-API discount (`DEFAULT_BATCH_DISCOUNT`).
* **Raises:**
  [**UnknownModelPrice**](#ek.agents.UnknownModelPrice) – if a model name is given with no rates for it.
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.episode_dollars(episode, , prices=None, price=None, batch=False)

Monetary cost of one episode (0.0 if it carries no [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.episode_from_messages(messages, , task_id='', usage=None, output=None, final_state=None, latency_s=None)

Build a full [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode) from a transcript (+ optional usage/state).

`output` defaults to the last assistant text – the agent’s final answer.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### ek.agents.final_state_match(episode, gold)

Compare the episode’s **final state** with the goal state (the tau-bench oracle).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.from_deepeval_test_case(case, , task_id='')

Adapt a DeepEval `LLMTestCase`-shaped object into an [`Episode`](#ek.agents.Episode) (duck-typed).

Reads `.actual_output` and `.tools_called`. Note DeepEval phones home by default
(Confident-AI telemetry) – disable it before use if that matters to you; `ek` never
enables it.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### ek.agents.from_inspect_sample(sample, , task_id='')

Adapt an Inspect AI `EvalSample`-shaped object into an [`Episode`](#ek.agents.Episode) (duck-typed).

Reads `.messages`, `.output` and `.id` – no `inspect_ai` import required, so scoring
an Inspect log never drags the harness into `ek`’s dependency closure.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### ek.agents.is_episode(x)

Duck-typed episode check (core never imports this module – it checks the *shape*).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.judge_validation(judge_labels, human_labels, , level='nominal')

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

### ek.agents.load_agent_baseline(name, , rootdir=None)

Load a named agent baseline (or `None`).

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.agents.load_prices(source)

Parse a LiteLLM-format price catalog into a `{model: ModelPrice}` table.

`source` is a path to (or an already-parsed mapping of) LiteLLM’s MIT-licensed
`model_prices_and_context_window.json`. We read the **data file only** and never
import the `litellm` SDK – its `enterprise/` subtree is a proprietary carve-out,
and the SDK drags a large transitive tree in just to look up a rate.

Entries lacking token rates (embeddings, unpriced models) are skipped.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Example

```pycon
>>> table = load_prices({"m": {"input_cost_per_token": 1e-6,
...                            "output_cost_per_token": 2e-6}})
>>> table["m"].output
2e-06
```

### ek.agents.match_calls(pred, gold, , canon=None)

Assign predicted calls to gold calls – the layer `FieldMetric` does not provide.

Calls are grouped by tool name (a call to the *wrong tool* is never a match). Within a tool
name, candidate pairs are scored and taken **globally best-first** (exact-argument matches
before partial ones, more agreeing arguments before fewer), so identical repeated calls are
interchangeable and a reordered pair is still matched – while a strong pairing can never be
stolen by a weaker one that merely happened to come first in the prediction.

This is a deterministic greedy approximation to the optimal assignment. It is exact for the
common cases (distinct calls, repeats, reorderings); a full min-cost bipartite matching
(Hungarian) could differ only on pathological many-to-many partial-overlap sets, and is not
worth a dependency here.

Returns `(pairs, unmatched_pred, unmatched_gold)`, `pairs` being `(pred_call, gold_call)`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> pairs, up, ug = match_calls([("s", {"q": "b"}), ("s", {"q": "a"})],
...                             [("s", {"q": "a"}), ("s", {"q": "b"})])
>>> len(pairs), up, ug          # order-insensitive within a tool name
(2, [], [])
```

### ek.agents.newcombe_difference(successes_a, n_a, successes_b, n_b, , z=1.96)

Newcombe score interval for the **difference** of two proportions `p_a - p_b`.

This is the right instrument for “did the agent get worse than its baseline?”, and it is
strictly better than the tempting shortcut of *checking whether two 95% CIs overlap*.
Non-overlap is not a 5% test – it is roughly a 0.5% one, so it buys its low false-alarm
rate by going nearly blind: on a 60-task suite it cannot see a 15-point success-rate drop.
A difference interval keeps the same false-alarm rate and recovers most of the power,
because the variance of a difference is not the sum of two interval widths.

Newcombe (1998) method 10: build each proportion’s Wilson interval, then

> lower = (p_a - p_b) - sqrt((p_a - l_a)^2 + (u_b - p_b)^2)
> upper = (p_a - p_b) + sqrt((u_a - p_a)^2 + (p_b - l_b)^2)

Returns `(lower, upper)`. The difference is a *confident regression* when `upper < 0`.

### Example

```pycon
>>> lo, hi = newcombe_difference(50, 100, 90, 100)   # 50% now vs 90% before
>>> hi < 0                                            # confidently worse
True
>>> lo, hi = newcombe_difference(89, 100, 90, 100)   # 89% vs 90% -- noise
>>> hi > 0
True
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### ek.agents.output_match(episode, gold)

Canonicalized exact match on the final answer (the GAIA-style oracle).

The weakest useful oracle, and the default only because it needs nothing. Prefer a
state-based or executable checker whenever the task has one.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.pairwise_judge(judge, a, b, , criteria='', swap=True)

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

### ek.agents.pass_at_k(, n, c, k)

Unbiased `pass@k`: probability that **at least one** of k samples succeeds.

The HumanEval estimator `1 - C(n-c, k) / C(n, k)` over `n` trials with `c`
successes. This is the **capability** metric – right when a single success is enough
(offline candidate generation behind a verifier).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Example

```pycon
>>> round(pass_at_k(n=10, c=1, k=1), 3)
0.1
>>> pass_at_k(n=10, c=10, k=5)
1.0
```

### ek.agents.pass_hat_k(, n, c, k)

Unbiased `pass^k`: probability that **all k** independent trials succeed.

The tau-bench estimator `C(c, k) / C(n, k)`; for a per-trial success probability `p`
it decays to `p**k`. This is the **reliability** metric – the one that matters when
consistency *is* the product.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Example

```pycon
>>> pass_hat_k(n=8, c=8, k=8)
1.0
>>> pass_hat_k(n=8, c=4, k=2)
0.21428571428571427
```

### ek.agents.per_million(input, output, , cached_input=None, reasoning=None)

Build a [`ModelPrice`](#ek.agents.ModelPrice) from the human-facing **USD per 1M tokens** figures.

* **Return type:**
  [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)

### Example

```pycon
>>> p = per_million(3.0, 15.0)          # $3 in / $15 out per 1M tokens
>>> round(p.input, 9), round(p.output, 9)
(3e-06, 1.5e-05)
```

### ek.agents.price_of(model, , prices=None)

Resolve a model’s rates from an injected catalog, or fail *actionably*.

`ek` deliberately ships no built-in price table (rates go stale monthly and would be
magic numbers). Supply one with `prices=`, or load the LiteLLM catalog with
[`load_prices()`](#ek.agents.load_prices).

* **Return type:**
  [`ModelPrice`](ek.agents.cost.html.md#ek.agents.cost.ModelPrice)

### ek.agents.recorded_success(episode, gold)

Trust a `success` already recorded on the episode (e.g. by an external harness).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.reliability(source, , k=1, slices=None, cost=None, warn_deterministic=True)

Compute `pass@k` / `pass^k` / success-rate over a k-trial run.

* **Parameters:**
  * **source** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Graded [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode) s (grouped by `task_id`), or an
    already-grouped `{task_id: [bool, ...]}` mapping.
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The k of pass@k / pass^k. Tasks with fewer than k trials are skipped (and
    counted in `detail['skipped']`) rather than silently mis-estimated.
  * **slices** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional `{task_id: slice_label}` for per-slice cuts.
  * **cost** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – An optional cost report (from [`ek.agents.cost_report()`](#ek.agents.cost_report)) to carry along –
    reliability without cost is only half the picture.
  * **warn_deterministic** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Warn when **no** task shows mixed outcomes across trials, in
    which case `pass^k` degenerates to `pass^1` and is not informative.
* **Return type:**
  [`ReliabilityReport`](#ek.agents.ReliabilityReport)

### Example

```pycon
>>> r = reliability({"t1": [True, True, False], "t2": [True, True, True]}, k=2)
>>> r.n_tasks, r.k
(2, 2)
>>> round(r.pass_hat_k, 3)      # t1: C(2,2)/C(3,2)=1/3 ; t2: 1.0
0.667
```

### ek.agents.run_suite(agent, tasks, , k=1, check=None, metrics=None, price=None, prices=None, run=None, seed=None, persist=False, run_id=None, rootdir=None)

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
  [`ReliabilityReport`](#ek.agents.ReliabilityReport)
* **Returns:**
  A [`ReliabilityReport`](#ek.agents.ReliabilityReport) carrying `pass@k`, `pass^k` (with
  a bootstrap CI), the success rate (with a Wilson CI), the per-slice cuts, and the cost
  report – because reliability without cost is only half the answer.

### ek.agents.save_agent_baseline(report, name, , rootdir=None)

Freeze a run as a named baseline (including its provenance, so the gate can guard it).

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the run evaluated nothing. An empty baseline is worse than no baseline –
      it is a *permanent free pass*, because every bound in it is the maximally-uncertain
      default and no later run can be shown to fall below it.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.agents.suite_grammar(tasks)

The union grammar over a whole task suite (later tools win on name collision).

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### ek.agents.tool_grammar(\*tools)

Build the Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) from [`ToolSpec`](#ek.agents.ToolSpec) s.

The resulting grammar is the frozen cost SSOT every agent metric reads: node cost =
the tool’s call weight, field cost = the argument’s importance.

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### Example

```pycon
>>> from ek.base import FieldSpec
>>> g = tool_grammar(ToolSpec("send", params={"to": FieldSpec("to")},
...                           destructive=True))
>>> g.node_cost("send")
10.0
```

### ek.agents.trajectory_from_messages(messages)

Parse a chat transcript into a [`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory).

Each assistant tool call becomes a [`Step`](ek.agents.base.html.md#ek.agents.base.Step), and the *next* tool/result
message becomes that step’s `observation`. A tool message carrying an error marks the step
as errored – error *recovery* across later steps is itself an evaluable signal.

* **Return type:**
  [`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory)

### ek.agents.wilson_interval(successes, n, , z=1.96)

Wilson score interval for a proportion – correct in the tails, unlike Wald/CLT.

The naive normal (“Wald”) interval is badly wrong at small n and near 0/1 – exactly
where agent success rates live. Wilson is the cheap, dependency-free fix.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> lo, hi = wilson_interval(0, 10)
>>> lo == 0.0, round(hi, 3)
(True, 0.278)
```

### Modules

| [`base`](ek.agents.base.html.md#module-ek.agents.base)             | Agent-evaluation data shapes: the episode (Layer B) and the task/tool grammar (Layer A).                                                                                                                                                      |
|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`bridge`](ek.agents.bridge.html.md#module-ek.agents.bridge)         | One-way adapters: turn whatever your agent already emits into an `ek` [`Episode`](#ek.agents.Episode).                                                                                                               |
| [`cost`](ek.agents.cost.html.md#module-ek.agents.cost)             | The cost model and the price SSOT: dollars per successfully completed task.                                                                                                                                                                   |
| [`harness`](ek.agents.harness.html.md#module-ek.agents.harness)       | The agent task-suite harness: k trials, per-slice cuts, and a gate that survives noise.                                                                                                                                                       |
| [`judge`](ek.agents.judge.html.md#module-ek.agents.judge)           | LLM-as-judge: a reference-free [`Signal`](ek.base.html.md#ek.base.Signal), a reference-based [`Metric`](ek.base.html.md#ek.base.Metric), and the validation that must precede either. |
| [`metrics`](ek.agents.metrics.html.md#module-ek.agents.metrics)       | Task-success and cost-per-successful-task: the two metrics that drive the whole thing.                                                                                                                                                        |
| [`toolcalls`](ek.agents.toolcalls.html.md#module-ek.agents.toolcalls)   | Tool-call correctness: BFCL-style AST matching over a *multiset* of calls.                                                                                                                                                                    |
| [`trajectory`](ek.agents.trajectory.html.md#module-ek.agents.trajectory) | Trajectory (process) evaluation: a cost-weighted **sequence** edit distance over steps.                                                                                                                                                       |
