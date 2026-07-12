---
name: ek-dev-agents
description: "How to build and extend ek's AGENT/ASSISTANT evaluation instance (ek/agents/) — the second concrete instance after ek.ocr. Covers the episode object (Episode/Trajectory/Step/Cost/RunProvenance as Layer B) and the task/tool grammar (ToolSpec/TaskSpec building a GraphGrammar as Layer A, where FieldSpec.importance is the cost of a wrong tool argument), the agent metrics (TaskSuccessMetric with an injected Checker oracle, CostPerSuccessMetric implementing Cost-of-Pass E[cost]/P(success), ToolCallMetric doing BFCL AST match over a multiset of calls, TrajectoryMetric doing a Needleman-Wunsch sequence edit distance), the reliability estimators (pass@k capability vs pass^k reliability — pure functions, NOT Metrics), the LLM-as-judge seams (JudgeSignal reference-free vs JudgeMetric reference-based vs pairwise_judge, plus judge_validation reusing Krippendorff/Cohen IAA), the agent harness (run_suite with k trials, and the variance-aware agent_regression_gate using Wilson/bootstrap intervals), and the duck-typed bridges (trajectory_from_messages, from_inspect_sample). Use when implementing, debugging, or extending anything under ek/agents/, adding an agent metric/signal/checker, scoring an episode or trajectory or tool call, computing pass^k or cost per successful task, evaluating an LLM judge, or wiring an external agent harness (Inspect AI, DeepEval, Ragas). Read BEFORE touching ek/agents/."
metadata:
  audience: developers
---

Authoritative spec for **ek's agent & assistant evaluation instance** (`ek/agents/`). You are building `ek` itself, not using it. Match the names and decisions here and in `misc/docs/ek_12 -- Agent-Evaluation Library Landscape & ek Integration Map.md` (the architecture report — **read it before writing agent code**). The concepts are mapped in `ek_07`; the details live in `ek_08` (task success), `ek_09` (judge/QE), `ek_10` (trajectory), `ek_11` (cost economics).

## The one-sentence thesis

Agent evaluation is the **same 2×2** `ek` already implements — (reference availability) × (granularity) — but the evaluated object is an **episode** (tool calls + observations ending in a final state) and the objective is **cost per successfully completed task**, not cost per token.

`ek.agents` mirrors `ek.ocr` exactly: it builds *on* `ek.base`, registers strategies in `ek.registry`, and adapts external harnesses through a one-way bridge. **It adds zero dependencies** — everything below is pure-python, and the bridges duck-type, so `ek` can score an Inspect/DeepEval run without importing either.

## The two layers, for agents

**Layer A — the task/tool grammar (the cost SSOT).** Do *not* invent a parallel schema. `ToolSpec` and `TaskSpec` are **builders** over the existing `GraphGrammar`:

- a `ToolSpec` renders to a `NodeType` whose `fields` are the tool's argument `FieldSpec`s;
- `FieldSpec.importance` = **the cost of getting that argument wrong**;
- `ToolSpec.destructive=True` defaults the call weight to `DESTRUCTIVE_WEIGHT` (10.0).

This is what makes ek's tool-call and trajectory metrics *cost-sensitive* — a wrong argument to a refund tool is not one unit of error. No off-the-shelf agent metric models this; it is the reason to build rather than borrow.

**Layer B — the episode.** `Episode(task_id, trajectory, output, final_state, cost, success, run, ...)`, where `Trajectory` is a sequence of `Step(tool, args, observation, error)`, `Cost` tallies tokens **by kind** (they are priced asymmetrically), and `RunProvenance` records the hidden eval variables.

**These shapes live in `ek/agents/base.py`, NOT `ek/base.py`.** `ek/base.py` is the zero-dependency, IE-schema-only SSOT; the OCR instance sets the precedent that instance shapes stay out of core and core duck-types them (`is_episode()`).

## Four rules that are easy to get wrong

These were all caught in an adversarial design review. **Do not "simplify" them back.**

### 1. `pass^k` is NOT a `Metric`

`Metric.__call__(pred, gold)` scores one pair and `evaluate()` aggregates independent cases — there is **no grouping key and no k-trials axis** anywhere in that flow. `pass@k`/`pass^k` are *cross-task* quantities over k trials of the **same** task. They ship as **pure functions** in `ek/agents/reliability.py`, and the **harness** owns the trial grouping, returning a `ReliabilityReport`.

- `pass_at_k(n, c, k) = 1 - C(n-c, k)/C(n, k)` — **capability** ("can it ever?"). HumanEval estimator.
- `pass_hat_k(n, c, k) = C(c, k)/C(n, k)` — **reliability** ("does it every time?"), decays to `p**k`. **This is the production number.**

Guard the degenerate case: `pass^k` presumes genuine run-to-run stochasticity. On a deterministic agent it collapses to `pass^1` and measures nothing — `reliability()` **warns** rather than reporting a falsely reassuring number.

### 2. `TrajectoryMetric` must NOT use the graph-edit-distance engine

Tempting (a trajectory *is* a typed graph) and wrong. `networkx.graph_edit_distance` — which powers the flagship `TypedGraphMetric` — is an **isomorphism search**:

- it **ignores step order** (node substitution matches on type+fields, ignoring identity) — the exact error a trajectory metric exists to catch;
- it raises above `DEFAULT_MAX_NODES = 60`;
- it is **timeout-nondeterministic** (returns a best-so-far bound);
- its denominator is polluted by synthetic ordering edges.

So **reuse the cost model, not the engine**: an O(n·m) **Needleman–Wunsch** sequence edit distance with per-step costs read from the grammar. Order-honoring, exact, deterministic, unbounded. Schemes (they *disagree* — choose deliberately): `in_order` (default), `exact`, `any_order`, `superset`.

A genuine DAG mode would justify the graph engine — but `Step` carries no dependency edges, so it is deliberately **not offered** rather than shipped as a mode that quietly ignores order.

### 3. Tool calls need a multiset matcher before any counting

`FieldMetric` compares records **keyed by field name** — the key *is* the alignment. Tool calls are a **keyless multiset**: two `search(q=…)` calls collide onto one key, and reordering/repeating silently miscounts. So:

1. `match_calls()` assigns predicted → gold calls (grouped by tool name; exact-argument matches claimed first so an approximate match cannot steal them);
2. **then** reuse `FieldMetric`'s TP/FP/FN accounting and micro-F1 aggregation.

The BFCL algorithm: name matches, expected args present with right values (case-insensitive, whitespace-folded), no hallucinated args; args matched **by name** (order irrelevant), element order **within a list value** relevant. Note the default canonicalizer here is `default_canonicalizer()`, unlike the IE metrics where `None` means "compare raw".

### 4. Only a reference-free, criteria-only judge is a `Signal`

- `JudgeSignal` — reference-free, binds `criteria` at construction, `__call__(output) -> float`. `cost_tier=5` (the most expensive family — escalate to it last). The `judge` is an **injected callable**; `ek` never imports an LLM SDK.
- A judge that needs gold is a **`Metric`** (`JudgeMetric`), on the `score()` side.
- **Pairwise** judging needs two outputs → `pairwise_judge()`, which **swaps the order and averages** (position bias is a double-digit effect, not a rounding error).
- **Hard Rule 1 applies**: a raw judge score is not a probability. `estimate_quality()` refuses to gate it uncalibrated.
- **`judge_validation()` is not optional** — it reuses the harness IAA (`krippendorff_alpha`, `cohen_kappa`) to certify a judge against human labels. An unvalidated judge is a liability, not a metric.

**Known facade limitation (documented, not silently relied upon):** `facade._base_confidence` means *all* raw signals and calibrates the single mean. Mixing a judge score with a logprob under one calibrator is not sound. Run the judge as the only signal, or calibrate it separately.

## Cost: the other half of the objective

`ek/agents/cost.py`. **Cost-of-Pass** = `E[cost] / P(success)` → `inf` when nothing succeeds (a model that never succeeds is not cheap, it is *unusable*).

- **No hardcoded prices.** Rates go stale monthly; they would be magic numbers. A `ModelPrice` is passed in or resolved from an injected catalog. `load_prices()` parses **LiteLLM's MIT price JSON — the data file, never the SDK** (its `enterprise/` subtree is a proprietary carve-out).
- With no rates supplied, dollar figures are `None`, **never a fabricated 0.0** (a silent zero makes a costly agent look free).
- `CostPerSuccessMetric` **must stamp `detail['higher_is_better'] = False`** — `cost_per_success` is not in `harness._LOWER_IS_BETTER`, so without the stamp the regression gate inverts direction and waves a cost regression straight through. There is a test pinning this.

## The harness and the variance-aware gate

Agent metrics are **stochastic random variables**; the core `regression_gate` is a scalar point comparison and is **unsound** for them (it flags noise and misses real regressions inside the tolerance band).

- `run_suite(agent, tasks, *, k, check, metrics, price, run, seed)` — k trials/task, injected checker, per-slice, persisted. `metrics=` injects the **suite grammar** so the Layer-A cost weights actually reach a tool-call/trajectory metric.
- `agent_regression_gate` — **test the DIFFERENCE, with both runs' uncertainty folded in.** There are three ways to do this and only one is right; the first two are bugs we have already shipped and fixed:
  1. ❌ **Our interval vs the baseline's point.** The baseline is *itself a noisy estimate*. A Wilson upper bound only reaches 1.0 when every trial passed, so one flake against a lucky baseline reads as a "confident regression".
  2. ❌ **Do the two 95% CIs overlap?** Seductive and wrong: non-overlap is **not a 5% test, it is roughly a 0.5% one**. It buys a calm gate by going blind — a 15-point drop on a 60-task suite becomes invisible. *A gate that cannot see a real regression is as broken as one that cries wolf.*
  3. ✅ **A two-sample interval on `current - baseline`.** `newcombe_difference()` for the success rate (needs the raw counts — which is why `_summary` persists `n_success`), and `difference_upper_bound()` for `pass^k` (a mean of per-task estimators, so Newcombe does not apply; combine both runs' SEs instead). Regression iff the upper bound of the change is below `-tolerance`. Measured: ~2x the power of (2), at a full-gate false-alarm rate of ~1–4% (under the 5% budget).
  An overlapping result with a lower point estimate is **not a clean pass** — it is an underpowered experiment, and the gate warns.
- **Empty runs and empty baselines are both fatal.** With no trials every bound is the maximally-uncertain default, so *nothing can be shown to fall below it*: an empty run would pass any gate, and an empty **baseline** is a permanent free pass for every future run (guarding only the current run just moves the hole upstream). `save_agent_baseline` refuses to freeze a zero-trial run; the gate rejects both.
- **`RunProvenance` guards comparability.** A tau-bench-style *user simulator is itself an LLM* — a hidden eval variable. The gate **refuses** (raises) to compare runs whose `simulator`, `suite_version`, or `scaffold` changed, exactly as the offline gate refuses to compare two different metrics.
- **Verifier isolation is a security property**: the checker must run where the agent's trajectory cannot reach it (a red-team produced a ~10-line `conftest.py` that pytest auto-loads and rewrote every test outcome to "passed", scoring near-perfect without solving a task).

## Checkers (the injected success oracle)

Success is a **final-state** check, not a surface-text check. `Checker = (episode, gold) -> bool`, resolved from the `checkers` registry namespace. Built-ins: `output` (canonicalized exact match — the weakest, and the default only because it needs nothing), `final_state`, `recorded`. **Real suites inject their own** (a DB-state comparison, a hidden-test run, a programmatic predicate) — `ek` never owns an LLM or a sandbox.

## Adding something new

- **A metric** → implement the `Metric` Protocol, stamp `detail['higher_is_better']`, give it an `aggregate()` that is a **ratio of sums, never a mean of ratios**, and `register("metrics", name, instance)` in `ek/agents/metrics.py::_register_defaults`.
- **A checker** → `@register("checkers", "<name>")` on an `(episode, gold) -> bool`.
- **A signal** → follow `ek-dev-add-signal`; set `cost_tier` honestly (cheap signals run first).
- **A bridge** → **duck-type it** (`trajectory_from_messages`, `from_inspect_sample`). The dependency direction is the `ek → ocracy` rule restated: **`ek → inspect_ai / deepeval / ragas`, never the reverse.**
- **A dependency** → consult `ek-dev-licensing` FIRST. Quarantined: **Arize Phoenix (Elastic-2.0** — source-available, not OSI; declares ELv2 in its License *field* but ships **no trove classifier**, so a classifier-only scanner misses it), LangSmith/Braintrust (proprietary SaaS — HTTP sinks only). DeepEval is permissive but **phones home**; disable telemetry.

## Where things are

| File | Role |
|---|---|
| `ek/agents/base.py` | Episode/Trajectory/Step/Cost/RunProvenance (Layer B) + ToolSpec/TaskSpec grammar builders (Layer A) |
| `ek/agents/reliability.py` | `pass_at_k`, `pass_hat_k`, `ReliabilityReport`, Wilson + bootstrap intervals |
| `ek/agents/metrics.py` | `TaskSuccessMetric`, `CostPerSuccessMetric`, the checkers, the registry wiring |
| `ek/agents/toolcalls.py` | `match_calls` (the multiset matcher) + `ToolCallMetric` (BFCL AST/F1) |
| `ek/agents/trajectory.py` | `TrajectoryMetric` (Needleman–Wunsch; the four schemes) |
| `ek/agents/cost.py` | `ModelPrice`, `dollars`, `cost_of_pass`, `cost_report`, `load_prices` |
| `ek/agents/judge.py` | `JudgeSignal`, `JudgeMetric`, `pairwise_judge`, `judge_validation` |
| `ek/agents/harness.py` | `run_suite`, `save_agent_baseline`, `agent_regression_gate` |
| `ek/agents/bridge.py` | duck-typed adapters (messages / Inspect / DeepEval → `Episode`) |
