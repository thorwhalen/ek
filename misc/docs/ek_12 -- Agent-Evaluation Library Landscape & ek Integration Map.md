# ek_12 — Agent-Evaluation Library Landscape & ek Integration Map

**Author:** Thor Whalen

**Status:** Architecture / integration brief — the *implementation* report of ek's agent-evaluation extension. It is the agent-domain analogue of -> ek_06 (the IE library landscape + reuse/wrap/BUILD allocation + license register). Where -> ek_07 states the concept, -> ek_08 defines outcome-based success metrics, and the reference-free judge/confidence machinery lives in -> ek_09 (backed by the IE-side -> ek_03), *this* report maps every agent-eval capability to a verified, licensed library, then gives the concrete `ek/agents/` subpackage design. This is the report you read when you are about to write code.

---

## TL;DR

A permissive agent-evaluation stack exists end-to-end **today**, and it is unusually clean on licensing: the harness layer (Inspect AI, MIT [14]), the metrics layer (DeepEval, Apache-2.0 [17]; langchain-ai/agentevals, MIT [19]; Ragas, Apache-2.0 [38]), the self-consistency layer (uqlm, Apache-2.0 [20]), and the price catalog (LiteLLM's root data file, MIT [22][23]) are all OSI-permissive and actively maintained. So ek should **WRAP** two or three of these behind an `ek[agents]` extra rather than reinventing them; **BUILD** only the handful of things nobody ships as an importable, cost-aware, cost-weighted primitive — the *cost-per-successfully-completed-task* aggregator, the unbiased `pass@k`/`pass^k` estimators, a tool-call AST/F1 metric, a trajectory typed-graph metric (which is just ek's existing cost-weighted `TypedGraphMetric` -> ek_02 lifted onto a `Trajectory`), and the judge-validation helper (which is just the harness IAA/Krippendorff -> ek_02 pointed at judge-vs-human agreement); and **IMPORT-ONLY / quarantine** the license landmines and SaaS-only platforms (Arize Phoenix under Elastic-2.0 [29]; LangSmith and Braintrust as proprietary hosted platforms whose only pip-installable pieces — autoevals, MIT [34] — are the parts you may touch). The deliverable is a new `ek/agents/` subpackage that mirrors `ek/ocr/` exactly: it builds ON `ek.base` (Layer-A task/tool grammar carrying value/cost weights; Layer-B episode metadata), registers agent metrics and signals in the strategy registry, and adapts external harnesses through a one-way `bridge.py` whose dependency direction is a hard rule — **ek → inspect_ai / deepeval / ragas, never the reverse** (the ek→ocracy dependency-direction rule: ek depends on ocracy via an extra, never the reverse; the agent bridge follows the same rule). The industry has already validated the WRAP-behind-adapters architecture: MLflow's `mlflow.genai.evaluate` natively wraps five third-party eval libraries exposing 60+ metrics [39]. The one thing every serious 2025–2026 source agrees ek must NOT skip is cost: a benchmark that scores accuracy without cost lets an agent chase tiny gains with unbounded API calls (HAL/Princeton [36]), which is precisely why ek promotes cost to a first-class metric axis rather than an afterthought.

---

## 1. The landscape, as a decision table

The agent-eval ecosystem splits along two orthogonal axes that a naive survey conflates. First, **eval-library vs observability-platform**: an eval-library is something you `import` and call in-process as a dependency (DeepEval, agentevals, Ragas, uqlm); an observability-platform ingests traces over HTTP and renders them in a UI (LangSmith, Braintrust, Phoenix, Langfuse, Weave). The architectural consequence for ek is decisive: *eval-libraries* go behind `ek[agents]` as wrapped strategies; *platforms* go behind an optional `ek[agents-obs]` tier and are spoken to over HTTP as sinks, never linked as core dependencies. Second, **what is evaluated**: a task-suite runner (does the agent finish the job?), a RAG evaluator (is the answer grounded?), a trajectory evaluator (did it take the right steps?), or a judge (is this free-text output good?). ek needs all four seams, and the table below is the roster verified against live PyPI/GitHub in mid-2026.

| Library / Platform | Evaluates | License + package | Shape | CI-gate story | Cost-aware? | Verdict for ek |
|---|---|---|---|---|---|---|
| **Inspect AI** (UK AISI) | task/agent/trajectory | **MIT** [14][16], `inspect-ai` (0.3.245, Py≥3.10) | `Task = Dataset+Solver+Scorer`; Solvers span one `generate()` → full tool-using ReAct/Deep agent; entry-point plugins register Scorers/Tools/Sandboxes [15] | CLI + Python; Docker/K8s sandboxes | partial (usage logged) | **WRAP** — primary task-suite runner |
| **DeepEval** (Confident AI) | task/RAG/agent/judge | **Apache-2.0** [17], `deepeval` | pytest-native; 30+ LLM-judge metrics incl. `ToolCorrectnessMetric`, `TaskCompletion`, `GEval` [18] | pytest → CI native | no | **WRAP** — metrics behind extra |
| **langchain-ai/agentevals** | trajectory | **MIT** [19], `agentevals` | `create_trajectory_match_evaluator` (strict / unordered / **graph** modes) + LangGraph thread extraction | library | no | **WRAP or design-cue** — trajectory match |
| **Ragas** | RAG | **Apache-2.0** [38], `ragas` | 7 metrics (faithfulness, answer/context relevancy…) [7] | library | no | **WRAP** — RAG-agent slice |
| **uqlm** (CVS Health) | judge/self-consistency | **Apache-2.0** [20][21], `uqlm` | black-box sample-and-vote scorers → [0,1] confidence | library | no | **WRAP** — self-consistency Signal |
| **LiteLLM** (BerriAI) | (price catalog) | **MIT** core; `enterprise/` carved out [22] | root `model_prices_and_context_window.json` data file [23] | data file | **yes (the SSOT)** | **VENDOR the JSON**, don't import the SDK |
| **promptfoo** | task/judge | **MIT** [24] (still MIT after OpenAI's Mar-2026 acquisition) | single `promptfooconfig.yaml`; deterministic asserts incl. `latency`, `cost` | CLI harness | **yes** | design-cue (declarative YAML), not a Python metric lib |
| **OpenAI Evals** | task | **MIT** framework [25]; registry data separately licensed | registry + framework | CLI | no | reference; **scan bundled data** |
| **lm-evaluation-harness** (EleutherAI) | task (few-shot) | **MIT** [26], `lm-eval` | task-centric runner | CLI | no | secondary reference (not agent-trajectory) |
| **HELM** (Stanford CRFM) | holistic multi-metric | **Apache-2.0** [27], `crfm-helm` | heavyweight framework | CLI | no | reference only (too heavy) |
| **DSPy** (stanfordnlp) | metrics/optimizers | **MIT** [28], `dspy` | programmatic metrics + optimizers | library | no | design-cue for programmatic metrics |
| **autoevals** (Braintrust) | judge/RAG | **MIT** [34], `autoevals` | standalone model-graded scorers | library | no | optional WRAP; the OSS half of a SaaS |
| **W&B Weave** | obs + eval | **Apache-2.0** [33], `weave` | tracing + eval SDK | HTTP + lib | usage traced | `ek[agents-obs]` (permissive) |
| **TruLens** (now Snowflake) | obs + eval | **MIT** [32], `trulens` | feedback-function eval | HTTP + lib | no | `ek[agents-obs]` (permissive) |
| **MLflow GenAI** (LF) | harness wrapping 5 libs | **Apache-2.0** [39], `mlflow` | `mlflow.genai.evaluate` wraps Ragas/DeepEval/Phoenix/TruLens/Guardrails | HTTP + lib | usage traced | **architecture precedent**, not a dep |
| **Arize Phoenix** | observability | **Elastic-2.0** [29] — NOT OSI | trace UI | HTTP | traced | **QUARANTINE** — import-only/HTTP-only |
| **Langfuse** | observability | **MIT core + EE** [31] | trace UI; EE features gated by key | HTTP | traced | `ek[agents-obs]`, pin MIT pkgs only |
| **LangSmith** (LangChain) | observability SaaS | **proprietary** | hosted | HTTP only | traced | HTTP sink only, never a dep |
| **Braintrust** | eval SaaS | **proprietary** core | hosted | HTTP only | traced | HTTP sink; use `autoevals` (MIT) instead |

Two observations shape the whole design. First, **Inspect AI is the standout WRAP candidate for the runner**: it is MIT [16], from the UK AI Security Institute (which uses it for nearly all its automated evals, and which Anthropic and DeepMind have adopted), its `Dataset → Task → Solver → Scorer` spine cleanly separates the *task suite* from the *agent under test* from the *grading*, and — crucially for ek's open-closed discipline — third-party Python packages register custom Scorers, Tools, Sandboxes, Approvers and Hooks via setuptools entry points with no manual wiring [15]. AISI itself ships add-on packages (`inspect_evals`, `inspect-swe`, `inspect-scout`), which is direct precedent for ek extending Inspect behind an extra. (Note the "200+ prebuilt evals" ship in the companion `inspect_evals` package, not the core — itself an example of the add-on pattern.) Second, **DeepEval already ships the two agent metrics ek would otherwise sweat over**: `ToolCorrectnessMetric` (with `should_exact_match` and `should_consider_ordering` flags, comparing `tools_called` against `expected_tools`) and `TaskCompletion` [18]. That does not make ek's tool-call metric redundant — ek needs an AST/F1 metric whose costs come from Layer-A importance weights, which DeepEval does not model — but it means the metrics extra should *offer* DeepEval's versions as registered strategies for teams who want a batteries-included default.

## 2. What the benchmarks tell us to build (not borrow)

The library table answers "what can I import?"; the benchmark literature answers "what must the imports compute?" Four load-bearing facts from -> ek_07 / -> ek_08, restated here only insofar as they constrain the *library* choice:

**Cost-per-successful-task is the objective, and it has a published economic framework.** "Cost-of-Pass" [35] formalizes the expected monetary cost to obtain a correct answer as `inference_cost / success_probability` in the retry model — combining per-query cost with success probability and attempt count. This is the theoretical grounding for ek's `CostPerSuccess` aggregator, and it settles a design question decisively: the aggregation is a **cost-weighted ratio, never a naive mean of per-episode costs**. HAL/Princeton reinforces the operational corollary: benchmarks that score accuracy without cost let agents "chase tiny gains with unbounded API calls," so leaderboards must compute the accuracy/cost Pareto frontier and favor cost-effective agents [36]. No permissive library ships this aggregator — it is ek's flagship BUILD.

**Two success estimators, both unbiased and combinatorial, neither shipped as an importable metric.** `pass@k` (Chen et al. 2021, HumanEval/Codex [1]) is the *capability* metric — at least one of k samples succeeds — with unbiased estimator `pass@k = E_task[1 − C(n−c, k)/C(n, k)]` over n samples per task with c correct, numerically stabilized by returning `1.0` when `n−c < k`. `pass^k` (tau-bench, Yao et al. 2024 [2]) is the *reliability* metric — *all* k independent trials of the same task succeed — with estimator `pass^k = E_task[C(c, k)/C(n, k)]`, decaying to `p^k`. The reliability gap these expose is the entire motivation for the cost-per-success framing: on tau-bench, GPT-4o achieves <50% task success averaged across the two domains (~61% pass^1 on retail, ~35% on airline), and reliability collapses to pass^8 < 25% on retail — a ~60% relative drop from pass^1 [2]. A 2026 reliability-science literature now formalizes this beyond pass@1 with Reliability Decay Curves and Meltdown Onset Points, finding Graceful Degradation Scores drop from 0.90 to 0.44 on software-engineering tasks across the duration range (vs 0.74→0.71 on document processing) [13]. ek must BUILD both estimators; the HuggingFace `evaluate` `code_eval` reference implementation of the pass@k stabilization is a fine template, but ek owns the code so the license is clean.

**Tool-call correctness is AST-matching, and BFCL is the reference design.** The Berkeley Function Calling Leaderboard (Patil et al., ICML 2025 [3]) parses the emitted call into an abstract syntax tree and — *without executing it* — verifies the function name matches, all required params are present, each argument's type and value are correct (case-insensitive string comparison with whitespace/punctuation normalization), and no params outside the doc were hallucinated; arguments are matched by name, so argument order is ignored (though element order *within* list values matters). This is exactly the algorithm `ek/agents/metrics.py:tool_call` should implement, scaled to thousands of functions without execution.

**Success is a final-state check, not a surface-text or tool-syntax check.** tau-bench grades by comparing the final database state against an annotated goal state plus required user-facing outputs [2]; SWE-bench grades by whether hidden unit tests flip to green after applying the model's patch [8]; GAIA grades by quasi-exact-match on a single answer [9]; WebArena/AgentBench grade functional task success in interactive environments [10][11]. AgentBoard adds a *progress-rate* metric capturing partial advancement through multi-turn tasks, relevant to a graded (non-binary) `trajectory_match` [12]. The consequence for ek's library allocation: the *checker* must be pluggable (a `Validator` -> ek_04 in ek terms), because different task suites carry different oracles, and the harness must not assume string match.

## 3. License register for agent-eval dependencies

This mirrors -> ek_06's register one level up. The good news, verified against primary LICENSE files, is that the **default set is entirely permissive** — there is no copyleft in the critical path, so the wrap-behind-extras plan does not collapse on licensing grounds.

| Dependency | License (verified) | Tier | Notes / trap |
|---|---|---|---|
| Inspect AI | MIT [14] | `ek[agents]` default | "Copyright (c) 2024 UK AI Security Institute" |
| DeepEval | Apache-2.0 [17] | `ek[agents]` default | hosted Confident AI SaaS is separate, not a code dep |
| agentevals | MIT [19] | `ek[agents]` default | — |
| Ragas | Apache-2.0 [38] | `ek[agents]` default | canonical repo is `explodinggradients/ragas`; a `vibrantlabsai` mirror exists — verify canonical before pinning |
| uqlm | Apache-2.0 [20] | `ek[agents]` default | PyPI `license_expression: Apache-2.0` |
| LiteLLM (data file only) | MIT [22] | vendor JSON | `enterprise/` subtree is proprietary — a **hard import boundary**; the root JSON is outside it and MIT-clean; retain the MIT notice when vendoring |
| autoevals | MIT [34] | optional | the OSS half of proprietary Braintrust |
| OpenAI Evals | MIT framework [25] | reference | **registry data may be separately licensed — scan bundled data, not just the package** |
| lm-eval-harness | MIT [26] | reference | — |
| HELM / crfm-helm | Apache-2.0 [27] | reference | heavyweight |
| DSPy | MIT [28] | reference | — |
| TruLens | MIT [32] | `ek[agents-obs]` | — |
| W&B Weave | Apache-2.0 [33] | `ek[agents-obs]` | — |
| MLflow | Apache-2.0 [39] | reference/optional | architecture precedent |
| promptfoo | MIT [24] | design-cue | CLI, not a Python metric API |
| **Arize Phoenix** | **Elastic-2.0** [29] | **QUARANTINE** | source-available, NOT OSI; forbids offering as a managed service |
| Langfuse | **MIT core + EE** [31] | `ek[agents-obs]`, pin MIT | EE (RBAC, audit logs, retention) needs a key |
| LangSmith | **proprietary SaaS** | HTTP sink only | never a dep |
| Braintrust | **proprietary** core | HTTP sink only | use `autoevals` (MIT) instead |

Two register nuances matter for the CI license gate, and they correct a common misconception. **Arize Phoenix is Elastic License 2.0 — source-available, not OSI-approved — with the signature clause forbidding you from offering the software "to third parties as a hosted or managed service"** [29]. It must stay out of ek's permissive core. But contrary to the intuition that ELv2 hides in a repo file, Phoenix's PyPI metadata explicitly sets the SPDX License field to `Elastic-2.0`, so a scanner reading the `License`/`License-Expression` field *will* catch it — the one caveat is that Phoenix ships **no** `License ::` trove classifier, so a naive gate that keys only off classifiers would miss it. **Configure the license gate to read the License field, not just classifiers.** The genuinely scanner-invisible traps are the ones flagged in -> ek_06's register (`zss` BSD-3-in-a-file, the `mistralai` namespace package, PubTabNet CDLA), not Phoenix. The real legal driver for quarantining Phoenix is not that ELv2 forbids library use (it permits use, copy, distribution and derivative works) — it is that (a) bundling a non-OSI dependency as a hard default pollutes ek's permissive-core story, and (b) any downstream offering ek as a hosted service could inherit the managed-service restriction. Hence: import-only / HTTP-only / opt-in extra, never a default. The gate should fail the build on GPL/AGPL/SSPL/non-commercial/Elastic in the `core..agents` closure, exactly as -> ek_06 specifies for `core..hitl`.

## 4. reuse / wrap / BUILD allocation for `ek.agents`

The discipline is the same as -> ek_06: **reuse > wrap > build**, and build only the connective tissue no library owns. Concretely:

**WRAP behind `ek[agents]` (permissive default extra):**
- **Inspect AI** as the task-suite runner — its `Dataset/Solver/Scorer` maps onto ek's harness, and its entry-point plugin system means ek's own Scorers can register into Inspect (and vice-versa) without either project depending on the other's internals.
- **DeepEval** as a metrics provider — offer `ToolCorrectnessMetric`, `TaskCompletion`, and `GEval` as registered ek `Metric`/`Signal` strategies for teams wanting a batteries-included judge, so ek's hand-built metrics are the *cost-aware* option, not the *only* option.
- **Ragas** as the RAG-faithfulness provider for the RAG-agent slice (faithfulness = fraction of answer statements supported by retrieved context [7]); cross-link the underlying reference-free QE machinery to -> ek_03 rather than re-deriving.
- **uqlm** as the self-consistency Signal provider — its black-box sample-and-vote scorers return [0,1] confidence, mapping directly onto `estimate_quality()`'s `Signal → calibrate → decide` cascade [20][21].
- **agentevals** as the trajectory-match provider (or design-cue) — its strict/unordered/**graph** modes are the reference for mapping trajectory equality onto ek's `TypedGraphMetric` [19].

**VENDOR (not import):** LiteLLM's root `model_prices_and_context_window.json` [23] — a standalone MIT data file carrying `input_cost_per_token`, `output_cost_per_token`, cache/image/computer-use costs, context windows, and capability flags for 2,500+ models across 100+ providers. ek parses it with any JSON reader, retaining the MIT notice, and *never imports the litellm SDK* — sidestepping the `enterprise/` proprietary carve-out entirely. The JSON supplies only the price catalog; "cost-per-successful-task" is ek's own composition on top (catalog rates × observed token usage ÷ ek-defined success counts).

**BUILD (the connective tissue nobody ships as a cost-aware, importable primitive):**
- The **`CostPerSuccess` aggregator** — the `evaluate()`-injected corpus aggregator implementing Cost-of-Pass [35] as a cost-weighted ratio.
- **`pass_at_k` and `pass_hat_k`** — the two unbiased combinatorial estimators [1][2], as registered `Metric`s.
- **`tool_call`** — an AST/F1 metric mirroring BFCL [3], whose per-field costs come from Layer-A `FieldSpec.importance`.
- **`trajectory_match`** — ek's existing cost-weighted `TypedGraphMetric` (-> ek_02) lifted onto a `Trajectory`; the tool-call-graph distance *is* the typed-graph distance, so ek reuses the engine, not rebuilds it.
- **The LLM-judge `Signal` wrapper + judge-validation helper** — a thin G-Eval-style chain-of-thought judge [4] wrapped as a `Signal`, plus a validation helper that reuses the harness IAA primitives (Krippendorff's α, Cohen's κ -> ek_02) to check judge-vs-human agreement, reproducing the MT-Bench methodology (GPT-4 judge ~85% human agreement, above the ~81% human-human agreement [5]).

**IMPORT-ONLY / quarantine:** anything copyleft/non-commercial/SaaS-only — Phoenix (ELv2), LangSmith and Braintrust (proprietary), Langfuse EE modules. These live behind `ek[agents-obs]` (for the permissive-licensed observability SDKs: Weave, TruLens, MIT-core Langfuse) or are spoken to over HTTP as optional sinks; Phoenix is never a default dependency.

The BUILD list is short *by design*, and it is short because ek reuses two engines it already owns: the cost-weighted typed-graph metric (-> ek_02) and the ROVER-style agreement seam (-> ek_03). Self-consistency (Wang et al. 2022 [6]) is sample-and-vote over final answers — the *exact* generalization of ROVER's N-way align-and-vote — so the self-consistency Signal reuses ek's ROVER machinery rather than rebuilding it; uqlm is the WRAP that provides it out of the box.

## 5. The concrete `ek.agents` instance design

`ek/agents/` mirrors `ek/ocr/` one-for-one: it builds ON `ek.base`, registers strategies in `ek/registry.py`, guards optional deps with `@requires_extra("agents")`, and adapts external harnesses through a one-way `bridge.py`. The dependency direction is the ek→ocracy dependency-direction rule made literal (ek depends on ocracy via an extra, never the reverse; the agent bridge follows the same rule): **ek → inspect_ai / deepeval / ragas via `ek[agents]`, never the reverse.**

### 5.1 Data shapes (built on `ek.base`)

**Layer A becomes a task/tool grammar.** `TaskSpec` and `ToolSpec` are the frozen typed-schema SSOT for a task suite: `ToolSpec` carries the allowed tools, their argument schemas (the AST-match ground truth), and — critically — per-tool and per-argument **value/cost weights** riding on `FieldSpec.importance`. This is what makes ek's `tool_call` metric *cost-sensitive*: calling a destructive tool with a wrong argument is not one unit of error, it is `importance`-weighted error, and the error-cost of an unnecessary API call feeds `CostPerSuccess`. `TaskSpec` carries the task-value weight (how much a completed task is worth) and the reference oracle (final-state goal, or a pluggable `Validator`). The grammar is frozen and never mutated by a run — exactly Layer A's contract.

**Layer B becomes an episode's metadata.** `Episode` (alias: the evaluated object) holds a `Trajectory` — an ordered sequence of `Step`s, each a `(tool_call, observation)` pair with `raw_signals`, `confidence`, `findings`, and `provenance` keyed by `NodePath`, exactly as `AnnotatedExtraction` does for an extracted payload. Alongside it rides `Cost` (tokens in/out, priced amount, retries, latency) and the final `decision`/`verdict`. This is the direct lift of ek's two-layer split: the `Trajectory` is scored offline against a gold trajectory (`score`/`evaluate`) and estimated online against consensus/judge (`estimate_quality`), sharing the same Layer-A grammar object.

These shapes are added to `ek.base` where they are general (a `Trajectory` is a typed graph; a `Cost` is domain-neutral) and to `ek/agents/` where they are agent-specific (`ToolSpec`, `Episode`).

### 5.2 `ek/agents/metrics.py`

Five registered `Metric` strategies, each `__call__(pred, gold, *, grammar=None) -> float | Score`:
- `task_success` — the final-state check; delegates to the `TaskSpec` oracle / `Validator` (-> ek_04).
- `pass_at_k` — `E_task[1 − C(n−c,k)/C(n,k)]` [1], the corpus capability metric.
- `pass_hat_k` — `E_task[C(c,k)/C(n,k)]` [2], the corpus reliability metric; this and `pass_at_k` are the aggregators `evaluate()` injects, **never a naive mean** — the same rule -> ek_08 states.
- `tool_call` — BFCL-style AST match + F1 [3], costs from Layer-A importance; can also delegate to DeepEval's `ToolCorrectnessMetric` [18] when the `deepeval` extra is present.
- `trajectory_match` — ek's cost-weighted `TypedGraphMetric` over the `Trajectory` (-> ek_02); supports strict / unordered / graph modes mirroring agentevals [19], and a graded progress-rate variant per AgentBoard [12].

### 5.3 `ek/agents/judge.py`

The LLM-judge `Signal` — a G-Eval-style chain-of-thought + form-filling judge [4] wrapped behind the `Signal` Protocol so it plugs into `estimate_quality()`'s cascade. Its output is a raw signal that must be *calibrated* before it decides anything (judge scores are not probabilities) — the calibration and the accept/flag/escalate decision are ek_03's machinery (-> ek_03, -> ek_09), reused verbatim; the cascade's confidence gate **is** the decide stage, and escalation **is** selective prediction. The `judge_validation` helper reuses the harness IAA primitives (Krippendorff's α / Cohen's κ -> ek_02) to certify a judge against human labels before it is trusted, reproducing MT-Bench's judge-vs-human agreement protocol [5]. This is non-optional because judge biases have measured, load-bearing effect sizes: position bias swings winrate 10–15 points (mitigate by swapping order and averaging), verbosity bias inflates preference for longer outputs by 15–30 points, self-preference runs 10–25% [40][5]. G-Eval itself reports only Spearman 0.514 with humans on summarization while flagging self-preference bias toward LLM-generated text [4] — so a judge Signal that is not validated and calibrated is a liability, not a metric.

### 5.4 `ek/agents/cost.py`

The cost model + price SSOT + `CostPerSuccess` aggregator. It vendors LiteLLM's `model_prices_and_context_window.json` [23] as the price catalog (retaining the MIT notice), computes per-episode `Cost` from observed token usage × catalog rates, and aggregates via Cost-of-Pass [35] — `expected_cost = inference_cost / success_probability` — into a cost-per-successfully-completed-task figure. Because HAL shows accuracy-without-cost is gameable [36], `evaluate()` reports the accuracy/cost pair (the Pareto point), never accuracy alone.

### 5.5 `ek/agents/harness.py`

The task-suite runner, paralleling `ek/harness.py`. `evaluate_agent_store` runs an agent callable over a `TaskSpec` store, per slice, producing `pass^k` + cost + latency per slice; `save_baseline` / `regression_gate` provide the golden-set CI gate (a reliability *and cost* regression fails the build); the IAA helpers (`krippendorff_alpha`, `cohen_kappa`, `percent_agreement`) serve judge validation. It must support **contamination-resistant task sets** — the same model moves 5–15 points on SWE-bench Verified depending on scaffolding [36], and OpenAI stopped evaluating models on SWE-bench Verified (Feb 2026), citing both ~59.4% flawed/over-strict test cases and frontier-model training-data contamination, and now recommends SWE-bench Pro (Scale); so `TaskSpec` pins the scaffold and the harness supports encrypted/held-out tasks (HAL encrypts all traces to prevent scraping-based contamination [36]; the methodology reference is "Establishing Best Practices for Building Rigorous Agentic Benchmarks" [37]).

### 5.6 `ek/agents/bridge.py`

The one-way adapter. It converts (a) a plain agent callable, (b) an Inspect AI `Task`/eval log, or (c) a DeepEval test case into an ek `Episode`. Dependency direction is the hard rule: `ek → inspect_ai / deepeval / ragas` via `ek[agents]`, **never reverse** — Inspect is the natural `bridge.py` target because its entry-point plugin system lets ek register Scorers *into* Inspect while ek imports Inspect, not the other way. MLflow's `mlflow.genai.evaluate` — which natively wraps Ragas, DeepEval, Phoenix, TruLens and Guardrails behind adapters exposing 60+ metrics [39] — is the industry proof that WRAP-behind-adapters, not BUILD-everything, is the correct architecture.

### 5.7 `pyproject` extras and gates

- `ek[agents]` — the permissive default extra: `inspect-ai`, `deepeval`, `ragas`, `uqlm`, `agentevals` (all MIT/Apache-2.0). The vendored LiteLLM JSON is a package data file, not a dependency.
- `ek[judge]` — optional heavier judge backends (e.g. a specific provider SDK) if a team wants a different judge than the default.
- `ek[agents-obs]` — the observability tier: `weave` (Apache-2.0), `trulens` (MIT), MIT-core `langfuse`. Phoenix is documented here as HTTP-only, never installed by default.
- `check_requirements` probes each extra and, on `ImportError`, raises the actionable `pip install ek[agents]` message via `@requires_extra("agents")`.
- The **CI license gate** extends -> ek_06's gate to the `core..agents` closure: fail on GPL/AGPL/SSPL/non-commercial/Elastic, reading the `License`/`License-Expression` field (not just trove classifiers, so Phoenix and any classifier-less ELv2 package are caught) [29].

## 6. The full capability → component → library integration table

| Capability | ek component | Facade / registry seam | Library allocation |
|---|---|---|---|
| Task-suite runner | `agents/harness.py` | `evaluate_agent_store`, `regression_gate` | **WRAP** Inspect AI (MIT [14]) |
| Task-success (final state) | `metrics.task_success` | `Metric` + `Validator` (-> ek_04) | **BUILD** oracle dispatch; task-specific checkers |
| Capability metric | `metrics.pass_at_k` | `Metric` aggregator (`evaluate`) | **BUILD** unbiased estimator [1] |
| Reliability metric | `metrics.pass_hat_k` | `Metric` aggregator (`evaluate`) | **BUILD** unbiased estimator [2] |
| Tool-call correctness | `metrics.tool_call` | `Metric`, costs from Layer-A | **BUILD** BFCL AST/F1 [3]; optional DeepEval `ToolCorrectness` [18] |
| Trajectory match | `metrics.trajectory_match` | `TypedGraphMetric` (-> ek_02) | **REUSE** ek engine; design-cue agentevals [19] |
| RAG faithfulness | `metrics` (RAG slice) | `Metric` / `Signal` (-> ek_03) | **WRAP** Ragas (Apache-2.0 [7][38]) |
| Self-consistency signal | `judge` / qe seam | `Signal` (-> ek_03), ROVER reuse | **WRAP** uqlm (Apache-2.0 [20]); **REUSE** ROVER seam |
| LLM-as-judge | `judge.py` | `Signal` → calibrate → decide (-> ek_03/ek_09) | **BUILD** G-Eval wrapper [4]; optional DeepEval `GEval` [18] |
| Judge validation | `judge.judge_validation` | harness IAA (-> ek_02) | **REUSE** Krippendorff/Cohen [5] |
| Confidence → escalation | `estimate_quality` cascade | decide = selective prediction (-> ek_03) | **REUSE** ek_03 calibration/conformal |
| Cost model + price SSOT | `cost.py` | `Cost`, price catalog | **VENDOR** LiteLLM JSON (MIT [23]); **BUILD** cost model |
| Cost-per-successful-task | `cost.CostPerSuccess` | `evaluate` aggregator (cost-weighted ratio) | **BUILD** per Cost-of-Pass [35] |
| Production drift/monitoring | (-> ek_05) | harness monitoring | **REUSE** ek_05 (CBPE, audit sampling) |
| External harness adapter | `bridge.py` | one-way ek → inspect/deepeval | **BUILD** adapters; architecture per MLflow [39] |
| Observability sink | `ek[agents-obs]` | HTTP sinks | **WRAP** Weave/TruLens (perm. [32][33]); Phoenix HTTP-only [29] |

## 7. Mapping to ek (the load-bearing bridge, restated)

Agent evaluation is the **same 2×2 as IE evaluation** — (reference availability) × (granularity) — but the evaluated object is an **episode/trajectory** (tool-calls + observations ending in a final state) and the objective is **cost per successfully completed task**, not per-token cost. Concretely, onto ek's spine:

- **Layer A (`GraphGrammar`) becomes the task/tool grammar** — `TaskSpec`/`ToolSpec` carrying allowed tools, arg schemas, and task-value/error-cost weights on `FieldSpec.importance`. Frozen SSOT; feeds constrained decoders/validators and cost-sensitive metrics exactly as in IE.
- **Layer B (`AnnotatedExtraction`) becomes the episode metadata** — the `Trajectory`, per-step `raw_signals`/`confidence`/`findings`/`provenance` keyed by `NodePath`, plus `Cost` and `verdict`, riding *alongside* the grammar, never mutating it.
- **`score()`/`evaluate()` get agent metrics** — `task_success`, `pass_at_k`, `pass_hat_k`, `tool_call`, `trajectory_match`; `evaluate()` aggregates via the metric's *own* aggregator (`pass^k`, `CostPerSuccess`), never a naive mean, with `per_slice` cuts.
- **`estimate_quality()` gets agent signals** — LLM-as-judge, self-consistency/agreement, RAG faithfulness; the cascade's confidence gate **is** the decide stage and escalation **is** selective prediction (-> ek_03).
- **The cost-weighted typed-graph metric generalizes to trajectory/tool-call-graph distance** — ek reuses `TypedGraphMetric`, not rebuilds it.
- **ROVER agreement generalizes to self-consistency** (sample-and-vote [6]) — reuse the ROVER seam; uqlm is the WRAP.
- **The strategy registry + DI + `@requires_extra` pattern is unchanged** — `Metric`, `Signal`, `Calibrator`, `DecisionPolicy`, `Validator` Protocols resolved by name and injected keyword-only with smart defaults; `ek[agents]` is a permissive extra, `ek[agents-obs]` the heavier tier.
- **Persistence and the license gate are unchanged** — dol JSON stores under `~/.local/share/ek/`, and the CI license gate extends to `core..agents`.

## 8. Contested evidence, flagged honestly

Three caveats a senior reviewer should hold. **First, LLM-judge validity is genuinely contested**: G-Eval's 0.514 Spearman [4] is a weak correlation, and position/verbosity/self-preference biases have double-digit effect sizes [40][5] — which is why the judge Signal is quarantined behind mandatory validation + calibration, not trusted raw. **Second, `pass^k` has a sampling caveat**: it presumes the *same* task is run n ≥ k times with genuine stochasticity between runs (in tau-bench that comes from the LM-simulated user + agent [2]); on tasks with no run-to-run variance, `pass^k` degenerates to `pass^1` and tells you nothing new — the harness must ensure real stochasticity or report the caveat. **Third, benchmark contamination and scaffold sensitivity are pervasive**: SWE-bench Verified moves 5–15 points on scaffolding alone [36], and OpenAI stopped evaluating models on it (Feb 2026), citing both ~59.4% flawed/over-strict test cases and training-data contamination (now recommending SWE-bench Pro); tau-bench itself had a Few-Shot agent invalidated by data leakage [36]. The design responses — pin the scaffold in `TaskSpec`, support encrypted/held-out task sets, always report the accuracy/cost Pareto point — are not polish; they are what separates a trustworthy agent-eval harness from a leaderboard-gaming one.

---

## References

[1] Chen M, Tworek J, Jun H, et al. Evaluating Large Language Models Trained on Code. arXiv, 2021. <https://arxiv.org/abs/2107.03374>

[2] Yao S, Shinn N, Razavi P, Narasimhan K. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, ICLR 2025. <https://arxiv.org/abs/2406.12045>

[3] Patil SG, Mao R, Yan F, et al. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML (PMLR v267), 2025. [proceedings.mlr.press/v267/patil25a.html](https://proceedings.mlr.press/v267/patil25a.html)

[4] Liu Y, Iter D, Xu Y, Wang S, Xu R, Zhu C. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. EMNLP 2023. arXiv:2303.16634. <https://arxiv.org/abs/2303.16634>

[5] Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS Datasets & Benchmarks 2023. arXiv:2306.05685. <https://arxiv.org/abs/2306.05685>

[6] Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. arXiv:2203.11171. <https://arxiv.org/abs/2203.11171>

[7] Es S, James J, Espinosa-Anke L, Schockaert S. Ragas: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217, 2023. <https://arxiv.org/abs/2309.15217>

[8] Jimenez CE, Yang J, Wettig A, et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770. <https://arxiv.org/abs/2310.06770>

[9] Mialon G, Fourrier C, Swift C, Wolf T, LeCun Y, Scialom T. GAIA: A Benchmark for General AI Assistants. ICLR 2024. arXiv:2311.12983. <https://arxiv.org/abs/2311.12983>

[10] Zhou S, Xu FF, Zhu H, et al. WebArena: A Realistic Web Environment for Building Autonomous Agents. ICLR 2024. arXiv:2307.13854. <https://arxiv.org/abs/2307.13854>

[11] Liu X, Yu H, Zhang H, et al. AgentBench: Evaluating LLMs as Agents. ICLR 2024. arXiv:2308.03688. <https://arxiv.org/abs/2308.03688>

[12] Ma C, et al. AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. NeurIPS Datasets & Benchmarks 2024. [proceedings.neurips.cc PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/877b40688e330a0e2a3fc24084208dfa-Paper-Datasets_and_Benchmarks_Track.pdf)

[13] Khanal A, Tao Y, Zhou J. Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents. arXiv:2603.29231, 2026. <https://arxiv.org/abs/2603.29231>

[14] UK AI Security Institute. Inspect AI: A framework for large language model evaluations (MIT). GitHub, 2026. <https://github.com/UKGovernmentBEIS/inspect_ai>

[15] UK AISI. Inspect AI documentation (extensions, scorers). 2026. <https://inspect.aisi.org.uk/>

[16] inspect-ai 0.3.245 (MIT, Py≥3.10). PyPI, 2026. <https://pypi.org/project/inspect-ai/>

[17] Confident AI. DeepEval: The LLM Evaluation Framework (Apache-2.0). GitHub, 2026. <https://github.com/confident-ai/deepeval>

[18] DeepEval Docs. Tool Correctness Metric. 2026. <https://deepeval.com/docs/metrics-tool-correctness>

[19] LangChain. agentevals: Readymade evaluators for agent trajectories (MIT). GitHub, 2026. <https://github.com/langchain-ai/agentevals>

[20] CVS Health. uqlm: Uncertainty Quantification for Language Models (Apache-2.0). GitHub, 2026. <https://github.com/cvs-health/uqlm>

[21] Bouchard D, et al. UQLM: A Python Package for Uncertainty Quantification in Large Language Models. JMLR/TMLR; arXiv:2507.06196, 2025. <https://arxiv.org/abs/2507.06196>

[22] BerriAI. LiteLLM LICENSE (MIT core + enterprise/ carve-out). GitHub, 2026. <https://github.com/BerriAI/litellm/blob/main/LICENSE>

[23] BerriAI. model_prices_and_context_window.json (MIT). GitHub, 2026. <https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json>

[24] promptfoo. Test your prompts, agents, and RAGs (MIT). GitHub, 2026. <https://github.com/promptfoo/promptfoo>

[25] OpenAI. Evals: a framework for evaluating LLMs (MIT). GitHub, 2026. <https://github.com/openai/evals>

[26] EleutherAI. lm-evaluation-harness LICENSE (MIT). GitHub, 2026. <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE.md>

[27] Stanford CRFM. crfm-helm (Apache-2.0). PyPI, 2026. <https://pypi.org/project/crfm-helm/>

[28] stanfordnlp. DSPy (MIT). GitHub, 2026. <https://github.com/stanfordnlp/dspy>

[29] Arize AI. Phoenix (Elastic License 2.0). GitHub/PyPI, 2026. <https://github.com/Arize-ai/phoenix/blob/main/LICENSE>

[30] Arize. Phoenix self-hosting license (ELv2 managed-service restriction). 2026. <https://arize.com/docs/phoenix/self-hosting/license>

[31] Langfuse. Open-source strategy / license (MIT core + EE). 2026. <https://langfuse.com/docs/open-source>

[32] TruEra/Snowflake. TruLens LICENSE (MIT). GitHub, 2026. <https://github.com/truera/trulens/blob/main/LICENSE>

[33] Weights & Biases. Weave LICENSE (Apache-2.0). GitHub, 2026. <https://github.com/wandb/weave/blob/master/LICENSE>

[34] Braintrust. autoevals: model-graded evaluation (MIT). GitHub, 2026. <https://github.com/braintrustdata/autoevals>

[35] Erol MH, El B, Suzgun M, Yuksekgonul M, Zou J. Cost-of-Pass: An Economic Framework for Evaluating Language Models. arXiv:2504.13359, 2025. <https://arxiv.org/abs/2504.13359>

[36] Kapoor S, et al. Holistic Agent Leaderboard (HAL): The Missing Infrastructure for AI Agent Evaluation. arXiv:2510.11977; Princeton, 2025. <https://arxiv.org/pdf/2510.11977>

[37] Establishing Best Practices for Building Rigorous Agentic Benchmarks. arXiv:2507.02825, 2025. <https://arxiv.org/pdf/2507.02825>

[38] Ragas: Evaluation framework for RAG and LLM applications (Apache-2.0). PyPI, 2026. <https://pypi.org/project/ragas/>

[39] MLflow. Top 5 LLM and Agent Observability Tools in 2026 (mlflow.genai.evaluate wraps 5 eval libs; Apache-2.0). 2026. <https://mlflow.org/top-5-agent-observability-tools/>

[40] Zhao W, et al. Self-Preference Bias in LLM-as-a-Judge. arXiv:2410.21819, 2024. <https://arxiv.org/html/2410.21819v1>

[41] Sierra Research. tau2-bench (successor benchmark: domains + voice). GitHub, 2026. <https://github.com/sierra-research/tau2-bench>
