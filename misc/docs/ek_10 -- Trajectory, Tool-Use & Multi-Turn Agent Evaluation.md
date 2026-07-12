# Trajectory, Tool-Use & Multi-Turn Agent Evaluation

**Author:** Thor Whalen

**Status:** Draft — synthesis report ek_10 in the ek agent-evaluation corpus. Extends the ek Knowledge-Evaluation framework from information-extraction outputs to AI agents/assistants.

## TL;DR

Outcome grading answers *did the agent get the right final answer*. It is necessary and it is not enough. Two agents can reach the same final state — one in three grounded tool-calls, one after eleven flailing retries with a hallucinated tool, a redundant write, and a lucky recovery — and outcome grading scores them identically. The second agent is a production incident waiting to happen: higher token cost, higher latency, higher blast radius, lower reliability under resampling. **Process evaluation scores the *sequence* — the tool-calls, arguments, ordering, and decisions the agent made to get there.** The load-bearing observation for ek is that a trajectory is a *typed sequence that is also a graph*: steps are nodes carrying a tool type and an argument record, ordering and data-dependencies are edges, and the value of getting a step right (or the cost of getting it wrong) is a weight. That is *exactly* ek's Layer-A typed-graph-grammar territory (`ek/base.py`), and a `TrajectoryMetric` is not a new machine — it is the cost-weighted `TypedGraphMetric` (-> ek_02) with steps-as-nodes, plus a sequence-edit-distance mode that reuses the same global TP/FP/FN accumulation logic that CER/WER already use (-> ek_02). Reference-based trajectory matching has *the same partial-match plurality* as slot-matching did in ek_02 — exact / in-order / any-order / subset / superset schemes that **disagree by design** — and the practitioner libraries (LangChain `agentevals`, DeepEval) already ship them. Reference-free process signals (LLM-as-judge over the trajectory, step-level process reward models, self-consistency) flow through `estimate_quality()` and inherit ek_03's calibration and selective-prediction discipline wholesale. Tool-use *findings* — hallucinated tool, ungrounded argument, redundant call, missing precondition — are `Validator`s (-> ek_04). And the whole thing must be reported on a **cost-vs-accuracy Pareto curve, not a leaderboard number** [15], which is precisely ek's cost-per-successfully-completed-task objective made operational.

---

## 1. Why process, and why now

The ek thesis for agents (established in the corpus preamble and ek_07–ek_09) is that agent evaluation is the *same* 2×2 as IE evaluation — (reference availability) × (granularity) — but the evaluated object is an **episode/trajectory**: a sequence of tool-calls and observations ending in a final state, and the objective is **cost per successfully completed task**, not per-token cost. Reports ek_08/ek_09 own the *outcome* side (task-success, pass@k/pass^k, final-state reward). This report, ek_10, owns the *process* side: everything about *how* the agent got there.

The field has converged on this split independently. The anchor secondary source — the *Survey on Evaluation of LLM-based Agents* (arXiv:2503.16416) [1] — organizes the entire landscape around core agentic capabilities (planning, tool use, memory, self-reflection), notes that "reference-based methods … compare trajectories against expected optimal paths, with platforms such as LangSmith, Vertex AI, and AgentEvals supporting various alignment modes," and — critically for ek — flags the field's open gaps as **cost-efficiency, safety, robustness, and fine-grained scalable methods**. Those four gaps are the ek value proposition restated by an outside party. Process evaluation is where three of the four (cost-efficiency, robustness, fine-grained) live.

The reason process matters *operationally* and not just academically: **AgentBoard** (NeurIPS 2024 Oral, arXiv:2401.13178) [2] showed that binary success rate throws away the signal you most need for debugging. It introduced the first **fine-grained progress-rate metric** that tracks *how far along the sub-goals* an agent got, rather than only whether it finished — the process-granularity analogue of ek_02's partial-match decomposition, where an extraction gets partial credit for the fields it did get right. An agent that reaches 90% of the sub-goals on a task and one that reaches 10% both score 0 on success rate; progress rate separates "almost there, one bug" from "lost from step one." For a QE/monitoring system, that difference is the whole game.

---

## 2. Trajectory match schemes — and why they disagree

The first thing to internalize is that reference-based trajectory matching is **not one metric**. It is a *family* of schemes that give different scores to the same (predicted, reference) trajectory pair, and choosing among them is a modeling decision, not a default. This is structurally identical to ek_02's finding that slot/entity matching has exact-vs-partial-vs-type-only schemes that disagree — the same disagreement, one level up, over sequences of tool-calls instead of sets of spans.

The canonical, pip-installable reference implementation is LangChain's **`agentevals`** (MIT-licensed; `pip install agentevals`) [3][4]. Its `create_trajectory_match_evaluator(trajectory_match_mode=...)` factory offers exactly four reference-based modes, and the library also ships helper scorers (`exact_match_scorer`, `in_order_match_scorer`, `any_order_match_scorer`) that expose the same logic. A trajectory here is a list of OpenAI-style messages — `HumanMessage` / `AIMessage(tool_calls=[{id, name, args}])` / `ToolMessage(tool_call_id)`. The evaluator returns a **deterministic boolean** `{key, score: bool, comment}` and — note this for the cost story below — **does not measure tokens or cost** [4][7]. The five practitioner schemes, unified:

| Scheme | `agentevals` mode | Passes when… | ek analogue |
|---|---|---|---|
| **Exact** | `strict` | same messages, same order, same tool calls (content compared too, not just tool names) [8] | exact-match slot scoring (-> ek_02) |
| **In-order (subsequence)** | (helper `in_order_match_scorer`) | reference actions appear in order, extra actions allowed [5] | ordered edit-distance with insertions free |
| **Any-order (set)** | `unordered` | same tool-call set, order irrelevant | bag-of-slots F1 (-> ek_02) |
| **Subset** | `subset` | agent calls *only* tools from reference (no extras) | precision-flavored: penalize over-calling |
| **Superset** | `superset` | agent calls *at least* the reference tools (extras allowed) | recall-flavored: "did it hit the key steps" |

Orthogonal to the step-matching mode is **argument strictness**: `tool_args_match_mode` ∈ {`exact` (default), `ignore`, `subset`, `superset`}, with per-tool `tool_args_match_overrides` for custom comparators [3][4]. This is the two-dimensional strictness lattice ek needs to expose: *which steps must match* (the mode) × *how precisely their arguments must match* (the arg mode). A financial-transaction agent wants `strict` + `exact`; a research agent that may legitimately reorder web searches wants `unordered` + `subset` on the query argument.

Two schemes matter especially. **Subset** is the one that catches *unnecessary/redundant tool-calls* — it fails the moment the agent calls a tool not in the reference, which is how you penalize the eleven-retry agent from the TL;DR. **Superset** is the one you reach for when "I only care that a few key tools were called and I'm fine with extra scaffolding" — the LangSmith docs describe it in exactly those words [5]. The disagreement between subset and superset on the *same* trajectory is the whole point: subset punishes over-calling, superset forgives it. There is no universally correct choice; there is only a choice appropriate to the task's cost structure, which is why ek must make it a *parameter driven by Layer-A weights*, not a hardcoded default.

`agentevals` also ships a **graph** representation for LangGraph agents — `GraphTrajectory {inputs, results, steps: list[list[str]]}` with `graph_trajectory_strict_match()` — plus an **LLM-as-judge** variant (`create_trajectory_llm_as_judge`, `create_graph_trajectory_llm_as_judge`) with prebuilt `TRAJECTORY_ACCURACY_PROMPT` / `_WITH_REFERENCE` prompts and `continuous=True` for 0–1 scores [3]. That deterministic-graph-metric-vs-LLM-judge split is *the same split ek already has*: the reference-based `TypedGraphMetric` on the `score()`/`evaluate()` side versus the judge `Signal` on the `estimate_quality()` side. `agentevals` validates the architecture; ek generalizes it (cost weights, calibration, registry).

**DeepEval** (Apache-2.0; `deepeval` 4.x) [9] gives the deterministic complement. Its `ToolCorrectnessMetric` is **non-LLM in its core path** and reference-based: it compares `tools_called` against `expected_tools`. A verification correction is load-bearing here — DeepEval's own docs prose says the score is "(correctly used tools) / (total tools called)," but the **source code divides by `len(expected_tools)`**, making it a *recall-flavored proportional ratio, not precision and not F1* [10]. Strictness escalates via `ToolCallParams` (name-only by default; `INPUT_PARAMETERS` also checks args; `OUTPUT` also checks outputs), plus `should_consider_ordering` (a weighted-LCS over the expected order) and `should_exact_match` (1.0/0.0). An LLM is invoked *only* when `available_tools` is supplied, and then the final score is `min(deterministic_score, LLM_tool_selection_score)` [10]. DeepEval's companion `TaskCompletionMetric` is the opposite kind of thing — LLM-as-judge over the agent's *full trace* — and belongs to the outcome side (-> ek_08). Multi-turn/conversational versions of these are still an open feature request in DeepEval (issue #2223) [11], which is a small opening for ek to be ahead.

The design lesson ek takes: **the deterministic proportional scorers (agentevals modes, DeepEval ToolCorrectness) are `Metric`s; the LLM-judge trajectory scorers are `Signal`s.** Same object, two facades.

---

## 3. Process supervision vs outcome supervision

The deepest theoretical grounding for scoring the process comes from **"Let's Verify Step by Step"** (Lightman et al., OpenAI, arXiv:2305.20050; ICLR 2024) [12]. The result: **process supervision — feedback at each intermediate reasoning step — significantly outperforms outcome supervision — feedback only on the final answer — on MATH.** The process-supervised reward model (PRM) reaches **78.2%** on a representative subset of the MATH test set under Best-of-N selection (state this as a *reward-model / Best-of-N* result, not the base model solving 78% unaided) [12]. The training signal is **PRM800K**: ~800,000 step-level human correctness labels over ~75K solutions to ~12K problems, released **MIT-licensed** at `openai/prm800k` and therefore permissively reusable [12][13].

Two things ek inherits. First, the *concept*: a step-level verdict per trajectory step is exactly a per-node `finding` in ek's Layer-B `AnnotatedExtraction`. A PRM emits, per step, a correctness probability — that is a per-node confidence signal keyed by `NodePath`, riding alongside the trajectory grammar, which is the literal shape of Layer B. Second, the *reusable asset*: PRM800K is a ready-made, permissively-licensed dataset of step-level labels that ek can use to calibrate and validate any step-level process signal it builds — the same way ek_02's harness uses gold corpora for IAA. When your reference *is* step-level ("this step was correct / this step introduced the error"), you get a process reward model; when it is answer-level, you get outcome supervision. ek supports both because the granularity axis of the 2×2 is exactly that choice.

The caveat, stated honestly: process supervision beat outcome supervision *in a math-reasoning setting with clean step boundaries*. Tool-use trajectories have messier step boundaries (is a retry a new step or a continuation?), and the generalization of "process > outcome" to open-ended agents is not established at the same confidence. ek's stance is not "process supervision is always better" — it is "process supervision is a *distinct granularity* the framework must express," and the empirical which-is-better question is per-domain, answered by ek's own `evaluate()` with per-slice cuts.

---

## 4. Tool-use quality beyond binary correctness

"Did it call the right tool" is the floor. The quality dimensions that separate a robust agent from a fragile one are:

- **Tool selection** — right tool for the sub-goal (DeepEval `ToolCorrectnessMetric` name-level; agentevals `subset`/`superset`) [4][10].
- **Argument grounding** — arguments faithful to the conversation/observations, not hallucinated (DeepEval `INPUT_PARAMETERS`; agentevals `tool_args_match_mode`; DeepEval's LLM-based referenceless `ArgumentCorrectnessMetric`) [9][10].
- **Redundant/unnecessary calls** — over-calling, caught by `subset` mode and by step-count efficiency.
- **Error recovery & retrying** — did the agent recover from a failed call, and *how expensively* (retry count is a cost, not just a success flag).
- **Efficiency** — number of steps; Galileo and DeepEval both expose step-efficiency signals [9][16].
- **Hallucinated tools/args** — calling a tool that does not exist, or inventing an argument schema — a *finding*, a `Validator` output (-> ek_04), not a graded scalar.

The benchmarks that operationalize these, and what ek borrows from each:

**AgentBench** (Liu et al., ICLR 2024, arXiv:2308.03688) [14] is the multi-environment capability spread — 8 distinct interactive environments (Operating System, Database, Knowledge Graph, Digital Card Game, Lateral Thinking Puzzles; ALFWorld/House-Holding, WebShop; Mind2Web web browsing), each with task-specific success metrics, evaluated across 29 LLMs. Its finding — that poor long-term reasoning, decision-making, and instruction-following are the main obstacles — is a *process* diagnosis dressed as an outcome benchmark. ek's takeaway is architectural: the grammar (Layer A) must be per-environment (allowed tools + arg schemas differ per domain), which is exactly why the grammar is a dependency-injected SSOT and not a global constant.

**ToolSandbox** (Apple, arXiv:2408.04682) [17] is the one to study hardest for *process* scoring. It is **stateful** (a tool's behavior depends on world state — the messaging tool only works if cell service is on and battery is sufficient), has **implicit state dependencies between tools**, ships a **built-in user simulator** for on-policy conversational evaluation, and — the key move — uses a **milestone-based dynamic evaluation strategy that scores intermediate *and* final states over an arbitrary trajectory** [17]. Milestones-over-a-trajectory is *precisely* AgentBoard's progress rate made into a scoring rubric, and it maps directly onto ek: a milestone is a required sub-graph of the Layer-A grammar, and "did the trajectory hit this milestone" is a per-node finding aggregated by the metric's own aggregator. **Licensing trap, flagged loudly (-> ek_06):** ToolSandbox is under the **Apple Sample Code License** — a bespoke, non-OSI, non-SPDX license granting a "personal, non-exclusive license," expressly reserving patent rights ("no … patent rights … are granted") and restricting trademark use [18]. It technically permits internal reproduction/modification/redistribution, so it is not non-commercial copyleft, but its patent reservation and non-standard terms mean **it must never be vendored into ek's permissive core** — cite it, learn from the milestone design, reimplement the *idea* cleanly; do not copy the code.

**TRAJECT-Bench** (arXiv:2510.04550) [19] is the recent trajectory-*aware* tool-use benchmark that reports exactly the process diagnostics an ek `TrajectoryMetric` should emit: **tool-selection correctness, argument correctness, and dependency/order satisfaction**, over trajectories synthesized to vary in *breadth* (parallel calls) and *depth* (interdependent chains). That triple maps one-to-one onto ek's Layer-A grammar — arg schemas and order/dependency constraints are grammar edges — and onto Validators for the findings. (Marked medium-confidence in the dossier; treat as a design target, not a settled standard.)

---

## 5. Multi-turn and conversational evaluation

Single-turn tool-use is the easy case. Real assistants hold a goal across turns, retain instructions, and negotiate with a user who is *also acting*. Three benchmarks anchor the space.

**MT-Bench-101** (Bai et al., ACL 2024, arXiv:2402.14762) [20] is the fine-grained multi-turn dialogue benchmark: a **three-tier ability taxonomy — Perceptivity, Adaptability, Interactivity — decomposed into 13 tasks, 4,208 turns across 1,388 dialogues.** Its aggregation rule is the one ek should steal: a **minimum-score-taking metric across a dialogue — the weakest turn caps the score.** This is the multi-turn analogue of MT-Bench-101's insight that a single averaged score *hides* per-turn collapse; averaging lets one great turn paper over a catastrophic one. The min-rule is opinionated and correct for reliability-critical assistants, and it is exactly the kind of *non-mean aggregator* ek's `evaluate()` contract already mandates ("aggregate via the metric's OWN aggregator, never a naive mean"). Notably, MT-Bench-101 found that common alignment techniques do *not* obviously improve multi-turn ability — a warning that single-turn wins do not transfer.

**tau-bench** (Yao et al., Sierra, arXiv:2406.12045; ICLR 2025) [21] introduced the metric that reframes the whole reliability conversation, and it is a *process/consistency* metric even though it is computed over outcomes. **pass^k** = the probability that an agent solves the *same task on all k independently sampled trials*, estimated per-task by the combinatorial estimator `pass^k = E_task[ C(c,k) / C(n,k) ]` where c of n trials pass [21]. It is the deliberate inverse of **pass@k** — the *unbiased estimator* of "at least one of k succeeds," `pass@k = E[1 − C(n−c,k)/C(n,k)]` (Chen et al., Codex/HumanEval, arXiv:2107.03374) [22]. pass@k measures **capability** and *rises* toward 1 as k grows; pass^k measures **reliability** and *falls* toward 0. The empirical punchline, stated with the verification correction: a SOTA gpt-4o function-calling agent **succeeds on <50% of tasks on average** (pass^1 ≈ 61% retail, ≈ 35% airline) yet **drops to pass^8 < 25% in retail** [21] — reliability collapses well below single-shot capability. For ek's CI gate (-> ek_02 golden-set regression gate), *pass^k is the right gate metric*, not pass@1: you want to ship the agent that succeeds *every* time, not the one that succeeds *some* time. (Caveat, flagged: pass^k over a few hundred tasks is a small-sample estimate; its confidence interval is wide, and it inherits the benchmark-contamination and holdout-poverty concerns of §8.)

**tau2-bench** (Barres et al., Sierra, arXiv:2506.07982; repo `sierra-research/tau2-bench`, **MIT**) [23][24] is the one that names the deepest methodological problem in this report: **dual control.** Its new **Telecom** domain is modeled as a **Dec-POMDP where both the agent and the simulated user act with tools on a shared, dynamic environment** [23]. (Nuance from verification: dual-control is specifically the *new* Telecom domain; the inherited airline/retail domains remain single-control — only the agent acts. This strengthens, not weakens, the point.) Agents show significant performance drops moving from no-user to dual-control [23]. And this is where the **user-simulator-is-an-evaluation-variable** problem becomes unavoidable: if the user is itself an LLM, then a fraction of every task's difficulty and every reward's variance comes from the *simulator*, not the agent under test. Both tau2-bench and ToolSandbox invest heavily in **constraining the simulated user to tools and observable state** precisely to raise simulation fidelity and reduce that confound [23][17]. ek's stance, stated as policy: **a simulated user is a *source/hypothesis*, never ground truth.** In ek's data model the user simulator is an input source feeding the episode, and its outputs carry provenance and confidence like any other signal — never the gold reference against which reward is computed. The gold reference is the *environment/database end-state* (tau-bench's own reward basis: compare final DB state to annotated goal state [21]), which is simulator-independent.

---

## 6. Planning evaluation

Planning is a distinct sub-capability with its own gold standard for *validity*, and it is the one place where a formal verifier beats both string-match and LLM-judge. **PlanBench** (Valmeekam, Kambhampati et al., NeurIPS 2023 D&B, arXiv:2206.10498) [25] evaluates **plan validity and optimality via the external VAL validator over PDDL** — Blocksworld/Logistics domains plus obfuscated (surface-name-scrambled) variants — checking plans for validity *formally* rather than by surface match, across plan generation, cost-optimal planning, replanning, plan verification, and goal reformulation. LLM plan-generation "falls short even for SOTA models," and *worsens under surface-name obfuscation* — evidence the models pattern-match rather than plan [25]. PlanBench even calls for future *partial-correctness* metrics, which is the progress-rate/partial-match theme again.

The ek lesson: **when a formal verifier exists, use it — it is a deterministic `Metric` that returns validity/optimality, not a judge.** PlanBench's separation of *planning* from *retrieval* (via obfuscation) is a slicing strategy ek's `per_slice` cuts should replicate: evaluate the same agent on named vs obfuscated tasks and report the gap as a robustness slice. VAL-over-PDDL is a clean, permissively-usable external validator; subgoal decomposition and plan validity become a `Validator` family (-> ek_04) that emits per-step findings ("this step violates a precondition") rather than a single scalar.

---

## 7. Memory, self-reflection, and safety — closing the capability set

§1's four core agentic capabilities are planning, tool-use, **memory**, and **self-reflection**; the survey's [1] open gaps add **safety**. Planning (§6) and tool-use (§4) are covered above; this section covers the three this report has so far left implicit, each kept to a paragraph plus its ek mapping.

### Long-term memory

Multi-turn assistants must retain and correctly retrieve state *across sessions*, not just within one episode. **LongMemEval** (Wu et al., arXiv:2410.10813) [33] isolates five memory abilities — **information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention** (declining to answer when the memory does not hold the fact) — and shows a sharp accuracy drop for commercial assistants once long chat histories must be recalled; **LongMemEval-V2** (arXiv:2605.12493) [34] extends it. → maps to ek as a **reference-based recall check over an episode's retained state**: remembered facts are Layer-B `raw_signals` keyed by `NodePath`, "was the right fact recalled at the right turn" is a `score()` against a gold memory reference, and abstention is a `Validator` (answering when the memory is silent = a false-positive finding).

### Self-reflection / self-correction

Beyond selecting the right next step, a robust agent **detects and recovers from its own errors** — a failed tool-call, a wrong branch, an inconsistent observation. The evaluable quantity is **error-recovery quality**: given an injected or naturally-occurring failure, did the agent notice, and how expensively did it retry (retry count and re-planning cost, not just eventual success — the same cost lens as §4's redundant-call dimension). → maps to ek as a **trajectory-level Finding plus a recovery signal**: the failure and its resolution are per-step `findings` in Layer B (-> ek_04), and "recovered / did-not-recover / recovered-but-wastefully" is a reference-free `estimate_quality()` signal (-> ek_03), cost-weighted by the retry steps the Layer-A grammar prices.

### Agent safety and adversarial robustness

The survey flags **safety** as a top open gap, and the tau-bench framing (§5) makes the enterprise stake concrete: a **destructive tool call** — an irreversible database mutation, a wrong refund — is far costlier than a missing read. Two benchmark families operationalize the risk. **AgentDojo** (Debenedetti et al., NeurIPS 2024) [35] measures **prompt-injection** resistance — **97 realistic tasks and 629 security test cases** across banking, Slack, travel, and workspace environments, checking whether injected content in tool outputs hijacks the agent. **AgentHarm** (Andriushchenko et al., ICLR 2025) [36] measures **harmful tool-use** — whether an agent complies with malicious multi-step requests — and **OS-Harm** (arXiv:2506.14866) [37] extends the harm axis to **computer-use agents**. → maps to ek as **should-not-call semantics in the tool-call metric** (a forbidden or injected-goal tool call is a scored false-positive, weighted by its Layer-A error-cost) **plus safety `Validator`s** (prompt-injection-triggered call, destructive-without-confirmation, harmful-request compliance) emitting per-node findings — the same FLAG machinery as §4's tool-use findings (-> ek_04), tuned to the high error-cost end of the grammar.

---

## 8. Harness support and the trace-as-trajectory standard

The tooling landscape (full reuse/wrap/build allocation and license register in -> ek_06):

| Tool | License | What ek takes | Caution |
|---|---|---|---|
| **agentevals** (LangChain) [3] | MIT | trajectory match modes + arg modes + graph + judge — *wrap behind `Metric`/`Signal`* | boolean-only, **no cost/token measurement** [7] |
| **DeepEval** [9] | Apache-2.0 | deterministic `ToolCorrectnessMetric` (recall-ratio), step-efficiency, Pytest-native | multi-turn tool metrics still open [11] |
| **Inspect AI** (UK AISI) [26] | MIT | `Solver`/`Scorer`/`Store` maps *cleanly* onto ek's facades + Layer-B store | full harness, heavier than a metric lib |
| **tau2-bench** (Sierra) [24] | MIT | dual-control env, pass^k, DB-state reward basis | needs Python ≥3.12; benchmark not library |
| **OpenAI Evals** [27] | MIT | registry/CLI pattern | not trajectory-specialized |
| **PRM800K** [13] | MIT | step-level labels for calibrating process signals | dataset, not code |
| **OpenLLMetry / OTel GenAI** [28][29] | Apache-2.0 | **the vendor-neutral span schema for trajectory-with-cost** | standard, adopt directly |
| **Arize Phoenix** [30] | **Elastic License 2.0** | observability/eval, spans-as-steps | **ELv2 is source-available, NOT OSI** — opt-in extra with warning, never default (-> ek_06) |
| **ToolSandbox** (Apple) [18] | **Apple Sample Code License** | milestone-over-trajectory *design idea* | **non-OSI, patent-reserved — never vendor** (-> ek_06) |
| **Galileo Agentic Evals** [16] | SaaS (proprietary) | *feature reference*: measures cost + latency + tool-error | not a dependency |

**Inspect AI** [26] deserves emphasis: a `Task` = `Dataset` + `Solver` (anything from a single `generate()` to a full multi-turn tool-using ReAct agent) + `Scorer`, with a `Store` carrying per-sample agent state. That is ek's `evaluate(cases, …) -> Report` with a Layer-B state store, built by a national AI-safety institute under MIT. ek should be *interoperable* with Inspect (accept its trajectories, expose its scorers as `Metric`s) rather than compete with it.

The single most important infrastructure decision is **how you capture the trajectory in the first place**, and here the standard has arrived: **OpenTelemetry GenAI semantic conventions** [29], with the Apache-2.0 **OpenLLMetry** [28] as the reference implementation. The trace *is* the typed trajectory: a top-level `invoke_agent` span, child `chat` spans per LLM call, `execute_tool` spans per tool invocation, and `gen_ai.*` attributes carrying model, **token counts and costs**, finish reason, tool calls and results [28][29]. This is the answer to `agentevals`' cost-blindness: capture the episode as OTel spans and you get cost, latency, *and* the step structure co-located per step, for free, in a vendor-neutral schema. **ek should adopt the OTel GenAI span schema as the wire format for a Layer-B episode.** A span tree deserializes directly into an `AnnotatedExtraction` over a trajectory grammar: spans → nodes, parent/child and sequence → edges, `gen_ai.*` cost/token/latency attributes → per-node `raw_signals`, tool name/args → the node's typed payload.

**Two honestly-flagged pitfalls** the harness must defend against:

1. **LLM-as-judge bias** (medium-confidence evidence [31]). Trajectory and turn judges inherit measurable biases: **position bias** (~10–15 pt win-rate swing by option order — mitigate by swapping order and averaging), **verbosity bias** (longer outputs scored higher regardless of added content), and **self-preference** (~10–25% — a judge favors its own model's outputs). ek's answer is not to trust the judge but to route it through ek_03's machinery: a judge `Signal` must be **calibrated** before its score is trusted (-> ek_03), and validated for inter-rater agreement against humans using ek_02's IAA harness (Krippendorff's α, Cohen's κ) (-> ek_02). An uncalibrated judge is a *signal*, not a *metric* — that distinction is exactly what ek's two facades enforce.

2. **Cost-uncontrolled leaderboards** (high-confidence [15]). Kapoor et al., *AI Agents That Matter* (arXiv:2407.01502): calling models repeatedly to inflate accuracy incentivizes "extremely costly agents"; at substantially similar accuracy, **cost can differ by almost two orders of magnitude** yet goes unreported; simple baselines often Pareto-dominate complex SOTA; and because agent benchmarks "typically consist of only a few hundred samples," overfitting is "more severe than data contamination" [15]. The prescription — **report accuracy vs inference cost as a Pareto curve, not a single leaderboard number** [15][32] — is ek's cost-per-successfully-completed-task objective, verbatim, from an independent source. This is *why* Layer-A carries cost weights and *why* `evaluate()` must never collapse to a naive mean.

---

## 9. Mapping to ek

### How this lands in ek

**A trajectory is a Layer-B episode over a Layer-A task/tool grammar.** Concretely:

- **Layer A (GraphGrammar, frozen SSOT).** For agents this is a **task/tool grammar**: the allowed tools, their argument schemas (typed fields with `FieldSpec.importance`), the legal orderings and data-dependencies between steps (edges), and — the ek-specific lever — the **task-value and error-cost weights** per node/edge/field. A tool that mutates a database carries high error-cost; a read-only search carries low. These weights are what turn agentevals' *boolean* match into ek's *cost-weighted* score, and they are what let `subset` vs `superset` be a *derived* choice (penalize over-calling proportionally to each tool's cost) rather than a hardcoded flag. TRAJECT-Bench's tool-selection/argument/dependency triple [19] and PlanBench's preconditions [25] are grammar constraints; ToolSandbox's milestones [17] are required sub-graphs.

- **Layer B (AnnotatedExtraction, per-episode metadata).** The episode's runtime record, keyed by `NodePath`: the **trajectory** (the actual step sequence), per-step `raw_signals` (PRM step-scores [12], judge scores, self-consistency agreement), **token cost / latency** (from OTel `gen_ai.*` spans [29]), per-step `findings`, `provenance` (including *which source produced each observation — the user simulator being one such source, never gold*), and the final `decision`. Layer B rides alongside the grammar, never mutating it — exactly as the OCR instance does with `OcrResult`.

- **`score(pred, gold, *, grammar, metric, …) -> Score`.** Gets a **`TrajectoryMetric`** that is *not new machinery*. Two modes, both reusing existing ek code:
  - **Graph mode:** the cost-weighted `TypedGraphMetric` (-> ek_02) with **steps as nodes, ordering/dependency as edges, tool-type and arg weights read from the Layer-A grammar.** Trajectory-graph distance *is* the cost-weighted typed-graph edit distance the framework already builds — this is the direct generalization the brief calls for.
  - **Sequence mode:** a trajectory edit distance that reuses the **global TP/FP/FN accumulation machinery of CER/WER** (-> ek_02) conceptually — substitutions/insertions/deletions over `(tool, args)` steps, weighted by Layer-A cost. The five match schemes of §2 are parameterizations of this: `strict` = full edit distance with zero tolerance, `unordered` = set distance, `subset`/`superset` = asymmetric insertion/deletion costs.
  - Wrap `agentevals` modes and DeepEval `ToolCorrectnessMetric` behind the `Metric` Protocol and register them by name (-> ek_06 reuse discipline); the flagship cost-weighted metric is the must-build.

- **`evaluate(cases, *, metric, grammar, …) -> Report`.** Aggregates via the **metric's own aggregator, never a naive mean.** This is where **pass^k** [21] lives (reliability aggregator over resampled trials), where **MT-Bench-101's min-across-turns rule** [20] lives (weakest-turn aggregator), and where **AgentBoard progress rate** [2] lives (partial-credit aggregator). `per_slice` cuts replicate PlanBench's named-vs-obfuscated robustness slicing [25] and AgentBench's per-environment breakdown [14]. The golden-set regression gate (-> ek_02) should gate on **pass^k, not pass@1**.

- **`estimate_quality(extraction, *, sources, signals, calibrator, validators, policy, …) -> QualityReport`.** The reference-free process path. **Agent signals:** trajectory LLM-as-judge (agentevals' judge variant [3]), self-consistency / sample-and-vote — **which is ROVER agreement generalized; reuse `qe/rover`, do not rebuild** (-> ek_03) — and step-level process-reward faithfulness (PRM-style [12]). The **cascade's confidence gate IS the `decide` stage; escalation-to-human IS selective prediction** (-> ek_03), inheriting conformal / risk-coverage / cost-ratio ρ=c_FN/c_FP wholesale. Every judge signal is **calibrated before trust and validated by IAA against humans** (-> ek_02, ek_03) — the mitigation for §8's judge biases.

- **Tool-use findings = `Validator`s.** Hallucinated tool, ungrounded argument, redundant call, missing precondition, milestone-missed — each is a `Validator` emitting a per-node finding into Layer B (-> ek_04's FLAG-vs-CORRECT six-layer model). This is the flag side; automated trajectory repair (retry-planning) would be the correct side.

- **The user simulator is a source/hypothesis, never gold** — an input source with provenance and confidence, per §5. Gold is the environment end-state.

- **Registry + DI + `requires_extra`.** `TrajectoryMetric`, agent `Signal`s, and tool-use `Validator`s are `typing.Protocol`s resolved by name and injected keyword-only with smart defaults (open-closed). Heavy/non-permissive backends are guarded: `agentevals`/`inspect-ai`/DeepEval/tau2-bench are MIT/Apache and can be defaults-eligible extras; **Phoenix (ELv2) and anything touching ToolSandbox code are opt-in-only, license-gated, never default** — the CI license gate fails the build on any non-OSI/copyleft/non-commercial dependency reaching the `core..hitl` closure (-> ek_06).

- **Production monitoring** of live agent trajectories — drift in progress rate, tool-error rate, cost-per-success over time — flows into ek_05's monitoring/drift machinery (CBPE, audit sampling) (-> ek_05), fed by the same OTel span stream.

The one-sentence architecture: **capture the episode as OTel GenAI spans → deserialize into a Layer-B `AnnotatedExtraction` over a Layer-A task/tool grammar → `score()`/`evaluate()` it with a cost-weighted `TrajectoryMetric` (typed-graph or sequence edit distance) against a reference, or `estimate_quality()` it reference-free with calibrated judge/self-consistency signals and tool-use Validators — aggregating with pass^k / min-across-turns / progress-rate, never a naive mean, and reporting on a cost-vs-accuracy Pareto curve.**

---

## References

[1] Yehudai A, et al. A Survey on Evaluation of LLM-based Agents. arXiv:2503.16416, 2025. <https://arxiv.org/abs/2503.16416>

[2] Ma C, Zhang Z, et al. AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. NeurIPS 2024 (Oral). arXiv:2401.13178, 2024. <https://arxiv.org/abs/2401.13178> · [repo](https://github.com/hkust-nlp/AgentBoard)

[3] LangChain. [agentevals: Readymade evaluators for agent trajectories](https://github.com/langchain-ai/agentevals) (MIT). GitHub, 2024–2026.

[4] LangChain. [How to evaluate your agent with trajectory evaluations](https://docs.langchain.com/langsmith/trajectory-evals). LangSmith Docs, 2026.

[5] LangChain. Trajectory match modes (superset / subset / unordered / strict) and helper scorers. LangSmith Docs. <https://docs.langchain.com/langsmith/trajectory-evals>

[6] agentevals. [PyPI package](https://pypi.org/project/agentevals/) (MIT), 2026.

[7] LangChain. Trajectory evaluators return `{key, score: bool, comment}` (no cost/token measurement). LangSmith Docs. <https://docs.langchain.com/langsmith/trajectory-evals>

[8] LangChain agentevals. `strict` mode compares messages, order, and tool calls. GitHub. <https://github.com/langchain-ai/agentevals>

[9] Confident AI. [deepeval](https://pypi.org/project/deepeval/) (Apache-2.0), PyPI, 2026; agentic metrics (TaskCompletion, ToolCorrectness, step efficiency).

[10] Confident AI. [Tool Correctness metric](https://deepeval.com/docs/metrics-tool-correctness). DeepEval Docs, 2026. (Score divides by `len(expected_tools)` in source: recall-flavored proportional ratio, not F1.)

[11] Confident AI / community. Conversational Task Completion & Tool Correctness feature request. DeepEval GitHub issue #2223. <https://github.com/confident-ai/deepeval/issues/2223>

[12] Lightman H, Kosaraju V, Burda Y, Edwards H, Baker B, Lee T, Leike J, Schulman J, Sutskever I, Cobbe K. Let's Verify Step by Step. ICLR 2024. arXiv:2305.20050, 2023. <https://arxiv.org/abs/2305.20050> (78.2% on a representative MATH subset under Best-of-N with the PRM.)

[13] OpenAI. [PRM800K: 800K step-level correctness labels on MATH solutions](https://github.com/openai/prm800k) (MIT). GitHub, 2023.

[14] Liu X, et al. AgentBench: Evaluating LLMs as Agents. ICLR 2024. arXiv:2308.03688, 2023. <https://arxiv.org/abs/2308.03688> · [repo](https://github.com/THUDM/AgentBench)

[15] Kapoor S, Stroebl B, Siegel Z, et al. AI Agents That Matter. arXiv:2407.01502, 2024. <https://arxiv.org/pdf/2407.01502>

[16] Galileo. [Introducing Agentic Evaluations](https://galileo.ai/blog/introducing-agentic-evaluations). Galileo blog, 2025 (measures tool-selection quality, tool-call errors, session success, cost, latency).

[17] Lu J, Holleis T, Zhang Y, Aumayer B, Nan F, Bai F, Ma S, et al. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities. arXiv:2408.04682, 2024. <https://arxiv.org/abs/2408.04682> · [repo](https://github.com/apple/ToolSandbox)

[18] Apple. [ToolSandbox LICENSE — Apple Sample Code License](https://github.com/apple/ToolSandbox/blob/main/LICENSE) (non-OSI, non-SPDX; patent rights reserved; trademark restrictions). GitHub, 2024–2026.

[19] TRAJECT-Bench: A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use. arXiv:2510.04550, 2025. <https://arxiv.org/abs/2510.04550>

[20] Bai G, Liu J, et al. MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues. ACL 2024, pp. 7421–7454. arXiv:2402.14762. <https://aclanthology.org/2024.acl-long.401/>

[21] Yao S, Shinn N, Razavi P, Narasimhan K. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. ICLR 2025. arXiv:2406.12045, 2024. <https://arxiv.org/abs/2406.12045> (pass^k = C(c,k)/C(n,k); gpt-4o <50% avg, pass^8 <25% retail.)

[22] Chen M, et al. Evaluating Large Language Models Trained on Code (Codex/HumanEval; unbiased pass@k estimator). arXiv:2107.03374, 2021. <https://arxiv.org/abs/2107.03374>

[23] Barres V, Dong H, Ray S, Si X, Narasimhan K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025. <https://arxiv.org/abs/2506.07982>

[24] Sierra Research. [tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT). GitHub, 2025–2026.

[25] Valmeekam K, Marquez M, Olmo A, Sreedharan S, Kambhampati S. PlanBench: An Extensible Benchmark for Evaluating LLMs on Planning and Reasoning about Change. NeurIPS 2023 Datasets & Benchmarks. arXiv:2206.10498, 2022 (v4 2023). <https://arxiv.org/abs/2206.10498> · [repo](https://github.com/karthikv792/LLMs-Planning)

[26] UK AI Security Institute. [Inspect AI: A framework for LLM evaluations](https://github.com/UKGovernmentBEIS/inspect_ai) (MIT). GitHub / <https://inspect.aisi.org.uk/>, 2026.

[27] OpenAI. [Evals: framework for evaluating LLMs](https://github.com/openai/evals) (MIT). GitHub.

[28] Traceloop. [OpenLLMetry: OpenTelemetry-based GenAI observability](https://github.com/traceloop/openllmetry) (Apache-2.0). GitHub, 2026.

[29] OpenTelemetry. [Semantic conventions for generative AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) (invoke_agent / chat / execute_tool; token counts + costs). Docs; and [Inside the LLM Call: GenAI Observability with OpenTelemetry](https://opentelemetry.io/blog/2026/genai-observability/), 2026.

[30] Arize AI. [Phoenix: AI Observability & Evaluation](https://github.com/Arize-ai/phoenix) (Elastic License 2.0 — source-available, NOT OSI-approved). GitHub, 2026.

[31] Li Z, et al. Self-Preference Bias in LLM-as-a-Judge. arXiv:2410.21819, 2024. <https://arxiv.org/html/2410.21819v1> (position ~10–15 pt swing; verbosity bias; self-preference ~10–25%.)

[32] Narayanan A, Kapoor S. [AI leaderboards are no longer useful. It's time to switch to Pareto curves](https://www.normaltech.ai/p/ai-leaderboards-are-no-longer-useful). Normal Technology, 2024.

[33] Wu D, et al. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. arXiv:2410.10813, 2024. <https://arxiv.org/abs/2410.10813> (five memory abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention.)

[34] LongMemEval-V2. arXiv:2605.12493, 2026. <https://arxiv.org/abs/2605.12493>

[35] Debenedetti E, Zhang J, Balunović M, Beurer-Kellner L, Fischer M, Tramèr F. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. NeurIPS 2024 Datasets & Benchmarks. (97 tasks + 629 security test cases across banking/Slack/travel/workspace.)

[36] Andriushchenko M, et al. AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents. ICLR 2025.

[37] OS-Harm: A Benchmark for Measuring Safety of Computer-Use Agents. arXiv:2506.14866, 2025. <https://arxiv.org/abs/2506.14866>
