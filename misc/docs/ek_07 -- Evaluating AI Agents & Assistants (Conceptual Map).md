# Evaluating AI Agents & Assistants — A Conceptual Map

*The same 2×2 as information-extraction evaluation, but the object is an episode and the unit is cost-per-successful-task*

**Author:** Thor Whalen
**Status:** Conceptual map **+ reading guide** — the entry point into the agent-evaluation sub-series (`ek_08`–`ek_12`). It parallels the IE conceptual map (`information-extraction-evaluation-conceptual-map.md`) one rung up the abstraction ladder: where that map evaluated a *single extracted payload*, this one evaluates a *trajectory* that produces payloads (and side effects) over many steps. The body states the *concept* and the load-bearing thesis; the exhaustive tool rosters, formulas, license verdicts, and cost models live in the pointered reports (follow the `→ ek_0N` markers). Read §7 (mapping to ek) and §8 (reading guide) first if you want to know which report answers a given question.

---

## TL;DR

Agent evaluation is **not a new discipline** — it is the IE evaluation framework this project already built, lifted onto a different object. The same two independent axes still cut the space:

- **Reference availability** — do we have a gold outcome/trajectory to compare against (*reference-based*) or not (*reference-free*)?
- **Granularity** — are we scoring one run (*instance/episode level*) or aggregating over a task suite (*system level*)?

But three things change, and they change everything downstream:

1. **The evaluated object is an EPISODE (a trajectory), not a payload.** An agent run is a sequence of `(thought, tool-call, observation)` steps ending in a final state. You can grade the final state, the tool calls, the whole trajectory, or the task utility — a *ladder of evaluation objects* directly analogous to IE's string→field→graph ladder, and you climb it the same way: defer the boolean, keep numeric scores, weight by task value and error cost.
2. **The metric is COST PER SUCCESSFULLY COMPLETED TASK, not per-token cost.** Modern LLM applications are **compound AI systems** [8] — chained model calls, retrieval, tools, verifiers, memory. Reliability compounds *multiplicatively*: even 90 %-per-step reliability over 8 dependent steps yields ≈ 43 % end-to-end [14]. So a few points of per-step reliability dominate the end-to-end number, and the only honest denominator for cost is a *successful* task, not a token.
3. **Reliability, not capability, is the production metric.** The split between "can it ever do this?" and "does it do this *every single time*?" is the difference between **pass@k** (any of k trials succeeds — capability [1]) and **pass^k** (all k trials succeed — reliability [3]). τ-bench's headline is the whole thesis in one number: gpt-4o clears ≈ 61 % of retail tasks at pass¹ but only **≈ 25 % at pass⁸ in retail** [4] (airline is harder still, ≈ 35 % pass¹; the *cross-domain average* is < 50 % pass¹ — that "< 50 %" is the average, **not** the retail figure). Consistency collapses far below single-shot success.

The four corners of the 2×2 map cleanly onto ek's three facades plus its production-monitoring corner, and every new agent metric or signal plugs into the *existing* registry — no new architecture, just new strategies. The one genuinely new *build* is a trajectory/tool-call-graph distance, which is the cost-weighted typed-graph metric from `ek_02` with the nodes relabelled from fields to tool calls.

---

## 1. The evaluation 2×2 for agents

The IE map put your two needs in opposite corners of a reference-availability × granularity grid. Agents live in the identical grid; only the labels of the corners change, and each corner names a facade you already have.

| | **Reference-based** (gold outcome/trajectory available) | **Reference-free** (no gold at inference) |
|---|---|---|
| **System / suite level** | **Offline task-suite benchmarking** — τ-bench, BFCL, SWE-bench: run the agent over a fixed suite, aggregate with the metric's *own* aggregator, cut per-slice → `evaluate()` | **Production monitoring of an agent fleet** — success/cost/latency drift over live traffic → the `ek_05` corner |
| **Episode / instance level** | **Grading ONE episode vs a gold outcome or trajectory** — did this run reach the goal state / match the reference tool sequence? → `score()` | **Online QE / the cascade confidence gate** — LLM-as-judge, self-consistency, a verifier deciding accept/escalate on *this* run → `estimate_quality()` |

Two observations make this grid load-bearing rather than decorative.

**The reference-free corners dominate in production.** In IE, gold answers are scarce at inference; for agents they are *scarcer still*, because a gold **trajectory** is far more expensive to author than a gold string, and often ill-defined (many correct paths exist). So the reference-free instance corner — a verifier or judge deciding whether *this* episode is trustworthy — is where most operational value lives, and it is exactly `estimate_quality()`'s `signal → calibrate → validate → decide` cascade (→ `ek_03`, `ek_09`).

**Aggregation is never a naive mean.** τ-bench's pass^k is the canonical demonstration: the corpus-level reliability number is *not* the average of per-episode booleans — it is an unbiased combinatorial estimator over repeated trials (§4). This is precisely why `evaluate()` aggregates through the metric's own aggregator, and why per-slice cuts (by domain, task difficulty, tool subset) are mandatory rather than optional. A single mean hides the fact that reliability decays geometrically in the number of dependent steps.

---

## 2. Compound AI systems: why the unit is cost-per-successful-task

The 2019-era mental model — "evaluate the model" — is obsolete. Zaharia et al.'s BAIR framing [8] names the replacement: a **compound AI system** is "a system that tackles AI tasks using multiple interacting components, including multiple calls to models, retrievers, or external tools." State-of-the-art results now come from *systems*, not monolithic models — AlphaCode 2 generates up to ~1 M candidate solutions then filters; retrieval-augmented and tool-using agents chain many calls behind control logic. An agent is the most general such system: a control loop that decides, at each step, which tool to call with which arguments, and when to stop.

Two consequences fall directly out of this, and they are the spine of `ek_11`:

**Reliability compounds multiplicatively.** If an agent's task decomposes into *n* dependent steps each succeeding with probability *p*, end-to-end success is ≈ *pⁿ*. At *p* = 0.95 over 8 steps that is ≈ 66 %; at *p* = 0.90, ≈ 43 % [14]. The practical corollary is brutal: **a few points of per-step reliability dominate the end-to-end number far more than any single-step capability gain.** This is the quantitative engine behind pass^k's collapse (§4) and the reason τ-bench measures all-k-succeed reliability at all [3, 4].

**More calls is not monotonically better — it has an *optimum*.** Chen et al.'s "Are More LLM Calls All You Need?" (NeurIPS 2024) [7] proves that Vote and Filter-Vote strategies are **non-monotone** in the number of LM calls: "the performance of both Vote and Filter-Vote can first increase but then decrease as a function of the number of LM calls." The mechanism is diversity of query difficulty — more calls help *easy* queries but *hurt* hard ones, so a task mixing both exhibits an interior maximum. They derive an analytical scaling model that predicts the optimal call count from a handful of samples. The lesson for ek: the number of samples/retries is a *tunable strategy parameter with a cost-optimal setting*, not a "more is better" knob — and finding that setting is an evaluation task, not a guess.

Put together: the denominator for cost cannot be tokens, because tokens spent on a failed or abandoned episode are pure waste, and because the optimal amount of inference compute is itself a function of the success target. The honest unit is **cost (dollars) per successfully completed task** — *not* a raw token count. Token count is not interchangeable with dollars: input, output, cache-read, cache-write, and reasoning tokens are priced asymmetrically (output tokens typically ≈ 5× input), so a normalized "tokens per successful task" figure is at best a rough proxy — the real objective is *monetary cost* per successful task, and the accounting is deferred to `ek_11` (whose leaderboard-grade backends report Cost in USD and Latency directly [12]). The canonical formalization of this denominator is **Cost-of-Pass** (Erol et al. [15]): the expected monetary cost of obtaining a *correct* solution, E[cost] / P(success), which diverges to ∞ when the system never succeeds (P(success) = 0) — together with its **Frontier Cost-of-Pass**, the minimum achievable across the available models and the human expert. A reader of this map meets the concept here; `ek_11` builds directly on it (→ `ek_11`). Everything in `ek_11` is an elaboration of this denominator; everything in the two-layer data model (§7) exists to carry the per-episode token/latency/verdict metadata that lets you *compute* it.

---

## 3. The ladder of evaluation objects for agents

The IE map's central move was to climb a *ladder of evaluation objects* — from brittle string equality down to a cost-weighted typed-graph distance — deferring the boolean and weighting by cost. Agents have the identical ladder, one abstraction up. From cheapest/most-brittle to richest/most-expensive:

| Rung | Object graded | What it answers | Canonical implementation | Cross-ref |
|---|---|---|---|---|
| 1 | **Final-answer exact match** | Did the last message equal the gold string? | HumanEval-style string/unit-test check | brittle baseline |
| 2 | **Outcome / state-based grading** | Is the *world* in the goal state? | τ-bench (end DB state vs annotated goal) [3]; SWE-bench (repo tests pass) [11] | → `ek_08` |
| 3 | **Tool-call correctness** | Right tool, right args, right types? | BFCL AST match + executable eval [12, 13] | → `ek_10` |
| 4 | **Trajectory match** | Did the *sequence* of steps match a reference? | agentevals strict/unordered/subset/superset [dossier] | → `ek_10` |
| 5 | **Task utility / cost** | Was it worth it — value delivered minus cost? | cost-per-successful-task, tokens-per-success [12] | → `ek_11` |

Three principles ride on this ladder, all inherited verbatim from the IE map and all reconfirmed by the agent literature.

**Defer the boolean; grade outcomes over paths where you can.** Anthropic's "Demystifying evals for AI agents" is emphatic: "it's often better to grade what the agent produced, not the path it took" [10], because penalizing a creative-but-correct path is a false negative — exactly the false-negative problem that motivated the whole IE ladder. Rung 2 (outcome/state) is therefore the *default* rung for agents, not rung 4 (trajectory). τ-bench embodies this: it compares the **database state at end of conversation** against an annotated goal state, *not* the action sequence [3]. Trajectory match (rung 4) is reserved for cases where the *path itself* is the deliverable (safety audits, tool-use compliance, teaching correct procedure) — not as a proxy for success.

**Keep numeric scores; threshold late.** A 0/1 outcome throws away the graded signal that triage and calibration need. The nuance Anthropic adds is diagnostic, not just aesthetic: "a 0 % pass rate across many trials is most often a signal of a broken task, not an incapable agent" [10], and an eval saturated at 100 % only tracks regressions and gives no improvement signal. Both failure modes are invisible to a raw mean and visible only when you keep the distribution.

**Weight by task value and error cost.** The same cost-sensitivity that made a misspelled city ≠ two wrong digits on a donation amount applies here: a failed refund is not the same cost as a redundant clarifying question. This is where Layer A's importance/cost weights (§7) do their work — they are the `task-value / error-cost` weights the metric aggregator consumes, and they are the reason the trajectory-distance metric is *cost-weighted*, not raw.

---

## 4. Reliability metrics: pass@k vs pass^k

This is the single most important conceptual distinction in agent evaluation, and it is a clean generalization of the IE map's "defer the boolean / keep the distribution" principle to the setting of *stochastic re-runs*.

**pass@k — capability.** Introduced with HumanEval/Codex (Chen et al. 2021) [1], pass@k estimates the probability that **at least one** of k samples is correct. The naive estimator 1−(1−p̂)ᵏ is *biased* — it underestimates the true value "by a considerable margin," in the paper's words — so they use the unbiased, numerically-stable form

> pass@k := 𝔼_problems [ 1 − C(n−c, k) / C(n, k) ]

generating n ≥ k samples per problem (the paper uses n = 200) and counting c correct ones [1]. HumanEval is 164 hand-written problems (hand-written deliberately, to avoid training-set contamination), graded by unit tests (≈ 7.7 tests/problem — functional correctness, not string match). Codex-12B scores pass@1 = 28.8 %, pass@100 = 72.31 % [1]. pass@k is a **capability/potential** metric: it answers "could the system ever do this, given enough tries?"

**pass^k — reliability.** τ-bench (Yao, Shinn, Razavi, Narasimhan 2024) [3] introduces the complement: the probability that **all k** independent trials of the same task succeed. Its unbiased per-task estimator mirrors the pass@k construction:

> pass^k := 𝔼_task [ C(c, k) / C(n, k) ]

with c successes out of n trials, averaged over tasks [3]. For a task with per-trial success probability p, pass^k = pᵏ, so it **decays geometrically in k** — a 90 %-reliable agent drops to ≈ 0.9⁸ ≈ 43 % consistency at k = 8. This is the production metric: a customer-facing agent that succeeds "usually" but not "every time" is not shippable, and pass^k is the number that exposes it. τ-bench's headline — gpt-4o at **≈ 61 % pass¹ but ≈ 25 % pass⁸ in retail** (≈ 35 % pass¹ in the harder airline domain; the < 50 % pass¹ figure is the *cross-domain average*, not retail) [4] — is the empirical face of the compounding-reliability thesis (§2). The practitioner rule of thumb: **use pass@k when a single success is enough** (offline candidate generation with a verifier), **pass^k when consistency is the product** (anything customer-facing) [dossier].

**Coverage from repeated sampling — and its hard caveat.** Large Language Monkeys (Brown et al. 2024) [9] studies pass@k as *coverage* — the fraction of problems solved by any of k samples — and finds it scales with the number of samples over four orders of magnitude, "often log-linear and can be modelled with an exponentiated power law." DeepSeek-Coder-V2-Instruct on SWE-bench Lite rises from 15.9 % (1 sample) to **56 % (250 samples)**, beating the then-SOTA single-attempt 43 % [9]. GSM8K coverage with Llama-3-8B-Instruct reaches 95.3 % at 10 000 samples (MATH scales similarly but somewhat lower and less regularly) [9]. **The caveat is the whole ballgame:** coverage only converts to *usable* accuracy when a verifier can pick the right sample. "In domains without automatic verifiers, common methods for picking from a sample collection (majority voting and reward models) plateau beyond several hundred samples and fail to fully scale with the sample budget" [9]. **Verification is the bottleneck** — which is precisely why the reference-free QE corner (verifier/judge, → `ek_09`) is not a nicety but the thing that unlocks inference-time scaling. This is the exact same lesson the IE map drew about agreement/consensus, one level up.

**Where pass^k is contested.** pass^k assumes trials are i.i.d. and that "all k succeed" is the right reliability target; for tasks with legitimately non-deterministic acceptable outcomes it can under-credit a competent agent, and its variance at small n is high. Treat it as *the* reliability headline but always report the underlying pass¹ and the trial count n alongside it, and cut per-slice — a low pass^k concentrated in one hard slice is a different problem than uniform flakiness.

---

## 5. Outcome vs process (trajectory) supervision

The rung-2-vs-rung-4 tension has a name in the training literature — **outcome supervision vs process supervision** — and it recurs in evaluation. Outcome grading (rung 2) asks only whether the final state is right; process/trajectory grading (rung 4) inspects the steps. The evaluation default is outcome, for the false-negative reason in §3, but process grading is indispensable in three cases: (a) the path is the deliverable or the safety surface (did the agent call a destructive tool it shouldn't have?); (b) diagnosis — *why* did it fail, which step went wrong; (c) partial-credit shaping on long-horizon tasks where all-or-nothing outcome grading gives no gradient. τ²-bench sharpens this by separating *reasoning* errors from *communication/coordination* errors in a dual-control setting where both agent and user mutate shared state [6]. The full treatment — trajectory-match modes, tool-call AST/F1, multi-turn and dual-control grading — is `ek_10`; here it is enough to name the split and to insist that trajectory match is a *deliberate choice for path-sensitive tasks*, never the lazy default proxy for success.

---

## 6. Reference-free agent QE: the cascade gate, verifier, and judge

The reference-free instance corner (§1) is `estimate_quality()`'s territory, and it generalizes the IE map's `signal → calibrate → decide` cascade with three agent-specific signal families. The detail is `ek_09` (ties to `ek_03`); the map-level points are these.

**LLM-as-judge is the dominant signal — and it is biased.** A judge model grades an episode (or a rubric dimension of it) with no reference. It is cheap and general, but carries quantified, only-partially-mitigable biases: **position bias** (order of A/B swings win-rate ~10–15 pts, mitigated by averaging over both orders), **verbosity bias** (longer outputs inflated ~15–30 pts), and **self-preference** (a judge favors its own family — real but *highly variable in sign and magnitude* across models and datasets; some frontier models actually *under*-rate their own outputs) [dossier]. Present these as approximate, study-dependent ranges, never constants. The discipline that tames them is Hamel Husain's [dossier]: **binary pass/fail, not 1–5 Likert** ("if your evaluations consist of a bunch of metrics that LLMs score on a 1–5 scale, you're doing it wrong"); a single principal domain expert via *critique-shadowing*; and measuring judge↔human **precision and recall separately** (raw agreement misleads under class imbalance). His Honeycomb case reached > 90 % agreement in three iterations, but that is a *result*, not a universal threshold — ~75–90 % is the workable band; below ~75 % the judge adds more noise than it removes. Anthropic adds the structural rules: **one isolated judge per rubric dimension** (not one judge grading all dimensions at once), and give the judge an **"Unknown" escape hatch** to avoid hallucinated verdicts [10]. Judge *calibration* is `ek_03`; judge *validation via IAA/Krippendorff* reuses the `ek_02` harness — do not rebuild either.

**Self-consistency is ROVER, reused.** Sampling the agent k times and voting is the stochastic-LLM analog of ROVER's alignment-and-vote over N OCR engines — the IE map's second must-build. **Reuse it, don't rebuild it:** the ROVER agreement machinery generalizes directly to sample-and-vote self-consistency, and agreement is a strong label-free error predictor precisely because raw model probability is miscalibrated (→ `ek_03`). This is also the "verifier" that Large Language Monkeys shows is the bottleneck (§4).

**The cascade's confidence gate IS the decide stage; escalation IS selective prediction.** "Trust or Escalate" / Cascaded Selective Evaluation (Jung et al., ICLR 2025) [dossier] operationalizes this: start with a cheap weak judge, and "only when the model is not sufficiently confident, iteratively move on to a stronger model," routing the genuinely-uncertain cases to a human — with a *provable guarantee* of human-agreement via calibrated confidence and a gating threshold. That is exactly the accept/escalate/reject selective-prediction policy from `ek_03`, with the cost ratio ρ = c_FN/c_FP setting the threshold, and the cheap-then-expensive cascade is *also* the cost lever of `ek_11`. The confidence gate, the verifier, and the escalation policy are one object viewed three ways.

---

## 7. How this lands in ek — the conceptual bridge, spelled out

The thesis of this whole sub-series is that **agent evaluation requires no new ek architecture** — only new strategies plugged into the existing seams. Here is the mapping, component by component (the master orientation is `ek-dev-architecture`; the two-layer model and facades live in `ek/base.py`, the registry in `ek/registry.py`).

**Layer A (GraphGrammar) becomes a TASK/TOOL grammar.** In IE, Layer A is a frozen typed-schema SSOT carrying node/edge/field *types* plus importance/cost weights per element (`FieldSpec.importance`). For agents it carries exactly the same information about a different domain: the **allowed tools, their argument schemas, and the task-value / error-cost weights** — how much a given task is worth, how costly each error class is. It remains frozen, remains the SSOT, and continues to feed both the cost-sensitive metrics and the constrained decoders/validators (now: tool-argument validators and allowed-tool constrained decoding). Nothing about Layer A's design changes; only the vocabulary of its nodes.

**Layer B (AnnotatedExtraction) becomes EPISODE metadata.** In IE, Layer B is per-item runtime metadata (raw_signals, confidence, findings, provenance, decision) keyed by NodePath, riding alongside the grammar without mutating it. For agents it carries the **trajectory** (the step sequence), **per-step signals** (judge scores, self-consistency agreement), the **token cost and latency**, and the **verdict** — again keyed by path (now a step/tool path), again riding alongside the frozen grammar. This is the object that makes cost-per-successful-task *computable*: the token/latency/verdict fields are the raw material of `ek_11`'s denominator.

**The facades map over unchanged.**
- `score(pred, gold, *, grammar, metric, ...)` grades **one episode** against a gold outcome or trajectory — the reference-based instance corner. New metrics register here: task-success, tool-call F1/AST match, trajectory match. → `ek_08`, `ek_10`
- `evaluate(cases, *, metric, grammar, ...)` runs a **task suite**, aggregating through the metric's *own* aggregator (pass@k / pass^k are *exactly* such non-mean aggregators) with `per_slice` cuts — the reference-based system corner. → `ek_08`
- `estimate_quality(extraction, *, sources, signals, calibrator, validators, policy, ...)` runs the **reference-free cascade** on one episode: agent signals (LLM-judge, self-consistency, faithfulness) → calibrate → validate → decide. The confidence gate is the `decide` stage; escalation is selective prediction. → `ek_09`

**New metrics/signals plug into the registry, open-closed.** Every swappable behavior is already a `typing.Protocol` (Metric, Signal, Calibrator, DecisionPolicy, Validator) resolved by name and injected keyword-only with smart defaults. A task-success metric is a new `Metric`; an LLM-judge is a new `Signal`; the cascade escalation is a `DecisionPolicy`; a tool-argument checker is a `Validator`. Heavier or non-permissive agent-eval backends (a specific judge SDK, a benchmark harness) are guarded by `@requires_extra("<extra>")` and gated by the CI license check exactly as in `ek_06` — the licensing discipline is unchanged, and the register of traps for the agent layer (Arize Phoenix ELv2, LangSmith/closed backends) is `ek_12`.

**The cost-weighted typed-graph metric generalizes to trajectory distance.** This is the one genuinely new *build*, and it is barely new: the IE map's flagship must-build — a cost-weighted, type-aware distance over a typed graph, with per-type edit costs sourced from Layer-A importance (→ `ek_02`, `ek_06`) — becomes a **trajectory / tool-call-graph distance** simply by relabelling the nodes from fields to tool calls and sourcing the edit costs from the tool grammar's error-cost weights. The `networkx`/`apted`/`zss` cost-callable hooks are the same; we supply tool-aware cost functions instead of field-aware ones. Similarly, **ROVER agreement generalizes to self-consistency** — the second must-build is reused wholesale for sample-and-vote (§6). The harness (`evaluate_store`, `save_baseline`/`regression_gate` golden-set CI gate, IAA via `krippendorff_alpha`/`cohen_kappa`) applies unchanged: an agent regression gate is a golden episode-suite gate; judge validation is IAA over judge-vs-human labels.

The net claim: **ek's data model, facades, registry, and two must-builds already *are* an agent-evaluation framework** the moment you register agent strategies into them. The sub-series `ek_08`–`ek_12` fills in those strategies.

---

## 8. Reading guide — into ek_08–ek_12

This map is deliberately thin on tool rosters and formulas; each pointered report is where they live. Read them in this order for a build, or jump by corner of the 2×2.

**`ek_08` — Task success & outcome-based benchmarking (the reference-based corners).** The `evaluate()`/`score()` side for agents: outcome/state-based grading (τ-bench's end-state comparison [3, 4], τ²-bench's dual-control extension [6]), execution-based grading (SWE-bench's repo-test harness — 2 294 real GitHub issue/PR tasks from 12 Python repos, graded by FAIL_TO_PASS + PASS_TO_PASS [11]), and the pass@k/pass^k estimators [1, 3] as the metric aggregators. **Read this first if your question is "how do I benchmark my agent against a task suite and get a trustworthy reliability number?"** It also carries the contamination warnings (§ below) that `evaluate()` users must internalize.

**`ek_09` — LLM-as-judge, verifiers & reference-free agent QE (the reference-free instance corner).** The `estimate_quality()` cascade for agents: the judge-bias catalog and mitigations, Hamel's binary/critique-shadowing/precision-recall discipline, Anthropic's isolated-judge and Unknown-escape-hatch rules [10], self-consistency as reused ROVER, and Cascaded Selective Evaluation as the confidence gate. Ties tightly to `ek_03` (calibration, selective prediction, cost ratio ρ) and reuses the `ek_02` harness for judge validation. **Read this if your question is "can I trust *this* run without a gold answer, and when do I escalate to a human?"**

**`ek_10` — Trajectory, tool-call & multi-turn evaluation (process supervision).** The rung-3/rung-4 detail: BFCL's AST-match-plus-executable tool-call grading [12, 13], agentevals' strict/unordered/subset/superset trajectory-match modes, the trajectory/tool-call-graph distance build, and multi-turn/dual-control grading (τ²-bench [6]). **Read this if your question is "did the agent take the right steps / call the right tools," not just "did it end up right."**

**`ek_11` — Cost economics: cost-per-successful-task (the utility rung).** The §2 thesis made operational: compounding reliability [8, 14], the non-monotone optimum in LLM calls [7], coverage scaling and its verifier bottleneck [9], tokens-per-successful-task as the normalized unit [12], and CI eval gates that fail a run when *quality is fine but economics or latency exceed budget* [12]. **Read this if your question is "what does this agent actually cost to run reliably, and where is the optimum?"**

**`ek_12` — Libraries & integration map for agent eval.** The `ek_06` treatment for the agent layer: the reuse/wrap/build allocation and the license register. Permissive-OK harnesses (Inspect AI — MIT, UK AISI, whose Task = Dataset + Solver + Scorer maps almost 1:1 onto ek's facades; agentevals/openevals — MIT; DeepEval/Ragas — Apache-2.0; promptfoo/lm-eval-harness — MIT; τ-bench/SWE-bench — MIT; BFCL/`bfcl-eval` — Apache-2.0, one of the few that natively reports Cost and Latency [12, 13]) versus the traps (Arize Phoenix under Elastic License 2.0 — source-available, *not* OSI; LangSmith backend closed; note the corrected classification — the W&B Weave *SDK* is Apache-2.0, only its hosted backend is proprietary). **Read this to decide what to install, wrap, or build — and what to keep out of the permissive core.**

**A cross-cutting warning that belongs in every corner: benchmark contamination and reward hacking.** Offline agent benchmarks are contaminated in two distinct ways — *training-data leakage* (measurable ~10.6 % in SWE-bench Verified, ~8.7 % in base) and *runtime leakage / reward hacking* (a Cursor study found sealing git history and restricting internet dropped a top agent from 87.1 % to 73.0 % on SWE-bench Pro — 14.1 pts of leakage — with one system running `git log` to copy answers from commit history in 24.4 % of trajectories) [dossier]. OpenAI's Frontier Evals team **stopped reporting SWE-bench Verified** because "the score stopped meaning anything" [dossier]. The operational implications land on `evaluate()`: isolate every trial in a clean environment, seal answer channels, prefer held-out and freshly-authored tasks, and treat a suspiciously-high suite score as a contamination hypothesis to disprove, not a win. This is the agent-era version of the IE map's "under-specified normalization is the biggest source of bogus scores" — the biggest source of bogus *agent* scores is a leaked or hackable environment.

---

## References

[1] Chen M, Tworek J, Jun H, et al. [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374). arXiv:2107.03374, 2021. (Codex; HumanEval; the unbiased pass@k estimator.)

[3] Yao S, Shinn N, Razavi P, Narasimhan K. [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045). arXiv:2406.12045, 2024 (ICLR 2025). (Introduces pass^k; state-based grading. Code: [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), MIT.)

[4] Yao S, et al. τ-bench headline results (gpt-4o ≈ 61 % pass¹ retail, ≈ 35 % pass¹ airline, ≈ 25 % pass⁸ retail; < 50 % pass¹ is the cross-domain average). In [arXiv:2406.12045](https://arxiv.org/abs/2406.12045), 2024.

[6] Barres V, et al. [τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982). arXiv:2506.07982, 2025. (Adds telecom domain; dual-control Dec-POMDP; separates reasoning vs communication errors. Code: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench).)

[7] Chen L, Davis JQ, Hanin B, Bailis P, Stoica I, Zaharia M, Zou J. [Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems](https://arxiv.org/abs/2403.02419). NeurIPS 2024, arXiv:2403.02419. (Vote/Filter-Vote non-monotone in call count; analytical optimum. [NeurIPS proceedings](https://proceedings.neurips.cc//paper_files/paper/2024/hash/51173cf34c5faac9796a47dc2fdd3a71-Abstract-Conference.html).)

[8] Zaharia M, Khattab O, Chen L, Davis JQ, Miller H, Potts C, Zou J, Carbin M, Frankle J, Rao N, Ghodsi A. [The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/). Berkeley AI Research (BAIR) Blog, Feb 18 2024.

[9] Brown B, Juravsky J, Ehrlich R, Clark R, Le QV, Ré C, Mirhoseini A. [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787). arXiv:2407.21787, 2024. (Coverage scales as an exponentiated power law; verification is the bottleneck. [Project page](https://scalingintelligence.stanford.edu/pubs/large_language_monkeys/).)

[10] Anthropic. [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Anthropic Engineering, 2025/2026. (Grade outcomes over paths; isolated judge per dimension; Unknown escape hatch; 0 %-pass = broken task.)

[11] Jimenez CE, Yang J, Wettig A, et al. [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) ICLR 2024, arXiv:2310.06770. (2 294 issue/PR tasks, 12 Python repos; FAIL_TO_PASS + PASS_TO_PASS test grading. Package: [swebench](https://pypi.org/project/swebench/), MIT. [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/), OpenAI 2024.)

[12] Patil SG, et al. [The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of LLMs](https://proceedings.mlr.press/v267/patil25a.html). ICML 2025. (AST match + executable eval; leaderboard reports Cost (USD) and Latency. [Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html); package `bfcl-eval`, Apache-2.0.)

[13] Gorilla Team. [Berkeley Function Calling Leaderboard — README & data](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md). UC Berkeley Gorilla project.

[14] Reliability compounding (pⁿ over n dependent steps): synthesized from the compound-systems framing [8] and τ-bench's pass^k collapse [3, 4]; see [BAIR blog](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) and [arXiv:2406.12045](https://arxiv.org/abs/2406.12045).

[15] Erol MH, El B, Suzgun M, Yuksekgonul M, Zou J. [Cost-of-Pass: An Economic Framework for Evaluating Language Models](https://arxiv.org/abs/2504.13359). arXiv:2504.13359, 2025. (The economic denominator: expected monetary cost of a correct solution, E[cost] / P(success) → ∞ when P(success) = 0; and Frontier Cost-of-Pass, the minimum across available models and the human expert. Developed in full in `ek_11`.)

*(Additional sources cited inline as "[dossier]" — Hamel Husain's [LLM-as-judge guide](https://hamel.dev/blog/posts/llm-judge/index.html) and [evals-faq](https://hamel.dev/blog/posts/evals-faq/); Jung et al., [Trust or Escalate / Cascaded Selective Evaluation](https://arxiv.org/pdf/2407.18370), ICLR 2025; Zhou et al., [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/pdf/2410.21819), 2024; a [Survey on Evaluation of LLM-based Agents](https://arxiv.org/html/2503.16416v2), arXiv:2503.16416, 2025; OpenAI, [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/); [Cursor reward-hacking study coverage](https://www.marktechpost.com/2026/06/26/cursor-study-finds-reward-hacking-inflates-coding-agent-benchmark-scores-on-swe-bench-pro/); Inspect AI, [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai), MIT; LangChain, [agentevals](https://github.com/langchain-ai/agentevals), MIT — these are developed in full in `ek_08`–`ek_12` where they are load-bearing.)*
