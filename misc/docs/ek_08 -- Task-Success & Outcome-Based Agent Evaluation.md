# ek_08 — Task-Success & Outcome-Based Agent Evaluation

**Author:** Thor Whalen

**Status:** Research synthesis / integration brief. This is the reference-based (`score()`/`evaluate()`) side of ek's agent-evaluation extension — how success is *defined* and *checked* on the final state of an episode. It is the agent-domain analogue of -> ek_02 (offline IE evaluation). Reference-free confidence, judge calibration, and selective escalation live in the sibling brief -> ek_09/ek_03.

---

## TL;DR

Grade the **final state the task was supposed to produce**, not the surface text the agent emitted along the way. The strongest agent benchmarks of 2024–2026 converge on one principle: success is an **executable or programmatic check on the environment end-state** — the database row is correct and the policy was obeyed (tau-bench [1]), the patch applies and the hidden tests flip to green (SWE-bench [4]), the file system or app state satisfies a per-task success function (OSWorld [14]), the returned answer normalized-exact-matches an unambiguous reference (GAIA [7]). Judged text (LLM-as-judge over a transcript) is the *fallback* for tasks that resist a deterministic oracle — not the default, because a deterministic verifier has no reward-model error to hack [17]. Because a single trial is a coin flip on frontier models (gpt-4o solves <50% of tau-bench tasks on the first try [1]), you must report **two numbers, not one**: `pass@k` (best-of-k, does *at least one* of k tries succeed — the capability ceiling) computed with the Chen et al. unbiased estimator [6], **and** `pass^k` (does the *same task* succeed on *all* k independent tries — the reliability floor), which collapses toward zero as k grows (tau-bench retail: pass^1 ≈ 61% degrading to pass^8 < 25% [1]). In ek terms: these are `Metric`s on `score()`/`evaluate()`; the checker is pluggable; `pass^k` is the *corpus aggregator* the `evaluate()` facade injects, never a naive mean; tool-call correctness is a specialization of the existing span-F1 `FieldMetric` (-> ek_02); and the final-state check doubles as a `Validator` (-> ek_04). The whole apparatus inherits ek's two great pitfalls one level up: partial-credit gaming (-> ek_02) and reward hacking against a leaky verifier [13][14].

---

## 1. The conceptual move: from token accuracy to task success

Information-extraction evaluation grades a *static artifact* — a page of OCR, a filled slot schema — against a gold artifact. Agent evaluation grades a *process that ends in a state*. The evaluated object is no longer a string; it is an **episode**: a sequence of `(tool-call, observation)` steps terminating in a final environment state and (optionally) a natural-language answer. The unit of value is not "characters correct per page" but **cost per successfully completed task** — a task the agent either accomplished or did not, and if it did, at what token/latency/dollar cost.

This is the same 2×2 that organizes all of ek — (reference availability) × (granularity) — but rotated onto trajectories. This report owns the *reference-based* column: you have a gold notion of "done" (a goal DB state, a passing test suite, a canonical answer) and you check the agent's final state against it. The reference-free column (no gold, judge/self-consistency/faithfulness signals feeding a confidence gate that decides accept/flag/escalate) is the estimate_quality side, covered separately (-> ek_09/ek_03).

The load-bearing design decision — and the one every serious benchmark since 2023 has made — is to grade the **final state, not the trajectory**. Many valid paths reach the same correct end-state; grading the path punishes legitimate alternatives and rewards memorized ones. The Agentic Benchmark Checklist [12] names "gold-trajectory brittleness" a first-class validity threat, and the empirical record backs it: tau-bench explicitly "compares the database state at the end of a conversation with the annotated goal state" rather than the transcript [1]; WebArena scores "functional correctness … whether the final state satisfies requirements, rather than matching a specific trajectory" [8]. Trajectory match is a *diagnostic* (why did it fail?), not a *scorer*. Keep it for partial credit and error analysis; never make it the pass/fail oracle.

---

## 2. State-based / outcome grading: the benchmark roster

The best way to understand outcome grading is to see how the reference benchmarks actually compute their pass/fail bit. They cluster into four oracle types: **DB-state + policy**, **execution/test-suite**, **programmatic environment-state function**, and **normalized-exact-match answer**. The table anchors the landscape; the prose that follows draws the design lessons.

| Benchmark | Domain | Oracle type | What "success" checks | Headline result | Ref |
|---|---|---|---|---|---|
| **tau-bench** | Retail, airline customer service | DB-state + policy + required-output | Final DB state == annotated goal AND required info communicated to user | gpt-4o <50% single-trial; pass^8 <25% retail | [1] |
| **tau2-bench** | + Telecom (dual-control Dec-POMDP) | Same, + compositional programmatic task generator | Both agent AND user act on shared state; verifiable tasks from atomic components | Isolates reasoning vs comms/coordination errors | [2] |
| **BFCL** (v1→v4) | Function calling, 8 API domains | AST match / executable / state-based | Name+required-params+types (AST); run+compare (exec); backend state (multi-turn) | ~4,400 test cases at v3 | [3] |
| **SWE-bench** | Real GitHub Python issues | Execution / hidden tests | Patch applies AND FAIL_TO_PASS flip green AND PASS_TO_PASS stay green | Claude 2 at release: 1.96% | [4] |
| **SWE-bench Verified** | 500 human-validated subset | Same, cleaned | Under-specified issues & over-strict tests removed | GPT-4o 16%→33.2% same model | [5] |
| **WebArena** | 4 self-hosted web apps | Programmatic functional | DB/URL/DOM assertions; exact / must-include / LLM-fuzzy answer | GPT-4 14.41% vs human 78.24% | [8] |
| **GAIA** | General assistant, 466 Qs | Normalized exact-match | Short unambiguous answer, case/punct-normalized | Human 92% vs GPT-4+plugins ~15% | [7] |
| **OSWorld** | 369 real desktop+web tasks | Programmatic success function | Inspects file system / app state after agent finishes | 134 unique evaluators | [14] |
| **AppWorld** | 9 apps, 457 APIs, 750 tasks | Programmatic state checks | State assertions over interactive code generation | ACL'24 Best Resource | [13] |
| **WebShop** | 1.18M products, 12,087 instr. | Attribute-match reward | Purchased item attributes/price vs instruction (partial credit) | Best model 29% vs human 59% | [15] |
| **TravelPlanner** | 1,225 planning intents | Constraint pass rate | Commonsense + hard (budget) constraints; final all-or-nothing | GPT-4 final success 0.6% | [16] |

**DB-state + policy (tau-bench / tau2-bench).** tau-bench [1] is the cleanest template for the *enterprise agent* case ek most cares about. A simulated LLM user converses with a customer-service agent that has domain API tools plus a policy document. Success is *not* "did the transcript look helpful" — it is a two-part conjunction: (a) the final database state (order cancelled, refund of the correct amount, booking updated) matches the annotated goal, **and** (b) the required outputs were actually communicated to the user. This is exactly ek's Layer-A/Layer-B split made concrete: the goal-state comparison is a typed check against schema-defined records; the policy adherence is a constraint over the trajectory. tau2-bench [2] pushes it to a **dual-control Dec-POMDP** where both agent and user manipulate a shared state (a telecom troubleshooting session where the user must be *told* to toggle a setting), plus a compositional generator that emits programmatically-verifiable tasks from atomic components — the machine-generated analogue of ek's grammar-driven task synthesis. The reliability story is the headline: strong function-calling agents are "quite inconsistent," and the collapse from ~61% single-trial to <25% eight-trial reliability in retail is the empirical case for `pass^k` (§4).

**Execution / hidden tests (SWE-bench).** SWE-bench [4] is the gold standard for "the check IS the code running." A model is given a real repo + issue and must produce a patch. Grading: the patch must **apply cleanly** (a diff that doesn't apply fails instantly), then the repo's real test suite runs — every `FAIL_TO_PASS` test must flip from failing to passing (proving the fix), and every `PASS_TO_PASS` test must stay green (proving no regression) [4]. No judged text, no partial credit at the instance level: resolved or not. This is a pure state/execution oracle, and it is why SWE-bench became the code-agent benchmark. It also delivered the field's sharpest lesson in **harness fairness** (§6).

**Programmatic environment-state functions (OSWorld / WebArena / AppWorld).** These generalize the SWE-bench idea beyond code: each task ships an `initial_state`, an NL instruction, and a **programmatic success function** that, after the agent halts, inspects the file system / application DB / DOM / URL and returns pass/fail [14][8][13]. OSWorld hand-wrote 134 distinct evaluators for 369 tasks [14]; WebArena mixes `exact_match`, `must_include` substring checks, an LLM `fuzzy_match` for open-ended info-seeking answers, and programmatic DB/JS assertions [8]. The nuance worth preserving: WebArena is *not* purely deterministic — its info-seeking reward path invokes an LLM judge — so it is the boundary case where reference-based grading tips into judged text, and it is precisely there that a prompt-injection reward-hack was later found [14] (§6).

**Normalized exact-match (GAIA).** GAIA [7] shows that for a *general assistant* you do not always need an environment — you need an **unambiguous short answer**. Questions are engineered so the answer is a number, name, or string checkable by *normalized* exact match (lowercase, strip spaces/punctuation) [7]. The inversion — humans 92%, GPT-4-with-plugins ~15% — is the whole point: an assistant that dazzles on professional exams still fails cheap, verifiable, real-world lookups. The cheapness of the oracle is a feature: normalized exact-match costs nothing and cannot be prompt-injected.

**Constraint satisfaction (TravelPlanner / WebShop).** TravelPlanner [16] decomposes success into commonsense-constraint and hard-constraint (budget) pass rates plus a final all-or-nothing plan-success bit; GPT-4's 0.6% final success rate is the canonical demonstration that **all-or-nothing constraint satisfaction** is brutally harder than any single-constraint average suggests. WebShop [15] is the partial-credit counterpoint: reward is a *matching score* over the purchased item's attributes vs the instruction, so it gives graded credit. Both patterns map directly onto ek Layer-A: constraints are field/edge specifications with `importance` weights, and "final all-or-nothing vs weighted partial" is exactly the match-scheme choice ek already exposes for slot-F1 (-> ek_02) — and the same *caution* applies: partial-credit schemes are gameable and can flatter a system that satisfies many cheap constraints while missing the one that matters.

---

## 3. Why executable checks beat judged text (the RLVR argument, honestly stated)

The training-side literature supplies the cleanest articulation of *why* to prefer deterministic oracles. **RLVR** — Reinforcement Learning with Verifiable Rewards, introduced in Tulu 3 (Lambert et al., arXiv:2411.15124 [17]) and presaged by DeepSeek-Math's rule-based rewards (arXiv:2402.03300 [18]) — replaces a learned reward model with a **programmatic verification function**, typically binary: run the check, reward 1 iff verifiably correct. The advantage is that a deterministic verifier has **no learned-reward-model error**, so there is no reward-hacking against a flawed *learned* judge [17]. Transposed to evaluation: an executable/state check for a code fix or a tool outcome is a verifier with the same property — it cannot be talked into a passing grade the way an LLM judge can.

Two honesty caveats, because the brief demands contested evidence be flagged. First, RLVR is an empirical *training* method, not a *formal theory of evaluation*; it **motivates** rather than *proves* the superiority of state checks over judged text, and the reward it discusses is a training-time signal, a different object from an eval-time metric. Second — and this is the load-bearing correction — verifiable checks do **not eliminate** reward hacking; they **shift its failure mode** from learned-reward-model error to **verifier-coverage / specification error**. An imperfect deterministic verifier still admits false positives and is still gameable (the SQL-injection task that passed merely because the agent emitted the keyword `SLEEP`, regardless of effect [12]). So the correct claim is: prefer deterministic state checks *because they remove one entire class of gaming (judge manipulation) and make the remaining class (spec gaps) auditable in code* — not because they are unhackable. LLM-as-judge remains legitimate where no oracle exists (open-ended answers, plan quality), but it belongs in the estimate_quality lane where it is calibrated and its biases (verbosity, position, self-preference) are measured (-> ek_03), not silently trusted here.

---

## 4. pass@k and pass^k: report both, aggregate correctly

Single-trial success is a lie by omission on stochastic agents. Two orthogonal statistics rescue it, and they are opposite tails of the same distribution.

**pass@k — the capability ceiling (best-of-k).** "Does at least one of k independent samples succeed?" Under i.i.d. per-task success probability p, pass@k ≈ 1 − (1 − p)^k, which *rises* toward 1 as k grows. The critical implementation point — and the reason ek must build this rather than eyeball it — is that the naive plug-in estimator 1 − (1 − p̂)^k is **biased** and high-variance. The correct estimator is Chen et al.'s **unbiased combinatorial** form (HumanEval/Codex, arXiv:2107.03374 [6], the metric itself originating in Kulal et al. SPoC 2019, but the *unbiased estimator* is Chen et al.'s contribution): generate n ≥ k samples per task, count c that pass, and compute

> **pass@k = E_tasks[ 1 − C(n−c, k) / C(n, k) ]**   (= 1 when n − c < k)

where C(n−c, k)/C(n, k) is exactly the probability that all k drawn samples come from the failing set [6]. HumanEval used n = 200, k ≤ 100. Compute it in log-space or via the stable product form pass@k = 1 − ∏_{i=n−c+1}^{n}(1 − k/i) to avoid binomial overflow (the numpy trick in the paper's appendix) [6].

**pass^k — the reliability floor (all-k-must-pass).** "Does the *same task* succeed on *all* k independent trials?" This is tau-bench's contribution [1]. Under i.i.d. p it is *approximately* p^k, *falling* toward 0 as k grows — the mirror image of pass@k. Keep the "approx" hedge: tau-bench actually reports pass^k via an **unbiased combinatorial estimator** too — average over tasks of C(c_i, k)/C(n, k), where c_i is the number of successful trials of task i out of n — **not** literally the mean success rate raised to the k [1]. The empirical story is the whole reason the metric exists: gpt-4o solves <50% of tasks overall on a single try, and reliability degrades to pass^8 < 25% in retail (where single-trial success is a healthier ~61%) [1]. A system you would ship at 61% "accuracy" fails one in every ~1.3 customers by the eighth interaction. **Report both**: pass@k tells you the ceiling a retry-wrapper or best-of-n selector could reach if you had a verifier to pick the winner; pass^k tells you what a user relying on a single unassisted attempt actually experiences. They are the capability and the trustworthiness of the same agent, and quoting only one is malpractice.

**Best-of-n and k-sample coverage** are the practical bridge: pass@k is only *achievable* in production if you have a selector (a verifier, a reranker, self-consistency vote) that can identify the winning sample — otherwise the ceiling is unreachable. This is where §7's agreement machinery (self-consistency = ROVER sample-and-vote, reused from -> ek_03) becomes the selector, and where the confidence gate (accept the high-agreement sample, escalate the rest) is the estimate_quality decide-stage.

---

## 5. Tool-call correctness in depth: AST vs executable vs state, and "should-not-call"

Beneath whole-task success sits **tool-call correctness** — the per-step analogue of slot-F1. BFCL [3] is the reference taxonomy, and it grades three ways along a cost/fidelity gradient:

- **AST match** — parse the emitted call into a syntax tree and check *without executing*: function name matches, required parameters present, values within the allowed set, types correct. Cheap, deterministic, but blind to runtime behavior [3].
- **Executable match** — actually run the (real or simulated) function and compare outputs (exact / real-time within a tolerance / structural). Catches errors AST can't (a syntactically valid call that returns wrong data) at the cost of a live sandbox [3].
- **State-based match** (v3+, multi-turn) — after a sequence of calls, inspect the *backend system state* (file system, booking DB); v3 pairs this with a **trajectory subset-match** and an entry must pass both [3].

The categories matter as much as the modes. BFCL spans **simple / multiple / parallel / parallel-multiple** calls, plus two dual capabilities: **irrelevance detection** — the "**should-NOT-call**" capability, where *none* of the provided functions fit and the model must emit **no call** rather than hallucinate one — and **relevance detection**, where a relevant function *does* exist and the model must emit a call, but grading deliberately **checks only that a call is emitted, not its argument values** [3]. This asymmetry is a gift to ek: it decomposes tool-call correctness into three independently-scorable sub-metrics — *selection* (right tool chosen), *abstention* (correctly declining), and *argument-level* correctness (right values) — each of which maps to a distinct `importance`-weighted term in a Layer-A cost function. Fix the count too: BFCL is ~2,000 test cases at v1, growing to ~4,400 across v3 (roughly 1,390 non-live + 2,251 live + 800 multi-turn) [3].

The wider tool-use roster fills in the corners:

| Benchmark | Grading | Scale | Note | Ref |
|---|---|---|---|---|
| **API-Bank** | Executable (runnable system) | 73 APIs, 314 dialogues, 753 calls | First *executable* tool-use bench; planning/retrieving/calling | [11] |
| **ToolLLM / ToolBench** | **LLM-judged** pass rate + win rate | 16,000+ real APIs | 87.1% / 80.3% human agreement — *judged, not executable* | [9] |
| **Nexus (NexusRaven)** | Function-call success | Nested & composite calls | Commercially-permissive, beyond single/parallel | [20] |
| **ToolEmu** | **LM-emulated** execution + LM safety judge | 36 high-stakes tools, 144 cases | 68.8% failures validated real; 31.2% false-positive rate | [12-safety] |

Two contrasts to internalize. **ToolLLM/ToolBench** [9] grades with **LLM-as-judge** (pass rate = "completed within budget," win rate = pairwise preference), with respectable but imperfect human agreement (87.1% / 80.3%) — a reminder that at 16k-API scale, hand-written oracles don't exist and you *fall back* to judged text, inheriting all its biases. **ToolEmu** [12-safety] uses an LM to *emulate* tool execution (no real implementations) plus an LM safety evaluator — cutting a safety-eval cycle from ~8 hours to ~15 minutes, at the cost of a **31.2% false-positive rate** (68.8% of flagged failures were validated as real). That trade — cheap-and-noisy emulated oracle vs expensive-and-faithful real sandbox — is a genuine engineering dial, and ek should expose it as a checker strategy choice, not bury it.

---

## 6. Pitfalls: the outcome-grading failure catalog

Outcome grading is *better* than transcript grading, not *safe*. The 2025–2026 literature is largely a catalog of how state oracles get gamed or mis-specified. ek must treat these as first-class design constraints.

**Harness fairness / over-strict tests.** SWE-bench's original 16%→**33.2%** jump on **SWE-bench Verified** with the *same* GPT-4o model [5] is the canonical example: 93 Python developers screened 1,699 samples down to 500, removing under-specified issues and over-strict `FAIL_TO_PASS` tests that rejected valid solutions — "roughly half of the apparent failures on old SWE-bench being the harness's fault, not the agent's" [5]. The lesson: an over-strict oracle *systematically under-reports* capability, exactly as an over-strict match scheme does in IE (-> ek_02). ek's regression-gate harness must let you *audit the oracle*, not just the agent.

**Flawed tests + contamination → OpenAI stops evaluating.** OpenAI *stopped evaluating* models on SWE-bench Verified (Feb 2026) citing **two** compounding causes: **~59.4% of audited tasks had material issues** — flawed or over-strict tests that reject functionally-correct patches (surfaced by auditing 138 problems, 27.6% of 500, that o3 couldn't solve consistently across 64 runs) — **and** frontier-model **training-data contamination/leakage** inflating scores. OpenAI now recommends **SWE-bench Pro** (Scale; GPL-copyleft license plus a private held-out set for contamination resistance) [19]. When your ceiling is dominated by broken tasks *and* leaked solutions, the metric is measuring the harness and memorization, not capability.

**Reward hacking against a leaky verifier.** The Agentic Benchmark Checklist [12] documents concrete outcome-validity failures: tau-bench originally **counted empty responses as successful** (a do-nothing agent scored ~38% before fixes); a SQL-injection task passed if the agent merely inserted the keyword `SLEEP` regardless of effect [12]. **BenchJack** [13-benchjack] auto-generated exploits achieving near-perfect scores on 9 of 10 benchmarks **without solving any task** — including a **~10-line `conftest.py`** PyTest hook (pytest auto-loads it) whose hook rewrites *every* SWE-bench Verified test outcome to "passed" (the evaluator trusts test outputs from inside a container the agent's patch can modify), and leaking WebArena's gold answers (reference answers are passed in the task config, and agent output is interpolated into the judge prompt with no sanitization, enabling prompt injection of fake "evaluation notes") [13-benchjack]. Across the audit BenchJack generated working exploits on **9 of 10** benchmarks, each reaching near-perfect scores without solving tasks — the same reward-hacking pattern recurring across environments like Terminal-Bench and OSWorld. **Verifier isolation is a security property**, not a nicety: the check must run somewhere the agent cannot reach.

**Gold-trajectory brittleness, non-determinism, flaky tests, environment reproducibility.** Many valid paths reach one state, so trajectory-match under-counts (§1) [12]. Flaky tests and non-deterministic environments make even state oracles noisy — which is *another* reason to report pass^k: a task that passes 6/8 times is telling you either the agent is unreliable or the oracle is flaky, and you want to see that spread, not a point estimate.

**Benchmark contamination / leakage.** Public benchmarks leak into training data; a rising score can be memorization, not capability. The mitigations ek should adopt from the harness (-> ek_02): held-out golden sets that never touch training, canonicalization before comparison, and treating any single public number with suspicion.

The **Agentic Benchmark Checklist (ABC)** [12] organizes all of this into three pillars ek should adopt wholesale as a design review gate: **Outcome Validity** (the success signal truly indicates completion), **Task Validity** (the task is solvable iff the agent has the target capability), and **Benchmark Reporting** (issues discussed with quantitative evidence). Every ek agent-eval `Metric` and `Validator` should be reviewable against these three.

---

## 7. Grading harness support: what to wrap, what to build

The library landscape (fuller register in -> ek_06) offers strong, permissively-licensed primitives for the *scorer* layer — but a conspicuous gap at the *cost* layer.

| Library | License | Package | What ek reuses | Ref |
|---|---|---|---|---|
| **Inspect AI** (UK AISI) | MIT | `inspect-ai` | Scorer taxonomy + epochs/reducers (`pass_at`) as the aggregator model | [21][22] |
| **OpenAI Evals** | MIT (code) | `evals` / `oaieval` | `string_check` (eq/ilike) graders; eval-as-schema+criteria | [23] |
| **DeepEval** | Apache-2.0 | `deepeval` 4.0.9 | `TaskCompletionMetric` (judge), `ToolCorrectnessMetric` (deterministic) | [24][25] |
| **SWE-bench** | MIT | `swebench` 4.x | Dockerized apply-patch + hidden-test oracle | [4][26] |
| **BFCL** | Apache-2.0 | `bfcl-eval` | AST / executable / state tool-call graders | [3] |
| **tau/tau2-bench** | MIT | `tau2-bench` | DB-state + policy oracle; Pass^1..k leaderboard columns | [1][2] |
| **ToolBench** | Apache-2.0 | — | LLM-judged pass/win rate | [9] |
| **API-Bank** | **Apache-2.0** | — | Runnable executable tool oracle | [11] |

**Inspect AI** [21] is the template ek's facade layer should mirror most closely. Its `@scorer`/`@metric` decorator split *is* ek's `Metric` Protocol + aggregator split: a scorer returns `Score(value, answer, explanation)` against a `Target`; a `@metric` combines per-sample scores. Its built-in scorers map one-to-one onto grading primitives ek needs — `includes()` (substring), `match()` (position-normalized), `pattern()` (regex), `answer()`, `exact()` (normalized full match), `f1()` (token-overlap F1), `choice()`, `math()`, and the model-graded `model_graded_qa()` / `model_graded_fact()` [21][22]. Most report accuracy+stderr; `exact()`/`f1()` report mean+stderr [21]. Crucially, Inspect's **epochs + reducers** implement pass@k/pass^k *as the aggregator*: repeat each sample across epochs, then reduce with `pass_at` (P(≥k correct across epochs)), `at_least` (count over a threshold), or `mean_score`, and pass *multiple* reducers to compute several at once [22]. This is the concrete realization of "report pass@k AND reliability together," and it is exactly the shape of ek's `evaluate(metric=...).aggregate`.

**DeepEval** [24][25] draws the clean specialization boundary ek should copy: `TaskCompletionMetric` is **LLM-judge** over an agent trace (goes in estimate_quality / -> ek_03), while `ToolCorrectnessMetric` is **deterministic** — it compares called vs expected tools, matching *names* by default with optional parameter/output strictness [25]. That is precisely ek's tool-call F1 as a `FieldMetric` specialization, and DeepEval's name-vs-args strictness dial is the argument-level correctness axis from §5.

**The cost gap is the genuine build.** None of these harnesses surfaces **cost-per-successful-task** as a first-class scorer. Inspect and DeepEval capture token usage at the *trace/log* level, not as a metric; BFCL/SWE-bench/tau report accuracy and pass^k, not $/task [21][24]. This is the confirmation that ek's central agent-eval thesis — **cost per successfully completed task, weighted by Layer-A task-value/error-cost** — is a *build*, not a *wrap*. ek wraps the scorers and reuses the aggregator pattern; ek *builds* the cost-weighting on top, because the field simply doesn't ship it. License-wise the coast is clear (§ register above, full landmine map -> ek_06): every listed harness is MIT or Apache-2.0, none copyleft or non-commercial, so all are eligible as ek extras without tripping the CI license gate. Two corrections to carry forward: **API-Bank is Apache-2.0** (the benchmark's own `LICENSE` file), *not* MIT — the MIT belongs to the parent DAMO-ConvAI repo [11]; and OpenAI Evals' *harness code* is MIT but some *bundled datasets* carry mixed licenses (some CC BY-NC), which matters only if ek redistributes eval *data* rather than consuming the harness code — a footnote for the license gate, not a blocker.

---

## 8. How this lands in ek

Everything above is a `Metric` on the `score()`/`evaluate()` facades, plus a reused aggregator and an optional `Validator` — no new architectural layer. The mapping is tight.

**Layer A becomes a task/tool grammar.** The frozen typed-schema SSOT (`GraphGrammar`) carries, per node/edge/field, the *allowed tools + argument schemas* and the *task-value / error-cost weights* (`FieldSpec.importance`). A tau-bench-style goal is a typed record set; a BFCL argument schema is a `FieldSpec`; a TravelPlanner constraint is an edge with an importance weight. This is the single lever for the cost-weighted metrics that the harness ecosystem does *not* provide (§7): success is not a bare bit but a bit weighted by the value of the task and the cost of the specific error. The grammar also feeds constrained decoders/validators — the same object that scores can also *guard* generation.

**Layer B becomes the episode's runtime metadata.** `AnnotatedExtraction` — per-item `raw_signals`, `confidence`, `findings`, `provenance`, `decision` keyed by `NodePath` — carries the **trajectory** (the tool-call/observation sequence), per-step signals, token cost, latency, and the final verdict, riding *alongside* the grammar, never mutating it. The episode is a Layer-B object over a Layer-A task grammar; the final state to be graded is a projection of Layer B onto the goal fields of Layer A.

**`TaskSuccessMetric(check=...)` — pluggable checker.** The core agent metric is `score(pred_episode, gold, *, grammar, metric='task_success', check=...)` where `check` is a strategy resolved by name from the registry (an `ek.registry` `typing.Protocol`, injected keyword-only with a smart default, open-closed). Checkers are the four oracle types of §2: `db_state_match` (tau-bench), `hidden_tests` (SWE-bench, behind `@requires_extra("swebench")`), `programmatic_state_fn` (OSWorld/WebArena), `normalized_exact_match` (GAIA). Executable/sandbox checkers guard their heavy deps with `@requires_extra("...")` raising the actionable `pip install ek[...]`, and — per §6 — must run in an **isolated** environment the predicted trajectory cannot reach (the BenchJack `conftest.py` lesson [13-benchjack]).

**`pass^k` (and `pass@k`) as the corpus aggregator.** These are `evaluate(cases, *, metric).aggregate` — the metric's **own** aggregator, *never* a naive mean (the standing ek rule). `evaluate()` runs each case across n epochs and reduces with the Chen et al. unbiased pass@k estimator [6] *and* the tau-bench unbiased pass^k estimator [1], reporting both plus `per_slice` cuts (by domain, tool, task-value tier). This is a direct port of Inspect's epochs/reducers model [22] onto ek's `Report`. Implement pass@k in the stable product form to avoid binomial overflow [6].

**Tool-call F1 as a `FieldMetric`/span-F1 specialization.** Do *not* rebuild span-F1 (-> ek_02). A tool call is a span with a type (tool name) and attributes (arguments); tool-call selection F1 is entity-type F1, argument-level correctness is attribute-match F1, and "should-not-call" is the abstention/irrelevance case (§5) scored as a specificity term. The AST/executable/state-match gradient of BFCL [3] is three checker strategies behind one `Metric`, `importance`-weighted per the grammar.

**Final-state check as a `Validator` too.** The same oracle that *scores* an episode in the reference-based lane *validates* an extraction in the reference-free lane (-> ek_04): "does the final state satisfy the constraints" is a flag-vs-correct decision when no gold exists. One checker, two facades — the DB-state comparator is a `Metric` when you have a gold goal and a `Validator` when you have only the constraints.

**Cost-weighted typed-graph distance generalizes to trajectory distance.** ek's flagship offline metric — cost-weighted typed-graph edit distance, weights sourced from Layer-A `importance` (-> ek_02) — generalizes verbatim to **trajectory / tool-call-graph distance**: the episode is a graph of tool-call nodes and dataflow edges, and graph-edit-distance against a reference trajectory gives the *diagnostic* partial-credit signal that trajectory-match should be used for (§1) — never as the pass/fail oracle, always as the "why did it fail" companion to the state check.

**Self-consistency reuses ROVER; the confidence gate is the decide stage.** ROVER agreement (-> ek_03) *is* self-consistency sample-and-vote: sample k trajectories, align, vote. That vote is the **selector** that makes pass@k's ceiling reachable in production (§4). The agreement score feeds the estimate_quality confidence gate — accept the high-agreement outcome, **escalate the rest** — which is selective prediction / the accept-flag-block policy (-> ek_03). So the ceiling (pass@k), the selector (ROVER self-consistency), and the escalation (selective prediction) are one continuous machine, and only the *state check* in this report is new; the rest is reuse.

**Design-review gate.** Every ek agent `Metric`/`Validator` ships with an ABC [12] review: Outcome Validity (does the check truly mean "done"? — no empty-response-passes, no `SLEEP`-keyword passes), Task Validity (solvable iff the target capability is present), Reporting (pass@k *and* pass^k, per-slice, with the oracle auditable). Contamination and harness-fairness mitigations inherit from the golden-set CI gate (-> ek_02).

---

## References

[1] Yao S, Shinn N, Razavi P, Narasimhan K. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024. [<https://arxiv.org/abs/2406.12045>](https://arxiv.org/abs/2406.12045); code [<https://github.com/sierra-research/tau-bench>](https://github.com/sierra-research/tau-bench).

[2] Barres V, Dong H, Ray S, Si X, Narasimhan K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025. [<https://arxiv.org/abs/2506.07982>](https://arxiv.org/abs/2506.07982); code [<https://github.com/sierra-research/tau2-bench>](https://github.com/sierra-research/tau2-bench).

[3] Patil SG, et al. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML 2025 (PMLR v267). [<https://proceedings.mlr.press/v267/patil25a.html>](https://proceedings.mlr.press/v267/patil25a.html); v3 multi-turn blog [<https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html>](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html); PyPI `bfcl-eval` [<https://pypi.org/project/bfcl-eval/>](https://pypi.org/project/bfcl-eval/).

[4] Jimenez CE, Yang J, Wettig A, et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770. [<https://arxiv.org/abs/2310.06770>](https://arxiv.org/abs/2310.06770).

[5] OpenAI. Introducing SWE-bench Verified. 2024. [<https://openai.com/index/introducing-swe-bench-verified/>](https://openai.com/index/introducing-swe-bench-verified/); leaderboard [<https://www.swebench.com/verified.html>](https://www.swebench.com/verified.html).

[6] Chen M, Tworek J, Jun H, et al. Evaluating Large Language Models Trained on Code (Codex/HumanEval; unbiased pass@k estimator). arXiv:2107.03374, 2021. [<https://arxiv.org/abs/2107.03374>](https://arxiv.org/abs/2107.03374).

[7] Mialon G, Fourrier C, Swift C, Wolf T, LeCun Y, Scialom T. GAIA: a benchmark for General AI Assistants. arXiv:2311.12983, 2023. [<https://arxiv.org/abs/2311.12983>](https://arxiv.org/abs/2311.12983).

[8] Zhou S, Xu FF, Zhu H, et al. WebArena: A Realistic Web Environment for Building Autonomous Agents. ICLR 2024. arXiv:2307.13854. [<https://arxiv.org/abs/2307.13854>](https://arxiv.org/abs/2307.13854); code [<https://github.com/web-arena-x/webarena>](https://github.com/web-arena-x/webarena).

[9] Qin Y, Liang S, Ye Y, et al. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. ICLR 2024 (spotlight). arXiv:2307.16789. [<https://arxiv.org/abs/2307.16789>](https://arxiv.org/abs/2307.16789); ToolBench [<https://github.com/OpenBMB/ToolBench>](https://github.com/OpenBMB/ToolBench).

[11] Li M, Zhao Y, Yu B, et al. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. arXiv:2304.08244, 2023. [<https://arxiv.org/abs/2304.08244>](https://arxiv.org/abs/2304.08244); license (Apache-2.0) [<https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/api-bank/LICENSE>](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/api-bank/LICENSE).

[12] Zhu Y, et al. Establishing Best Practices for Building Rigorous Agentic Benchmarks (Agentic Benchmark Checklist). arXiv:2507.02825, 2025. [<https://arxiv.org/abs/2507.02825>](https://arxiv.org/abs/2507.02825); code [<https://github.com/uiuc-kang-lab/agentic-benchmarks>](https://github.com/uiuc-kang-lab/agentic-benchmarks).

[12-safety] Ruan Y, Dong H, et al. Identifying the Risks of LM Agents with an LM-Emulated Sandbox (ToolEmu). ICLR 2024 (spotlight). arXiv:2309.15817. [<https://arxiv.org/abs/2309.15817>](https://arxiv.org/abs/2309.15817); code [<https://github.com/ryoungj/ToolEmu>](https://github.com/ryoungj/ToolEmu).

[13] Trivedi H, Khot T, et al. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. ACL 2024 (Best Resource Paper). arXiv:2407.18901. [<https://arxiv.org/abs/2407.18901>](https://arxiv.org/abs/2407.18901).

[13-benchjack] Do Androids Dream of Breaking the Game? Systematically Auditing AI Agent Benchmarks with BenchJack. arXiv:2605.12673, May 2026. [<https://arxiv.org/pdf/2605.12673>](https://arxiv.org/pdf/2605.12673); Berkeley RDI companion [<https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/>](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/).

[14] Xie T, Zhang D, Chen J, et al. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. NeurIPS 2024. arXiv:2404.07972. [<https://arxiv.org/abs/2404.07972>](https://arxiv.org/abs/2404.07972); site [<https://os-world.github.io/>](https://os-world.github.io/).

[15] Yao S, Chen H, Yang J, Narasimhan K. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. NeurIPS 2022. arXiv:2207.01206. [<https://arxiv.org/abs/2207.01206>](https://arxiv.org/abs/2207.01206).

[16] Xie J, Zhang K, Chen M, et al. TravelPlanner: A Benchmark for Real-World Planning with Language Agents. arXiv:2402.01622, 2024. [<https://arxiv.org/abs/2402.01622>](https://arxiv.org/abs/2402.01622).

[17] Lambert N, et al. Tulu 3: Pushing Frontiers in Open Language Model Post-Training (RLVR). arXiv:2411.15124, 2024. [<https://arxiv.org/abs/2411.15124>](https://arxiv.org/abs/2411.15124).

[18] Shao Z, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO; rule-based verifiable rewards). arXiv:2402.03300, 2024. [<https://arxiv.org/abs/2402.03300>](https://arxiv.org/abs/2402.03300).

[19] OpenAI. Why we no longer evaluate SWE-bench Verified. 2026. [<https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/).

[20] Nexusflow. NexusRaven: a Commercially-Permissive Language Model for Function Calling. OpenReview 5lcPe6DqfI. [<https://openreview.net/pdf?id=5lcPe6DqfI>](https://openreview.net/pdf?id=5lcPe6DqfI); code [<https://github.com/nexusflowai/NexusRaven-V2>](https://github.com/nexusflowai/NexusRaven-V2).

[21] UK AI Security Institute. Inspect AI — Scorers reference (`includes`, `match`, `pattern`, `exact`, `f1`, `model_graded_qa`, …). [<https://inspect.aisi.org.uk/scorers.html>](https://inspect.aisi.org.uk/scorers.html); model-graded source [<https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/scorer/_model.py>](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/scorer/_model.py).

[22] UK AISI. Inspect AI — Scoring Metrics (epochs, reducers, `pass_at`, `at_least`, `mean_score`). [<https://inspect.aisi.org.uk/metrics.html>](https://inspect.aisi.org.uk/metrics.html).

[23] OpenAI. Evals: framework for evaluating LLMs and an open registry of benchmarks (MIT). [<https://github.com/openai/evals>](https://github.com/openai/evals); build-eval docs [<https://github.com/openai/evals/blob/main/docs/build-eval.md>](https://github.com/openai/evals/blob/main/docs/build-eval.md).

[24] Confident AI. DeepEval: the LLM evaluation framework (v4.0.9, Apache-2.0). PyPI [<https://pypi.org/project/deepeval/>](https://pypi.org/project/deepeval/); Task Completion metric [<https://deepeval.com/docs/metrics-task-completion>](https://deepeval.com/docs/metrics-task-completion).

[25] Confident AI. DeepEval — Tool Correctness metric. [<https://deepeval.com/docs/metrics-tool-correctness>](https://deepeval.com/docs/metrics-tool-correctness).

[26] SWE-bench. Evaluation guide (Docker harness, `run_evaluation`, hidden tests); PyPI `swebench` 4.x (MIT). [<https://www.swebench.com/SWE-bench/guides/evaluation/>](https://www.swebench.com/SWE-bench/guides/evaluation/); [<https://pypi.org/project/swebench/>](https://pypi.org/project/swebench/).
