# ek_11 — Cost per Successful Task: The Economics of Agent Evaluation

**Author:** Thor Whalen

**Status:** Research synthesis / integration brief. This is the *objective-function* report of ek's agent-evaluation extension — how to measure and optimize **expected cost per successfully-completed task**, and how to *evaluate* the routing / cascade / test-time-compute levers that move it. It sits underneath the capability reports: -> ek_08 defines and checks task success (`score()`/`evaluate()`), -> ek_09/ek_03 owns the reference-free confidence gate that a cascade's escalation *is*. This report makes **cost** a first-class Layer-B quantity and gives `evaluate()` an economic aggregator.

---

## TL;DR

**The denominator is a successfully completed task, not a token.** Per-token price is the wrong unit of account for an agent: a "cheaper" model that fails more often, retries more, escalates more, or emits 22% more output tokens to reach the same answer is a *cost regression*, and a per-token leaderboard will hide that. The load-bearing metric is **cost-of-pass** — the expected monetary cost to obtain one correct solution, `CoP = C_m(p) / R_m(p)` (expected cost of an attempt over probability of success), which goes to **infinity** when success rate is zero, correctly signaling infeasibility [1][2]. Report it *jointly* with reliability (two distinct compounding effects, not one: *within* an episode, end-to-end success compounds as `p^n` over n dependent steps; *across* k independent full-task retries, `pass^k ≈ p^k` is the reliability floor -> ek_08) and latency (TTFT + throughput), because these three axes trade against each other and a single scalar always lies. Two empirical facts reorganize the whole design: (1) test-time compute has an **optimum, not a ceiling** — Vote/Filter-Vote accuracy rises *then falls* as you add LLM calls, because more calls help easy queries and hurt hard ones [3], and compute-optimal allocation by difficulty is >4× more efficient than best-of-N and can beat a 14×-larger model under a matched FLOPs budget [4]; (2) the routing/cascade levers that exploit this (FrugalGPT up to 98% cost cut [5], RouteLLM >85% on MT-Bench at 95% of GPT-4 quality [6][7]) are themselves objects you must **evaluate**, and the right way to evaluate a router is the area under its non-decreasing cost-quality convex hull — RouterBench's AIQ [8]. In ek: cost is Layer-B episode metadata; `CostPerSuccessMetric` is an `evaluate()` aggregator returning `E[cost]/P(success)`; `FieldSpec.importance` generalizes to task value / error cost so the *same* cost-weight lever drives the economics; router evaluation is `evaluate()` with an AIQ-style Pareto aggregator plus a `regression_gate` on the frontier (-> ek_08 harness); and the price catalog is a small pluggable SSOT (LiteLLM's `model_prices` [9], OpenRouter `/models` fallback [10]), never a hardcode.

---

## 1. Why the denominator changed

Classical IE evaluation grades a static artifact and, when it costs anything, the cost is a fixed infrastructure line item — a page went through an OCR engine, the engine's amortized cost per page is roughly constant. Agent evaluation inverts this. The evaluated object is an **episode** (a trajectory of `(tool-call, observation)` steps ending in a final state -> ek_08), and the cost of producing that episode is a *runtime random variable*: it depends on how many tool calls the agent made, how many of them failed and were retried, how long its reasoning traces ran, whether a verification pass fired, and — crucially — whether the whole thing even succeeded. Two agents with identical single-attempt accuracy can differ 10× in what it actually costs to get a correct answer out of them.

So the unit of account is not "dollars per 1M tokens" but **dollars per successfully completed task**. The distinction is not pedantic. Kapoor et al.'s *AI Agents That Matter* [11] is the empirical indictment: accuracy-only agent leaderboards create perverse incentives because "SOTA agents are needlessly complex and costly." On HumanEval, a simple *warming* GPT-4 baseline hit 93.2% at a total cost of about $2.45, while the LATS agent reached only 88.0% at **$134.50**, and LDB matched 93.3% at $6.36 — and, damningly, "the cost of running these agents isn't a top-line metric reported in any of these papers" [11]. The complexity bought nothing; the leaderboard rewarded it anyway because it did not look at the second axis. The paper's prescription is exactly ek's: report agents on an **accuracy-vs-cost Pareto frontier**, and test dumb baselines first.

The economic framing has a formal home. **Cost-of-Pass** (Erol, El, Suzgun, Yuksekgonul, Zou; arXiv:2504.13359, ICLR 2026) [1][2] defines, per problem `p` and model `m`, `v(m,p) = C_m(p) / R_m(p)` where `C_m(p)` is the expected monetary cost of one inference attempt and `R_m(p)` is the probability the attempt is correct — "the expected monetary cost to obtain one correct solution." The formula is worth internalizing for three reasons. First, it *folds the failure rate into the denominator*: halving your per-attempt cost while doubling your failure rate is a wash, and the metric says so. Second, when `R_m(p) = 0` the cost-of-pass is **infinite**, "appropriately signaling infeasibility" [2] — a model that cannot solve a task at any price is not cheap, it is unusable, and a per-token metric that reports it as "$0.003/call" is actively misleading. Third, the paper defines a **frontier cost-of-pass** as the minimum CoP across all available models *or a human expert* (priced at the approximate cost of hiring one), which converts "is this agent worth deploying" into a concrete comparison against both the model menu and the human floor. Their empirical findings map cleanly onto routing policy: lightweight models are cheapest for basic quantitative tasks, large models win on knowledge-intensive tasks, and reasoning models win on complex quantitative tasks *despite* higher per-token cost — because per-token price is the wrong denominator [1]. Notably, inference-time methods (majority voting, self-refinement) "rarely justified their overhead" versus model-level innovation on their benchmarks, with the budget-aware TALE-EP a telling exception — a caveat I return to in §3.

There is a reference implementation (`mhamzaerol/Cost-of-Pass` [2]), and the metric is defined per-instance then aggregated over a benchmark distribution — which is exactly the shape ek's `evaluate()` aggregator wants.

---

## 2. The metric family: cost, reliability, latency — reported together

An agent's operating point lives in a **three-dimensional** space (quality × cost × latency), and any honest evaluation reports a point (or a frontier) in that space, never a projection onto one axis. The members of the family:

| Metric | Definition | What it captures | Pitfall it guards against |
|---|---|---|---|
| **Cost-of-pass** `C/R` [1] | E[cost per attempt] / P(success) | Expected $ to first correct solution | Cheap-but-wrong models; infeasibility (→∞) |
| **E[cost \| success]** | Expected total spend on episodes that succeed, folding retries/escalation/verification/failed tool calls | True production spend per delivered outcome | Ignoring retry & verification tax |
| **pass@k** (Chen et al. unbiased) [12] | `E[1 − C(n−c,k)/C(n,k)]`, n≥k samples, c correct | Capability *ceiling* (best-of-k) | The naive `1−(1−p̂)^k` plug-in, which is biased (downward, by Jensen) and high-variance [12] |
| **pass^k** [11] (-> ek_08) | P(same task succeeds on *all* k independent tries) ≈ `p^k` | Reliability *floor* across k retries (distinct from within-episode `p^n` step-compounding) | Reporting a lucky single run |
| **Latency** (TTFT + tok/s) [13] | time-to-first-token; sustained throughput | Interactive cost; wall-clock budget | Treating a slow agent as free |

Two of these deserve emphasis because they are routinely conflated. **pass@k** is the unbiased estimator from the Codex paper (arXiv:2107.03374) [12]: generate `n ≥ k` samples per problem, count `c` correct, and report `1 − C(n−c,k)/C(n,k)` averaged over problems. It corrects the tempting plug-in `1−(1−p̂)^k` with `p̂ = c/n`, which is *biased and high-variance* — and biased **downward** for `k ≥ 2` because `f(x)=1−(1−x)^k` is concave, so Jensen's inequality gives `E[f(p̂)] ≤ f(E[p̂])` [12]. The estimator samples without replacement from the `n` draws. It answers "if I could verify and keep the best of k tries, how often would one succeed?" — the *capability ceiling*.

**pass^k** answers the opposite, production-relevant question: "if I run the same task k times, does it succeed *every* time?" For a single-trial success probability `p`, `pass^k ≈ p^k` across k independent full-task retries — the *across-trial* reliability floor. This is **distinct** from *within-episode* compounding, with which it is routinely conflated: for an n-step episode where each step succeeds independently with probability `p`, end-to-end success is `p^n`, which collapses toward zero as episodes get longer (-> ek_08 documents tau-bench retail degrading from `pass^1 ≈ 61%` to `pass^8 ≈ 25%`). Both are multiplicative erosions of reliability, but one is over *retries of the whole task* (`p^k`) and the other over *dependent steps inside a single episode* (`p^n`) — keep them separate. Cost-of-pass and pass^k are the two numbers you *cannot* omit: the first tells you what the correct answer costs on average, the second tells you how often you can trust it without a human in the loop. METR's **time-horizon** metric [14] adds the difficulty axis that makes both legible: the *50%-time-horizon* is the human-expert task duration at which an agent succeeds half the time, fit via a logistic of success against human task length. Near-100% success sits under ~4-minute human tasks and drops below 10% above ~4 hours; the frontier 50%-horizon has doubled roughly **every 7 months** over 2019–2025 (with METR's own later analyses flagging a *faster* recent doubling, ~4 months in 2024–2025), and o3 was measured at about **1.5 hours (~90 min)** [14]. The point for ek: task *difficulty* (here proxied by human duration) is the variable that both compute allocation and reliability compounding are functions of, so it belongs in your slicing (`per_slice`) alongside cost.

The **verbosity trap** is where these axes couple most viciously. LLM-as-judge signals — which -> ek_09/ek_03 uses as a reference-free confidence source — carry a documented *length bias*: judges systematically score longer answers higher even when the extra length adds no information, and this "cannot be fixed with prompt engineering alone" — it needs human-calibrated data [15][16]. Combine that with the pricing asymmetry in §5 (output tokens cost ~5× input) and you get the TL;DR's warning made concrete: **a quality gain that a judge attributes to a 22%-longer answer is very likely a cost regression that the judge is blind to.** You must *measure* output-length inflation as a cost, not let a length-biased judge launder it as quality. This is why cost is a hard, counted Layer-B quantity and not something you infer from a quality model.

### Statistical rigor: error bars on `pass^k` and cost-per-success

Every number above is an *estimate* from a finite suite — a few hundred tasks, a handful of trials each — and reporting it as a bare point invites both false regressions and false wins. Success rate is a Bernoulli proportion, so it carries a confidence interval: use a **Wilson** (or Bayesian) interval rather than the naive Wald/CLT one, which under-covers badly in the small-`n`, near-0/near-1 regime agent suites live in, and **bootstrap** the cost-per-success ratio, whose sampling distribution is skewed and heavy-tailed because the denominator `P(success)` can be small [35]. And because tasks are drawn in *clusters* (multiple trials per task, multiple tasks per domain), the observations are not independent — you must use **clustered standard errors**, or you will report intervals several times too narrow [35]. "Adding Error Bars to Evals" (Miller) [35] is the practical reference: paired/variance-reduced comparisons when two agents run the same tasks, plus CIs that actually cover.

The reliability-science framing sharpens what to *measure over time*. "Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents" [36] (10 models, 23,392 episodes, 396 tasks, 4 duration buckets, 3 domains) turns the axes above into curves via four metrics: the **Reliability Decay Curve** (RDC, success vs horizon), the **Variance Amplification Factor** (VAF, how run-to-run variance grows with horizon), the **Graceful Degradation Score** (GDS), and the **Meltdown Onset Point** (MOP). The headline: GDS falls **0.90 → 0.44** on software engineering across the duration range but only **0.74 → 0.71** on document processing — degradation is domain-specific, and a single-domain scalar hides it. This is the empirical case for slicing (`per_slice` by domain × duration) and for reporting *variance*, not just means.

**→ maps to ek as:** the harness `regression_gate` (-> ek_08; -> ek_02 golden-set CI) must be **variance-aware** — it compares *confidence intervals*, not point estimates, flagging a regression only when the new frontier's interval separates from the baseline's (Wilson on `pass^k`, bootstrap on cost-per-success, clustered by task/domain). A gate on bare points either cries wolf on noise or waves through a real drop hidden inside the interval.

---

## 3. Test-time compute has an optimum, not a ceiling

The naïve mental model — "spend more inference compute, get more accuracy" — is wrong in a way that directly determines the objective function's shape. Three papers triangulate the correct model.

**Repeated sampling scales, but only with a verifier.** *Large Language Monkeys* (Brown et al., arXiv:2407.21787) [17] shows that **coverage** — the fraction of problems solved by *any* of k samples — scales log-linearly over four orders of magnitude, well modeled by an exponentiated power law. On SWE-bench Lite, DeepSeek-Coder-V2-Instruct rises from 15.9% (1 sample) to **56%** (250 samples), beating the 43% single-attempt SOTA of pricier frontier models; and "amplifying the cheaper DeepSeek model with five samples is more cost-effective and solves more issues than paying a premium for one sample from GPT-4o or Claude 3.5 Sonnet" [17]. This is the cost argument for repeated sampling. **But** coverage is a pass@k-style ceiling that assumes you can *verify* which sample is right — with an automatic verifier (unit tests, a proof checker) you realize it; without one, sample-selection methods (majority voting, reward models) plateau after a few hundred samples [17]. Whether you *have* a deterministic verifier is therefore the single most important economic fact about a task (-> ek_08's oracle taxonomy; -> ek_04 for verifier construction).

**Voting is non-monotone.** When you *don't* have a verifier and fall back to sample-and-vote, Chen et al.'s *Are More LLM Calls All You Need?* (arXiv:2403.02419) [3][18] proves the counterintuitive result: Vote and Filter-Vote accuracy "can first increase but then decrease as a function of the number of LM calls," because "more LM calls lead to higher performance on easy queries, but lower performance on hard queries, and non-monotone behavior can emerge when a task contains both types" [18]. Intuitively, majority vote amplifies the model's *modal* answer — great when the mode is correct (easy items), actively harmful when the mode is a confident wrong answer (hard items). They fit an analytical scaling model that predicts, from a handful of samples, the call count that *maximizes* performance. The engineering consequence is stark: **there is an optimal number of calls, and blindly cranking it up burns tokens to lose accuracy.**

**Compute-optimal allocation by difficulty.** Snell et al. (arXiv:2408.03314) [4][19] close the loop: allocate test-time compute *per-prompt by difficulty* (search against a process/verifier reward model vs. sequential revision of the response distribution), and a "compute-optimal" strategy is **>4× more efficient** than a best-of-N baseline, and under a FLOPs-matched budget "can be used to outperform a 14× larger model" on problems where the base model already has a nonzero success rate [4]. Overthinking easy items wastes tokens; under-thinking hard ones wastes the whole attempt. The lever is *difficulty-adaptive* allocation, which is exactly what a router or cascade implements.

The synthesis: the cost-per-success surface is **not monotone in compute**, it has an interior optimum whose location depends on item difficulty. That is why the optimizer is a *router* — and why, per §4, you must evaluate the router itself. (This also contextualizes Cost-of-Pass's finding [1] that generic inference-time methods "rarely justified their overhead": undirected majority voting is on the *wrong side* of the non-monotone curve for hard items; the exception, budget-aware TALE-EP, wins precisely because it *matches* compute to difficulty rather than spending it uniformly.)

---

## 4. Evaluating the cost-optimizer: routing and cascades

The mechanisms that exploit §3 form a well-mapped design space. The umbrella is Dohan et al.'s *Language Model Cascades* (arXiv:2207.10342) [20], which formalizes chain-of-thought, scratchpads, verifiers, STaR, selection-inference, and tool use as **probabilistic programs** — graphical models over string-valued random variables — giving a single theoretical frame for "compositions of repeated LM interactions." The concrete systems:

| System | Mechanism | Reported economics | License | Ref |
|---|---|---|---|---|
| **FrugalGPT** | LLM cascade: query models in increasing cost order, stop when a learned reliability scorer `g(q,a)→[0,1]` (a fine-tuned DistilBERT) passes a per-stage threshold | matches GPT-4 at **up to 98%** cost cut, or +4% accuracy at equal cost | — (paper; TMLR) | [5] |
| **RouteLLM** | Router trained on Chatbot Arena preference data selects strong-vs-weak model per query | **>85%** cost cut on MT-Bench at 95% of GPT-4 quality (~14% of MT-Bench queries to GPT-4 vs a Mixtral-8×7B weak model; ~54% on MMLU) | Apache-2.0 | [6][7] |
| **Hybrid LLM** | Route by predicted query difficulty with a **test-time-tunable** quality threshold | up to 40% fewer large-model calls at no quality drop; 22% fewer at 1% BART-score drop | — (ICLR 2024) | [21] |
| **AutoMix** | Small model self-verifies (few-shot); a POMDP **meta-verifier** arbitrates escalation | **>50%** compute cost cut at comparable performance | — | [22] |

Two design principles recur across all four, and both are ek machinery you already have. First, **a cascade's escalation decision IS selective prediction**: the confidence gate that decides "accept this cheap answer vs. escalate to the expensive model" is precisely the accept/flag/escalate decision of ek's `estimate_quality()` cascade, and the FrugalGPT scorer is a learned correctness predictor with a threshold — a `DecisionPolicy` over a `Calibrator`'d signal (-> ek_09/ek_03). Second, **the routing score must be calibrated** for the threshold to mean anything: an uncalibrated escalation score sends too many easy queries up (cost regression) or too many hard ones through (quality regression). Do not rebuild the calibration stack — reuse -> ek_03's temperature/isotonic/conformal machinery. Hybrid LLM's *test-time-tunable* threshold [21] is the router-side analogue of ek's risk-coverage operating point: one knob slides the whole system along the cost-quality tradeoff without retraining.

Now the part that is genuinely *evaluation of the optimizer*, and the reason this report exists. A router is not scored by a single accuracy number — it is scored by the **whole cost-quality curve it can trace out** as you sweep its threshold. **RouterBench** (Hu et al., arXiv:2403.12031) [8][23] gives the canonical aggregator: **AIQ (Average Improvement in Quality)**, the *normalized area under the non-decreasing convex hull* of the router's operating points in the cost-quality plane:

```
AIQ(R_θ) = 1/(c_max − c_min) ∫_{c_min}^{c_max} R̃_θ(c) dc
```

where `R̃_θ` is the non-decreasing convex hull of the router's `(cost, quality)` points, built by (1) linear interpolation between any two operating points via probabilistic mixing — you can hit an intermediate cost by randomly routing a fraction of queries to each of two configs — and (2) enforcing monotonicity: for `c₂ ≥ c₁` require `q₂ ≥ q₁`, discarding dominated points [8]. RouterBench backs this with **>405,000** inference outcomes across commonsense/knowledge/conversation/math/coding/RAG tasks [8]. The engineering takeaways for ek are precise:

- **The frontier, not a point, is the deliverable.** Evaluating a router means producing its non-decreasing convex hull and reducing it to one scalar (AIQ) so two routers are comparable — this is `evaluate()` with a Pareto/AIQ aggregator, not a naive mean.
- **Monotonicity is enforced, not assumed.** Dominated operating points are pruned; a well-behaved router's frontier only moves up-and-right.
- **Probabilistic mixing gives you the interpolated frontier for free** — you don't need to train a router at every cost level, you interpolate between the ones you have.

RouteLLM ships as `routellm` (Apache-2.0) [7] with `[serve]` and `[eval]` extras and four routers (similarity-weighted Elo, matrix factorization, BERT classifier, LLM classifier); RouterBench's code is `withmartian/routerbench` [8]. Both are permissive and safe for an ek extra (-> ek_06 license register; -> ek-dev-licensing).

---

## 5. Cost accounting mechanics: why cost must be a per-token-type lookup

You cannot compute `E[cost]` from a single price. The billing model has **asymmetries** that make cost a structured, per-token-type quantity — which is exactly why it must be a Layer-B measurement backed by a price SSOT, never a hardcoded constant.

- **Output ≫ input.** Output tokens cost several times input tokens — uniformly **5×** across Anthropic's current lineup (e.g. claude-sonnet-4-5 at $3/MTok input vs $15/MTok output; Opus $5/$25; Haiku $1/$5), with the wider 4–6× band reflecting cross-provider variation [9][24]. This asymmetry is the mathematical reason a 22%-longer answer is a cost regression (§2).
- **Cached input is cheap; cache writes are not.** A cache *read* (hit) costs ~**10%** of the base input rate (90% off), but a cache *write* costs **1.25×** (5-minute TTL) or **2×** (1-hour TTL) the base input rate [24]. Caching only discounts *reused* reads, not the whole prompt — so the cost model must distinguish input / cached-read / cache-write.
- **Batch APIs are ~50% off, and discounts stack.** The Batch API is 50% off both input and output for results returned asynchronously, and it *stacks* with caching: a cached batch read can cost as little as **~5%** of a standard non-cached request (0.1 × 0.5) [24].
- **Reasoning / "thinking" tokens are billed at the output rate** [24] — and often dominate the bill on hard items, which is the direct financial expression of the §3 non-monotonicity: over-sampling reasoning on easy items is pure waste.

Encoding all of this by hand is a maintenance disaster and a source of silent drift. The right move is a **price SSOT**. LiteLLM's `model_prices_and_context_window.json` [9][25] is the ready-made one: a machine-readable, MIT-licensed catalog covering 100+ providers with exactly the fields you need — `input_cost_per_token`, `output_cost_per_token`, `cache_read_input_token_cost`, `cache_creation_input_token_cost`, `output_cost_per_reasoning_token`, `*_flex` (batch) fields — plus `max_input_tokens` / `max_output_tokens`. It is runtime-loadable and overridable via the `LITELLM_MODEL_COST_MAP_URL` env var, so "cheapest model for tier X" becomes a *lookup*, not a hardcode. **OpenRouter's `/models`** endpoint [10] is the fallback catalog (400+ models; `pricing.prompt` / `pricing.completion` / `pricing.image` / `pricing.request` as *per-token USD strings* — strings, deliberately, to avoid float error — plus context limits). This is precisely the open-closed, config-driven pluggable table CLAUDE.md's design principles demand: no magic numbers, sensible default, overridable source.

**Latency is a cost dimension too.** For interactive agents, time-to-first-token (TTFT) and sustained throughput (tokens/sec) are real costs — a correct answer that arrives too late has failed a latency SLA as surely as a wrong answer failed a quality one. Artificial Analysis operationalizes this (§6): `time-per-task = output tokens / output speed` [13]. For batch/offline agents, the relevant unit is throughput per GPU-second. Either way, latency rides alongside dollar cost on the episode.

---

## 6. Cost-aware leaderboards and tracing

Two ecosystems matter here: **measurement methodologies** (how public leaderboards fold cost in) and **tracing backends** (how you record per-episode cost+quality+latency in production).

**Cost-aware leaderboards.** HELM (Liang, Bommasani, Lee et al., arXiv:2211.09110) [26] pioneered efficiency as a first-class metric, reporting both *actual* runtime and an *idealized* runtime on uniform A100/Megatron hardware — a **denoised** number (provider hardware/software with performance-variation noise removed) and an **idealized** one (modeled piecewise-linear in input tokens plus linear in output tokens) that isolates model efficiency from serving-stack noise and enables hardware-independent cross-model comparison; training efficiency is reported as tCO₂ [26]. HELM is `crfm-helm` (Apache-2.0), pip-installable [26]. **Artificial Analysis** [13] is the current practical reference for the three-axis view: separate Intelligence Index (quality), price, and speed, where price is a *blended* $/Mtok using a default **7:2:1** cache:input:output ratio, plus Output Speed (tok/s) and TTFT; it publishes an explicit weighted **cost-per-task** (input + cache-hit + cache-write + reasoning + answer token prices divided by task count) and `time-per-task` [13]. This 7:2:1 blend is itself a reusable default for ek's cost model. METR's time-horizon [14] (§2) is the difficulty-normalized companion.

**Tracing backends** are where Layer-B episode telemetry actually lands. The license map is a minefield, so the table is a license register as much as a feature comparison (-> ek_06; -> ek-dev-licensing):

| Backend | License | Cost handling | ek verdict |
|---|---|---|---|
| **Langfuse** | MIT core (`/ee` excepted) | Auto-computes generation cost at ingestion from a model-price table; breaks cost/latency down by user/session/model/prompt-version; OTel + LiteLLM integrations [27][28] | **Recommended default** tracing backend |
| **Helicone** | Apache-2.0 | Meters *requests*; self-host via Docker/K8s free [29] | Safe permissive alternative |
| **Arize Phoenix** | **Elastic License 2.0** (source-available, *not* OSI; no-managed-service clause) | OpenInference cost tracking | **Opt-in extra only; never a default; flag in license gate** [30] |
| **LangSmith** | Proprietary SaaS (client SDK is MIT) | Meters *base traces*; ~$2,514/mo at 1M base traces | Cannot depend on the *platform*; SDK-only [31] |
| **Weave** | OSS core + cloud upsell | — | Extra, with care |

Two traps to internalize. **Arize Phoenix is Elastic License 2.0** — source-available, not open-source, with a clause prohibiting offering it as a managed service. It must be an opt-in ek extra, never a default, and it must trip ek's CI license gate (-> ek-dev-licensing; -> ek_06). **LangSmith** is a proprietary hosted service (its `langsmith-sdk` client is MIT, so the blocker is the *platform*, not a vendorable library) — you cannot make it a library backend. And note the metering units differ — Langfuse counts *events*, Helicone counts *requests*, LangSmith counts *base traces* — so cross-tool cost comparisons are **not 1:1** [29].

On the eval-harness side, two facts shape build-vs-borrow. **Inspect AI** (`inspect-ai`, MIT, UK AISI) [32] is the strongest pip-installable agent-eval harness that *natively* measures cost: its Task = Dataset + Solver + Scorer model supports agent evals (including driving external agents like Claude Code / Codex CLI), and it enforces **per-sample limits on total tokens AND total dollar cost** via `set_model_cost()` / `--model-cost-config`, logging every prompt/response/token-count for audit — a candidate to WRAP behind ek's `evaluate()`/harness rather than rebuild (-> ek_08 makes the same call for the success-checking side). Conversely, **DeepEval** (Apache-2.0) and **Ragas** both track cost but expose it only *per-run*, not as a cost-per-success aggregator — Ragas even needs a custom `TokenUsageParser` because LangChain LLMs don't return usage uniformly [33]. That per-run-only gap is *precisely* what `CostPerSuccessMetric` fills: none of them return `E[cost]/P(success)`.

---

## 7. Honest caveats

- **LLM-judge biases contaminate any cost objective that uses a judge signal.** Position bias (10–15 points; mitigate with swap augmentation — evaluate A-vs-B *and* B-vs-A, accept only consistent verdicts), verbosity/length bias (needs human-calibrated data, not prompt engineering), and self-preference bias are all documented [15][16]. Because the judge is often the escalation gate of a cascade (§4), a biased judge doesn't just mis-score — it *mis-routes*, spending money in the wrong direction. Calibrate and validate the judge (-> ek_03 for calibration; -> ek_02 harness for IAA/Krippendorff judge validation) before trusting it in the objective.
- **Benchmark validity threats compound the cost distortion** [11]: missing holdout sets (overfitting to the eval), shortcut/hardcoded policies (WebArena's top agent STeP "hardcodes policies"), non-independent environments (rate-limited Reddit violating task independence), harness scoring bugs (LATS and STeP marked incorrect tasks correct), and eval runs so expensive they're rarely rerun (SWE-Agent full eval "could cost over $8,000 for a single run") [11]. The implication for ek's harness: a **golden holdout + a CI cost budget + verified scoring** are non-negotiable (-> ek_08; -> ek_02 golden-set CI).
- **`pass^k` and coverage caveats** carry over from -> ek_08: `pass^k` assumes independent trials (correlated failures make it optimistic), and coverage-based cost arguments [17] evaporate without a verifier. Whether the task has a deterministic oracle is the fork in the road.
- **Cost-of-Pass's "inference-time methods rarely justify overhead" finding is benchmark-specific** [1] and does *not* contradict §3 — undirected voting is on the wrong side of the non-monotone curve, while difficulty-matched compute (compute-optimal scaling [4], TALE-EP) is on the right side. Don't over-read it into "test-time compute doesn't pay"; read it as "*undirected* test-time compute doesn't pay."

---

## How this lands in ek

**Cost is a first-class Layer-B quantity.** The `AnnotatedExtraction` metadata that -> ek_08 populates with the trajectory and verdict gains a structured **cost record** per episode: input/cached-read/cache-write/reasoning/output token counts, dollar cost (priced through the SSOT), retry count, escalation count, verification-pass count, failed-tool-call count, and latency (TTFT + total). This rides alongside the Layer-A grammar, never mutating it — the same architectural rule as every other Layer-B signal (-> ek-dev-architecture).

**`FieldSpec.importance` generalizes to task value / error cost.** This is the keystone reuse. In IE, `FieldSpec.importance` (Layer A) weights how much a field matters for cost-sensitive scoring. In the agent domain, the *same frozen typed-schema SSOT* carries **task value** (the payoff of completing this task) and **error cost** (the penalty of a wrong final state — asymmetric, since a wrong booking costs more than a failed one). The identical cost-weight lever that drives cost-sensitive IE metrics (-> ek_02, -> ek-dev-add-metric) now drives the economic objective — no new mechanism, just a new interpretation of the weights, exactly the open-closed generalization the two-layer model was built for.

**`CostPerSuccessMetric` is an `evaluate()` aggregator.** It implements the `Metric` Protocol with its *own* corpus aggregator (never a naive mean — a mean of per-episode costs would ignore the failures entirely): over a corpus it returns `E[cost] / P(success)` = cost-of-pass [1], plus the `E[cost | success]` variant that folds retries/escalation/verification, reported jointly with `pass^k` (-> ek_08) and latency. When `P(success)=0` it returns infinity, preserving the infeasibility signal [2]. It is resolved by name and injected keyword-only through the registry (-> ek-dev-add-metric), guarded by `@requires_extra` where it needs a pricing dependency.

**The price catalog is a pluggable SSOT.** A small `PriceTable` strategy (Protocol) wraps LiteLLM's `model_prices` JSON [9] with OpenRouter `/models` [10] as fallback — config-driven, overridable, no hardcoded rates (CLAUDE.md open-closed principle). `cost(usage, model) = Σ tokens_of_type × rate_of_type` over the five token types of §5. This is the object that makes "cheapest model for tier X" a lookup.

**Router evaluation is `evaluate()` with a Pareto/AIQ aggregator + a frontier `regression_gate`.** When the evaluated object is a *router or cascade* rather than a single agent, `evaluate()` sweeps its threshold, builds the non-decreasing cost-quality convex hull, and reduces it to AIQ [8]. The harness's `save_baseline` / `regression_gate` (-> ek_08 harness; -> ek_02 golden-set CI) then gates CI on the **frontier**: a change that dominates the old frontier passes; one that pushes any operating point down-and-right fails the build. This is the router-eval analogue of a regression test.

**The cascade's confidence gate *is* the decide stage.** An escalation policy (FrugalGPT's threshold [5], AutoMix's POMDP meta-verifier [22], Hybrid LLM's tunable knob [21]) is a `DecisionPolicy` over a `Calibrator`'d agent signal — accept the cheap answer, or escalate = selective prediction. Do not rebuild: reuse `estimate_quality()`'s signal→calibrate→validate→decide cascade and -> ek_03's conformal/risk-coverage machinery. The routing score must be calibrated for the threshold to carry a guarantee.

**Self-consistency reuses ROVER.** Sample-and-vote self-consistency [34] is the agreement signal ek already has as ROVER (-> ek_03; -> ek-dev-architecture must-builds) — sample n trajectories, vote on the final answer, use vote margin as a reference-free confidence signal. Reuse it; don't rebuild. But respect §3's non-monotonicity: the vote count is a *tuned* parameter matched to item difficulty, not a crank to max out.

**Wrap, don't rebuild, the harness.** Inspect AI (MIT) [32] already runs agents under per-sample token *and dollar* budgets with full audit logging; it is the natural thing to wrap behind `evaluate_store` for the run-the-predictor-over-a-store loop, leaving ek to own the economic aggregation (`CostPerSuccessMetric`, AIQ, the frontier gate) that Inspect and DeepEval/Ragas [33] don't provide.

---

## References

[1] Erol MH, El B, Suzgun M, Yuksekgonul M, Zou J. Cost-of-Pass: An Economic Framework for Evaluating Language Models. arXiv:2504.13359; ICLR 2026. [<https://arxiv.org/abs/2504.13359>](https://arxiv.org/abs/2504.13359); OpenReview [<https://openreview.net/forum?id=vC9S20zsgN>](https://openreview.net/forum?id=vC9S20zsgN).

[2] Cost-of-Pass reference implementation. GitHub `mhamzaerol/Cost-of-Pass`. [<https://github.com/mhamzaerol/Cost-of-Pass>](https://github.com/mhamzaerol/Cost-of-Pass).

[3] Chen L, Davis JQ, Hanin B, Bailis P, Stoica I, Zaharia M, Zou J. Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems. arXiv:2403.02419, 2024. [<https://arxiv.org/abs/2403.02419>](https://arxiv.org/abs/2403.02419).

[4] Snell C, Lee J, Xu K, Kumar A. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv:2408.03314, 2024. [<https://arxiv.org/abs/2408.03314>](https://arxiv.org/abs/2408.03314).

[5] Chen L, Zaharia M, Zou J. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176, 2023 (TMLR 2024). [<https://arxiv.org/abs/2305.05176>](https://arxiv.org/abs/2305.05176).

[6] Ong I, Almahairi A, Wu V, Chiang WL, Wu T, Gonzalez JE, Kadous MW, Stoica I. RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665; ICLR 2025. [<https://arxiv.org/abs/2406.18665>](https://arxiv.org/abs/2406.18665).

[7] LMSYS Org. RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing (blog); repo `lm-sys/RouteLLM` (Apache-2.0). [<https://www.lmsys.org/blog/2024-07-01-routellm/>](https://www.lmsys.org/blog/2024-07-01-routellm/); [<https://github.com/lm-sys/RouteLLM>](https://github.com/lm-sys/RouteLLM).

[8] Hu QJ, Bieker J, Li X, Jiang N, Keigwin B, Ranganath G, Keutzer K, Upadhyay SK. RouterBench: A Benchmark for Multi-LLM Routing System. arXiv:2403.12031, 2024; code `withmartian/routerbench`. [<https://arxiv.org/abs/2403.12031>](https://arxiv.org/abs/2403.12031); [<https://github.com/withmartian/routerbench>](https://github.com/withmartian/routerbench).

[9] BerriAI. LiteLLM `model_prices_and_context_window.json` (price/context SSOT; MIT). [<https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json>](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

[10] OpenRouter. Models API and pricing reference. [<https://openrouter.ai/docs/guides/overview/models>](https://openrouter.ai/docs/guides/overview/models); [<https://openrouter.ai/pricing>](https://openrouter.ai/pricing).

[11] Kapoor S, Stroebl B, Siegel ZS, Nadgir N, Narayanan A. AI Agents That Matter. arXiv:2407.01502, 2024. [<https://arxiv.org/abs/2407.01502>](https://arxiv.org/abs/2407.01502).

[12] Chen M, Tworek J, Jun H, et al. Evaluating Large Language Models Trained on Code (Codex; pass@k estimator). arXiv:2107.03374, 2021. [<https://arxiv.org/abs/2107.03374>](https://arxiv.org/abs/2107.03374).

[13] Artificial Analysis. Language Model Benchmarking Methodology (Intelligence Index, 7:2:1 price blend, Output Speed, TTFT, cost-per-task). [<https://artificialanalysis.ai/methodology>](https://artificialanalysis.ai/methodology).

[14] Kwa T, West R, et al. (METR). Measuring AI Ability to Complete Long Tasks. arXiv:2503.14499, 2025. [<https://arxiv.org/abs/2503.14499>](https://arxiv.org/abs/2503.14499); [<https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/>](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/).

[15] Sigl S. The 5 Biases That Can Silently Kill Your LLM Evaluations (And How to Fix Them). [<https://www.sebastiansigl.com/blog/llm-judge-biases-and-how-to-fix-them/>](https://www.sebastiansigl.com/blog/llm-judge-biases-and-how-to-fix-them/).

[16] Self-Preference Bias in LLM-as-a-Judge. arXiv:2410.21819, 2024. [<https://arxiv.org/html/2410.21819v1>](https://arxiv.org/html/2410.21819v1).

[17] Brown B, Juravsky J, Ehrlich R, Clark R, Le QV, Ré C, Mirhoseini A. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. arXiv:2407.21787, 2024. [<https://arxiv.org/abs/2407.21787>](https://arxiv.org/abs/2407.21787).

[18] Chen L, Davis JQ, et al. Are More LLM Calls All You Need? (as [3]). [<https://arxiv.org/abs/2403.02419>](https://arxiv.org/abs/2403.02419).

[19] Snell C, Lee J, Xu K, Kumar A. (HTML version of [4]). [<https://arxiv.org/html/2408.03314v1>](https://arxiv.org/html/2408.03314v1).

[20] Dohan D, Xu W, Lewkowycz A, Austin J, Bieber D, Gontijo Lopes R, Wu Y, Michalewski H, Saurous RA, Sohl-Dickstein J, Murphy K, Sutton C. Language Model Cascades. arXiv:2207.10342, 2022. [<https://arxiv.org/abs/2207.10342>](https://arxiv.org/abs/2207.10342).

[21] Ding D, Mallick A, Wang C, Sim R, Mukherjee S, Ruhle V, Lakshmanan LVS, Awadallah AH. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. arXiv:2404.14618; ICLR 2024. [<https://arxiv.org/abs/2404.14618>](https://arxiv.org/abs/2404.14618).

[22] Madaan A, Aggarwal P, et al. AutoMix: Automatically Mixing Language Models. arXiv:2310.12963, 2023. [<https://arxiv.org/abs/2310.12963>](https://arxiv.org/abs/2310.12963).

[23] Hu QJ, et al. RouterBench (ar5iv HTML). [<https://ar5iv.labs.arxiv.org/html/2403.12031>](https://ar5iv.labs.arxiv.org/html/2403.12031).

[24] Anthropic. Prompt caching and pricing — Claude Platform Docs (cache-read 10%, cache-write 1.25×/2×, Batch 50%, reasoning billed as output, output 5× input). [<https://platform.claude.com/docs/en/build-with-claude/prompt-caching>](https://platform.claude.com/docs/en/build-with-claude/prompt-caching); [<https://platform.claude.com/docs/en/about-claude/pricing>](https://platform.claude.com/docs/en/about-claude/pricing).

[25] BerriAI. LiteLLM token usage & pricing docs. [<https://docs.litellm.ai/docs/completion/token_usage>](https://docs.litellm.ai/docs/completion/token_usage).

[26] Liang P, Bommasani R, Lee T, et al. Holistic Evaluation of Language Models (HELM). arXiv:2211.09110, 2022; efficiency metrics docs (crfm-helm, Apache-2.0). [<https://arxiv.org/abs/2211.09110>](https://arxiv.org/abs/2211.09110); [<https://crfm-helm.readthedocs.io/en/latest/metrics/>](https://crfm-helm.readthedocs.io/en/latest/metrics/).

[27] Langfuse. `langfuse/langfuse` (MIT core). [<https://github.com/langfuse/langfuse>](https://github.com/langfuse/langfuse).

[28] Langfuse. Token & Cost Tracking documentation. [<https://langfuse.com/docs/observability/features/token-and-cost-tracking>](https://langfuse.com/docs/observability/features/token-and-cost-tracking).

[29] Helicone. Open-source LLM observability platform (Apache-2.0). [<https://github.com/helicone/helicone>](https://github.com/helicone/helicone).

[30] Arize AI. Phoenix (`arize-phoenix`, Elastic License 2.0). [<https://github.com/arize-ai/phoenix>](https://github.com/arize-ai/phoenix).

[31] LangChain. LangSmith (proprietary SaaS; client SDK MIT). [<https://www.langchain.com/langsmith>](https://www.langchain.com/langsmith).

[32] UK AI Security Institute. Inspect AI: A framework for large language model evaluations (`inspect-ai` v0.3.245, MIT). [<https://github.com/UKGovernmentBEIS/inspect_ai>](https://github.com/UKGovernmentBEIS/inspect_ai); [<https://inspect.aisi.org.uk/>](https://inspect.aisi.org.uk/).

[33] Confident AI. DeepEval (Apache-2.0); Ragas cost-analysis docs. [<https://github.com/confident-ai/deepeval>](https://github.com/confident-ai/deepeval); [<https://docs.ragas.io/en/stable/howtos/applications/_cost/>](https://docs.ragas.io/en/stable/howtos/applications/_cost/).

[34] Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171; ICLR 2023. [<https://arxiv.org/abs/2203.11171>](https://arxiv.org/abs/2203.11171).

[35] Miller E. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640, 2024. [<https://arxiv.org/abs/2411.00640>](https://arxiv.org/abs/2411.00640).

[36] Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents. arXiv:2603.29231, 2026. [<https://arxiv.org/abs/2603.29231>](https://arxiv.org/abs/2603.29231).
