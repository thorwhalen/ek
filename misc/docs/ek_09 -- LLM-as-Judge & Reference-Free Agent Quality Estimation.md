# ek_09 — LLM-as-Judge & Reference-Free Agent Quality Estimation

**Author:** Thor Whalen

**Status:** Research synthesis / integration brief. This is the reference-free (`estimate_quality()`) side of ek's agent-evaluation extension — how to estimate the quality of an open-ended agent output when there is *no checkable outcome and no gold*. It is the agent-domain analogue of -> ek_03 (reference-free QE for information extraction). The reference-based side — where "done" is programmatically checkable — lives in the sibling brief -> ek_08.

---

## TL;DR

When an agent's output has no executable oracle — a written brief, a research summary, a customer reply, a plan — you cannot `score()` it against gold. You must *estimate* its quality from the output itself. Two families of reference-free signal do this. The first is the **LLM-as-judge**: a model reads the output (and maybe a rubric or a competitor output) and emits a verdict. It works — GPT-4 reaches ~85% agreement with human experts on the non-tie subset of MT-Bench, matching the ~81% human-human rate [1] — but it is a *biased instrument* that must be **validated against humans, de-biased, and calibrated before any of its scores are allowed to gate a decision.** Judges favor the first option presented, longer answers, their own generations, and superficially-polished-but-wrong instruction violations; on hard reasoning they sit near chance [1][3][6][7]. The second family is **structural, judge-free signals** — self-consistency / sample-and-vote [8], sampling-based hallucination detection (SelfCheckGPT [9]), and RAG faithfulness / groundedness (Ragas [11], TruLens, RAGChecker) — which need no gold *and* no trusted grader, only the model's own output distribution. The load-bearing thesis: **all of this is the exact same `signal → calibrate → decide` pipeline ek already ships for OCR confidence (-> ek_03), with a judge as one (expensive, high-`cost_tier`) signal among many.** Judge validation reuses the harness IAA machinery — `krippendorff_alpha`, `cohen_kappa` (-> ek_02). Judge scores flow through the same `Calibrator` and `DecisionPolicy` (accept/flag/block) as any other signal, and **Hard-Rule-1 — never gate on an uncalibrated signal — applies to judges with full force.** Prefer binary verdicts over Likert, validate on a held-out labeled set with per-class rates (TPR/TNR) not raw accuracy, and use a **panel of cheap diverse judges** (PoLL) over one expensive one for both bias and cost [7]. Where a deterministic verifier exists (-> ek_08), use it; the judge is the *fallback*, not the default.

---

## 1. Why reference-free, and where the judge sits

-> ek_08 owns the case where "done" is a programmatic bit: the DB row is right, the tests flip green, the answer normalized-exact-matches. That is the happy path and you should always reach for it first. But a large and growing fraction of what agents produce has no such oracle: a synthesized research report, a drafted email, a summary, a recommendation, a multi-paragraph explanation. There is no hidden test suite for "is this a *good* brief." Here you are in the reference-free quadrant — the same quadrant -> ek_03 mapped for OCR/IE, where the object was a page of extracted text with no gold transcript. The machinery is identical; only the object changes. In -> ek_03 the reference-free object was an `AnnotatedExtraction`; here it is an **episode / trajectory ending in an open-ended artifact**, and the quality question is "should a human trust and ship this, or should it be flagged / escalated / blocked?" — which is exactly **selective prediction** (-> ek_03), now applied to agent output.

Two reference-free signal families answer that question. They differ in what they trust:

- **LLM-as-judge** trusts *another model's judgment* of the output. Powerful and general, but it imports a second model's biases and failure modes into your metric. This is the bulk of this report because it is the most-used and the most-abused.
- **Structural / self-referential signals** trust only *the model's own output distribution*: sample the same prompt several times and measure agreement (self-consistency [8], SelfCheckGPT [9], `uqlm` [13]), or decompose the answer into claims and check each against the retrieved context (RAG faithfulness [11]). No trusted grader is required, which is why these are often the *cheaper and more robust* first line of defense.

The right architecture uses both in a **cascade** (a cost-sensitive theme throughout ek): a cheap structural signal (self-consistency, faithfulness) runs on every item; an expensive judge runs only on the items the cheap gate is uncertain about. **The cascade's confidence gate *is* the `decide` stage; escalation to a human or to a stronger judge *is* selective prediction.** Anthropic's own agent-eval guidance lands in the same place: prefer outcome checks; use model-based graders only for nuance, and "always give the LLM a way out — an instruction to return *Unknown* when it does not have enough information" [17] (the judge's own abstention is a first-class signal, mapping directly onto ek's flag verdict).

---

## 2. LLM-as-judge foundations

### 2.1 The founding validation

The paper that made LLM-as-judge respectable is Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS 2023) [1]. Two artifacts: **MT-Bench**, 80 multi-turn questions across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, humanities/social-science knowledge), and **Chatbot Arena**, crowdsourced pairwise "battles" scored into Elo / Bradley-Terry ratings. The headline validation: on the **non-tie subset (setup S2)**, GPT-4-as-judge agrees with human experts ~**85%** of the time — *higher* than the ~**81%** human-human agreement — while under the tie-inclusive setup S1 the numbers are ~70% (GPT-4-vs-human) and ~63% (human-human) [1]. Read carefully: the "judge matches humans" claim is specifically the tie-excluded subset. That is the correct claim to make, and it is genuinely strong evidence that a frontier judge is a usable instrument for *general chat quality*. It is *not* evidence that judges work on hard reasoning or factuality — a distinction §4 makes central.

### 2.2 The three protocols: pointwise, pairwise, reference-guided

There are three ways to elicit a judgment, and the choice matters more than the model:

| Protocol | What the judge sees | Output | Best for | Weakness |
|---|---|---|---|---|
| **Pointwise / direct scoring** | One output (+ rubric) | Score (binary or scale) | Faithfulness, policy violations, objective checks; scales to production streams | No anchor; scores drift; harder to calibrate across items [16] |
| **Pairwise comparison** | Two outputs A/B | Which is better (+ ties) | Subjective dimensions (tone, coherence, persuasiveness); model ranking | O(n²) comparisons; position bias; needs a baseline to compare against [16] |
| **Reference-guided** | Output + a gold/reference answer | Score or verdict | Math/reasoning where a worked solution exists | Needs a reference — drifts toward -> ek_08 territory [1] |

Empirically, **pairwise comparison yields more stable results and smaller judge-human gaps than pointwise scoring for subjective dimensions**, while **direct scoring is more versatile for objective checks like faithfulness and policy violations** [16]. The engineering implication for ek: the judge protocol is itself a strategy choice, injected keyword-only, not hard-wired. For a production `estimate_quality()` stream you usually want *pointwise binary* (see §5); for offline model comparison inside `evaluate()` (-> ek_08) you want *pairwise*.

### 2.3 G-Eval: the rubric + CoT + form-filling judge

Liu et al., *G-Eval* (EMNLP 2023, arXiv:2303.16634) [2] is the canonical structured pointwise judge. Its recipe: (1) a task introduction + evaluation criteria, (2) **auto-generated chain-of-thought "evaluation steps"** derived from that criteria, then (3) a **form-filling** pass that emits a score. Its distinctive engineering trick is the score de-quantization: rather than take the single integer the model emits, G-Eval computes a **probability-weighted sum over the discrete rating tokens**, `score = Σ_i p(s_i)·s_i` where `p(s_i)` is the (renormalized) token probability of each candidate integer score [2]. This captures the judge's uncertainty instead of forcing a hard bucket, and it lifts correlation on nuanced dimensions. With a GPT-4 backbone, G-Eval reaches **average Spearman 0.514 across the four SummEval dimensions** (coherence, consistency, fluency, relevance) [2] — a large jump over prior automatic metrics. Two caveats to carry into ek: (a) 0.514 is the *aggregate* across four dimensions, not a single-dimension score [2]; (b) the probability-weighting mechanic needs token log-probs, which many production chat APIs no longer expose — so real deployments often fall back to a single sampled integer, losing exactly the uncertainty signal that made G-Eval good. That log-prob channel, where available, is a natural raw signal for ek's `Calibrator`.

### 2.4 Binary vs Likert: the practitioner verdict

Academic G-Eval uses 1–5 scales; production practice has swung hard the other way. The strongest field guidance — Hamel Husain's widely-cited LLM-judge writeups — is blunt: "If your evaluations consist of a bunch of metrics that LLMs score on a 1–5 scale (or any other scale), you're doing it wrong" [18]. The reasoning is sound: adjacent Likert points (is this a 3 or a 4?) are subjective and inconsistent across annotators and across runs, they hide uncertainty in the mushy middle, and they are not *actionable* — a "3.5" tells you nothing to do. **Binary pass/fail forces clearer thinking and more consistent labeling** [18][19]. ek's default `JudgeSignal` should therefore be binary; a scalar/Likert judge is an opt-in variant, and if used, its scale must be squashed through a `Calibrator` (isotonic/Platt) before it can gate anything. This is not a stylistic preference — it is what makes the judge *validatable* (§4), because per-class rates (TPR/TNR) are defined on a binary label, not a 5-point scale.

---

## 3. Judge biases — be blunt, and cite the magnitudes

A judge is a *systematically* biased instrument, and the biases are large enough to invert conclusions if ignored. All magnitudes below trace to Zheng et al.'s own tables [1] (Eugene Yan's survey [16] reproduces them; MT-Bench is the primary source).

| Bias | What it is | Measured magnitude [1] | Mitigation |
|---|---|---|---|
| **Position bias** | Judge favors the answer in a particular slot (usually first) | GPT-4 only **65.0%** consistent when A/B are swapped; **30.0%** first-biased. GPT-3.5 46.2% consistent / 50.0% first-biased. Claude-v1 23.8% consistent / **75.0%** first-biased | Run both orders, average / require agreement; drop inconsistent pairs (that is the S2 setup) |
| **Verbosity / length bias** | Judge prefers longer answers at equal quality | "Repetitive list" attack fools GPT-3.5 and Claude-v1 **91.3%** of the time vs GPT-4 **8.7%** | Length-control, penalize redundancy in rubric, calibrate |
| **Self-enhancement / self-preference** | Model prefers its *own* generations | GPT-4 ~**+10%** own-win-rate over human assignment; Claude-v1 ~**+25%**; GPT-3.5 none measurable | Use a *different-family* judge; use panels (§6); never judge with the model under test |
| **Sycophancy / rubric-wording sensitivity** | Judge agrees with asserted framing; verdict shifts with prompt phrasing | Paraphrasing instructions to balance preference toward 0.5 improved Spearman ρ by ~17% (Mistral-7B), ~10% (Llama-3-8B) [16] | Randomize/paraphrase prompt; constrain rubric; validate per-prompt |

Two honesty notes the dossier's adversarial pass insists on. First, on the two position-bias numbers per model: **consistency** (agreement under A/B swap) and **first-position rate** are *different metrics* — Claude-v1 is 23.8% consistent *and* 75.0% first-biased, not "24–70%" [1]. Second, on self-enhancement: Zheng et al. explicitly hedge — "due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias" [1]. So present the +10%/+25% figures as *suggestive*, not established effect sizes. That hedge does *not* weaken the underlying warning, because the **mechanism** is separately established: Panickssery, Bowman & Feng (*LLM Evaluators Recognize and Favor Their Own Generations*, NeurIPS 2024, arXiv:2404.13076) show via **fine-tuning experiments** a **linear correlation between a model's self-recognition capability and the strength of its self-preference bias**, with controlled experiments indicating the link is causal rather than a straightforward confound (studied on GPT-4 and Llama 2) [4]. The load-bearing consequence: **using a model's own judge scores as a reward signal risks self-reinforcement** — the model learns to prefer what it recognizes as its own, not what is good. For ek this is a hard design constraint: a `JudgeSignal` should default to a *different model family* than any agent it evaluates, and this should be a checked, not merely documented, default.

---

## 4. Judge validation — the load-bearing discipline

Everything above is worthless if you deploy a judge without measuring whether *this* judge, on *your* task, agrees with *your* humans. This is the single most important operational message of this report: **a judge is a measurement instrument and must be calibrated against ground truth before use, exactly like any sensor.**

### 4.1 How to measure judge-vs-human agreement (reuse the harness)

Measure agreement with **chance-corrected metrics on a held-out labeled set**, not raw accuracy. Raw agreement is "generally not recommended and can be misleading when classes are imbalanced" [18] — and pass/fail labels are almost always imbalanced (most outputs pass). Use **Cohen's kappa** for judge-vs-one-human (or human-vs-human) and **Krippendorff's alpha** for multi-rater / partial-label settings — precisely the IAA functions ek's harness already ships (`cohen_kappa`, `krippendorff_alpha`, `percent_agreement`; -> ek_02). This is a *direct reuse*, not a rebuild: judge validation is IAA where one "rater" happens to be a model. Beyond kappa, track **per-class rates — True Positive Rate and True Negative Rate — on the held-out set** [18][20], because a judge can have a great TPR (catches good outputs) and a terrible TNR (misses defects), and the aggregate hides it. Concretely, the field guidance is that a validated judge needs **100+ labeled examples and ongoing (weekly, then monthly) maintenance** [18][20] — validation is not one-and-done. And when you *report* those agreement numbers, do it with proper statistical rigor — confidence intervals on the judge-vs-human rate, not a bare point estimate — per the reporting guidance in *How to Correctly Report LLM-as-a-Judge Evaluations* [15]; this reuses the same statistical-reporting discipline (CIs, clustered/error-barred evals) that ek_11 makes central for its metrics.

### 4.2 The precision/recall asymmetry that will bite you

The most important empirical fact about judges as *defect detectors*: they have **high precision but low recall on defects**. On MT-Bench general chat, GPT-4-judge agreement with humans is 85% (Arena 83–87%) — but on **faithfulness / factual consistency it collapses**: Spearman ρ ≈ 0.55 on faithfulness, ρ ≈ 0.27–0.46 on factual consistency; GPT-3.5-turbo separates factual from hallucinated summaries only 58.5% of the time [16]. Decomposed: LLM-evaluators identify **>95% of consistent summaries (high precision) but only 30–60% of inconsistent ones (low recall)** [16]. The engineering consequence is stark and must be wired into ek's decision policy: **a judge saying "looks good" is weak evidence of correctness — it misses 40–70% of real defects.** Never let a bare "pass" from a judge auto-accept a high-stakes item. This is why structural signals (§7) and the cost-ratio `ρ = c_FN/c_FP` framing (-> ek_03) matter: when a false negative is expensive, the judge alone cannot carry the accept decision.

### 4.3 Meta-eval benchmarks — validate the judge, don't ship them

There is a growing family of benchmarks whose job is to grade *judges*. Use them to *select and validate* a `JudgeSignal` offline — they are datasets + code, **not runtime metric libraries**.

| Benchmark | What it tests | Headline finding | License | Ref |
|---|---|---|---|---|
| **JudgeBench** (ICLR 2025) | *Objective* correctness pairs (knowledge/reasoning/math/code) | Vanilla GPT-4o **50.86%** (≈ random); best general judge Claude-3.5-Sonnet **64.29%**; reasoning models better (o1-preview 75.4%, o3-mini-high 80.9%); **~31%** spread best-vs-worst | code+data, ScalerLab | [3] |
| **RewardBench** | 2,850 prompt-chosen-rejected trios; Chat / Chat-Hard / Safety / Reasoning | Saturated — GPT-4o **83.3%** overall; reasoning subsets up to **~97%** (likely contamination) | code Apache-2.0, data ODC-BY | [5] |
| **LLMBar** (ICLR 2024) | 419 instruction-following pairs; Natural + Adversarial (deceptive superficial appeal) | Even best evaluators have substantial room; **judge reliability is prompt-sensitive** (rankings flip across prompts) | princeton-nlp/LLMBar | [6] |
| **MT-Bench-human** | 3K expert votes on MT-Bench | The founding judge-vs-human agreement set | — | [1] |

Two corrections the adversarial pass forced, worth stating precisely so ek's docs don't propagate errors: (1) the JudgeBench near-random number is **vanilla GPT-4o at 50.86%**, *not* 56% — 56.57% is the *Arena-Hard-prompted* judge; and the 64.29% ceiling is the best *general-purpose* judge, not a "hardest split," since reasoning models climb to ~81% [3]. (2) RewardBench's headline is **GPT-4o 83.3% overall** (the 91.96% figure sometimes quoted is not in the paper); "up to ~97%" is the *reasoning subset*, not overall [5]. The takeaway for ek: **judges are near-chance on hard objective reasoning (JudgeBench) yet saturated on preference alignment (RewardBench)** — so a judge you validated on chat quality tells you *nothing* about its trustworthiness on math/logic. Validate per task type; don't extrapolate.

---

## 5. Rubric-based judging, judge confidence, and calibration

The best practitioner methodology is not "pick an off-the-shelf metric" — it is **align the judge to one domain expert via iterative critique-and-label**. Husain's Honeycomb case study: an LLM-judge prompt seeded with few-shot critiques from a single domain expert reached **>90% agreement with that expert in 3 iterations** [18]. The process — the expert makes binary pass/fail calls and writes critiques "detailed enough to use in a few-shot prompt," then you fold those into the judge and re-measure — is the "benevolent dictator" pattern: in most organizations one or two people's judgment is the ground truth, and the judge's job is to *replicate that specific person*, not embody a universal notion of quality [18]. This is a strong argument for ek treating the rubric/criteria as **injected configuration** (open-closed), not a hardcoded metric.

The corollary is a named anti-pattern: **off-the-shelf prefab metrics (Ragas defaults, BERTScore, ROUGE) used without validation are "endemic" abuse** — "all you get is you don't know what they actually do; in the best case they waste your time, in the worst case they create an illusion of confidence that is unjustified" [18]. BERTScore and ROUGE in particular are "not useful for evaluating LLM outputs in most AI applications" [18]. ek's stance follows: prefab metrics are *offered* behind the registry, but the *default posture* is "validate before you trust," and the docs must say so loudly (-> ek_03's Hard-Rule-1).

**Judge confidence and calibration.** A judge can emit more than a verdict: G-Eval's probability-weighted score [2] is a confidence signal; a pairwise judge's margin is one; the self-consistency of *repeated* judge calls is another (run the judge 3× at temperature and measure verdict agreement — the same sample-and-vote trick as §7). Whatever the raw signal, **it must pass through the same `Calibrator` as OCR confidence** (temperature scaling / Platt / isotonic; -> ek_03) before a `DecisionPolicy` consumes it. A raw judge score of "0.8" is not an 80%-correct probability until you have fit a calibration map on labeled data and checked it on a reliability diagram. This is the crux of the ek mapping: **judges are just another miscalibrated signal source, and ek already owns the calibration + decision machinery** (-> ek_03).

**Cost, and when a cheap judge suffices.** Judging is not free, and the cost is material: full tau-bench-airline evaluation runs cost **\$34.58–\$180.49 per model run**, versus a human-expert baseline of ~\$2.06/task [21]; a self-hosted 72B judge at ~10 runs/week is roughly \$240–470/week on 2×H100 vs \$600–1,200/week via commercial APIs [21]. This directly motivates the cascade: **run a cheap judge (or a judge-free structural signal) first, escalate to an expensive judge only on the confidence gate.** A cheap judge suffices when the task is easy for judges (general chat quality, obvious format/policy checks) and the cost of a judge error is low; reserve the expensive judge for the hard, high-stakes middle. Note the tooling gap: the pure metric libraries (Ragas, DeepEval metrics, TruLens, uqlm) **do not natively report token cost per judged item** — cost tracking lives in the observability/harness tier (Langfuse, Phoenix, promptfoo, Inspect) [22]. So a `JudgeSignal` must set its `cost_tier` explicitly and cost aggregation belongs in ek's harness, not in the metric.

---

## 6. Panels of judges (PoLL): cheaper *and* less biased

The most useful single lever for both cost and bias is to **replace one big judge with a panel of small diverse ones**. Verga et al. (Cohere), *Replacing Judges with Juries* (arXiv:2404.18796) [7]: a panel of three models from *different families* — Command-R + GPT-3.5 + Claude-3-Haiku — aggregated by vote/average, **matches or beats a single large GPT-4 judge on human correlation (Cohen's kappa) across 6 datasets and 3 judge settings, at roughly 1/7th the cost, with lower intra-model (self-preference) bias** [7]. The heterogeneity is the point: no single family's self-preference or idiosyncrasy dominates the panel, so the aggregate is less biased than any member. Two honesty caveats from the adversarial pass: GPT-4 was atypically *weak* in this particular reference-based QA setup (each individual small judge also beat it, and on HotpotQA a single Haiku edged out the full panel), so **the panel's core value is bias/variance reduction, not a large raw-accuracy jump**; and the ">7×cheaper" figure is based on 2024 API list prices, not an intrinsic property [7]. There is no canonical PoLL pip package — it is implemented ad hoc (e.g. via HuggingFace `distilabel` recipes) or inside harnesses [7] — so **ek should BUILD a thin `PanelJudgeSignal` that composes N registered `JudgeSignal`s and aggregates their verdicts**, reusing the ROVER/agreement aggregation it already has (§7, -> ek_03). This is a wrap-and-compose, not a from-scratch build.

---

## 7. Judge-free structural signals: self-consistency, hallucination, faithfulness

The second reference-free family needs *no trusted grader* — only the model's own output distribution. These are often the right *first* line of defense because they cannot import a judge's biases.

### 7.1 Self-consistency / agreement (the ROVER generalization)

Wang et al., *Self-Consistency* (ICLR 2023, arXiv:2203.11171) [8]: sample a diverse set of reasoning paths at temperature, then **majority-vote (marginalize) over final answers** instead of greedy decoding. Accuracy gains are large — GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2%, StrategyQA +6.4%, ARC-challenge +3.9% [8] — but the point *for evaluation* is different: **the vote margin is itself a confidence signal**. 3/3 agreement is far stronger evidence of correctness than a 2–1 split. This is *exactly* ek's ROVER agreement generalized: ROVER votes over multiple OCR engines' outputs of the same input (-> ek_03); self-consistency votes over multiple samples of the same model on the same input. **Reuse the ROVER aggregator; do not rebuild it** — the input is a bag of candidate outputs, the output is a consensus + an agreement score, and that score feeds the same `Calibrator`/`DecisionPolicy`. For agents, the candidates are sampled trajectories or sampled final answers; low cross-sample agreement is a strong flag/escalate signal.

### 7.2 Sampling-based hallucination detection

SelfCheckGPT (Manakul et al., EMNLP 2023, arXiv:2303.08896) [9] operationalizes self-consistency as a **zero-resource, black-box hallucination detector**: sample N stochastic responses; a fact the model actually knows recurs consistently across samples, while a hallucinated one diverges and contradicts. Five variants (BERTScore, QA, n-gram, NLI, LLM-prompting); the NLI and prompt variants achieve the best AUC-PR and strongest human correlation [9]. No external DB and no logits needed — pure black-box, which is why it wraps cleanly as a `Signal`. FActScore (Min et al., EMNLP 2023, arXiv:2305.14251) [10] is the *granular groundedness* counterpart: decompose a long generation into **atomic facts** and report the fraction supported by a reliable knowledge source (ChatGPT bios score only 58% supported; the automated estimator is within <2% of human FActScore) [10]. Note the important distinction the dossier flags: **FActScore requires a knowledge base, so it is groundedness-*with-reference*, not purely reference-free** — it sits at the boundary with -> ek_08 and belongs to ek as a `Validator` when a knowledge source exists, and a `Signal` otherwise.

### 7.3 RAG faithfulness / groundedness (a huge reference-free sub-area)

When the agent has *retrieved context*, faithfulness — does the answer follow from the context, or did the model make it up? — is a reference-free signal computable *without gold*. **Ragas** (Es et al., EACL 2024, arXiv:2309.15217, Apache-2.0) [11] defines the reference-free **RAG triad**: **faithfulness** = (# answer claims supported by retrieved context) / (total claims) — a direct hallucination detector; **answer relevancy** = generate N reverse-questions from the answer and take mean cosine similarity to the original question; **context precision/recall** over retrieved sentences [11]. TruLens (MIT) frames the complementary live-observability "RAG triad" of context relevance / groundedness / answer relevance as feedback functions. The reference-free RAG landscape, with licenses (verified 2026-07-11) that matter for ek's opt-in-extra policy (-> ek_06):

| Library | Signature capability | Reference-free? | License | ek fit | Ref |
|---|---|---|---|---|---|
| **Ragas** | RAG triad: faithfulness, answer_relevancy, context_precision/recall | Yes | **Apache-2.0** | Wrap metrics as `Signal`/`Validator` in `ek[rag]` | [11] |
| **DeepEval** | G-Eval class + RAG metrics; Pytest-style `assert` | Yes | **Apache-2.0** | Consume `GEval` as a `JudgeSignal`; RAG metrics as `Signal` | [23] |
| **TruLens** | RAG triad feedback funcs; live observability | Yes | **MIT** | Feedback funcs → `Signal`; production monitoring (-> ek_05) | [24] |
| **RAGChecker** | Claim-level entailment; retriever-vs-generator diagnostics | Yes | **Apache-2.0** | Diagnostic `Validator` | [25] |
| **ARES** | Fine-tuned judge classifiers + **PPI** confidence intervals | Yes | verify LICENSE | Statistical-correction judge; needs anchor labels | [12] |
| **SelfCheckGPT** | Sampling-consistency hallucination detection | Yes | **MIT** | `Signal` in `ek[agreement]` | [9] |
| **FActScore** | Atomic-fact factual precision | No (needs KB) | **MIT** | `Validator` when KB exists | [10] |
| **uqlm** | Black-box self-consistency + judge + ensemble scorers | Yes | **Apache-2.0** | Already ek's `[agreement]` backend | [13] |
| **lm-polygraph** | ~40 white-box (logit/hidden-state) UQ methods | Yes | **MIT** | White-box UQ `Signal` (needs model internals) | [14] |

**ARES** (Saad-Falcon et al., NAACL 2024, arXiv:2311.09476) [12] deserves a call-out: it fine-tunes lightweight LM judges on synthetic query-passage-answer data and uses **Prediction-Powered Inference (PPI)** with a few hundred human labels to give *confidence intervals* on scores and *debias* judge predictions — the same statistical-correction spirit as ek's harness. It is the principled answer to "my judge is biased and I have a small labeled anchor set." Verify its LICENSE before defaulting (dossier confidence medium) [12].

### 7.4 Black-box vs white-box UQ, and what ek already ships

Two complementary toolkits bracket the UQ space. **uqlm** (CVS Health, arXiv:2507.06196, Apache-2.0) [13] is the **black-box self-consistency** library ek *already ships behind `[agreement]`*: four scorer families — black-box (semantic entropy, number of semantic sets, non-contradiction probability, entailment probability, BERTScore, exact-match rate, cosine similarity), white-box (token-probability), LLM-as-judge, and tunable ensembles [13]. Because it needs only generated text, it works over hosted APIs and is the natural home for the self-consistency §7.1 and PoLL §6 aggregations. **lm-polygraph** (Fadeeva et al., EMNLP 2023, arXiv:2311.07383, MIT) [14] is the **white-box** counterpart — ~40 UQ/calibration methods needing logits/hidden states — for when ek runs against an open-weights agent and can see internals. The design guidance: default to uqlm (black-box, API-friendly) in `ek[agreement]`; offer lm-polygraph as an opt-in white-box extra for local models.

---

## 8. Honest caveats: where reference-free signals mislead

This report would be malpractice without a blunt caveats section, because reference-free signals *feel* authoritative and are the easiest to over-trust.

- **Judges are near-chance on hard reasoning.** JudgeBench: vanilla GPT-4o ≈ 50.86%, best general judge 64.29% [3]. If your agent's output requires verifying a hard numeric/logical claim, the judge is not a reliable oracle — you need a deterministic checker (-> ek_08) or a human.
- **Judges have low recall on defects.** They pass 30–70% of genuinely unfaithful outputs [16]. A judge "pass" is weak positive evidence; weight it accordingly in the cost-ratio `ρ` framing (-> ek_03).
- **Self-preference is real and mechanistically causal.** Do not judge a model with itself; the bias grows with self-recognition [4]. Panels with diverse families mitigate but don't eliminate it [7].
- **"Criteria drift" is a production pitfall.** Evaluation criteria *shift after you see model outputs* — eval is "an iterative, human-driven sensemaking process, not a static target" [19]. Mitigation: re-run error analysis on 100+ *fresh* traces on every feature/prompt/model change; monitor weekly for new systems, then monthly (-> ek_05).
- **Benchmark contamination / saturation.** RewardBench reasoning subsets near 97% are "likely inflated by data contamination" [3][5]; a saturated benchmark cannot discriminate judges.
- **Rigid graders penalize valid answers.** Anthropic reports Opus 4.5 scoring only 42% on CORE-Bench due to grading that penalized `96.12` when expecting `96.124991…` [17] — a reminder that even the *reference* side has judgment calls, and that model-based graders need calibration against humans and an *Unknown* escape hatch [17].
- **The self-consistency confound.** A confidently-wrong model produces *consistent* wrong samples — 3/3 agreement on a shared misconception reads as high confidence. Self-consistency measures *stability*, not *correctness*; it is a necessary-not-sufficient signal, best combined with an independent faithfulness check.

---

## 9. Mapping to ek

This section ties the report to ek's data model, facades, and registry (the CONCEPTUAL BRIDGE). The one-line thesis: **a judge is one `Signal`; faithfulness is a `Signal`/`Validator`; validation reuses the harness IAA; and everything flows through the same `estimate_quality()` calibrate→decide pipeline as OCR confidence.**

**The object.** For agents, Layer B (`AnnotatedExtraction`) becomes the **episode's metadata** — the trajectory, per-step signals, token cost, latency, and the final open-ended artifact — riding alongside a Layer A **task/tool grammar** (allowed tools + arg schemas + task-value/error-cost weights). This report operates on that Layer-B object with no gold: `estimate_quality(extraction, *, sources, signals, calibrator, validators, policy)` → `QualityReport`.

**JudgeSignal implements the `Signal` Protocol.** A judge is resolved by name from the registry and injected keyword-only (open-closed). It has three levers baked in as smart defaults: (1) `cost_tier` set **high** explicitly — the metric libraries don't report per-call cost, so ek owns that (§5) [22]; (2) protocol = binary pointwise by default (§2.4) [18], with pairwise/reference-guided as variants; (3) a **different model family** than any agent under evaluation, to dodge self-preference [4]. Optional judge backends (DeepEval's `GEval` [23], Ragas [11]) live behind `@requires_extra("rag")` / `@requires_extra("judge")` raising the actionable `pip install ek[...]`.

**Faithfulness / groundedness = a `Validator` (or `Signal`).** When a knowledge source or retrieved context exists, atomic-fact / claim-entailment checks (FActScore [10], RAGChecker [25], Ragas faithfulness [11]) are `Validator`s in the six-layer FLAG-vs-CORRECT sense (-> ek_04). When no source exists, sampling-consistency (SelfCheckGPT [9]) is a pure `Signal`.

**Judge validation reuses the harness IAA.** Selecting and validating a `JudgeSignal` is IAA where one rater is a model: reuse `krippendorff_alpha`, `cohen_kappa`, `percent_agreement` (-> ek_02) on a 100+-item held-out labeled set, and track TPR/TNR per class [18][20]. The meta-eval benchmarks (JudgeBench [3], RewardBench [5], LLMBar [6]) are **offline validation datasets**, not runtime deps — a `harness` routine that scores a candidate judge against them, never a shipped dependency.

**Self-consistency reuses ROVER.** Sample-and-vote agreement (§7.1) is ek's existing ROVER aggregator (-> ek_03) applied to samples of one model instead of outputs of several engines. `PanelJudgeSignal` (PoLL, §6) composes N `JudgeSignal`s and aggregates with the same voting machinery [7][8]. Build the thin composition; reuse the aggregation.

**One calibrate→decide pipeline.** Judge scores, self-consistency margins, and faithfulness ratios are all raw signals that flow through the **same `Calibrator`** (temperature/Platt/isotonic) and the **same `DecisionPolicy`** (accept / flag / block) as OCR confidence (-> ek_03). The judge's own *Unknown* verdict [17] maps directly onto the flag decision; escalation to a human or a stronger judge *is* selective prediction; the cascade's confidence gate *is* the decide stage. **Hard-Rule-1 applies to judges with full force: no gating on an uncalibrated signal.** A judge that says "0.8" gates nothing until that 0.8 is calibrated to a real probability on labeled data. This is not new machinery — it is the identical pipeline ek built for -> ek_03, with a judge slotted in as one (expensive, biased, must-be-validated) signal among many.

**Licensing (-> ek_06).** Verified 2026-07-11: safe permissive backends are Ragas (Apache-2.0) [11], DeepEval (Apache-2.0) [23], uqlm (Apache-2.0) [13], RAGChecker (Apache-2.0) [25], TruLens (MIT) [24], SelfCheckGPT (MIT) [9], FActScore (MIT) [10], lm-polygraph (MIT) [14], Inspect AI (MIT). **License trap: Arize Phoenix is Elastic License 2.0 (ELv2), NOT OSI-approved** — it forbids offering the software as a managed service, and even its `phoenix-evals` sub-package is ELv2, so there is no permissive escape hatch. Phoenix cannot enter ek's core or a default extra [26]; if used at all it is a clearly-flagged opt-in. **Langfuse** core is MIT *except* the `/ee` directories (separately commercially licensed) — pin to core, avoid `ee/` [27]. This is exactly the build/borrow/wrap discipline and license-landmine register ek_06 owns.

---

## 10. Recommendations (opinionated)

1. **Default to structural signals first, judge second.** Run self-consistency (uqlm, `ek[agreement]`) [13] and, where context exists, Ragas faithfulness [11] on every item. Escalate to a judge only on the confidence gate. Cheaper, less biased, no grader to validate.
2. **Ship a *binary pointwise* `JudgeSignal`** with a rubric injected as config, aligned to one domain expert via iterative critique-and-label [18]. Not Likert; not a prefab metric taken on faith.
3. **Validate every judge before it gates anything.** 100+ labeled items, Cohen's kappa + TPR/TNR (reuse harness IAA, -> ek_02), re-validated on schedule to catch criteria drift [18][19][20]. Score candidate judges against JudgeBench/LLMBar offline [3][6].
4. **Prefer a panel of cheap diverse judges (PoLL) over one expensive judge** [7]. Build `PanelJudgeSignal` over the ROVER aggregator; use different families to suppress self-preference [4].
5. **Never judge a model with itself**, and never feed a model's own judge scores back as a reward signal — the self-preference is mechanistically causal [4].
6. **Calibrate, then decide.** Judge/consistency/faithfulness scores → one `Calibrator` → one `DecisionPolicy` (accept/flag/block) → escalate on flag (-> ek_03). Hard-Rule-1: no uncalibrated gating.
7. **Set `cost_tier` high on judges and account cost in the harness, not the metric** [21][22]; use judges as the cascade's expensive tail, not its default.
8. **Keep licenses clean** (-> ek_06): permissive backends only in defaults; Phoenix (ELv2) and Langfuse `/ee` are opt-in-or-avoid [26][27].

---

## References

[1] Zheng L, Chiang W-L, Sheng Y, Zhuang S, Wu Z, Zhuang Y, et al. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). NeurIPS 2023 Datasets & Benchmarks Track. arXiv:2306.05685.

[2] Liu Y, Iter D, Xu Y, Wang S, Xu R, Zhu C. [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634). EMNLP 2023 (ACL Anthology 2023.emnlp-main.153). arXiv:2303.16634.

[3] Tan S, Zhuang S, Montgomery K, Tang W, Cuadron A, Wang C, et al. [JudgeBench: A Benchmark for Evaluating LLM-Based Judges](https://arxiv.org/abs/2410.12784). ICLR 2025. arXiv:2410.12784. Code: [ScalerLab/JudgeBench](https://github.com/ScalerLab/JudgeBench).

[4] Panickssery A, Bowman SR, Feng S. [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076). NeurIPS 2024. arXiv:2404.13076.

[5] Lambert N, Pyatkin V, Morrison J, Miranda L, Lin BY, Chandu K, et al. [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787). arXiv:2403.13787. Code: [allenai/reward-bench](https://github.com/allenai/reward-bench) (code Apache-2.0, data ODC-BY).

[6] Zeng Z, Yu J, Gao T, Meng Y, Goyal T, Chen D. [Evaluating Large Language Models at Evaluating Instruction Following (LLMBar)](https://arxiv.org/abs/2310.07641). ICLR 2024. arXiv:2310.07641. Code: [princeton-nlp/LLMBar](https://github.com/princeton-nlp/LLMBar).

[7] Verga P, Hofstätter S, Althammer S, Su Y, Piktus A, Arkhangorodsky A, et al. [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL)](https://arxiv.org/abs/2404.18796). arXiv:2404.18796.

[8] Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D. [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). ICLR 2023. arXiv:2203.11171.

[9] Manakul P, Liusie A, Gales MJF. [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896). EMNLP 2023. arXiv:2303.08896. Code: [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt) (MIT).

[10] Min S, Krishna K, Lyu X, Lewis M, Yih W, Koh PW, et al. [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251). EMNLP 2023. arXiv:2305.14251. Code: [shmsw25/FActScore](https://github.com/shmsw25/FActScore) (MIT).

[11] Es S, James J, Espinosa-Anke L, Schockaert S. [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217). EACL 2024 (System Demonstrations). arXiv:2309.15217. Code: [explodinggradients/ragas](https://github.com/explodinggradients/ragas) (Apache-2.0).

[12] Saad-Falcon J, Khattab O, Potts C, Zaharia M. [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476). NAACL 2024. arXiv:2311.09476. Code: [stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES) (pkg `ares-ai`).

[13] Bouchard D, et al. [UQLM: A Python Package for Uncertainty Quantification in Large Language Models](https://arxiv.org/abs/2507.06196). arXiv:2507.06196. Code: [cvs-health/uqlm](https://github.com/cvs-health/uqlm) (Apache-2.0).

[14] Fadeeva E, Vashurin R, Tsvigun A, et al. [LM-Polygraph: Uncertainty Estimation for Language Models](https://arxiv.org/pdf/2311.07383). EMNLP 2023 (Demonstrations). arXiv:2311.07383. Code: [IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph) (MIT).

[15] [How to Correctly Report LLM-as-a-Judge Evaluations](https://arxiv.org/abs/2511.21140). arXiv:2511.21140, 2025.

[16] Yan E. [Evaluating the Effectiveness of LLM-Evaluators (aka LLM-as-Judge)](https://eugeneyan.com/writing/llm-evaluators/). eugeneyan.com, 2024.

[17] Anthropic. [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Anthropic Engineering, 2025.

[18] Husain H. [Creating a LLM-as-a-Judge That Drives Business Results / Using LLM-as-a-Judge For Evaluation](https://hamel.dev/blog/posts/llm-judge/). Hamel's Blog, 2024.

[19] Husain H, Shankar S. [LLM Evals FAQ: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/). Hamel's Blog, 2026.

[20] Husain H, Shankar S. [LLM Evals FAQ — judge validation (TPR/TNR, Cohen's kappa)](https://hamel.dev/blog/posts/evals-faq/). Hamel's Blog, 2026.

[21] qaskills. [tau-bench Agent Evaluation Guide (2026): pass^k, cost per task](https://qaskills.sh/blog/tau-bench-agent-evaluation-guide-2026). 2026.

[22] LangChain. [LangSmith Evaluation: Online Evaluators, Cost & Latency Tracking](https://docs.langchain.com/langsmith/evaluation). LangChain Docs, 2026.

[23] Confident AI. [DeepEval: The LLM Evaluation Framework (Apache-2.0, v4.0.9)](https://pypi.org/project/deepeval/). PyPI / [confident-ai/deepeval](https://github.com/confident-ai/deepeval), 2026.

[24] TruEra/Snowflake. [TruLens: Evaluation and Tracking for LLM Experiments and AI Agents (MIT)](https://github.com/truera/trulens). GitHub, 2026.

[25] Ru D, et al. [RAGChecker: A Fine-grained Framework for Diagnosing RAG (Apache-2.0)](https://github.com/amazon-science/RAGChecker). amazon-science/RAGChecker, 2024.

[26] Arize AI. [Phoenix: AI Observability & Evaluation (Elastic License 2.0)](https://github.com/Arize-ai/phoenix). GitHub, 2026.

[27] Langfuse. [Open source AI engineering platform (MIT core, /ee enterprise)](https://github.com/langfuse/langfuse). GitHub, 2026.
