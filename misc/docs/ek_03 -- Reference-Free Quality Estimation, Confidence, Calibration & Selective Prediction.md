# R3 — Reference-Free Quality Estimation, Confidence, Calibration & Selective Prediction
*for Information-Extraction (IE) and OCR Systems*

**Author:** Thor Whalen

> **How to use this file:** This is a research/landscape document, not a code library. It leads with a decision flow, separates pre-LLM (classical) from LLM-era methods throughout, and flags the cheapest reliable options first. Save as `R3-reference-free-QE.md`.

---

## TL;DR

- **You can almost always get an actionable quality signal at inference time, even with no gold answer — but raw confidence is not it.** The reliable pattern is a three-stage pipeline: **(1) obtain a confidence signal** (intrinsic posterior/logprob, ensemble agreement, an auxiliary QE model, or a deterministic verifier), **(2) calibrate it** so the number means a probability (temperature/Platt/isotonic; measure with ECE), and **(3) turn it into a decision** with a distribution-free guarantee (conformal prediction / conformal risk control) and a selective-prediction operating point on the risk–coverage curve.
- **Prefer the cheapest reliable signal first.** If the extractor already emits well-behaved per-unit confidence (Google Cloud Vision, Azure Document Intelligence, Mathpix), you need only calibrate + conformalize — no LLM, no ensemble. If it emits *uncalibrated* confidence (Tesseract), calibrate first. If it emits token logprobs (TrOCR, pix2tex, GPT-4o, Mistral OCR), aggregate logprobs (length-normalized geometric mean) then calibrate. Only when the extractor emits **nothing usable** (Claude Vision, OCR.space, Apple Vision) should you spend money on an external signal — and even then, deterministic verifiers (schema/range/cross-field/dictionary) and multi-engine agreement (ROVER) beat LLM-as-judge for most fields.
- **Conformal prediction is the keystone that makes "flag the worst X% with a known error rate" rigorous.** Split conformal gives finite-sample, distribution-free coverage with a one-line quantile; conformal risk control extends it to monotone losses (false-negative rate, token-F1). All you need is a modest exchangeable calibration set of labeled examples — typically a few hundred per field type.

---

## The Decision Flow (read this first)

```
                 ┌─────────────────────────────────────────────┐
                 │  What confidence does the extractor emit?    │
                 └─────────────────────────────────────────────┘
                                    │
   ┌────────────────┬──────────────┼───────────────┬───────────────────────┐
   ▼                ▼              ▼               ▼                       ▼
WELL-BEHAVED    UNCALIBRATED   TOKEN LOGPROBS   LINE-LEVEL ONLY        NOTHING USABLE
per-unit conf.  per-unit conf. (generative)     (CTC line score)      (no conf/logprob)
(GCV, Azure,    (Tesseract)    (TrOCR, pix2tex,  (Paddle/Easy/Rapid)   (Claude Vision,
 Mathpix)                       GPT-4o, Mistral)                        OCR.space, Apple)
   │                │              │               │                       │
   │                │              ▼               │                       │
   │                │   aggregate token logprobs    │                      │
   │                │   (length-normalized geo-mean)│                      │
   │                │              │               │                       │
   ▼                ▼              ▼               ▼                       ▼
 [AGGREGATE to the field/record level: min over chars, mean, or learned pooling]
                                    │
                                    ▼
                 ┌─────────────────────────────────────────────┐
                 │  CALIBRATE on a labeled holdout              │
                 │  • binary "field correct?" → Platt/isotonic  │
                 │  • logits available → temperature scaling    │
                 │  • measure ECE + reliability diagram         │
                 └─────────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────────────┐
                 │  DECIDE with a guarantee                     │
                 │  • split conformal → p-value / prediction set│
                 │  • conformal risk control → bound FNR/F1     │
                 │  • selective prediction → pick operating pt  │
                 │    on the risk–coverage curve (cost-aware)   │
                 └─────────────────────────────────────────────┘
                                    │
                       accept  ◄────┼────►  flag (HITL)  ────►  block/reject
```

**When the extractor emits NOTHING usable**, take the external-signal branch, cheapest first:
1. **Deterministic verifiers** (≈ free): schema/type checks, range/format (regex, checksums like IBAN/Luhn), cross-field consistency (totals = sum of line items), dictionary/gazetteer lookup, n-gram language-model perplexity.
2. **Multi-engine agreement** (cost = N engines): run 2–3 cheap OCR engines, align with ROVER, treat agreement as confidence.
3. **Auxiliary QE model** (one model load): a trained reference-free estimator (COMET-QE/CometKiwi lineage idea, ported to IE) or a fine-tuned token classifier predicting "field correct?".
4. **LLM self-consistency / LLM-as-judge** (most expensive): sample K times and vote; or ask a judge model. Use only where 1–3 are insufficient.

---

## 1. Confidence Sources

The reference-free signal comes from four families. I separate **pre-LLM / classical** from **LLM-era** methods throughout.

### 1a. Intrinsic confidence

**Pre-LLM / classical.** Discriminative OCR and sequence labelers expose internal posteriors:
- **Per-character / per-word posteriors.** Engines such as Google Cloud Vision and Azure Document Intelligence emit rich per-unit confidence; Tesseract emits per-symbol/word confidence (and alternatives via its choice iterator) that is *locally informative but globally uncalibrated*. A patent on OCR output verification (US 9,384,423) describes the confidence score as, for a probabilistic model, "the posterior probability of the output text string given the image data," and for a character-classification system "the arithmetic or geometric average of the individual character classification scores" — and notes it is "used to trigger a 'reject' decision … where the confidence is below a threshold." This is exactly the selective-prediction reject option, decades before the name.
- **CTC engines** (Tesseract, EasyOCR, RapidOCR, PaddleOCR) produce frame/line scores but **no token logprobs by construction** — there is no autoregressive factorization to read off. PaddleOCR/RapidOCR/EasyOCR give only a line-level recognition score, which is *non-local*: a single bad character is diluted across the whole line, which empirically weakens its value for error detection.
- **NER/CRF sequence labelers** give per-token marginals (forward–backward) and a global sequence score (Viterbi). These are the IE analogue of OCR posteriors; the CRF captures label dependencies (e.g., I-PER cannot follow B-LOC) so token errors are correlated.

**LLM-era.** Generative extractors expose **sequence-level and token-level logprobs**:
- **How to recover them.** TrOCR and pix2tex are generative locals whose token logprobs are directly recoverable from the decoder. OpenAI GPT-4o/4.1 *can* return per-BPE-token logprobs (`logprobs=True`), though this is flaky on image inputs and can be empty under strict `json_schema` decoding. Claude Vision returns **no logprobs and no confidence** — zero intrinsic QE. Mistral OCR emits opt-in confidence that is **logprob-derived**: when `confidence_scores_granularity` is set, the OCR response includes named fields `average_page_confidence_score`, `minimum_page_confidence_score`, and (at word granularity) `word_confidence_scores` — i.e., aggregates of token-level log-probabilities.
- **How to aggregate token logprobs into a field/sequence score.** A raw sum of log-probabilities is length-biased — longer fields look worse because every token adds a negative term. The standard fix is **length normalization**: divide the summed log-prob by `length^α`. Google's NMT system (Wu et al. 2016) found "α ∈ [0.6–0.7] was usually found to be best." For a per-field quality score I recommend the **length-normalized geometric mean of token probabilities** (equivalently, `exp(mean log p)` — the perplexity-inverse), which is exactly the family Mistral OCR reports:

  ```
  field_score = exp( (1/T) * Σ_{t=1..T} log p(token_t) )      # geometric mean of token probs
  # or with a tunable length penalty α:
  field_score = exp( (Σ_t log p(token_t)) / T^α )
  ```
  Use **min over tokens** when you care about the *weakest* character (catching a single transposed digit in an amount); use the geometric mean when you care about *overall* field plausibility. Aggregation choice is a modeling decision to validate on held-out data — keep it pluggable.

**Cost/signal.** Intrinsic confidence is essentially **free** (already computed during inference). The catch: it is typically **miscalibrated** (Section 2), and for RLHF-tuned LLMs it is *systematically overconfident*. Treat it as a raw score to be calibrated, never as a probability.

### 1b. Ensemble / agreement-based

**Pre-LLM / classical — ROVER.** NIST's **Recognizer Output Voting Error Reduction (ROVER)** (Fiscus, 1997) combines outputs of multiple recognizers when no reference is available. It runs in two modules: (1) an **alignment** module merges the system outputs into a single **word transition network (WTN)** via iterative dynamic-programming alignments; (2) a **voting** module scores each branch point and "selects the best scoring word (with the highest number of votes) for the new transcription." Voting can use frequency of occurrence alone or also confidence scores. The alignment complexity is roughly `O(N·l·L·L')` in the number of recognizers and sequence lengths, which is why ROVER is historically run with only a handful of systems. **Applicability to OCR:** direct — OCR is a transcription task with the same structure as ASR, and ROVER's vote agreement doubles as a *confidence signal* (positions where engines disagree are exactly where to flag). Later work (LV-ROVER) adds lexicon verification; Jalalvand et al. (2017) add a trained QE model to rank ROVER inputs when decoder confidences are unavailable or "tend to over estimate the real quality of the recognized words."

**LLM-era — self-consistency and sampling-variance uncertainty.**
- **Self-consistency** (Wang et al., 2022) samples a diverse set of reasoning paths at temperature > 0 and takes the **plurality/majority vote** over final answers. It "boosts the performance of chain-of-thought prompting with a striking margin … including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%)" (with PaLM-540B the absolute GSM8K jump was 56.5% → 74.4%). For IE, the analogue is sampling K extractions and voting per field; vote share is a confidence proxy. Notably, Wang et al. found that voting beat weighting by the model's own probabilities "because the conditional probability generated from the language model is not well calibrated" — a direct argument for agreement over raw confidence.
- **Semantic entropy** (Farquhar et al., *Nature* 2024) computes uncertainty over **meanings** rather than token strings: sample multiple generations, cluster them by **bidirectional entailment**, and compute entropy over clusters. High semantic entropy flags "confabulations — which are arbitrary and incorrect generations." The method "works across datasets and tasks without a priori knowledge of the task, requires no task-specific data and robustly generalizes to new tasks not seen before." Cost is real: the original used ten generations per prompt plus a quadratic number of entailment checks — a 5-to-10-fold increase in computation that "hinders practical adoption"; cheaper variants exist (Semantic Entropy Probes, Kossen et al. 2024; Bayesian estimation on a budget, 2025).

**Cost/signal.** Ensemble/agreement gives a strong signal but multiplies inference cost by N (engines or samples). For OCR, **deterministic multi-engine ROVER** is far cheaper than LLM sampling and should be tried first.

### 1c. Trained auxiliary QE models — the Machine-Translation QE lineage

Reference-free **Quality Estimation (QE)** is a mature field built around the **WMT QE shared tasks** (annual since 2012). The lineage:
- **Feature-engineered era:** **QuEst** (Specia et al., 2013) and **QuEst++** (Specia et al., 2015) extracted linguistic features for support-vector-regression / decision-tree regressors. **deepQuest** (Ive et al., 2018) brought neural (and document-level) QE.
- **Neural predictor-estimator era:** **OpenKiwi** (Kepler et al., 2019, PyTorch) implemented the winning WMT 2015–18 systems (QUETCH, NuQE, Predictor-Estimator, and stacks) for word- and sentence-level QE. **TransQuest** (Ranasinghe et al., 2020) used cross-lingual transformers (XLM-R) and won WMT 2020 sentence-level QE, reporting roughly a 0.1–0.2 Pearson gain over OpenKiwi and 0.3–0.4 over QuEst++.
- **COMET-QE / CometKiwi era (current SOTA):** **CometKiwi** (Rei et al., WMT 2022, `Unbabel/wmt22-cometkiwi-da`) is a **reference-free** estimator predicting quality "based solely on the source sentence and the generated output," built on the COMET framework joined with OpenKiwi's predictor-estimator architecture and a word-level tagger. It won all WMT 2022 QE subtasks; the scaled-up 2023 version (`wmt23-cometkiwi-da-xl`) extended SOTA to word/span/sentence granularity.

**What transfers to OCR/IE.** The architecture is the transferable idea: a **trained model that takes (input, output) and predicts a quality score without a reference**. For OCR, the "source" is the image (or image features) and the "output" is the recognized text; for IE, the source is the document and the output is the structured field. A practical port: fine-tune a small token classifier (or a BERT-style model that ingests OCR confidence as a feature) to predict "is this field/token correct?". The **ConfBERT** work (Hemmer et al., DAS 2024) does exactly this — it "incorporates OCR confidence scores into token embeddings" for post-OCR error detection, and importantly found that for **well-calibrated OCRs, a simple confidence-only baseline** (pick the threshold that maximizes validation F1) can match or beat the heavier model. That is the cheapest-reliable-option principle, empirically validated; the same paper documents large ECE gaps between commercial OCRs (well-calibrated) and open-source ones.

**Cost/signal.** Training cost + one model load at inference. Strong signal, but only worth it when intrinsic + verifier signals are insufficient. The trained estimator must itself be calibrated.

### 1d. Verifier / checker signals (deterministic — cheapest of all)

Pre-LLM, deterministic, and frequently the **highest ROI**:
- **Schema / type checks**: is the field the right type, length, enum?
- **Range / format**: regex, date validity, checksums (Luhn for cards, IBAN, ISBN).
- **Cross-field consistency**: invoice total = Σ line items; tax = rate × subtotal; dates ordered.
- **Dictionary / gazetteer / lexicon** lookup (LV-ROVER's lexicon verification is this idea inside the voting loop).
- **n-gram language-model perplexity** as a cheap fluency/plausibility score.

These produce a hard pass/fail or a cheap continuous score, require no labels to *build*, and catch the errors that confidence scores miss (a confidently-misread but plausible digit). They should be the **first** line of every IE QE stack.

---

## 2. Calibration — making the score *mean* something

A confidence of 0.9 should mean "correct 90% of the time." Modern neural nets violate this badly. **Guo et al. (2017), "On Calibration of Modern Neural Networks"** showed "modern neural networks, unlike those from a decade ago, are poorly calibrated," with depth, width, weight decay, and Batch Normalization all implicated, and that they are systematically **overconfident**.

### Post-hoc calibration methods

- **Temperature scaling** (Guo et al. 2017): rescale logits by a single scalar `T` chosen to minimize NLL on a validation set: `softmax(z/T)`. It is "surprisingly effective," "takes a millisecond," "can be implemented in 2 lines," and **does not change the argmax** (accuracy is unchanged). `T>1` softens overconfidence; `T→0` is hard-thresholding (overconfident), `T→∞` is uniform (underconfident). Requires access to logits. This is the default first thing to try when logits are available.
- **Platt scaling**: fit a logistic regression (sigmoid) mapping the score → correctness. Works on any scalar score (no logits needed), so it is the natural choice for aggregated OCR confidence or aggregated logprobs. Temperature scaling is a single-parameter special case of Platt scaling.
- **Isotonic regression**: non-parametric monotonic fit; more flexible than Platt, needs more calibration data, can overfit on small sets. **Histogram binning** is the simplest binning method and produces discrete predictions whose ECE can be estimated without further binning.

### Measuring calibration

- **Expected Calibration Error (ECE)**: bin predictions by confidence, take the weighted average gap between mean confidence and mean accuracy per bin. Caveat: ECE is sensitive to binning and can mislead (confidence reliability diagrams can misrepresent effective miscalibration); histogram-binning methods allow unbinned estimates.
- **Reliability diagrams**: plot accuracy vs. confidence per bin; the diagonal is perfect calibration.
- For OCR specifically, the **Detection-ECE (D-ECE)** extends ECE to localized/bounding-box outputs.

### Sequence- and field-level calibration is harder

Calibrating a *single* classification is easy; calibrating a **structured/sequence** output is not. A sequence has many tokens, each with its own posterior, and the quantity you actually care about — "is the whole field/record correct?" — is a different (and rarer) event than "is this token correct?". Practical guidance:
- Calibrate at the **granularity of the decision**. If you gate on fields, build a binary "field-correct" target and calibrate the aggregated field score against it (Platt/isotonic). Don't calibrate per-token and hope it composes.
- For NER/sequence labeling, per-token calibration is necessary but not sufficient; structure (CRF label dependencies) means token errors are correlated.
- For LLM token probabilities, **RLHF makes things worse**: RLHF-tuned models are "overconfident with a more sharpened output probability" and even verbally overconfident, because reward models used for PPO "exhibit inherent biases towards high-confidence scores regardless of the actual quality of responses" (Leng et al. 2024). Pre-trained (pre-RLHF) base models are typically better calibrated (Kadavath et al. 2022). Conversely, Tian et al. (2023) found that for some strong RLHF models, **verbalized** confidence can be "better-calibrated than the model's conditional probabilities" — worth testing per model.

### Libraries (current capabilities)

- **scikit-learn**: `CalibratedClassifierCV` (Platt via `sigmoid`, or `isotonic`) with cross-validation; `calibration_curve` for reliability diagrams. The pragmatic default for any scalar-score calibration. BSD-3.
- **netcal** (`EFS-OpenSource/calibration-framework`): a Python 3 library purpose-built "for measuring and mitigating miscalibration of uncertainty estimates." Implements binning (`HistogramBinning`, `IsotonicRegression`, BBQ), scaling (`TemperatureScaling`, `LogisticCalibration`/Platt, Beta calibration), metrics (`ECE`, `MCE`, `ACE`, `MMCE`), and crucially **D-ECE** for detection/localized outputs — relevant to OCR bounding boxes. Multi-class is handled one-vs-all.
- **torch-uncertainty**: PyTorch-native temperature scaling and broader uncertainty tooling, for in-training and post-hoc use.
- For distribution-free post-hoc calibration without distributional assumptions, **df-posthoc-calibration** (histogram binning with unbinned ECE estimates).

---

## 3. Conformal Prediction — distribution-free thresholds with a guarantee

Calibration makes a score *mean* something on average; **conformal prediction (CP)** turns it into a **decision with a finite-sample, distribution-free coverage guarantee**. This is the rigorous way to do "flag the worst X% with a known error rate."

### Split (inductive) conformal prediction

Introduced by Vovk, Papadopoulos et al. (2002–2005), **split CP** is the workhorse. Given a held-out **calibration set** of `n` labeled examples (exchangeable with test data — the only assumption), and a **nonconformity score** `s(x,y)` (e.g., `1 − calibrated_confidence`):

```
1. Compute calibration scores  s_1, ..., s_n.
2. Set q̂ = the ⌈(n+1)(1−α)⌉ / n  empirical quantile of {s_i}.
3. Prediction set:  C(x) = { y : s(x,y) ≤ q̂ }.
Guarantee:  P( Y_test ∈ C(X_test) ) ≥ 1 − α     (marginal, finite-sample).
```

The guarantee holds for **any** score function and **any** data distribution, with a tiny `O(1/n)` slack. For a *gating* application you often don't need full prediction sets: calibrate a per-instance **conformal p-value** and flag anything below threshold — coverage then tells you precisely what fraction of truly-bad items you catch. Key limitation: the guarantee is **marginal**, not **conditional** — distribution-free *conditional* coverage is provably impossible (Barber et al. 2019), so per-field guarantees require class-conditional/Mondrian calibration (calibrate separately per field type).

### Conformal Risk Control (CRC)

**Angelopoulos, Bates, Fisch, Lei & Schuster (2022)** extend CP "to control the expected value of any monotone loss function." Instead of coverage, you can bound the **false-negative rate**, **token-level F1**, or graph distance — exactly the metrics IE cares about; the worked examples in the paper "bound the false negative rate, graph distance, and token-level F1-score." The procedure generalizes split CP and is tight up to `O(1/n)`. PyTorch reference code is published (`aangelopoulos/conformal-risk`). For IE: "guarantee that the flagged-as-OK set has FNR ≤ 5%" is a CRC statement.

### CP for sequences, NER, and structured output

- **NER:** Singer, Sengupta & Pazdernik (2026) adapt sequence-labeling NER to produce "uncertainty-aware prediction sets … guaranteed to contain the correct labeling with a user-specified confidence level," with both full-sequence and subsequence variants and **class-conditional** coverage accounting for sentence length, language, entity type, and entity count. This is the most directly relevant recent CP-for-IE result (treat as preliminary until independently reproduced).
- **Ordinal/structured:** conformal risk control for ordinal classification exists; hierarchical-label CP returns an intermediate node that implicitly represents a confidence set.

### CP for LLMs / conformal factuality

- **Conformal Language Modeling** (Quach, Fisch, Schuster, Yala, Sohn, Jaakkola & Barzilay, ICLR 2024): calibrates a **stopping rule** for sampling LM outputs so the candidate set "covers at least one acceptable response," plus a rejection rule to drop low-quality samples — for open-domain QA, summarization, and radiology report generation.
- **Conformal Factuality** (Mohri & Hashimoto, ICML 2024): frames correctness as entailment and uses CP as a **back-off algorithm** that "progressively makes LM outputs less specific (and expanding the associated uncertainty sets)" by removing low-confidence sub-claims until a high-probability correctness guarantee (80–90%) holds. "Applies to any black-box LM and requires very few human-annotated samples." Code: `tatsu-lab/conformal-factual-lm`.
- **Conformal abstention** (Yadkori et al. 2024) uses conformal risk control to upper-bound hallucination risk. Caveat: language generation **violates exchangeability** (conditional, recursive) and the output space is combinatorial, so most CP-for-NLP works operate at the sentence/claim level or reformulate as multiple-choice.

### Libraries (current, 2026)

- **MAPIE** (`scikit-learn-contrib/MAPIE`, BSD-3, maintained by Capgemini Invent): scikit-learn-compatible intervals (regression, time series), prediction sets (classification), and risk control (multilabel, segmentation). **MAPIE v1 is live** with "major changes to the API"; as of June 2026 the latest documented build is **v1.4.1**. "It implements peer-reviewed algorithms that are model and use case agnostic and possesses theoretical guarantees under minimal assumptions on the data and the model." The default choice for an sklearn-centric IE stack. (Its published "MAPIE Roadmap 2026" lists risk control for "LLM-as-Judge and image segmentation" plus exchangeability tests as *planned, not yet shipped* — verify availability at build time.)
- **crepes** (`henrikbostrom/crepes`, BSD-3): model-agnostic conformal **classifiers, regressors, and predictive systems** (CPS — full CDFs), with standard, normalized, and **Mondrian** (class-conditional) variants. Wraps any sklearn-style model. Ideal when you want class-conditional (per-field-type) coverage. NumPy-based (no GPU).
- **TorchCP** (`ml-stat-Sustech/TorchCP`, LGPL-3.0): PyTorch-native, GPU-accelerated, with explicit **LLM and GNN** support, CP-specific training, and online prediction — "achieving up to 90% reduction in inference time on large datasets." Use when your extractor is a PyTorch model and you want CP on-device.
- **puncc** (`deel-ai/puncc`, MIT): "Predictive UNcertainty Calibration and Conformalization," scikit-learn/PyTorch/TF-compatible via wrappers; regression, classification, anomaly detection. Slower than TorchCP on large data.
- **nonconformist** (`donlnz/nonconformist`): the classic implementation, sklearn-compatible — but **effectively unmaintained** (last PyPI release June 2017, docs self-described as "severely deprecated"). Use MAPIE or crepes instead for new work.

---

## 4. Selective Prediction — the decision rule

**Selective prediction** (a.k.a. reject option, prediction-with-abstention, classification-with-rejection) lets the system **abstain** rather than emit a low-confidence answer. This is the formal home of "accept / flag / block."

### Formalism and the risk–coverage curve

A selective predictor is a pair `(f, g)`: a predictor `f` and a **selection function** `g(x) ∈ {0,1}`. The common choice is a threshold on a confidence-rate function: `g(x) = 1[ conf(x) ≥ τ ]`. Two quantities trade off:
- **Coverage** `φ = E[g(x)]` — the fraction of inputs you act on automatically.
- **Selective risk** `R(f,g) = E[loss · g] / E[g]` — the error rate *among the accepted*.

The **risk–coverage curve** plots one against the other. You **choose the operating point** `τ` to hit a target risk (e.g., ≤1% error on the auto-accepted slice) or a target coverage (e.g., auto-process 80%, route 20% to humans).

### Softmax-response baseline and SelectiveNet

- **Geifman & El-Yaniv (2017), "Selective Classification for Deep Neural Networks"**: the **softmax-response (SR)** baseline thresholds max-softmax; they give a method to "set a desired risk level" so the classifier "rejects instances as needed, to grant the desired risk (with high probability)." SR and MC-dropout perform similarly.
- **SelectiveNet** (Geifman & El-Yaniv, 2019): a three-headed network (prediction / selection / auxiliary) trained **end-to-end** for a target coverage, optimizing the selective risk directly. It beats SR and MC-dropout on the risk–coverage trade-off — e.g., "on the Cats vs. Dogs dataset, SelectiveNet showed a 48.16% improvement over SR at 0.80 coverage," with "8.5% at 0.95 coverage compared to SR, and a maximal advantage of 26.8% compared to MC-dropout at 0.75 coverage." Caveat: it requires *retraining* with architectural/loss changes — for an off-the-shelf extractor, threshold-on-calibrated-confidence + conformal is far cheaper and usually sufficient.

### Selective prediction for extraction and generation (LLM-era)

- **OOD detection & selective generation** (Ren et al. 2022) propose the **Quality-vs-Abstention curve** for conditional LMs: at abstention rate α, remove the lowest-scoring α-fraction and measure quality of the rest — the generation analogue of risk–coverage.
- **Selective Generation for Controllable LMs** (Lee et al., NeurIPS 2024) learns an entailment-aware selective generator controlling the **hallucination rate as a false-discovery-rate** with a probabilistic guarantee, and flags **metric misalignment** — that exact-match/F1 undercount valid answers in generation, breaking naive selective prediction. Important for IE where multiple surface forms are valid (e.g., "Jan 3 2024" vs "2024-01-03").
- **P(IK) / self-evaluation** (Kadavath et al., 2022, "Language Models (Mostly) Know What They Know," Anthropic): models can be trained to predict **P(IK)** — "the probability that 'I know' the answer to a question, without reference to any particular proposed answer." "Larger models are well-calibrated on diverse multiple choice and true/false questions," and self-evaluated **P(True)** shows "encouraging performance, calibration, and scaling." Caveat: models "struggle with calibration of P(IK) on new tasks."
- **Deep ensembles** (Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017): a simple, parallelizable alternative to Bayesian NNs that "produces well-calibrated uncertainty estimates which are as good or better than approximate Bayesian NNs" and expresses "higher uncertainty on out-of-distribution examples." The classical ensemble route to both a confidence signal and abstention.

### Connecting operating-point choice to cost

The operating point should be **cost-sensitive**. If `c_FN` is the cost of an undetected error and `c_FP` the cost of a needless human review, the cost-optimal posterior threshold derives from the ratio `ρ = c_FN/c_FP`. With a **calibrated** probability `p` of correctness, accept when the expected cost of accepting is below the expected cost of routing to a human — a direct function of `ρ`. This is why calibration (Section 2) is a prerequisite: without it, the threshold has no defensible meaning. Conformal prediction (Section 3) then lets you *guarantee* the resulting risk rather than merely estimate it.

---

## 5. Practical Recipe — recommended stacks by extractor capability

The goal: a **calibrated, actionable per-field quality score**, cheapest reliable option first. Throughout, the **deterministic verifier layer (1d) runs first and always** — it is free and catches confident-but-wrong errors that no confidence signal will.

### (a) Extractor emits well-behaved per-unit confidence — Google Cloud Vision, Azure Document Intelligence, Mathpix

```
per-unit conf ──► aggregate to field (min for "weakest link"; mean/geo-mean for overall)
              ──► calibrate field score (Platt or isotonic) against "field-correct" labels
              ──► split conformal / CRC to set the flag threshold at target risk
              ──► selective gate (accept / HITL / block) on the risk–coverage curve
```
No LLM, no ensemble. **This is the cheapest complete stack.** Validate the aggregation (min vs. mean) on held-out data. Mathpix's dual calibrated confidence for STEM is a rare case of *already-calibrated* output — still verify ECE on your data before trusting it.

### (b) Extractor emits *uncalibrated* per-unit confidence — Tesseract

```
per-symbol/word conf ──► CALIBRATE FIRST (isotonic or Platt) — Tesseract conf is locally
                          informative but globally meaningless raw
                     ──► aggregate to field ──► conformal threshold ──► selective gate
```
Tesseract's `GetChoiceIterator` alternatives add a second signal (margin between top-1 and top-2). The ConfBERT result applies: once calibrated, a **confidence-only threshold** chosen on validation F1 is a strong, cheap baseline — try it before anything heavier.

### (c) Extractor emits token logprobs — TrOCR, pix2tex, GPT-4o/4.1, Mistral OCR

```
token logprobs ──► aggregate: length-normalized geometric mean  exp(Σ log p / T^α)
                   (use MIN over tokens for "catch one bad digit"; geo-mean for overall)
              ──► calibrate (Platt on the scalar; or temperature scaling if you have logits)
              ──► conformal / CRC ──► selective gate
```
Mistral OCR already does the aggregation (`average_page_confidence_score` / `word_confidence_scores` are exp-of-mean-logprob aggregates) — you still must calibrate it. For GPT-4o, expect logprobs to be **flaky on image inputs / empty under strict `json_schema`**; have a fallback (treat as case (d)). For RLHF'd LLMs, **calibrate aggressively** (overconfidence) and consider testing verbalized confidence as an alternative signal.

### (d) Extractor emits NOTHING usable — Claude Vision, OCR.space, Apple Vision (and GPT-4o when logprobs are empty)

External signal, **cheapest first**:
```
1. DETERMINISTIC VERIFIERS (≈ free): schema, range/regex/checksum, cross-field totals,
   dictionary/gazetteer, n-gram LM perplexity.            ◄── do this always, first
2. MULTI-ENGINE AGREEMENT (cost = N cheap engines): run 2–3 OCR engines, align with
   ROVER, use vote agreement as confidence; disagreement positions = flags.
3. AUXILIARY QE MODEL (one model load): a trained reference-free estimator
   (COMET-QE/CometKiwi idea ported to IE; or a ConfBERT-style "field-correct" classifier).
4. LLM SELF-CONSISTENCY / LLM-AS-JUDGE (most expensive): sample K extractions and vote;
   or semantic-entropy clustering for a confabulation flag; or an LLM judge.
   ──► whichever signal you pick, CALIBRATE it, then conformal/selective gate.
```
The ordering reflects the stated preference: **deterministic and classical methods before LLMs**. ROVER multi-engine voting is dramatically cheaper than LLM sampling and, for OCR, often as effective. Reserve self-consistency / semantic entropy / LLM-as-judge for fields where verifiers and agreement genuinely cannot reach.

### Cross-cutting engineering notes (for a composable Python QE layer)

- **Facade + dependency injection.** Define a single `quality_estimate(extractor_output) -> CalibratedScore` facade; inject the *signal source* (intrinsic / ensemble / auxiliary / verifier), the *calibrator* (Platt / isotonic / temperature), and the *decision policy* (conformal / selective) as swappable strategies. Progressive disclosure: the default path (case a) is two calls; the hard path (case d) composes more plugins.
- **The calibration + conformal layers are extractor-agnostic** — they consume a scalar score and a labeled calibration set, so they're shared infrastructure across all four cases. This is where MAPIE/crepes (sklearn-compatible) plug in cleanly; netcal/scikit-learn handle calibration.
- **Keep a labeled calibration set** (a few hundred exchangeable examples per field type) as a first-class asset; it powers both calibration and conformal guarantees, and class-conditional (Mondrian) calibration restores per-field-type validity.
- **Monitor drift.** Conformal coverage holds under exchangeability; when the document distribution shifts, re-calibrate. Track realized coverage vs. target as a production metric.

---

## Recommendations (staged, with thresholds)

1. **Build the deterministic verifier layer first** (schema/range/checksum/cross-field/dictionary). It is free, needs no labels, and catches confident errors. *Threshold to escalate:* if verifiers alone leave residual field error above your tolerance, add a confidence signal.
2. **Use intrinsic confidence where it exists; calibrate it; measure ECE.** *Threshold:* if post-calibration ECE stays high (> ~0.05) or the reliability diagram is non-monotone, the signal is too weak — escalate to ensemble/auxiliary.
3. **Adopt split conformal / CRC for thresholds as soon as you have ~a few hundred labeled examples** per field type. Set α from business risk tolerance; use Mondrian/class-conditional calibration per field type. *Threshold:* if realized coverage drifts below target in production, re-calibrate (distribution shift).
4. **Choose the selective-prediction operating point from cost.** Compute `ρ = c_FN/c_FP`; set the accept threshold on the *calibrated* probability. Publish the risk–coverage curve so stakeholders pick coverage vs. risk explicitly.
5. **Escalate to LLM-based signals (self-consistency, semantic entropy, LLM-as-judge) only for the residual hard fields** where verifiers + agreement + auxiliary QE are insufficient. Budget for the K× cost and calibrate the resulting signal like any other.
6. **Engineer it as a composable layer** (facade + injected signal/calibrator/policy plugins) so the same calibration+conformal core serves every extractor, and swapping a vendor (e.g., Claude → Azure) changes only the signal-source plugin.

---

## Caveats

- **Marginal vs. conditional coverage.** Standard conformal guarantees are *marginal* (averaged over the test distribution). Distribution-free *conditional* (per-field, per-document-type) coverage is provably impossible (Barber et al. 2019); approximate it with class-conditional/Mondrian calibration, and state clearly which guarantee you ship.
- **Exchangeability is the load-bearing assumption.** It breaks under distribution shift (new document templates, new languages) and is *inherently* violated by autoregressive LLM generation. Conformal-for-LLM methods work around this at the claim/sentence level; treat their guarantees as approximate for free-form output.
- **Calibration is dataset- and model-specific and decays.** A temperature/Platt fit on one document population will not transfer to another; recalibrate on shift. RLHF-tuned LLMs are systematically overconfident, and their logprobs may be unavailable, empty, or unreliable on image inputs.
- **Line-level OCR confidence is non-local.** For PaddleOCR/EasyOCR/RapidOCR, a single bad character is diluted across a line score, weakening error localization — prefer engines with per-unit confidence, or add a verifier/ensemble signal.
- **Ensemble and sampling methods cost N×.** Self-consistency and semantic entropy are powerful but expensive (semantic entropy: ten samples + quadratic entailment checks, a 5–10× compute increase); cheaper variants exist but verify they hold up on your data.
- **Some cited 2026 capabilities are forward-looking.** MAPIE's announced LLM-as-Judge risk control and related 2026 roadmap items are *planned*, not shipped — verify availability at build time. The NER full-/subsequence conformal work (Singer et al. 2026) is very recent; treat its empirical claims as preliminary until independently reproduced.
- **"Free" intrinsic confidence is never free of miscalibration.** Never gate on raw posteriors or raw logprobs; the calibration step is non-optional.

---

## REFERENCES

[1] Fiscus J. [A Post-Processing System to Yield Reduced Word Error Rates: Recognizer Output Voting Error Reduction (ROVER)](https://www.nist.gov/publications/post-processing-system-yield-reduced-word-error-rates-recognizer-output-voting-error). NIST, 1997.
[2] Jalalvand S, Negri M, Falavigna D, Matassoni M, Turchi M. [Automatic Quality Estimation for ASR System Combination](https://arxiv.org/abs/1706.07238). arXiv:1706.07238.
[3] [LV-ROVER: Lexicon Verified Recognizer Output Voting Error Reduction](https://arxiv.org/pdf/1707.07432). arXiv:1707.07432.
[4] Rei R, Treviso M, Guerreiro NM, Zerva C, et al. [CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task](https://arxiv.org/abs/2209.06243). arXiv:2209.06243.
[5] Rei R, Guerreiro NM, Pombal J, et al. [Scaling up CometKiwi: Unbabel-IST 2023 Submission for the QE Shared Task](https://arxiv.org/pdf/2309.11925). arXiv:2309.11925.
[6] Unbabel. [COMET: A Neural Framework for MT Evaluation](https://github.com/Unbabel/COMET). GitHub.
[7] Kepler F, Trénous J, Treviso M, et al. [OpenKiwi: An Open Source Framework for Quality Estimation](https://ar5iv.labs.arxiv.org/html/1902.08646). arXiv:1902.08646.
[8] Ranasinghe T, Orasan C, Mitkov R. [TransQuest: Translation Quality Estimation with Cross-lingual Transformers](https://arxiv.org/pdf/2011.01536). arXiv:2011.01536.
[9] Guo C, Pleiss G, Sun Y, Weinberger KQ. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). arXiv:1706.04599.
[10] Pleiss G. [Neural Network Calibration](https://geoffpleiss.com/blog/nn_calibration.html). Blog (temperature-scaling explainer).
[11] Leng S, et al. [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/abs/2410.09724). arXiv:2410.09724.
[12] Tian K, et al. [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from LMs Fine-Tuned with Human Feedback](https://arxiv.org/pdf/2305.14975). arXiv:2305.14975.
[13] Angelopoulos AN, Bates S. [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511). arXiv:2107.07511.
[14] Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. [The limits of distribution-free conditional predictive inference](https://arxiv.org/pdf/1903.04684). arXiv:1903.04684.
[15] Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. [Conformal Risk Control](https://arxiv.org/abs/2208.02814). arXiv:2208.02814.
[16] Angelopoulos AN. [conformal-risk (code)](https://github.com/aangelopoulos/conformal-risk). GitHub.
[17] Singer M, Sengupta S, Pazdernik K. [Uncertainty Quantification for NER via Full-Sequence and Subsequence Conformal Prediction](https://arxiv.org/abs/2601.16999). arXiv:2601.16999.
[18] Quach V, Fisch A, Schuster T, Yala A, Sohn JH, Jaakkola TS, Barzilay R. [Conformal Language Modeling](https://arxiv.org/abs/2306.10193). arXiv:2306.10193 (ICLR 2024).
[19] Mohri C, Hashimoto T. [Language Models with Conformal Factuality Guarantees](https://arxiv.org/abs/2402.10978). arXiv:2402.10978 (ICML 2024).
[20] Yadkori YA, et al. [Mitigating LLM Hallucinations via Conformal Abstention](https://arxiv.org/pdf/2405.01563). arXiv:2405.01563.
[21] Geifman Y, El-Yaniv R. [Selective Classification for Deep Neural Networks](https://arxiv.org/pdf/1705.08500). arXiv:1705.08500 (NeurIPS 2017).
[22] Geifman Y, El-Yaniv R. [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://arxiv.org/pdf/1901.09192). arXiv:1901.09192 (ICML 2019).
[23] Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D. [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). arXiv:2203.11171.
[24] Farquhar S, Kossen J, Kuhn L, Gal Y. [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0). Nature 630, 625–630 (2024).
[25] Kossen J, et al. [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/abs/2406.15927). arXiv:2406.15927.
[26] Ren J, et al. [Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://arxiv.org/pdf/2209.15558). arXiv:2209.15558.
[27] Lee M, et al. [Selective Generation for Controllable Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/5a6815122f533193a022cbc41786c1cc-Paper-Conference.pdf). NeurIPS 2024.
[28] Kadavath S, et al. (Anthropic). [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221). arXiv:2207.05221.
[29] Lakshminarayanan B, Pritzel A, Blundell C. [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474). arXiv:1612.01474 (NeurIPS 2017).
[30] Hemmer A, Coustaty M, Bartolo N, Ogier J. [Confidence-Aware Document OCR Error Detection (ConfBERT)](https://arxiv.org/abs/2409.04117). arXiv:2409.04117 (DAS 2024).
[31] [System and method for OCR output verification](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9384423). US Patent 9,384,423.
[32] Wu Y, et al. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144). arXiv:1609.08144 (length normalization).
[33] Huang J, Song J, Zhou X, Jing B, Wei H. [TorchCP: A Python Library for Conformal Prediction](https://arxiv.org/abs/2402.12683). arXiv:2402.12683 (JMLR v26).
[34] [MAPIE — Model Agnostic Prediction Interval Estimator](https://github.com/scikit-learn-contrib/MAPIE). scikit-learn-contrib (BSD-3).
[35] Boström H. [crepes: conformal classifiers, regressors, and predictive systems](https://github.com/henrikbostrom/crepes). GitHub (BSD-3).
[36] Mendil M, Mossina L, Vigouroux D. [puncc — Predictive UNcertainty Calibration and Conformalization](https://github.com/deel-ai/puncc). DEEL-AI (MIT).
[37] Linusson H, et al. [nonconformist](https://github.com/donlnz/nonconformist). GitHub (classic, now unmaintained).
[38] EFS-OpenSource. [net:cal calibration framework](https://github.com/EFS-OpenSource/calibration-framework). GitHub.
[39] scikit-learn. [Probability calibration: CalibratedClassifierCV / calibration_curve](https://scikit-learn.org/stable/modules/calibration.html).
[40] Campos D, et al. [Conformal Prediction for Natural Language Processing: A Survey](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00715/125278/Conformal-Prediction-for-Natural-Language). TACL (MIT Press).