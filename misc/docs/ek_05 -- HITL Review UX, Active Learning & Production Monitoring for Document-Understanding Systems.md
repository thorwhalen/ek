# HITL Review UX, Active Learning & Production Monitoring for Document-Understanding Systems
### R5 — A Pattern Catalog and Design Guide for the Human-Review and Monitoring Layer of Information-Extraction Pipelines
**Author: Thor Whalen**

> This is the fifth report (R5) in a series accompanying a conceptual map of an information-extraction evaluation system. It targets a Python architect / AI R&D practitioner building an OCR + LLM/ML extraction pipeline with confidence scoring, validation, and human review. Progressive disclosure: the **Pattern Catalog** below is the actionable core; the five deep-dive sections that follow give the evidence and the tooling detail.
>
> *(To save as a `.md` file: copy the full contents of this report into a text editor and save as `R5_HITL_Review_UX_Active_Learning_Monitoring.md`.)*

---

## TL;DR
- **Display calibrated confidence, not raw model scores, and pair soft visual cues with cognitive-forcing friction.** Confidence numbers help humans calibrate trust only when the numbers are actually calibrated; miscalibrated confidence and fluent explanations both *increase* over-reliance, and humans cannot detect miscalibration on their own [1][2][3][9]. Build your review layer around a risk–coverage operating point: auto-accept the high-confidence head, hard-gate the low-confidence tail, and route the uncertain middle to humans with provenance (bounding boxes + multiple raw OCR/model outputs) one click away [10][14][20].
- **You get a lot "for free" from currently-maintained annotation tools, but not the loop.** Label Studio (Apache-2.0) gives bounding-box review, OCR transcript editing, pre-annotation overlays and an ML-backend; Argilla (Apache-2.0, now Hugging Face) gives confidence-filtered queues and semantic search; Prodigy gives scriptable uncertainty-sampling active learning. But the *automated active-learning loop, cost×uncertainty triage, calibration, and drift monitoring* you must wire yourself, and in Label Studio the live AL loop is Enterprise-only [25][26][30][31].
- **Monitor accuracy without labels using proxy signals, but treat them as alarms, not measurements.** Confidence-Based Performance Estimation (NannyML/CBPE) gives an unbiased accuracy estimate *only if probabilities are calibrated*; otherwise use prediction drift, abstention/reject rates and validator-firing rates as leading indicators, and confirm true error rate with a small statistically-designed audit sample [33][34][37][41].

---

## PATTERN CATALOG

Each pattern lists **(a) when to use**, **(b) the signal it consumes**, and **(c) the affordance** (the concrete UI/UX element or interaction).

### A. Confidence & Uncertainty Display

**P1 — Calibrated traffic-light field badge**
- **When:** Every extracted field in a review UI, once your confidence scores have been calibrated (temperature scaling / isotonic) and validated against held-out accuracy.
- **Signal:** Calibrated per-field confidence (post-hoc calibrated probability), bucketed into 3 bands.
- **Affordance:** Green / amber / red dot or left-border on the field, where the band boundaries map to *measured* accuracy thresholds (e.g., green ≥ 99% empirical precision). Show the numeric calibrated % on hover. Never color by raw softmax. Rationale: confidence cues calibrate trust only when calibrated [1][2].

**P2 — Value-Suppressing Uncertainty Palette (VSUP) for heatmaps**
- **When:** Document-level or region-level confidence heatmaps overlaid on the page image.
- **Signal:** Spatial confidence/uncertainty per token or region.
- **Affordance:** A VSUP that collapses color distinctions as uncertainty rises (uncertain regions fade into a neutral "confidence fog"), so reviewers cannot read precise values off low-certainty areas — empirically nudges more cautious decisions than independent bivariate encodings [4].

**P3 — Linguistic + numeric dual label**
- **When:** Confidence shown to non-technical reviewers or in audit exports.
- **Signal:** Calibrated probability.
- **Affordance:** Show *both* a calibrated number ("92%") and a controlled verbal band with a fixed, documented numeric range (IPCC-style: "likely = 66–90%"). Numeric precision is preferred by recipients for consequential decisions; bare verbal terms vary wildly between readers [5][6].

**P4 — Hypothetical Outcome Plot (HOP) / ensemble for distributional fields**
- **When:** A field has a genuine distribution of candidate values (e.g., multiple model passes or OCR engines disagree on a number).
- **Signal:** The set of candidate outputs and their probabilities.
- **Affordance:** Animated or stacked draws of the candidate values, letting reviewers "count" frequencies rather than read error bars — HOPs improved multivariate probability judgments by an estimated 35–41 percentage points over error bars/violin plots in controlled work [7][8].

### B. Soft Signals vs. Hard Gates

**P5 — Three-zone risk–coverage router**
- **When:** Always, as the backbone control. Choose two thresholds from your risk–coverage curve.
- **Signal:** Calibrated confidence / selective-prediction score.
- **Affordance:** Auto-accept above τ_high (no human), human-review band between τ_low and τ_high (soft signal), hard-gate/auto-reject below τ_low (block + mandatory escalation). Thresholds chosen to hit a target selective risk at maximum coverage (Selective Accuracy Constraint) [10][11][12].

**P6 — Soft nudge (warning, non-blocking)**
- **When:** Medium-risk fields where the cost of a false block (reviewer friction) exceeds the cost of a missed error.
- **Signal:** Validator firing (e.g., checksum soft-fail, format anomaly), mid-band confidence.
- **Affordance:** Inline amber warning with a one-line reason and a "looks right" dismissal — never a modal. Reserve interruption for genuine high-severity cases to avoid alert fatigue: drug-safety alerts are overridden in 49–96% of cases (van der Sijs et al., *JAMIA* 2006), and a 2024 meta-analysis of 16 studies puts the pooled physician override rate at 90% (95% CI 85–95%) [13][22].

**P7 — Hard gate (block + structured override)**
- **When:** High-severity fields (payment amount, legal identifiers) below τ_low, or hard-validator failure (e.g., IBAN check digit fails).
- **Signal:** Abstention/reject flag or hard-validator failure.
- **Affordance:** Block submission; require either a correction or a *structured* override reason chosen from a fixed list of 5–10 codes (not free text), which both reduces friction and generates tunable telemetry on why gates fire [13].

**P8 — Cognitive-forcing checkpoint**
- **When:** High-stakes accept actions where over-reliance is the dominant risk.
- **Signal:** A high-confidence AI suggestion on a high-impact field.
- **Affordance:** Force an action before the AI value is revealed or accepted (e.g., reviewer must commit a glance/decision first, or answer "why might this be wrong?"). Cognitive-forcing functions reduce over-reliance on incorrect AI, though they cost time and are least preferred by users — apply selectively [3][9].

### C. Provenance / Drill-Down

**P9 — Click-to-source bounding box**
- **When:** Every extracted field.
- **Signal:** Stored normalized bounding-box coordinates (0–1) per field.
- **Affordance:** Clicking a field scrolls the page image to and highlights the exact source region; store coordinates once at extraction so any later render/audit can reconstruct the highlight without re-running extraction [14][16].

**P10 — Raw-transcript disagreement diff**
- **When:** A field is derived from multiple OCR engines or model passes.
- **Signal:** Multiple raw outputs aligned at the character/token level.
- **Affordance:** Side-by-side panel with disagreement spans highlighted, plus a one-click "accept engine X" choice; character-level voting consensus can be pre-computed and shown as the default, with the split exposed for adjudication [17][18][19].

**P11 — Consensus/agreement indicator**
- **When:** Multi-engine or multi-pass extraction.
- **Signal:** Inter-model agreement (e.g., vote split, consensus entropy).
- **Affordance:** A small "3/3 agree" vs. "2/1 split" badge that doubles as a triage signal. Disagreement among independently-reasoning models is a strong error predictor: in *Design and Evaluation of Multi-Agent AI Oracle Systems* (arXiv:2605.30802, 1,189 KalshiBench prediction-market questions, Architecture A), "When all three models independently produce the same answer, accuracy reaches 88.34%. When the vote splits 2-1, accuracy drops to 57.82%, barely above chance" — a 30.5-point gap, with the unanimous high-confidence cell reaching 97.87% on 563 questions [19][20].

### D. Review-Queue Triage & Active Learning

**P12 — Cost × uncertainty priority queue**
- **When:** Whenever human review capacity is the bottleneck.
- **Signal:** Calibrated uncertainty × business cost/severity of the field or document.
- **Affordance:** A ranked work queue where items are ordered by expected risk (uncertainty × cost), not arrival order; low-confidence items in high-value document regions float to the top [16][21].

**P13 — Correction-as-gold capture**
- **When:** Every human correction.
- **Signal:** The human-corrected value + bounding box.
- **Affordance:** Persist each correction as labeled training/eval data with provenance; feed into an active-learning loop. Tools provide the capture and webhook; you own the retraining trigger [25][30].

**P14 — Uncertainty / diversity sampling for the AL loop**
- **When:** Selecting which corrected items to retrain on or which unlabeled docs to send for labeling.
- **Signal:** Model uncertainty (entropy/margin/least-confidence) and representativeness/diversity.
- **Affordance:** A batch selector that combines uncertainty sampling with a diversity/density term to avoid redundant, outlier-heavy batches — pure uncertainty sampling is biased toward outliers and can underperform random in batch mode [21][23][24].

### E. Production Monitoring

**P15 — Confidence-distribution / abstention dashboard**
- **When:** Continuous production monitoring, especially when labels are delayed/absent.
- **Signal:** Distribution of calibrated confidence, abstention/reject rate, validator-firing rates over time.
- **Affordance:** Time-series panels with reference-window comparison and alerts; a rising reject rate or a leftward-shifting confidence distribution is a label-free leading indicator of degradation [33][34][35].

**P16 — Label-free performance estimate (CBPE/DLE)**
- **When:** You need an accuracy number but ground truth is delayed.
- **Signal:** Calibrated prediction probabilities (classification) or a trained loss-estimator (regression).
- **Affordance:** A NannyML-style estimated-accuracy line with confidence bands; valid *only* under calibration + covariate-shift assumptions — display the assumption status next to it [37][38].

**P17 — Sample-for-audit estimator**
- **When:** Periodically, to ground-truth your proxy signals and estimate true error rate.
- **Signal:** A random (or stratified) sample of production outputs.
- **Affordance:** A scheduled audit queue that draws a statistically-sized sample, routes it to senior review, and reports the error rate with a confidence interval; use acceptance-sampling / sequential curtailment to minimize inspection volume [41][42].

---

## DEEP DIVE 1 — Confidence Visualization & Calibration

**The core empirical result: confidence helps only if it is calibrated, and humans cannot police calibration themselves.** The foundational study is Zhang, Liao & Bellamy (FAccT 2020), who ran two incentivized MTurk experiments on a UCI income-prediction task using a 2×2×2 design (8 conditions: show/hide confidence × show/hide prediction × full/partial model; Exp 1 N=72). AI accuracy in the trials was 75% (lowered from test accuracy by stratified sampling across confidence levels), and participant baseline accuracy averaged 65% with "only 14 of 72 participants under 60%" [1]. They found that "confidence score can help calibrate people's trust in an AI model, but trust calibration alone is not sufficient to improve AI-assisted decision making" [1]. The calibration effect on agreement was statistically significant (people agreed more on high-confidence cases and less on low-confidence ones), but this did *not* translate into a significant improvement in joint human-AI accuracy. They also warn that "confidence scores are not always well calibrated in ML classifiers" and that local explanations produced no perceivable trust-calibration effect [1].

A forthcoming controlled follow-up (Li et al., CHI 2026, N=126×2) sharpens the warning: "miscalibrated AI confidence impairs users' appropriate reliance and reduces AI-assisted decision-making efficacy, and AI miscalibration is difficult for users to detect." Users "over-rely on overconfident AI and under-rely on underconfident AI." Communicating the calibration level helps users detect miscalibration but depresses trust enough to cause under-reliance, so it doesn't improve net decision quality [2]. The practical implication for an extraction pipeline is unambiguous: **calibrate confidence (temperature scaling, isotonic regression) before display, and treat the calibration as a first-class, monitored artifact.**

**Explanations and fluent rationales can backfire.** Bučinca et al. (CSCW 2021) showed people "overrely on the AI: they accept an AI's suggestion even when that suggestion is wrong," and "adding explanations to the AI decisions does not appear to reduce the overreliance and some studies suggest that it might even increase it" [9]. Explanations are read "as a general signal of competence — rather than being evaluated individually for their content" [3]. Their remedy — cognitive forcing functions — reduced over-reliance but were "the conditions that [participants] preferred and trusted the least," and they also reduced reliance on *correct* predictions [9]. Counter-explanations (reasons the AI might be wrong) are a gentler alternative that reduces over-reliance [3].

**Color, palettes and uncertainty visualization.** Naïve green→red scales risk two failures: (1) implying precision the model doesn't have, and (2) channel interference when value and uncertainty are encoded independently. Value-Suppressing Uncertainty Palettes (Correll, Moritz & Heer, CHI 2018) solve this by allocating fewer color distinctions as uncertainty grows, so high-uncertainty values literally cannot be read precisely — a crowdsourced study showed VSUPs "encourage people to more heavily weight uncertainty information in decision-making" [4]. For distributional fields, Hypothetical Outcome Plots animate discrete draws; users of HOPs were "an estimated 35 to 41 percentage points more accurate" than error-bar/violin-plot users on multivariate probability judgments, and HOPs helped untrained observers detect trends in ambiguous data [7][8]. The drawback is sampling error from finite frames and the cognitive cost of integrating across frames [8].

**Numeric vs. verbal vs. visual expression.** A large literature (reviewed in *Trends in Cognitive Sciences*, 2022) finds senders prefer verbal probabilities but "verbal probabilities can be easily misunderstood," and assigning imprecise numeric ranges to words does not fix it; "when making consequential decisions, recipients prefer (precise) numeric probabilities" [6]. Meta-analyses show a phrase like "very likely" spans ~70–95% across readers [5]. **Recommendation:** lead with a calibrated number, optionally annotate with a *controlled* verbal band whose numeric range is fixed and documented (IPCC model), and reserve color purely as a coarse triage cue, not a precision channel.

## DEEP DIVE 2 — Soft Signals vs. Hard Gates (Selective Prediction)

**Frame the whole review layer as selective prediction.** A selective model pairs a predictor with a gating function: predict if confidence ≥ τ, otherwise abstain/reject. The trade-off is captured by the **risk–coverage curve** — selective risk (error on the accepted set) as a function of coverage (fraction predicted on) [10][12]. Summary metrics: Area Under the Risk–Coverage curve (AURC), and the **Selective Accuracy Constraint (SAC)** — the maximum coverage achievable at a target accuracy [10]. This is exactly how to *choose operating points*: fix the accuracy your downstream process requires (e.g., 99.5% on payment amounts), then read off the coverage and the corresponding threshold; everything below the threshold is gated to humans or rejected.

**Soft vs. hard, by cost asymmetry.** The choice between a non-blocking nudge (P6) and a blocking gate (P7) should follow the cost asymmetry of the field. Where a false block costs reviewer friction but a missed error is cheap/recoverable, use a soft signal. Where a missed error is expensive or irreversible, use a hard gate with mandatory correction or escalation. Escalation policies should be tiered: second reviewer → senior reviewer → auto-reject, triggered by confidence band, disagreement, or repeated validator firing.

**Warning design and alert fatigue.** The clinical-decision-support literature is the richest evidence base and a cautionary tale: clinicians override 49–96% of alerts, and alert fatigue stems from both "cognitive overload" and "desensitization from repeated exposure to the same alert" [13][22]. Practical mitigations transfer directly: tier alerts by severity, suppress duplicates, fire in-workflow at the decision point, and replace free-text override boxes with "a fixed list of 5 to 10 structured codes" that generate tunable telemetry [13]. A conservative triggering rate preserves the signal value of the alerts that do fire — Project Hermes reports "47 signals over 60 days (0.78/day) represented a manageable query rate" [13].

## DEEP DIVE 3 — Provenance & Drill-Down

**Store geometry once, reconstruct forever.** Modern extraction APIs return per-field polygons as normalized coordinates (0–1, top-left origin), which "enable visual verification by highlighting where the model found each value" [14]. Persisting these alongside the value means "when a downstream process, a reviewer, or a compliance audit needs to verify a value, you can reconstruct the exact page region without re-running extraction" [16]. For native PDFs, deterministic text coordinates (e.g., PyMuPDF `search_for`) give exact, unambiguous boxes — "page 4, paragraph 3" is fuzzy; a bounding box is not — enabling bidirectional grounding between pixel-space and text-space [15].

**Grounding is the trust mechanism.** Source-linking ("every claim a model makes is anchored to a verifiable location") is what makes hybrid text+vision pipelines trustworthy and auditable, and is increasingly built into document VLMs that emit explicit grounding tags / bounding boxes [15][14]. For review UX, the affordance is click-to-source highlighting (P9), confidence-weighted by region so low-confidence regions, not just low-confidence tokens, surface first [16].

**Multiple raw outputs and disagreement.** Presenting several OCR-engine or model-pass outputs side-by-side accelerates adjudication. Character-level voting consensus (as in Calamari's confidence-based voting) can reduce character error rates 30–50% versus a single model and should be the pre-computed default, with the disagreement spans exposed for the human [18][17]. A label-free agreement signal — "Consensus Entropy" across VLMs — "identifies trustworthy OCR outputs without labels," because "correct predictions converge while errors diverge"; on OCRBench-style evaluation, CE "improves quality verification F1 scores by 15.2, a 42.1% gain over VLM-as-Judge" [20]. Surface the vote split as both a diff view (P10) and a triage badge (P11).

## DEEP DIVE 4 — Review-Queue Triage & Active Learning

**Triage = cost × uncertainty.** Order the human queue by expected risk, combining calibrated uncertainty with the business cost/severity of the field or document, so scarce reviewer time attacks the highest-expected-loss items first (P12). Combine with region-aware confidence so a low-confidence token in a high-value field outranks a low-confidence token in a footnote [16].

**Active-learning query strategies and their caveats.** The canonical families are uncertainty sampling (least-confidence, margin, entropy), query-by-committee (QBC, maximal disagreement among a committee), expected model change, and diversity/representativeness [21]. The crucial practical caveats: (1) **uncertainty sampling is biased toward the current model and prone to querying outliers**, and "may not be sufficient in batch active learning due to the redundancy of instances" [23][24]; (2) **batch-mode AL needs an explicit diversity/density term** to avoid labeling near-duplicate uncertain points [23][24]; (3) **AL can underperform uniform/random sampling** in some settings, so benchmark against random before trusting it [24]; (4) **cold start** — early models are too weak to rank uncertainty well — and **sampling bias** — the labeled set drifts away from the production distribution. QBC mitigates the single-model bias of uncertainty sampling but is computationally costly for deep models [24].

**Tooling: what's free vs. what you build.** All four named tools are currently maintained as of 2025–2026:

| Tool | License | Doc/OCR/bbox fit | HITL & AL out-of-the-box | Build-it-yourself |
|---|---|---|---|---|
| **Label Studio** (HumanSignal) | Apache-2.0 (Community) [26] | Strong: `RectangleLabels` + per-region `TextArea` for OCR; PDF→paginated images; Tesseract/YOLO/MMDetection ML-backends; interactive smart pre-annotation [29][32] | Pre-annotation overlays, confidence/score task sampling, ML-backend webhook to retrain; **but the automated active-learning loop is Enterprise-only** — Community can only "manually sort tasks and retrieve predictions to mimic an active learning process" [30] | Calibration, cost×uncertainty ranking, drift monitoring, the closed AL loop (in Community) |
| **Argilla** (Hugging Face) | Apache-2.0 [31] | Text/NLP-first; records with predictions; deploys on HF Spaces in clicks | Pre-annotation with model predictions, **filter by high prediction score to validate fast**, search + semantic similarity to find critical subsets, push/pull to HF Hub for retraining; widely used "specifically for active learning" [25][28] | Native bounding-box document review; the retraining trigger; calibration; monitoring |
| **Prodigy** (Explosion) | Proprietary, perpetual one-time license (entry ~$390–$490, +12 mo upgrades) [27] | NLP-first; PDF/OCR and Segment-Anything plugins exist | Scriptable, built-in **active-learning** with uncertainty sampling; self-hosted; recipes are Python | Web-scale multi-reviewer governance; document bbox review depth; monitoring |
| **doccano** | MIT [27] | Text classification/NER/seq2seq; **no native PDF/bbox**; needs external OCR | Basic REST API, model pre-labeling import; no adjudication/IAA dashboard | Almost everything beyond basic text span labeling; note slowed release cadence (latest tagged v1.8.5) [27] |

Other notable currently-maintained options: **Kili, Labelbox, Encord, TagTog** (native PDF rendering for document annotation) [25]; **Rossum** (commercial, invoice/document IDP with built-in human-in-the-loop validation UI and confidence-driven review); and LLM-eval tools with human review such as **Arize Phoenix** (open-source, OTel-native; "annotations… let you attach human feedback and automated LLM evaluations to any span or trace") [39][40].

**Bottom line on "free vs. build":** the *editor surface* (bounding-box drawing, OCR transcript editing, side-by-side review, pre-annotation overlays, correction capture, score-based task sampling) is free from Label Studio/Argilla/Prodigy. The *intelligence layer* — calibrating confidence, computing cost×uncertainty priority, closing the active-learning retraining loop, and label-free drift monitoring — is what you must build (or buy Enterprise/Rossum for).

## DEEP DIVE 5 — Production Monitoring (Label-Free Drift)

**Three kinds of shift.** *Data (covariate) drift* is a change in P(X) — the inputs; *concept drift* is a change in P(y|X) — the input→output relationship; *label shift* is a change in P(y). Concept drift directly degrades accuracy and is the hardest to catch without labels [35][36]. In an OCR+LLM extraction pipeline, drift shows up as new document templates, new vendors, scan-quality changes, or model/version updates.

**Proxy signals that best approximate accuracy degradation.** When ground truth is delayed or absent, the recommended tiering (per Evidently) is: use labels when you have them; otherwise monitor **prediction drift** (a shifting output distribution signals upstream change); when labels are indefinitely unavailable, weight toward **input data-quality / distribution signals** and **feature-attribution drift** [33][35]. For an extraction system, the highest-value proxies are: the **calibrated-confidence distribution** (a leftward shift = rising uncertainty), the **abstention/reject rate** (P15), and **validator-firing rates** (checksums, format/range validators) over time — these fire without any human labels and tend to move before measured accuracy collapses.

**Label-free performance estimation.** NannyML offers two methods: **CBPE (Confidence-Based Performance Estimation)** for classifiers, which "leverages the confidence score of the predictions" to estimate any confusion-matrix-based metric, and **DLE (Direct Loss Estimation)** for regression, which trains a "nanny" model to estimate the monitored model's loss [37][38]. The critical caveat, stated by NannyML itself: CBPE "is unbiased estimator of performance assuming … the monitored model returns well-calibrated probabilities" [38]. NannyML's newer PAPE and multi-calibrated M-CBPE extend this under covariate shift [34][38]. For multivariate drift, NannyML uses PCA-reconstruction error to catch correlation changes that univariate tests miss [34].

**Tooling (currently maintained).** **Evidently AI** is "an open-source framework (Apache 2.0) with 40M+ downloads," offering drift reports with KS/PSI/Wasserstein tests and prediction/data/concept-drift presets; the company raised a $15M Series A led by DN Capital in December 2024 [33][35]. **NannyML** (open-source + Cloud) — label-free performance estimation (CBPE/DLE) and PCA-based multivariate drift [37][34]. **whylogs/WhyLabs** — lightweight statistical profiles enabling drift detection "without storing raw data," good for privacy-sensitive streams [33]. **Arize / Phoenix** — OpenTelemetry-native LLM/ML observability with tracing, evals, and human annotations [39][40]. **deepchecks** — test-suite-driven validation with distribution-shift checks and dashboards [40]. Common PSI guidance: PSI < 0.1 = no significant change; 0.1–0.2 = investigate [33].

**Sample-for-audit to estimate true error.** Proxy signals must be periodically ground-truthed. Draw a random or stratified sample of production outputs, send to senior review, and report the error rate with a confidence interval (P17). Two statistically-grounded approaches: (1) **confidence-interval estimation** of the error rate — but achieving tight bounds "can require surprisingly large sample sizes"; (2) **acceptance sampling**, especially "sequential sampling with curtailment," which "overall requires far less inspection" because it stops as soon as a batch can be accepted/rejected [41]. Example: 30 errors in 5,000 sampled items → 0.6% estimated error, 95% CI (0.41%, 0.86%) [42]. Use the audit to recalibrate confidence and to validate (or invalidate) your label-free estimates.

---

## RECOMMENDATIONS (staged, with thresholds that change them)

**Stage 0 — Foundations (do first).**
1. **Calibrate confidence** (temperature scaling, then isotonic if needed) and verify with a reliability diagram + Expected Calibration Error before any score is displayed or used for gating. *Threshold to revisit:* if post-deployment ECE drifts above your tolerance (e.g., >0.05) or the audit (Stage 3) shows band accuracies no longer match band labels, recalibrate.
2. **Plot the risk–coverage curve** per high-value field and pick τ_high/τ_low from a target selective risk (SAC). *Threshold:* set τ_high where empirical precision on the accepted head meets the downstream requirement (e.g., 99.5%).

**Stage 1 — Review UX (build the editor on an existing tool).**
3. Adopt **Label Studio** (Apache-2.0) as the review surface for bounding-box + OCR-transcript review; wire your extractor as an ML-backend for pre-annotation overlays [29][32]. Use **Argilla** instead if your fields are text-centric and you want HF-Hub-native dataset versioning [25][31].
4. Implement P1 (calibrated badge), P9 (click-to-source bbox), P10/P11 (disagreement diff + consensus badge). Add P8 cognitive-forcing only on the highest-stakes accept actions [9].
5. Replace any free-text override with structured override codes (P7) from day one [13].

**Stage 2 — Triage & active learning.**
6. Implement the **cost×uncertainty priority queue** (P12) — this is build-it-yourself; the tools won't rank by business cost.
7. Capture every correction as gold (P13). Start the AL loop with **uncertainty + diversity** sampling (P14), and **benchmark against random sampling** — if AL doesn't beat random on your held-out curve, stay with random [24]. *Threshold:* adopt AL only when it demonstrably reduces labels-to-target-accuracy versus random.

**Stage 3 — Monitoring & audit.**
8. Stand up **Evidently** for prediction/data drift and **NannyML CBPE/DLE** for label-free performance estimation; gate CBPE trust on your calibration status [33][37][38].
9. Dashboard the three best proxies: calibrated-confidence distribution, abstention/reject rate, validator-firing rates (P15). *Threshold to act:* alert when reject rate or a drift statistic exceeds its reference-window control limit (e.g., PSI > 0.2) [33].
10. Run a **scheduled audit sample** (P17) using acceptance sampling with curtailment; use it to estimate true error with a CI and to validate the proxies [41][42]. *Threshold:* if audited error exceeds the contracted rate, tighten τ_low (more gating) and trigger retraining.

## CAVEATS
- **Most HCI evidence is from adjacent domains** (income prediction, clinical decision support, intelligence analysis, data-viz tasks), not document extraction specifically; effect sizes may not transfer cleanly. Treat the patterns as well-motivated hypotheses to A/B test on your own reviewers.
- **The Li et al. (2026) miscalibration study is a preprint** (CHI 2026, not yet final-published); cite as forthcoming [2].
- **Confidence is not uniformly available or meaningful.** LLM token probabilities are often poorly calibrated and may not exist for closed APIs; regression outputs lack a native confidence, which is why DLE trains a separate estimator [38]. Calibration is a prerequisite for nearly every pattern here.
- **CBPE and similar estimators assume calibration and (often) covariate-shift-only conditions**; under concept drift they can be wrong in the optimistic direction — always back them with periodic labeled audits [38][41].
- **Tooling status moves fast.** Prodigy pricing figures come from third-party aggregators and should be verified on the vendor page; doccano's release cadence appears slowed (latest tagged v1.8.5); Label Studio's automated AL loop is Enterprise-gated [27][30].
- **Active learning is not free lunch:** sampling bias, cold start, and batch redundancy can make a naïve AL loop worse than random sampling; the corrected-data feedback loop can also entrench model blind spots if you only ever review what the model is unsure about [23][24].

---

## REFERENCES
1. Zhang Y, Liao QV, Bellamy RKE. Effect of Confidence and Explanation on Accuracy and Trust Calibration in AI-Assisted Decision Making. FAccT 2020. [doi/10.1145/3351095.3372852](https://dl.acm.org/doi/10.1145/3351095.3372852)
2. Li et al. Understanding the Effects of Miscalibrated AI Confidence on User Trust, Reliance, and Decision Efficacy. CHI 2026 (preprint). [arXiv:2402.07632](https://arxiv.org/abs/2402.07632)
3. Beyond Explainable AI (XAI): An Overdue Paradigm Shift and Post-XAI Research Directions. [arXiv:2602.24176](https://arxiv.org/pdf/2602.24176)
4. Correll M, Moritz D, Heer J. Value-Suppressing Uncertainty Palettes. CHI 2018. [doi/10.1145/3173574.3174216](https://dl.acm.org/doi/10.1145/3173574.3174216)
5. Words of estimative probability — overview of verbal-probability range variation. [globalsecurity.org](https://www.globalsecurity.org/intell/ops/probability.htm)
6. Communicating uncertainty using words and numbers. Trends in Cognitive Sciences 2022. [cell.com](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(22)00060-2)
7. Hullman J et al. Hypothetical Outcome Plots Outperform Error Bars and Violin Plots. PLOS One 2015. [journals.plos.org](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142444)
8. Kale A et al. Hypothetical Outcome Plots Help Untrained Observers Judge Trends in Ambiguous Data. IEEE TVCG 2019. [idl.uw.edu](https://idl.uw.edu/papers/hops-trends)
9. Bučinca Z, Malaya MB, Gajos KZ. To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI. CSCW 2021. [arXiv:2102.09692](https://arxiv.org/abs/2102.09692)
10. Cattelan LFP et al. On Selective Classification Under Distribution Shift (risk-coverage, AURC, SAC). [openreview](https://openreview.net/pdf?id=FiqXqKR26c)
11. Geifman Y, El-Yaniv R. SelectiveNet: A Deep Neural Network with an Integrated Reject Option. ICML 2019. [scribd mirror](https://www.scribd.com/document/921352028/SelectiveNet-a-Deep-Network-With-an-Integrated-Reject-Option)
12. Franc V et al. Optimal Strategies for Reject Option Classifiers. [arXiv:2101.12523](https://arxiv.org/pdf/2101.12523)
13. Reducing alert fatigue in clinical decision support (structured override codes, tiering); Project Hermes manageable-query-rate figure [arXiv:2602.18643]. [mindbowser.com](https://www.mindbowser.com/reduce-cdss-alert-fatigue-clinical-decision-support/)
14. Mindee. Polygons (Bounding Boxes) — overlay for validation/review. [docs.mindee.com](https://docs.mindee.com/models/optional-features/polygons-bounding-boxes)
15. Grounding in Document Extraction (PyMuPDF). [medium.com/@pymupdf](https://medium.com/@pymupdf/grounding-in-document-extraction-ada1bb367af5)
16. Pulse AI. Word and Cell Level Bounding Boxes — source linking, confidence-weighted review queues. [runpulse.com](https://www.runpulse.com/blog/word-and-cell-level-bounding-boxes-are-now-generally-available)
17. A Multi-Evidence, Multi-Engine OCR System. [researchgate](https://www.researchgate.net/publication/252980446_A_multi-evidence_multi-engine_OCR_system)
18. Technical Analysis of Modern Non-LLM OCR Engines (Calamari confidence voting, CER reduction). [intuitionlabs.ai](https://intuitionlabs.ai/articles/non-llm-ocr-technologies)
19. Design and Evaluation of Multi-Agent AI Oracle Systems (agreement vs split accuracy: 88.34% vs 57.82%). [arXiv:2605.30802](https://arxiv.org/pdf/2605.30802)
20. Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying OCR (+15.2 F1, 42.1% gain). [arXiv:2504.11101](https://arxiv.org/pdf/2504.11101)
21. Active Learning query strategies overview (uncertainty, QBC, expected model change). [arXiv:1808.01412](https://arxiv.org/pdf/1808.01412)
22. Override rate of drug-drug interaction alerts in CDS: systematic review & meta-analysis (pooled 90%, 95% CI 85–95%); AHRQ PSNet Alert Fatigue primer. [psnet.ahrq.gov](https://psnet.ahrq.gov/primer/alert-fatigue)
23. Query-by-committee improvement with diversity and density in batch active learning. ScienceDirect. [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0020025518303700)
24. An Expanded Benchmark … the Edge of Uncertainty Sampling for Active Learning. [arXiv:2306.08954](https://arxiv.org/html/2306.08954v3)
25. Document/text annotation tool comparisons (Label Studio, Argilla, Prodigy, doccano, TagTog, Kili). [labelyourdata.com](https://labelyourdata.com/articles/document-annotation-tools)
26. Label Studio repository (Apache-2.0). [github.com/HumanSignal/label-studio](https://github.com/HumanSignal/label-studio)
27. Annotation tool licensing/maintenance (Prodigy perpetual license; doccano MIT). [prodi.gy/buy](https://prodi.gy/buy)
28. Argilla — active learning and HF integration. [argilla.io](https://argilla.io/)
29. Label Studio — Interactive bounding-box OCR with Tesseract backend. [labelstud.io](https://labelstud.io/guide/ml_tutorials/tesseract)
30. Label Studio — Set up an active learning loop (Enterprise). [docs.humansignal.com](https://docs.humansignal.com/guide/active_learning.html)
31. Argilla on Hugging Face Spaces (Apache-2.0, free/open-source). [huggingface.co](https://huggingface.co/docs/hub/spaces-sdks-docker-argilla)
32. Label Studio ML backend (OCR/YOLO/MMDetection, interactive pre-annotation). [github.com/HumanSignal/label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend)
33. Evidently AI — data drift detection and proxy signals without labels (Apache-2.0, 40M+ downloads). [evidentlyai.com](https://www.evidentlyai.com/ml-in-production/data-drift)
34. NannyML — detecting data drift (PCA reconstruction, performance estimation). [nannyml.readthedocs.io](https://nannyml.readthedocs.io/en/main/tutorials/detecting_data_drift.html)
35. Evidently AI — concept drift and proxy metrics. [evidentlyai.com](https://www.evidentlyai.com/ml-in-production/concept-drift)
36. Drift Detection in Robust ML Systems. Towards Data Science. [towardsdatascience.com](https://towardsdatascience.com/drift-detection-in-robust-machine-learning-systems/)
37. NannyML GitHub — CBPE/DLE label-free performance estimation. [github.com/nannyml/nannyml](https://github.com/nannyml/nannyml)
38. NannyML — CBPE assumptions (well-calibrated probabilities). [nannyml.readthedocs.io](https://nannyml.readthedocs.io/en/v0.4.1/how_it_works/performance_estimation.html)
39. Phoenix — human annotations on traces/spans. [arize.com/phoenix](https://arize.com/phoenix/)
40. Arize Phoenix overview + deepchecks LLM. [deepchecks.com](https://deepchecks.com/llm-tools/arize-phoenix/)
41. On Efficient and Statistical Quality Estimation for Data Annotation (acceptance sampling, sequential curtailment). [arXiv:2405.11919](https://arxiv.org/pdf/2405.11919)
42. Assessing the accuracy of the Australian Senate count (error-rate CI example). [arXiv:2205.14634](https://arxiv.org/pdf/2205.14634)