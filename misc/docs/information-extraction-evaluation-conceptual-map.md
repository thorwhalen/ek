# Information Extraction Evaluation — A Conceptual Map

*OCR treated as a special (especially noisy) case of a general problem*

**Author:** Thor Whalen
**Status:** Conceptual map **+ reading guide** — updated after the deep research (Reports `ek_01`–`ek_06`). The original framing held up; **inline notes mark where the research corrected, sharpened, or filled in** the pre-research scoping. The body states the *concept* and the research's *correction*; the exhaustive tool rosters, numbers, and license verdicts live in the pointered report (follow the `→ ek_0N` markers). The two closing sections (§9 cross-cutting findings, §10 reading guide) are the map *into* the six reports — read them first if you want to know which report answers a given question.

---

## TL;DR

Evaluating an information-extraction (IE) system splits cleanly along **two independent axes**:

- **Reference availability** — do we have a gold answer to compare against (*reference-based*) or not (*reference-free*)?
- **Granularity** — are we scoring one item (*instance-level*) or aggregating over a corpus (*system-level*)?

Your two stated needs sit in opposite corners of that 2×2:

| | **Reference-based** (gold available) | **Reference-free** (no gold at inference) |
|---|---|---|
| **System / corpus level** | **Need #1: offline benchmarking** — pick the system/parameters | Production monitoring & **drift** detection (the corner you didn't mention) |
| **Instance level** | Per-item scoring during eval; spot-checks; error analysis | **Need #2: online confidence / quality estimation → triage & HITL** |

Three ideas do most of the work:

1. **Choose the right object and the right comparison function.** The biggest lever is *not* a better metric on raw strings — it's moving evaluation **down to the structured payload** you actually consume downstream, and weighting errors by their **real cost** (a misspelled city ≠ two extra digits on a donation amount).
2. **Separate the front-end from the rest.** OCR is just one error-prone *transducer* feeding the pipeline. Everything except the image→text error model is **source-agnostic** and applies equally to PDF, DOCX, XLSX, tables, and DB responses. The other two transducers worth naming are **ASR** (speech→text — the closest analog) and **table/structured-document extraction** (with its own structured metric).
3. **Calibration is the hidden prerequisite, and the comparator you most want doesn't exist off-the-shelf.** *(Research-level lesson.)* Almost every raw signal — OCR confidence, LLM logprob, validator firing — is **uncalibrated**, and calibration turns out to be load-bearing for the gate (§3), the UX (§6), **and** monitoring (§8) alike. Meanwhile the single most useful comparator — a **cost-weighted, type-aware distance over your typed graph** — is the one thing no library ships (→ `ek_02`, `ek_06`). It is the principal *build*, not a configuration choice.

A fourth actor is increasingly the *extractor itself*: when an LLM does the extraction, it needs its own reference-free QE — **faithfulness / groundedness / hallucination detection**. The research adds a blunt caveat: such an extractor often emits **no usable intrinsic confidence at all** (§7), so external QE is frequently the *only* option, not a complement.

---

## 1. The two needs, named precisely

### 1.1 Offline benchmarking (Need #1)
*A.k.a. reference-based evaluation, gold-standard evaluation, system-level / corpus-level evaluation, an "eval harness."*

You run candidate systems (and parameter settings) over a fixed corpus or a **stratified sample** of it, score against ground truth, and **select**. This is the experimentation/evaluation system you described. Its outputs are comparative metrics, **error analysis**, and — done well — **per-slice** results (by source type, language, scan quality) and **regression tests** so a "better" system doesn't silently get worse on some slice. → `ek_02`

### 1.2 Online quality estimation (Need #2)
*A.k.a. reference-free quality estimation (**QE**), confidence/uncertainty estimation, inference-time / runtime evaluation, in-the-loop quality control.*

At inference you have **no gold answer**, so you must *estimate* quality from signals available at runtime. "Quality estimation" is the term of art (it crystallized in machine translation, where reference-free QE is a whole subfield — the **QuEst → OpenKiwi → TransQuest → CometKiwi** lineage, all reference-free; → `ek_03`). The estimate drives a **decision**: accept, flag for human verification, or block. That decision rule is **selective prediction** (§3.3).

> **The unifying frame for Need #2 is reference-free QE feeding a selective-prediction policy.** Everything you listed — character confidences, multi-system agreement, plausibility checks, color-coded UX — is either a *source of the estimate* or a *consumer of the decision*.

---

## 2. What are we comparing, and with what function? (your central insight)

Your observation that string equality causes "too many false negatives" is the heart of the matter. The fix is to recognize there is a **ladder of evaluation objects**, and to climb it:

1. **Exact match** (string equality) — brittle baseline; high false-negative rate.
2. **Edit-distance metrics** — Levenshtein, normalized into **CER (Character Error Rate)** and **WER (Word Error Rate)**. Graded, not boolean. *Two research corrections:* (a) CER/WER are **not bounded in [0,1]** — a hypothesis longer than the reference pushes them past 100%, which is itself a useful **hallucination signal**; (b) the canonical tool is **`jiwer`** (with **`rapidfuzz`** as its C++ backend — *not* a parallel library; and `rapidfuzz` is the one that uniquely offers **cost-weighted** edit distance). Accumulate WER **globally**, never average per-batch (that's mathematically wrong). The GPL `Levenshtein`/`python-Levenshtein` packages are a license landmine `rapidfuzz` (MIT) replaces. → `ek_02`, `ek_06`
3. **Normalized / canonicalized comparison** — lowercasing, Unicode normalization, whitespace/diacritic folding, number/date canonicalization *before* comparing. *The research promotes this from a ladder rung to load-bearing infrastructure:* under-specified normalization is **the single biggest source of bogus scores** (it can swing results by tens of points, and a reused English normalizer **catastrophically corrupts** Indic/abugida scripts). Treat the canonicalizer as a versioned, tested, schema-aware component — see §5. → `ek_02`
4. **Semantic comparison** — embedding similarity / **STS**, **BERTScore**, **LLM-as-judge**. *Sharpened by the research:* for the **structured IE targets this project cares about**, tolerance for minor variants is delivered by **ANLS / ANLS\*** (normalized edit distance, ~0.5 threshold) and canonicalization — **not** by embeddings (→ `ek_02`). The embedding/LLM-judge rung is more relevant to free-text generation; keep it lower-priority for IE (and an LLM judge needs its own validation).
5. **Structured / field-level comparison** — compare the **extracted records**, field by field: **slot error rate**, entity/attribute **precision/recall/F1**, and a **shape-specific** metric for nested structure (tables vs. nested JSON vs. typed graph). Two things a map should flag: the **partial-match schemes disagree** (`seqeval` vs. `nervaluate`), so choose one deliberately; and tables have *two* rival metrics — **TEDS** (use **TEDS-Struct** to isolate structure from OCR content) and **GriTS** — the canonical reminder to compare the *structure*, not the flattened string. Exact **graph edit distance is NP-hard**; use a documented approximation. The library roster per shape → `ek_02` (and what to install → `ek_06`).
6. **Task / utility comparison** — does the downstream decision come out right? The ultimate ground truth.

Two principles ride on top of this ladder:

- **Defer the boolean.** Keep **numerical scores** (or multidimensional feature vectors) as far downstream as possible and threshold *late*. Premature `True/False` throws away information you need for triage and calibration. The research confirms this instinct repeatedly (e.g. raw CER/WER carrying a hallucination signal that a boolean would discard).
- **Cost-sensitive / importance-weighted metrics** (utility-weighted, task-aware). Weight per-field and per-error-type by real-world cost. **This is where the research most sharply corrects the map:** there is **no turnkey library** that takes a typed schema + a per-field/per-type cost matrix and returns a weighted typed-graph distance. The libraries (`zss`, `networkx` GED) expose **cost-callable hooks** only — you supply the cost *functions*. Treat cost-weighted typed-graph scoring as a **build**, not a knob (→ `ek_02`, `ek_06`; it's one of the two confirmed must-builds).

**Takeaway:** the ideal comparator operates on the *structured downstream payload* with *type-aware, cost-weighted* distances — and it should be the same object whether you're benchmarking (vs. gold) or estimating quality (vs. a consensus/expectation). The research gives this a concrete shape: a **two-layer object** — **Layer A** a frozen *graph grammar / schema* (the SSOT carrying node/edge/field types **and** importance weights, reused by metrics *and* by constrained decoders), **Layer B** the *extraction-and-verification metadata* (provenance, raw signals, calibrated confidence, validator findings, the accept/flag/block decision) riding alongside it. The same `(grammar, estimates)` pair is scored offline vs. gold and estimated online vs. consensus. → `ek_06`

---

## 3. Where the online signal comes from (confidence, calibration, decision)

Mental model from the research: **signal → calibrate → decide**. The signal stage is extractor-specific; calibration and the decision rule are shared, extractor-agnostic infrastructure.

### 3.1 Sources of a reference-free quality signal
- **Intrinsic model confidence** — per-character/word posteriors, or sequence/token log-probabilities for a generative extractor. *Correction: not "always available."* Genuine **per-symbol** confidence exists in only **Google Cloud Vision and Tesseract**; Mathpix/Mistral reach per-word; many engines stop at line level; and **Claude Vision, OCR.space, and Apple Vision emit nothing usable**. For generative extractors the **only** intrinsic channel is token logprobs, which are themselves fragile (OpenAI's are flaky/empty on images and empty under strict `json_schema`; only Mistral exposes a logprob-*derived* score). When you do have logprobs, **length-normalize** before aggregating (raw sums are length-biased; a **MIN-over-tokens** variant catches a single bad digit). → `ek_01`, `ek_03`
- **Ensemble / agreement-based** (your "multiple OCR compared against each other"). *Promoted by the research from "one source among equals" to often the **best** source* — agreement beats raw model probability precisely **because** that probability is miscalibrated (self-consistency, Wang 2022 → `ek_03`). The classical sequence method is **ROVER** (alignment + per-position voting, from ASR); for stochastic LLMs the analog is **self-consistency** (sample-and-vote). Try **deterministic, N-engine ROVER first** — it's far cheaper than LLM sampling. Agreement is a strong, label-free error predictor (a multi-LLM benchmark: 3/3 agreement ≈ 88% vs. a 2-1 split ≈ 58%; and label-free "Consensus Entropy" verifies OCR with **no labels**) → `ek_05`. *Caveat: no maintained, permissively-licensed ROVER exists — it's the second confirmed build, and `uqlm` covers black-box LLM self-consistency off-the-shelf (`lm-polygraph` adds white-box logit uncertainty)* → `ek_06`.
- **Auxiliary QE model** — a trained estimator predicting quality from `(input, output)` features. The MT-QE lineage (CometKiwi et al.) is the mature template; **ConfBERT** is the realized OCR/IE port (and shows a simple confidence-threshold can match a heavy model *when* the OCR is well-calibrated). → `ek_03`
- **Verifier / checker signals** — rule, schema, and consistency checks (§4) that fire when an output is implausible. The research calls these the **highest-ROI first line** — run them always; they catch confidently-misread-but-plausible errors that *no* confidence signal will catch.

### 3.2 Calibration (the step that makes the green→red gradient *mean* something)
Raw confidences are usually **miscalibrated** — a "0.9" isn't 90% correct (substantiated across the OCR inventory: nearly every engine is uncalibrated; **Mathpix is the sole calibrated emitter**). **Calibration** fixes the score→correctness-probability mapping. *Sharpened:* it's **not one uniform step** — pick the method by available input (**temperature scaling** needs logits; **Platt/isotonic** work on any scalar, so they're the right tool for aggregated OCR confidence/logprobs), and calibrate **at the granularity of the decision** (build a binary "field-correct" target; don't calibrate per-token and hope it composes). Measure with **ECE** (binning-sensitive — treat cautiously; **D-ECE** for localized/bbox outputs). Default backend: **`netcal`** (full roster and license notes → `ek_06`). → `ek_03`

A stronger, model-agnostic option: **conformal prediction** — distribution-free, finite-sample guarantees that turn "flag the worst X%" into a principled threshold. *Three load-bearing qualifications the research adds:* (a) the guarantee is **marginal, not conditional** — distribution-free *per-field* coverage is **provably impossible** (Barber 2019); per-field-type validity needs **class-conditional / Mondrian** calibration (`crepes`); (b) the IE-relevant variant is **Conformal Risk Control**, which bounds the losses you actually care about (FNR, token-F1, graph distance), not vanilla coverage; (c) for **free-form LLM** output exchangeability is violated, so conformal-for-LLM works only at claim/sentence granularity. Tooling: **`MAPIE`** by default, `crepes` for per-field-type (Mondrian) coverage; the only graph/sequence option (`TorchCP`) is **LGPL — quarantine it**. → `ek_03`, `ek_06`

### 3.3 The decision rule: selective prediction
*A.k.a. prediction with a reject option, classification with rejection, abstention.* You **accept, abstain (route to human), or reject** based on the (calibrated) quality estimate. The governing object is the **risk–coverage curve**; pick the operating point. *The research supplies the missing quantitative bridge to §2's cost theme:* the cost-optimal accept threshold on a **calibrated** probability derives from the **cost ratio ρ = c_FN / c_FP** (undetected-error cost vs. needless-review cost) — which is exactly why calibration is a hard prerequisite for the gate to mean anything (→ `ek_03`). To operationalize the threshold, fix the required accuracy on a high-value field and read off the coverage — the **AURC / Selective Accuracy Constraint (SAC)** framing (→ `ek_05`).

---

## 4. Validation & correction (your "smoothing" idea, named)

Your generalization of numerical neighborhood smoothing to text is precisely the classical **noisy-channel model**, and it's Bayesian as you guessed:

> argmaxᵥ P(*v* | *o*) ∝ P(*o* | *v*) · P(*v*)
> where *o* = observed (OCR/extracted) value, *v* = true value, P(*o*|*v*) = **error model** (e.g., an OCR confusion matrix), P(*v*) = **language/structure prior**.

**The organizing axis the research adds: FLAG vs. CORRECT.** Layers 0–4 below are overwhelmingly **deterministic flag-or-coerce** (auditable, cheap); only the learned/LLM layer freely *invents* content, is stochastic, needs its own evaluation, and can degrade text. Build the prior **bottom-up and stop as early as it works** — many fields (codes, dates, amounts, enums) are *largely handled* by Layers 0–2 with no LM at all, deterministically and auditably (though even a schema-valid value can still be wrong). → `ek_04`

Your "context-specific smoothing" is the **prior** P(*v*), and it strengthens in layers:

- **Layer 0 — Canonicalization** (dates/units/numbers/Unicode). Cheap, deterministic, and — per §2/§5 — the highest-leverage correctness lever.
- **Layer 1 — Schema / type / range priors** (your strongest point) — a "donation amount" is a *number* in a *range*; a date has a *format*; a country code is an *enum*. Express as declarative validators (**`pydantic`**, **`pandera`**, `jsonschema`, Great Expectations), with the **same `pydantic` model reused as SSOT** for generation, parsing, *and* validation.
- **Layer 2 — Lexicon / gazetteer priors** — dictionaries, name lists, controlled vocabularies (your enum columns), with fuzzy resolution (`rapidfuzz`, `symspellpy`).
- **Layer 3 — Language-model prior** — n-gram **perplexity** (`KenLM`) or **masked-LM pseudo-log-likelihood** (`minicons`, Salazar 2020). *Correction: LM priors are strong **flags** but weak **correctors alone** — they detect/re-rank but need a separate candidate generator to fix anything; and **domain-adapting the prior is essential, not optional** (an out-of-domain LM flags style, not errors). Use character/byte-level models for OCR.*
- **Layer 4 — Constrained / structured generation** — grammar/JSON-schema-constrained decoding. *Two corrections:* (i) it guarantees **well-formedness, not correctness** — a schema-valid record can be entirely wrong, and native OpenAI/Anthropic structured outputs don't enforce `pattern`/`format`/`minimum`; over-constraining can **degrade reasoning** on some tasks (a *contested* finding — dottxt argues it's a prompt/schema artifact; treat the tax as real but mitigable by reasoning in free text first, then emitting structure). (ii) **Constrain-on-generate** (`outlines`/`xgrammar` masking illegal tokens at decode time) is mechanistically distinct from **post-hoc validate-and-retry** (`instructor` wrapping the API + a `pydantic` model). Note the tension with §3: strict `json_schema` can **empty the logprobs**.
- **Layer 5 — Learned seq2seq / LLM correction** — the only content-*inventing* layer. *Correction: this is research-grade with genuinely **mixed** evidence* — wins (CLOCR-C >60% CER reduction) coexist with failures (GPT-4 "not effective," "no free lunches," unusable for Finnish). LLMs hallucinate and "fix" genuine source misspellings; **never assume improvement** — evaluate per language/domain slice.

Two reference-free validators that span the layers:

- **Cross-field / cross-record consistency** — integrity constraints: line items sum to the total, dates ordered, foreign keys resolve. Cheap, powerful, reference-free.
- **Cross-source corroboration / triangulation** — *the research frames this as the single most powerful reference-free signal:* the same fact from two independent extractors agreeing is positive evidence; disagreement is a high-value flag and the trigger to escalate to Layer 5.

Also new: **statistical anomaly detection** on extracted numeric values (**Benford's law**, isolation forests / robust z-scores via `pyod`) as reference-free **flags** (flag-only — route to review, never auto-edit).

**Pre-LLM vs. LLM-era (you asked for both):**
- *Pre-LLM:* edit distance, n-gram LMs, noisy-channel correction (Kernighan/Church/Brill; Norvig as the toy version), HMM/CRF, finite-state transducers, dictionary lookup, regex/grammar validation. **Prefer these where they suffice** — cheaper, deterministic, auditable. The research makes this concrete: Layers 0–2 catch *most* errors with no LM. The post-OCR subfield has **exactly two** ICDAR competitions (2017, 2019 — not an open-ended series).
- *LLM/embedding era:* embedding similarity, MLM correction, LLM-as-judge, retrieval-augmented validation, self-consistency, generative QE. More capable in places, but stochastic, costly, and themselves in need of evaluation.

---

## 5. Ground truth & the eval harness (infrastructure for Need #1)

Reference-based evaluation is only as good as its gold standard — and the research shows the harness has **two co-equal pillars**, not one:

- **Annotation quality** — **double-annotation + adjudication**; measure **inter-annotator agreement (IAA)** to know your ceiling. *Selection rule the research adds:* default to **Krippendorff's α** (it alone handles missing/incomplete data and any measurement level — the real-world case); use **Cohen's κ** only for exactly two raters/nominal/complete data, **Fleiss' κ** for a fixed rater panel. Conventions: α ≥ 0.8 good, 0.67–0.8 tentative.
- **A versioned, schema-tuned canonicalizer** — the *higher-leverage* error source (§2 rung 3), auditable per-script (Mark-class/diacritic handling) and versioned **alongside** the gold set. Under-specified normalizers make cross-benchmark numbers non-comparable.

Plus:

- **Sampling & slicing** — stratify by source type, language, quality band; report per-slice. **OmniDocBench** (1651 PDF pages; 10 doc types / 5 layouts / 5 languages with attribute tags) is the concrete slice-aware instrument. → `ek_02`
- **Regression & "golden-set" testing** — freeze cases so changes can't silently regress a slice. Tooling: **`DeepEval`** (pytest-native "goldens"), **`promptfoo`** (YAML assertions + weights), **LangSmith** (datasets + tracing); practical pattern is a 20–50-example slice-stratified set for sub-5-min CI, and you'll typically need **two tools** (a CI gate + a dataset platform). Wiring *structured* metrics (TEDS/GriTS/ANLS\*) into CI is custom work.
- **Anchor numbers** (what "good" looks like): LayoutLMv3 entity-F1 ≈ 90–92 on FUNSD, 96–97 on CORD — but **flat entity-F1 overstates** quality where grouping/nesting matters (KIEval, TreeForm).
- **Eval-driven development** — treat the harness as a first-class, versioned artifact, not a one-off script. → `ek_02`

---

## 6. Human-in-the-loop & UX (consuming the decision)

This is where calibration + selective prediction surface to your reviewer. The research turns your §6 bullets into a **17-pattern catalog (P1–P17)** and adds hard HCI caveats the map lacked — with one meta-caveat: most of that evidence comes from *adjacent* domains (clinical alerts, data-viz, income prediction), so treat the patterns as well-motivated hypotheses to A/B-test on your own reviewers, not settled law. → `ek_05`

- **Calibrated confidence visualization** — your green→red gradients/heatmaps. *Two corrections:* (i) **calibration is necessary but not sufficient** — it improves *trust-calibration* but does **not by itself** improve joint human-AI accuracy (FAccT 2020); pair it with friction/gating. (ii) Use **color only as a coarse triage cue, not a precision channel** — prefer a **Value-Suppressing Uncertainty Palette (VSUP)** and lead with a calibrated **numeric** value (optionally IPCC-style fixed verbal bands); use **Hypothetical Outcome Plots** for distributional fields (~35–41 pts more accurate than error bars).
- **Over-reliance is a first-class hazard** *(new):* fluent explanations can **increase** over-reliance (read as a generic competence signal). Mitigate with **cognitive-forcing checkpoints** (commit before the AI value is revealed) and **counter-explanations** — applied selectively, since forcing functions are the least-preferred condition.
- **Soft signals vs. hard gates** — nudges for the uncertain band; hard gates past threshold. *Sharpened:* choose soft-vs-hard by **cost asymmetry**, and design gates to avoid **alert fatigue** (clinical decision support sees **49–96%** of alerts overridden; pooled ~90%): tier by severity, suppress duplicates, fire in-workflow, and replace free-text overrides with **5–10 structured override codes** that yield tunable telemetry.
- **Drill-down / provenance overlays** (your drill-down idea) — the image with the **bounding box** highlighted, multiple raw OCR transcripts with disagreement spans, and the extracted fields, all **linked back to source**. *Mechanic the research adds:* store per-field geometry **once as normalized 0–1 coordinates** at extraction time so any later audit reconstructs the highlight without re-running OCR (`PyMuPDF.search_for` for deterministic native-PDF coordinates). Note (§7) that provenance quality is **bimodal** across engines — some "bounding boxes" are model-guessed, not real.
- **Review-queue triage** — order the human's work by (cost × uncertainty), not arbitrarily.
- **Active-learning loop** — human corrections feed back as new gold. *Correction: not a free lunch* — uncertainty sampling is biased toward outliers and the current model and **can underperform random**; combine uncertainty **with a diversity/density term** and **benchmark against random** before adopting.
- **Tooling** — the **editor surface is free, the intelligence layer is build-yourself**: **Label Studio** (Apache-2.0; bbox + OCR; the active-learning loop is wireable via Community webhooks, with only the enterprise edition/SaaS license-restricted — `ek_05` and `ek_06` read its AL story slightly differently, so confirm before relying on it), **Argilla**, **Prodigy**, **doccano**. You still build calibration, cost×uncertainty ranking, the closed loop, and monitoring. → `ek_05`, `ek_06`

---

## 7. Separation of concerns — what's general vs. OCR-specific

**Source-agnostic (applies to PDF, DOCX, XLSX, tables, DB responses, web scrapes, and OCR alike):** the evaluation ladder (§2), QE + calibration + selective prediction (§3), validation/consistency/correction (§4 minus the OCR error model), the gold-standard harness (§5), and the HITL/UX layer (§6). Build these once, at the extraction-output level.

**OCR-specific add-ons (the extra error component you flagged):**
- The **image→text error model** P(*o*|*v*) — character confusion, segmentation/layout errors.
- **Per-character / per-word recognizer confidences** — *qualify heavily:* genuinely fine-grained native confidence is the **exception** (per-symbol only in Google Cloud Vision + Tesseract; per-word in Mathpix/Mistral; word/line in Azure/Textract/Paddle/Rapid/Easy; **none** usable in OCR.space/Apple Vision/pix2tex/Claude) — and nearly all of it is **uncalibrated**. → `ek_01`
- **User-tunable LM/lexicon priors at the front-end are essentially Tesseract-only** (DAWG, `--user-words`, char whitelist). For the other 14 systems the §4 lexicon/schema priors must be applied **externally (post-OCR)**. → `ek_01`
- **Post-OCR correction** as a dedicated stage — the **two** ICDAR competitions (2017, 2019). → `ek_04`
- **Image-region provenance** — bounding boxes for drill-down, but **bimodal**: real geometry from Google/Tesseract/Azure/Textract/Paddle/Rapid/Easy/Mathpix; **model-guessed** (unreliable, esp. on PDFs) from Claude/OpenAI; **char-offset only** (figure bboxes, no text geometry) from Mistral OCR; **none** from TrOCR/pix2tex. A system-selection constraint, not a given. → `ek_01`
- **Table structure you can actually score with TEDS/GriTS** comes only from PaddleOCR PP-StructureV3, Azure DI, and AWS Textract; most local engines emit no cell structure. → `ek_01`, `ek_02`

**The two other specializations worth naming:**
- **ASR — speech-to-text.** The closest structural twin: a noisy transducer to text, scored by **WER**, with native confidences, **ROVER** voting, and a mature **QE** literature. Most of the OCR machinery (ROVER, PLL rescoring, noisy-channel framing) is borrowed from here; mine it. → `ek_03`, `ek_04`
- **Table / structured-document extraction.** Its own structured metrics — **TEDS *and* GriTS** — the canonical illustration of "compare the structure, not the flattened string." → `ek_02`

**And the new transducer:** when an **LLM is the extractor**, *it* is a noisy source too. Its reference-free QE is **faithfulness / groundedness / hallucination detection**. *Caveat the research forces:* it frequently emits **no intrinsic confidence** (Claude Vision: none; OpenAI logprobs flaky/empty on images), so external QE — self-consistency (`uqlm`), cross-model agreement, verifiers — is often the **only** signal, not a complement to logprobs. → `ek_01`, `ek_03`

---

## 8. The corner you didn't mention: production monitoring

The reference-free × system-level cell is **drift / distribution-shift monitoring**: without labels, watch aggregate signals for degradation and **sample-for-audit** to refresh gold. The research adds the rigor: → `ek_05`

- **Name the shift.** **Covariate** drift P(X) and **label** shift P(y) are visible from distributions; **concept** drift P(y|X) is the dangerous one — **undetectable from inputs alone** and it degrades accuracy directly.
- **Label-free accuracy estimation has a precondition.** NannyML **CBPE** estimates metrics from confidence but is unbiased **only if probabilities are calibrated** (ties back to §3.2), and can be **optimistically wrong** under concept drift — always back it with periodic **labeled audits**.
- **Highest-value proxy signals** (all move *before* measured accuracy collapses): the **calibrated-confidence distribution**, the **abstention/reject rate**, and **validator-firing rates** (§4).
- **Sample-for-audit efficiently** — **acceptance sampling with sequential curtailment** needs far less inspection than fixed-CI estimation; use the audit to recalibrate and to validate the label-free estimates.
- **Tooling & a vocabulary note:** Evidently, NannyML, Arize Phoenix, whylogs (→ `ek_05`); for a conformal angle on drift, `crepes` provides **test martingales** (→ `ek_03`/`ek_06`). Note the **vocabulary split** the reports surface: **conformal prediction** is the §3/`ek_03` framing, while the monitoring/HITL layer (`ek_05`) speaks **risk–coverage / SAC** and **CBPE** — complementary tools, not the same one.

---

## 9. Cross-cutting findings the research established

Six through-lines run across all six reports — read these as the load-bearing updates to the original map:

1. **Cheapest-reliable-first is validated as a discipline, not a preference.** Deterministic verifiers run *always* and are highest-ROI (§3.1); validation builds bottom-up and *stops early* (§4); ROVER before LLM sampling (§3.1). This is your "avoid LLMs when we can," made concrete. → `ek_03`, `ek_04`
2. **Two things you most want do not exist off-the-shelf — they are *the* builds.** (a) A **cost-weighted, type-aware distance over a typed graph** (§2); (b) a maintained, permissive **ROVER** engine for cross-engine voting (§3.1). Everything else is mostly reuse/wrap. → `ek_02`, `ek_06`
3. **Calibration is the universal prerequisite** — for the gate (§3.3), the UX's trustworthiness (§6), and label-free monitoring (§8) — yet nearly every raw signal is uncalibrated (Mathpix is the lone calibrated OCR emitter). Calibrate *at the decision's granularity*, before display, and monitor calibration itself. → `ek_03`, `ek_05`
4. **Agreement/consensus is a first-class, label-free, often-best signal** — preferred over raw confidence *because* raw confidence is miscalibrated, and quantified (a multi-LLM benchmark: 3/3 agreement ≈ 88% vs. 2-1 ≈ 58%; and label-free "Consensus Entropy" lifts OCR-verification F1 by +15.2). It appears as a confidence source (§3), the strongest validator (§4), and a UX/monitoring signal (§6/§8). → `ek_03`, `ek_04`, `ek_05`
5. **One SSOT object threads the whole system.** The same typed schema drives constrained decoding, post-hoc validation, *and* scoring; the two-layer (grammar / extraction-metadata) design (§2 takeaway) is the concrete realization of "the same comparator object, offline and online." → `ek_04`, `ek_06`
6. **Permissive licensing is a real constraint with hidden traps.** A zero-copyleft default stack exists end-to-end, but six landmines (e.g. GPL `python-Levenshtein`, LGPL `TorchCP`, non-commercial `surya-ocr`) — some **invisible to license scanners** because their terms live in repo files, not PyPI metadata — must be quarantined behind opt-in extras. → `ek_06`

---

## 10. Reading guide to the deep research

Six reports, filed as `ek_01`–`ek_06` (originally drafted as R1–R6). The companion `prompts/deep-research-plan-ie-evaluation.md` holds the self-contained prompts and run order.

- **`ek_01` — OCR Systems Capability Inventory** *(backs §7).* A 15-system inventory scored on one axis: what **native QE signal** each emits (confidence/logprobs), plus geometry, table structure, LM/lexicon tunability, and deployment/license — organized by **native QE richness** (turnkey-confidence cloud engines, a feature-source middle tier, a minimal tier, and a logprob-gated VLM/LLM group). **Go here when** choosing or characterizing an OCR/VLM front-end: which engine gives a usable (and how-fine-grained) confidence, *real* bbox provenance, or TEDS-able cell structure, at what cost/license.
- **`ek_02` — Reference-Based (Offline) Evaluation** *(backs §2, §5).* An output-type → metric → library decision guide (CER/WER/ANLS, slot-F1, TEDS/GriTS, tree/graph edit, ANLS\*/metametric), plus the gold-standard harness (canonicalization, IAA, slicing, golden-set CI). **Go here when** picking/implementing a reference-based metric for a given object, or building the offline harness — and to confirm that **cost-sensitive typed-graph scoring must be built**.
- **`ek_03` — Reference-Free QE, Confidence, Calibration & Selective Prediction** *(backs §3).* The inference-time, no-gold pipeline: signal → calibrate → decide, organized by what the extractor actually emits, with conformal / conformal-risk-control / selective-prediction machinery and library choices. **Go here when** you need an actionable, calibrated, defensible accept/flag/block gate (and what to do when the extractor emits nothing).
- **`ek_04` — Post-Extraction Validation & Correction (incl. Post-OCR)** *(backs §4).* The six-layer cheapest→most-expensive noisy-channel pipeline organized by the **FLAG-vs-CORRECT** axis, with a 2026 per-layer tooling inventory and the honest (mixed) state of LLM post-OCR correction. **Go here when** you need to check/fix an extracted value without a gold answer, and to know which layer to stop at.
- **`ek_05` — HITL Review UX, Active Learning & Production Monitoring** *(backs §6, §8).* A 17-pattern catalog (P1–P17) plus HCI evidence: calibration-not-sufficient, over-reliance, alert fatigue, agreement-predicts-error, label-free drift with its calibration precondition, and the annotation/monitoring tool landscape. **Go here when** designing or auditing the human-review and monitoring layer.
- **`ek_06` — Library Landscape & Integration Map** *(cross-cutting; added after the original §9).* Every capability mapped to a verified pip-installable library (license/version/API as of mid-2026), a license register of six landmines, the two-layer facade architecture, and a reuse/wrap/**build** allocation — including the two confirmed must-builds. **Go here when** turning the map into running code, making build/borrow decisions, or keeping the dependency graph permissively licensed.
