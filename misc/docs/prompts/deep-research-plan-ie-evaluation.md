# Deep Research Plan — Information Extraction Evaluation Tooling

**Author:** Thor Whalen
**Goal:** Gather recent, state-of-the-art methods *and* concrete libraries so we can implement IE-evaluation tools by pooling existing components — covering both the **offline benchmarking** and **online quality-estimation** needs, with OCR as a special case.

---

## How to use this file

- **Shared context for every prompt:** attach `information-extraction-evaluation-conceptual-map.md` (the conceptual map). The prompts below assume the reader has it; they do *not* re-derive the framing.
- **Engine column:**
  - **Code** = run in **Claude Code**. Best for concrete repo/library/API/doc surveys and light hands-on inspection (actual output formats, what a tool really emits). Less copy-paste for you.
  - **AI** = run in **Claude.ai deep research**. Best for methodological/literature synthesis across recent papers — deeper research mode.
- **Output convention (all reports):** downloadable `.md`, authored as **Thor Whalen**, progressive-disclosure structure (TL;DR + decision points first). The **AI** reports additionally use **Vancouver-style numbered references** `[1], [2], …` with a `REFERENCES` section and hyperlinks `[name](url)`. The **Code** reports cite via inline links to repos/docs.

## Run order & dependencies (short version)

Almost everything is **parallelizable**. There is exactly one soft dependency worth respecting:

- **R1 (OCR inventory) is a useful early input** to R3, R4, and R6 (it tells them what confidence/structure each engine actually emits). It's fast and concrete, so **start R1 first or alongside the others**, and feed its output forward when ready. Nothing hard-blocks on it.
- **R6 (integration map) ideally runs last**, after R1–R5, since it stitches their findings into a concrete library plan. But it can start in parallel and be refined.

| ID | Title | Engine | Soft input from | Parallel? |
|----|-------|--------|-----------------|-----------|
| **R1** | OCR systems capability inventory (the 15) | **Code** | — | Yes |
| **R2** | Reference-based evaluation: metrics, structured/graph scoring, gold-standard harness | **AI** | — | Yes |
| **R3** | Reference-free QE, confidence, calibration & selective prediction | **AI** | R1 | Yes |
| **R4** | Post-extraction validation & correction (incl. post-OCR) | **AI** | R1 | Yes |
| **R5** | HITL review UX, active learning & production monitoring | **AI** | — | Yes |
| **R6** | Library landscape & integration map (pool it together) | **Code** | R1–R5 | Start in parallel, finalize last |

**If you want to minimize copy-paste:** R2–R5 *can* be run in Claude Code too, but you'll get deeper synthesis from Claude.ai deep research — so reserve Code for the two concrete-library reports (R1, R6) and run the four methodology reports (R2–R5) in Claude.ai.

---

## R1 — OCR Systems Capability Inventory *(Engine: Claude Code)*

> **Context:** Attached is a conceptual map for information-extraction evaluation. We are building tooling to evaluate OCR-based extraction, and need to know, per OCR system, what evaluation-relevant signals it natively exposes.
>
> **Task:** For each of the **15 systems below**, produce a structured profile focused strictly on *evaluation-relevant capabilities*. Read the repo/docs (and, where feasible, run a minimal example to inspect the actual output object) to determine — for each system:
> 1. **Rawest output available** — the lowest-level structured result it can return (e.g., per-character / per-glyph, per-word, per-line, per-block; with geometry?). Show the actual output schema/shape, not a prose summary.
> 2. **Confidence/likelihood signals** — does it emit per-character confidences? per-word? per-line? per-region? What scale/semantics, and how to access them in the API?
> 3. **Language model / smoothing / lexicon features** — built-in language models, dictionaries, allow-lists, domain/lexicon configuration, decoder parameters, anything that does language-sensitive correction.
> 4. **Structured output** — bounding boxes/polygons, reading order, layout/table structure, key-value pairs, JSON schema, hOCR/ALTO support.
> 5. **Deployment & constraints** — local vs. hosted API, license, on-device/privacy posture, rough cost model, language coverage, handwriting/math support.
>
> **Systems:** Tesseract (https://github.com/tesseract-ocr/tesseract), EasyOCR (https://github.com/JaidedAI/EasyOCR), RapidOCR (https://github.com/RapidAI/RapidOCR), PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR), ocrmac / Apple Vision (https://github.com/straussmaximilian/ocrmac), pix2tex / LaTeX-OCR (https://github.com/lukas-blecher/LaTeX-OCR), TrOCR handwritten (https://huggingface.co/microsoft/trocr-large-handwritten), OCR.space API (https://ocr.space/ocrapi), Google Cloud Vision (https://cloud.google.com/vision/docs/ocr), AWS Textract (https://aws.amazon.com/textract/), Azure AI Document Intelligence (https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview), Mistral OCR (https://mistral.ai/news/mistral-ocr), Mathpix (https://mathpix.com/), Anthropic Claude Vision (https://platform.claude.com/docs/en/docs/build-with-claude/vision), OpenAI GPT-4o/4.1 Vision (https://platform.openai.com/docs/guides/images).
>
> **Deliverable:** A `.md` report. Lead with a **comparison matrix** (rows = systems; columns = the 5 dimensions above, distilled to flags/short cells), then one short profile per system with the actual output schema and the exact API call/field to access confidences. End with a 5–8 line synthesis: which systems give us the richest QE signal natively, and which require a separate confidence layer. Note for the VLM/LLM-based ones (Claude/GPT/Mistral) whether they expose token-level logprobs or only free-form text, since that determines whether we get intrinsic confidence at all.

---

## R2 — Reference-Based Evaluation: Metrics, Structured/Graph Scoring & Gold-Standard Harness *(Engine: Claude.ai)*

> **Context:** Attached is a conceptual map for information-extraction evaluation (see §2 "evaluation ladder" and §5 "harness"). This report covers the **offline / reference-based** need.
>
> **Research questions:**
> 1. **Surface metrics** — current best practice and tooling for **CER/WER**, normalized edit distance, and text normalization/canonicalization pipelines (number/date/unicode folding) before scoring.
> 2. **Field/slot-level metrics** — slot error rate, entity/attribute precision-recall-F1, partial-match and span-overlap scoring; how leading information-extraction benchmarks score structured outputs.
> 3. **Structured-object metrics** — **TEDS** for tables; **tree/graph edit distance** and graph-matching metrics for nested or graph-shaped targets (we extract into a typed graph). How to define **cost-sensitive / importance-weighted** versions where per-field/per-type weights reflect downstream cost.
> 4. **Gold-standard methodology** — annotation workflows, double-annotation + adjudication, **inter-annotator agreement** (kappa, alpha), stratified sampling/slicing, and **regression / golden-set** testing.
> 5. **Eval frameworks** — current open-source evaluation harnesses/libraries suitable for document-extraction tasks (general LLM/IE eval frameworks, plus OCR/document-specific ones), with their strengths and gaps.
>
> **Deliverable:** `.md`, Vancouver references, authored as Thor Whalen. Lead with a **decision table**: for a given output object (string / fields / table / graph), which metric(s) to use and which library implements them. Favor recent (last ~2 years) sources and maintained libraries. Be explicit about what is *not* yet well-tooled (gaps we'd have to build).

---

## R3 — Reference-Free Quality Estimation, Confidence, Calibration & Selective Prediction *(Engine: Claude.ai)*

> **Context:** Attached is the conceptual map (see §3). This is the **online / reference-free** need: estimate output quality at inference with no gold answer, then decide accept / flag / block. *If available, also attach the R1 OCR inventory* — it tells us which intrinsic confidence signals each engine emits.
>
> **Research questions:**
> 1. **Confidence sources** — intrinsic (per-char/word posteriors; sequence/token logprobs for generative extractors), ensemble/agreement-based (multi-system voting incl. **ROVER**; for LLMs, **self-consistency** and sampling-variance uncertainty), and trained auxiliary **QE models** (incl. the machine-translation QE lineage, e.g., reference-free COMET-style estimators) — what transfers to OCR/IE.
> 2. **Calibration** — temperature/Platt/isotonic scaling, **ECE** and reliability diagrams; how to calibrate sequence-level and field-level confidences; current libraries.
> 3. **Conformal prediction** — distribution-free coverage guarantees for abstention/flagging in IE/sequence settings; how to set principled thresholds; current libraries (e.g., MAPIE/crepes/TorchCP-class tools) and their applicability.
> 4. **Selective prediction** — reject-option methods, **risk–coverage** analysis, choosing operating points; recent results on selective prediction for extraction/generation.
> 5. **Practical recipe** — given an extractor that emits (or doesn't emit) intrinsic confidence, what's the recommended stack to get a *calibrated, actionable* per-field quality score?
>
> **Deliverable:** `.md`, Vancouver references, authored as Thor Whalen. Lead with a **decision flow**: "what confidence can I get from this extractor → how to calibrate it → how to turn it into a flag/gate with a known error rate." Separate pre-LLM and LLM-era methods. Flag the cheapest reliable options first (consistent with avoiding LLMs where simpler methods suffice).

---

## R4 — Post-Extraction Validation & Correction (incl. Post-OCR) *(Engine: Claude.ai)*

> **Context:** Attached is the conceptual map (see §4). This report covers **reference-free validation and correction** of extracted values — the noisy-channel / prior-based "smoothing" layer. *Attach the R1 OCR inventory if available* (built-in lexicon/LM features per engine matter here).
>
> **Research questions:**
> 1. **Noisy-channel / post-OCR correction** — current state of the art in post-OCR error detection and correction (LM-based, seq2seq, and the recent LLM-based approaches; reference the ICDAR post-OCR correction lineage), with maintained tooling where it exists.
> 2. **Language-model priors** — surprisal/perplexity and masked-LM pseudo-likelihood for anomaly flagging and correction; domain-adapting the prior.
> 3. **Constrained / structured generation** — grammar- and JSON-schema-constrained decoding and validation-on-generate for generative extractors; current libraries (outlines/guidance/lm-format-enforcer/instructor-class tools) and tradeoffs.
> 4. **Schema/type/range & business-rule validation** — declarative validation stacks for extracted records (pydantic/jsonschema/pandera/great-expectations-class tools); how to express type/range/enum/regex/plausibility checks as reusable, composable validators.
> 5. **Cross-field & cross-source consistency** — integrity/consistency constraints (totals reconcile, dates ordered, referential integrity), and cross-source triangulation as a reference-free signal; anomaly-detection methods on extracted values.
>
> **Deliverable:** `.md`, Vancouver references, authored as Thor Whalen. Lead with a **layered checklist**: cheapest/deterministic validators first (type/range/regex/consistency), then LM-prior methods, then learned/LLM correction — with a library per layer. Be explicit about which methods *correct* vs. merely *flag*.

---

## R5 — HITL Review UX, Active Learning & Production Monitoring *(Engine: Claude.ai)*

> **Context:** Attached is the conceptual map (see §6 and §8). This report covers how confidence and validation surface to a human, and how the loop closes over time.
>
> **Research questions:**
> 1. **Confidence visualization** — patterns for trustworthy (calibrated) green→red cues, heatmaps, and uncertainty display in document-review interfaces; evidence on what actually drives correct human verification without over- or under-trusting.
> 2. **Soft signals vs. hard gates** — UX patterns for nudges vs. blocking thresholds tied to a risk–coverage operating point; escalation policies.
> 3. **Provenance / drill-down** — best practices for linking an extracted field back to its source (image bounding box, raw OCR transcript(s) with disagreement spans, source span), and presenting multiple raw outputs side-by-side for fast adjudication.
> 4. **Review-queue triage & active learning** — prioritizing human effort by (cost × uncertainty); active-learning loops where corrections become new gold; tooling for human-in-the-loop annotation/correction (Label Studio / Argilla / Prodigy / doccano-class tools) and their fit for document/OCR review.
> 5. **Production monitoring** — label-free **drift / distribution-shift** monitoring (confidence/abstention/validator-firing distributions), sample-for-audit strategies, and tooling.
>
> **Deliverable:** `.md`, Vancouver references, authored as Thor Whalen. Lead with a **pattern catalog** (each pattern: when to use, the signal it consumes, the affordance). Favor recent HCI/HITL findings and currently-maintained review tools. Note which patterns we can get "for free" from existing annotation tools vs. must build.

---

## R6 — Library Landscape & Integration Map *(Engine: Claude Code)*

> **Context:** Attached are the conceptual map and (ideally) the finished R1–R5 reports. We now want a concrete, *implementable* plan: which existing libraries to pool, and how they compose into evaluation tooling for both the offline and online needs.
>
> **Task:** Survey and verify (check PyPI/GitHub: maintenance, license, last release, API stability) the concrete libraries needed to implement:
> - surface & structured metrics (edit-distance/CER/WER, field-F1, TEDS, graph edit distance),
> - calibration & conformal/selective prediction,
> - validation & constrained generation,
> - HITL/annotation,
> - and the OCR engines from R1 behind a uniform interface.
>
> Then propose an **integration architecture** consistent with my Python conventions: a thin **facade** over pluggable backends, **functions-as-parameters** for swappable metrics/validators/confidence-sources, a **strategy/plugin** layout, tiered `extras_require` by dependency weight and license, and **progressive disclosure** (one-liners for the common case, full control underneath). Separate **graph grammar / schema** from **extraction-and-verification metadata** as in our existing two-layer design.
>
> **Deliverable:** `.md`, authored as Thor Whalen, with inline links to each library. Lead with a **dependency-tier table** (core / OCR / calibration / validation / HITL) flagging license and maintenance risk, then a small **architecture sketch** (interfaces + how a metric, a confidence source, and a validator plug in), then a prioritized build/borrow list (what to reuse as-is, wrap, or build). Call out any AGPL or otherwise license-incompatible dependencies explicitly, given permissive-license downstream goals.
