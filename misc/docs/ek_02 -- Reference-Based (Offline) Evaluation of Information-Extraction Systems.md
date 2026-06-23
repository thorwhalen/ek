# R2 — Reference-Based (Offline) Evaluation of Information-Extraction Systems

**Author: Thor Whalen**
**Series: Information-Extraction Evaluation Ladder — Report R2**
**Date: June 22, 2026**

## TL;DR
- Match the metric to the output object type: use **CER/WER + normalized edit distance** (jiwer, torchmetrics, HF evaluate, Whisper normalizer) for free strings; **entity/slot precision/recall/F1 with partial-match schemes** (seqeval, nervaluate) for fields/slots; **TEDS and GriTS** (PubTabNet, Microsoft Table-Transformer) for tables; and **tree/graph edit distance plus ANLS\\*** (zss, APTED, networkx GED, anls_star, metametric) for nested/typed-graph targets.
- The single biggest source of bogus scores is **canonicalization before scoring**: Unicode form, number/date folding, case, whitespace, and punctuation choices can swing scores by tens of points — the Whisper paper reports "WER drops of up to 50 percent usually due to a quirk such as a dataset's reference transcripts seperating contractions from words with whitespace." Build and version a schema-specific canonicalizer as a first-class component.
- The genuinely under-tooled gap is **cost-sensitive, importance-weighted, typed-graph evaluation**: there is no maintained, drop-in library that scores a nested typed graph with per-field/per-type business-cost weights. Teams must compose `metametric`, `zss`/APTED custom cost functions, or `anls_star` themselves.

## Decision Table — Output Type → Metric → Library

| Output object type | Recommended metric(s) | Library / tool that implements it | Notes |
|---|---|---|---|
| **String** (free text, OCR/ASR transcript, single field value) | Character Error Rate (CER), Word Error Rate (WER), normalized Levenshtein similarity / edit distance | `jiwer`, `torchmetrics` (`CharErrorRate`, `WordErrorRate`), HuggingFace `evaluate` (cer, wer wrapping jiwer), `fastwer`; `whisper_normalizer` for canonicalization | Always normalize first. jiwer ≥4.0 lets you score empty references (hallucination on silence). |
| **Single short answer vs. acceptable variants** | ANLS (Average Normalized Levenshtein Similarity) | DocVQA official scripts, `anls_star` (back-compatible) | ANLS thresholds at 0.5; tolerates minor OCR/spelling differences. |
| **Fields / slots** (key-value pairs, typed entities, spans) | Entity/field precision, recall, F1 (exact and partial); slot error rate | `seqeval` (CoNLL-style, IOB/IOBES strict mode), `nervaluate` (SemEval-2013: strict/exact/partial/type), HF `evaluate` (seqeval) | Choose type-aware vs type-agnostic and exact vs partial explicitly; they diverge widely. |
| **Fields with semantic grouping / nesting** | Group-aware entity F1; ANLS\\* over dict/list trees | `KIEval` (2025), `anls_star`, `TreeForm` (2024) | Flat entity F1 overstates KIE quality when grouping matters. |
| **Tables** (HTML/grid, merged cells) | TEDS, TEDS-Struct (structure-only); GriTS (topology/content/location) | TEDS: PubTabNet/IBM reference code, PaddleOCR `ppstructure`; GriTS: `microsoft/table-transformer` | TEDS is tree-edit on HTML; GriTS scores the table in native matrix form and yields precision/recall. |
| **Nested objects / typed graphs** (JSON, relation/event graphs, line items) | Tree edit distance (normalized), graph edit distance (GED), substructure-matching F1, ANLS\\* | `zss` (Zhang-Shasha), `apted`/APTED, `networkx` GED (`graph_edit_distance`, `optimize_graph_edit_distance`), `anls_star`, `metametric` | GED is NP-hard — use approximations. metametric derives MUC/B³/CEAF and custom metrics from dataclasses. |
| **Coreference / cluster-shaped output** | MUC, B³ (B-cubed), CEAF (φ3/φ4), BLANC, CoNLL avg | `conll/reference-coreference-scorers`, `corefud-scorer`, `metametric` (`coref_suite`) | CoNLL score = unweighted mean of MUC, B³, CEAFe F1. |

---

## 1. Surface Metrics (Strings)

### 1.1 CER, WER, normalized edit distance — best practice and tooling
CER and WER are edit-distance ratios: the count of substitutions + insertions + deletions to turn the prediction into the reference, divided by the number of reference characters (CER) or words (WER). A value of 0 is perfect; values can exceed 1.0 (100%) when the hypothesis is longer than the reference, which is often a hallucination signal.

The maintained tooling landscape:
- **jiwer** (jitsi/8x8, Apache-2.0) is the de-facto Python standard. It computes WER, MER (Match Error Rate), WIL/WIP, and CER using a fast RapidFuzz C++ edit-distance backend, exposes alignments (`process_words`, `process_characters`, `visualize_alignment`), and ships a CLI. As of v4.0 it defines behavior for empty references so you can test for hallucination on silent input. Its `jiwer.transforms` (Compose, Strip, RemoveMultipleSpaces, ReduceToListOfListOfWords, etc.) double as a normalization pipeline.
- **torchmetrics** provides `CharErrorRate` and `WordErrorRate` classes/functionals for in-training logging; correct usage requires global accumulation across the epoch rather than averaging per-batch WER (PyTorch Ignite's feature discussion notes that "naively averaging batch-level WER gives mathematically incorrect results because sequences have variable lengths").
- **HuggingFace `evaluate`** wraps jiwer for `cer`/`wer` and `seqeval` for entity tasks.
- **fastwer** is a fast C++ option but less actively maintained than jiwer.

### 1.2 Text normalization / canonicalization before scoring
This is the single highest-leverage and most error-prone step. Typical operations, in pipeline order:
1. **Unicode normalization** — NFC vs NFKC (compatibility decomposition folds ligatures, full-width forms, etc.).
2. **Case folding** — lowercasing.
3. **Whitespace** — collapse multiple spaces, strip, normalize newlines.
4. **Punctuation** — strip or standardize.
5. **Number folding** — map number words ↔ digits, normalize thousands separators/decimals.
6. **Date normalization** — canonical date formats.

Reference implementations:
- **Whisper's normalizers** — `EnglishTextNormalizer` and `BasicTextNormalizer`, available standalone via the `whisper_normalizer` PyPI package. `EnglishTextNormalizer` removes bracketed/parenthesized phrases, removes filler words (hmm/mm/uh/um), expands contractions, removes symbols/diacritics in Unicode categories M/S/P (except period, percent, currency), and converts numeric/currency expressions to Arabic-numeral forms.
- **jiwer transforms** — composable, used to standardize before WER.

### 1.3 Pitfalls
- **Normalization changes scores materially.** Per the Whisper paper (Radford et al., arXiv 2212.04356, Appendix C): "For several datasets, we observe WER drops of up to 50 percent usually due to a quirk such as a dataset's reference transcripts seperating contractions from words with whitespace." This makes cross-paper comparison invalid unless the exact normalizer is shared.
- **Normalizer overfitting.** OpenAI acknowledges its normalizer was co-developed with Whisper, risking overfit to Whisper's quirks; they cross-checked against the FairSpeech (Koenecke et al., 2020) normalizer and found they "perform similarly" on most datasets but diverge significantly on WSJ, CallHome, and Switchboard.
- **Language-specific damage.** The "What is lost in Normalization?" study (arXiv 2409.02449) shows Whisper's `BasicTextNormalizer` strips Unicode Mark-class characters (matras/vowel signs) in Indic scripts, destroying valid word forms. On Google FLEURS with Whisper-small, normalization produces "suspicious absolute WER reductions" of **21.9% for Hindi, 41.5% for Tamil, and 152.2% for Malayalam** — versus only 5.1% for English and 3.2% for Finnish — i.e., normalization tuned for English silently and massively corrupts WER for abugida scripts.
- **Standardization gap.** OCR/ASR benchmarks rarely fully specify their normalizer, so leaderboard numbers are often not comparable.

---

## 2. Field / Slot-Level Metrics

### 2.1 Slot error rate, entity P/R/F1
The workhorse for fields/slots is precision/recall/F1 over typed spans or key-value pairs. Two conventions matter and diverge:
- **Token/tag-level** (per-token label accuracy) vs **entity/span-level** (the whole span must match). seqeval is entity-level and CoNLL-compatible; SQuAD-style token F1 gives partial credit. The boundary/type/false-positive error taxonomy matters: a model that finds "Electric" instead of "General Electric" gets 0 under exact-span F1 but recall 0.5 / precision 1.0 (F1 ≈ 0.667) under token-level F1.
- **Field extraction F1 (alignment-free)** checks whether each extracted field value appears in ground truth; a single extra/missing character fails the field. Used by FUNSD, CORD, SROIE. Advantage: no alignment needed. Weakness, per the "Reading Order Independent Metrics" paper (arXiv 2404.18664): "it cannot assess the structure of nested entities. Additionally, it does not consider partial overlaps and is very strict for long entity blocks."

### 2.2 Partial-match and span-overlap scoring
- **seqeval** (chakki-works) — CoNLL `conlleval`-compatible entity F1; supports IOB1/IOB2/IOE1/IOE2/IOBES/BILOU and a strict mode keyed to a scheme. Default mode mimics conlleval.
- **nervaluate** (MantisAI) — implements the SemEval-2013 Task 9.1 scheme with four evaluation schemas: **strict** (span + type), **exact** (span, any type), **partial** (overlap, any type), **type** (type with span overlap). Counts COR/INC/PAR/MIS/SPU and computes Precision = (COR + 0.5·PAR)/ACT and Recall = (COR + 0.5·PAR)/POS for the partial schema. Note nervaluate and seqeval can disagree because seqeval ignores other-type tags at the tag level while nervaluate includes them.
- **Coreference / cluster metrics** — MUC (link-based), B³ (mention-based), CEAFm/CEAFe (optimal one-to-one cluster alignment via Kuhn-Munkres), BLANC. The CoNLL-2012 official `reference-coreference-scorers` (v8.01) and `corefud-scorer` (adds LEA, MOR) are the standards; `metametric`'s `coref_suite` reproduces MUC/B³/CEAFφ4 from dataclasses.

### 2.3 How leading document-IE benchmarks score structured outputs
- **CORD, FUNSD, SROIE** — entity-level / field-extraction F1. FUNSD: 4 entity types (key/value/header/other), 149 train / 50 test; SROIE: 4 fields (company, date, address, total), 626 train / 347 test; CORD: 30 labels in 4 groups, 800/100/100 split. Concrete strong-model numbers: **LayoutLMv3 reaches entity-level F1 of 90.29 (base) / 92.08 (large) on FUNSD and 96.56 (base) / 97.46 (large) on CORD** (Huang et al., LayoutLMv3, arXiv 2204.08387).
- **KIEval (2025, arXiv 2503.05488)** — adds group-level evaluation on top of entity F1; shows flat Entity F1 overstates KIE performance on CORD (which has grouping) while converging to Entity F1 on SROIE/FUNSD (no grouping). Built from interpretable TP/FP/FN.
- **TreeForm (2024, arXiv 2402.05282)** — represents FUNSD annotations as a tree (entities = nodes, links = labeled edges) and proposes an end-to-end F1 inspired by labeled attachment score (LAS).
- **DocVQA / DUE benchmark / KLEISTER** — DocVQA uses **ANLS** (Average Normalized Levenshtein Similarity), tolerant of minor OCR/spelling errors; the DUE benchmark aggregates DocVQA, InfographicVQA, KleisterCharity, DeepForm, WTQ, TabFact, PWC; DeepForm and similar IE tasks are scored by **F1**, KleisterCharity by **accuracy**.
- **ExStrucTiny (2026, arXiv 2602.12203)** — schema-variable structured IE from document images; combines ANLS for matched leaves, page accuracy, IoU + normalized proximity for bounding boxes, and normalized tree-edit distance for overall structure, with Hungarian matching of leaves. Notes "multiple JSON structures may convey the same information, so exact string matching or JSON object equality is insufficient for a fair evaluation."

---

## 3. Structured-Object Metrics

### 3.1 Tables — TEDS and GriTS
- **TEDS (Tree-Edit-Distance-based Similarity)** — introduced with PubTabNet (Zhong et al., IBM, ECCV 2020, arXiv 1911.10683). Tables are rendered as HTML trees; TEDS(Ta,Tb) = 1 − EditDist(Ta,Tb)/max(|Ta|,|Tb|). It captures both structure and cell-content (OCR) errors. **TEDS-Struct** ignores cell text to isolate structure accuracy (useful because "taking OCR errors into account may lead to an unfair comparison due to the different OCR models used by various TSR methods"). Reference code released July 2020 in the PubTabNet repo; PaddleOCR's `ppstructure` ships a TEDS evaluator. Strength: holistic single score robust to multi-hop cell misalignment. Weakness: requires HTML serialization, conflates structure with content unless using TEDS-Struct, and is sensitive to how complex spans are encoded.
- **GriTS (Grid Table Similarity)** — Smock et al., Microsoft (arXiv 2203.12555), shipped in `microsoft/table-transformer`. Scores the predicted table directly as a 2-D matrix via the 2D most-similar-substructure (2D-MSS) generalization of 2D-LCS (NP-hard), with a polynomial-time heuristic that returns upper/lower bounds ("in practice there is almost no difference between these bounds"). GriTS yields precision/recall and unifies three subtasks — cell **topology**, cell **content**, cell **location** — in one framework, enabling fairer cross-method comparison than TEDS.

### 3.2 Tree and graph edit distance for nested/graph targets
- **Tree edit distance** — `zss` (Zhang-Shasha, O(n⁴)) offers `simple_distance` and a richer `distance` with custom `insert_cost`/`remove_cost`/`update_cost` functions — the hook for cost-sensitive scoring. `apted`/APTED is tree-shape-independent and generally faster than Zhang-Shasha. Per the Wikipedia summary of the underlying theory, the Zhang-Shasha DP algorithm has worst-case time complexity O(n⁴).
- **Graph edit distance (GED)** — `networkx` provides `graph_edit_distance`, `optimize_graph_edit_distance` (generator of successive approximations), and `optimize_edit_paths`, all accepting `node_match`/`edge_match`/`node_subst_cost`/`edge_subst_cost` callables. GED is **NP-hard (and APX-hard)**; networkx's own docs warn "the problem of finding the exact Graph Edit Distance (GED) is NP-hard so it is often slow." Exact computation is feasible only for small graphs. Approximations: bipartite (Hungarian) assignment of node neighborhoods (cubic), quadratic-assignment formulations (IPFP, GNCCP), and neural GED regressors (which predict a similarity but cannot return an edit path).
- **ANLS\\*** (Peer et al., DeepOpinion, 2024, arXiv 2402.03848) — a universal drop-in generalization of ANLS for strings, tuples (best match), lists (Hungarian matching, penalizing missing/hallucinated items), and dicts (key-value with penalties for missing/hallucinated keys); arbitrarily nested. "The implementation of the ANLS\\* metric maps those complex structures into a tree and compares the ground truth tree against the predicted tree from the model." Open-source `anls_star` single-file implementation; back-compatible with classic ANLS scores. This is the closest thing to an off-the-shelf nested-JSON IE metric.
- **metametric** (Chen et al., EMNLP 2023, arXiv 2310.13793; PyPI Nov 2024) — a conceptual framework + library deriving metrics from dataclass structure via "matching of common substructures, possibly followed by normalization"; reproduces MUC/B³/CEAF and supports defining new structured metrics bottom-up via a decorator over an arbitrary dataclass.

### 3.3 Cost-sensitive / importance-weighted scoring
Where downstream cost differs by field/type, you want weighted scoring:
- **Custom edit costs.** `zss.distance` and networkx GED both accept per-node/per-edge cost callables, so you can encode "getting the invoice total wrong costs 10× a typo in a memo line." This is the principled route but you write the cost functions yourself.
- **Weighted F1 / utility-weighted scoring.** Assign per-type weights and compute a weighted average of per-type F1; promptfoo supports per-assertion `weight` so a structured test's final score is a weighted average ("the final score of the test case is calculated as the weighted average of the scores of all assertions").
- **No turnkey tool.** There is no maintained library that takes a typed schema + a cost matrix and returns an importance-weighted typed-graph distance. This is the principal build-it-yourself gap (see §6).

---

## 4. Gold-Standard Methodology

### 4.1 Annotation workflows: double annotation + adjudication
Best practice for a gold test set is independent double annotation followed by adjudication of disagreements by a senior annotator or committee. The DUE benchmark, for example, estimated human performance with two annotators per dataset, averaging their gold-validated scores. OmniDocBench used "automated tools, manual verification, and expert review."

### 4.2 Inter-annotator agreement (IAA)
- **Cohen's κ** — two annotators, nominal categories, fixed pairwise comparison.
- **Fleiss' κ** — extends to any fixed number of annotators (consistency of agreement).
- **Krippendorff's α** — most general: any number of coders, any measurement level (nominal/ordinal/interval/ratio), and crucially **handles missing data / incomplete annotation**, making it best for messy real-world annotation. Ranges 1 (perfect) to 0 (chance), negative = systematic disagreement.
- **Libraries:** `nltk.metrics.agreement` (`AnnotationTask.kappa()`, `.alpha()`), `statsmodels` (`fleiss_kappa`), the `krippendorff` PyPI package, the `disagree` library (kappa, Fleiss, Krippendorff, bidisagreements), the `agreement` package (numpy implementation of Cohen, Fleiss, Krippendorff, Gwet's gamma, Scott's pi, with weighted kernels, based on Gwet's *Handbook of Inter-Rater Reliability*), and `irrCAC`.
- **Acceptable thresholds:** common rules of thumb treat κ/α ≥ 0.8 as good/reliable and 0.67–0.8 as tentative, but these are conventions and domain-dependent.

### 4.3 Stratified sampling / slice-based evaluation
A single aggregate score hides failure modes. Stratify the eval set by document type, language, layout, vendor, field rarity, etc., and report per-slice metrics. OmniDocBench is explicitly built for this: **1651 PDF pages, covering 10 document types, 5 layout types, and 5 language types** (the current README/CVPR 2025 version; the original Dec-2024 v1.0 release had 981 pages / 9 doc types / 4 layout / 3 language), with page-level and block-level attribute tags enabling attribute-based slicing.

### 4.4 Regression / golden-set testing for IE/LLM pipelines
The mature pattern: maintain a versioned **golden dataset** of representative inputs with expected outputs, run evals on every PR/deploy, and fail the build if a metric crosses a threshold. Keep the CI golden set small (20–50 examples for sub-5-minute CI), expand from real production failures.
- **DeepEval** uses "goldens" — "a golden is a precursor to a test case," a `Golden` Pydantic object with `input`, `expected_output`, `context`, `expected_tools` collected into an `EvaluationDataset`; integrates with pytest (`assert_test`, `deepeval test run`) for CI gating; a Golden Synthesizer generates goldens from docs/contexts.
- **promptfoo** — declarative YAML test cases with pass/fail assertions, weighted assertions, JSON-schema validation; runs locally and in CI for multi-model regression testing.
- **LangSmith** — datasets + tracing for offline eval and production-regression surfacing (tightly coupled to LangChain).

---

## 5. Eval Frameworks

### 5.1 General LLM/IE eval harnesses
- **promptfoo** (MIT; reportedly acquired by OpenAI March 2026) — CLI/YAML, deterministic assertions (equals, contains, regex, JSON schema, latency, cost) plus model-graded (llm-rubric) assertions; supports per-assertion weights and a "Evaluating JSON Outputs" guide for validating structured outputs against schemas. Strength: fast CI gating and multi-model comparison; gap: not a structured-graph metric engine.
- **DeepEval** (Confident AI; "pytest for LLMs") — 14+ metrics (faithfulness, hallucination, JSON correctness, tool correctness, G-Eval custom rubric), pytest-native CI gating, golden datasets. Gap: metric thresholds set in dev "often break in production because the domain distribution differs."
- **OpenAI Evals**, **lm-evaluation-harness** (EleutherAI), **HELM** (Stanford) — benchmark-oriented harnesses; strong for task suites, weaker for bespoke nested-IE scoring.
- **Inspect** (UK AISI) — general agent/eval framework.
- **Ragas** — RAG-specific; relevant only if extraction is RAG-mediated.
- **Giskard**, **TruLens**, **Arize Phoenix**, **Braintrust**, **LangSmith** — observability/regression-tracking platforms. The emerging consensus (inference.net, Comet comparisons): pair a lightweight CI-gating tool (DeepEval/promptfoo/Ragas) with a dataset/regression platform (Braintrust/LangSmith/Arize) — "you almost certainly need two tools."

### 5.2 OCR / document-specific eval tooling
- **OmniDocBench** (opendatalab, CVPR 2025, arXiv 2412.07626) — the leading 2024–25 document-parsing benchmark + eval framework. Per-subtask metrics (from its config): text blocks → Normalized Edit Distance + BLEU + METEOR; display formulas → Edit distance + CDM; tables → TEDS + Edit distance; reading order → Edit distance; layout → mAP. Normalized Edit Distance is NED = 1 − Lev(s1,s2)/max(|s1|,|s2|). Tables converted to HTML for TEDS (uses PubTabNet implementation). Supports end-to-end, task-specific, and attribute-level slicing. Actively maintained (v1.7 as of 2026). A known limitation (per LlamaIndex analysis): continuous metrics against a single fixed ground truth "tend to punish small, harmless differences like punctuation, spacing, and line breaks."
- **CDM (Character Detection Matching)** — formula-recognition metric (Wang et al., Shanghai AI Lab, arXiv 2409.03643, Sept 2024). "CDM renders both the model-predicted LaTeX and the ground-truth LaTeX formulas into image-formatted formulas, then employs visual feature extraction and localization techniques for precise character-level matching, incorporating spatial position information" — avoiding the unfairness of BLEU/edit-distance on non-unique LaTeX. (Several secondary sources mis-expand the acronym as "Correctness and Discrepancy Metric" or "Concept-Driven Metric"; "Character Detection Matching" is authoritative per the primary paper.)
- **OCRBench / OCRBench v2** (Fu/Liu et al., arXiv 2501.00321) — bilingual, 10,000 human-verified QA pairs across 31 scenarios and 23 tasks (text recognition, VQA, KIE, HME recognition, text localization, reasoning); exact string matching for options-based questions. Per the paper, "Upon evaluating 38 state-of-the-art LMMs… 36 out of 38 models scored below 50 out of 100," exposing weak text localization and logical reasoning.
- **marker** (datalab-to/marker) — benchmarks via heuristic text-alignment scoring + LLM-as-judge ("We scored based on a heuristic that aligns text with ground truth text segments, and an LLM as a judge scoring method"); separate table benchmark on FinTabNet.
- **Nougat** (facebookresearch/nougat, arXiv 2308.13418) — ships `test.py` computing edit-distance accuracy, BLEU, METEOR ("Edit Distance (ED) based accuracy score… BLEU score… METEOR score"); paper also reports precision/recall/F1, broken out by All/Tables/Plain text/Math. "In this work we consider the normalized edit distance, where we divide by the total number of characters."
- **docTR** (mindee/doctr, Apache-2.0) — `doctr.utils.metrics`: `TextMatch` (word-level recognition accuracy at 4 normalization tolerances: raw, lower-case, anyascii, lower-case+anyascii), `LocalizationConfusion` (recall/precision/mean IoU, default thresh 0.5), `DetectionMetric`, and `OCRMetric` (end-to-end box-match + string-match).
- **PaddleOCR** — `tools/eval.py`: detection scored by "Precision, Recall, and Hmean (F-Score)"; recognition by sequence accuracy; `ppstructure` table module by TEDS.

---

## 6. Gaps — What Is NOT Yet Well-Tooled (build-it-yourself)

1. **Cost-sensitive, importance-weighted typed-graph edit distance.** No maintained library accepts a typed schema + per-field/per-type cost matrix and returns a weighted distance/similarity. You must compose `zss.distance`/APTED/networkx-GED custom cost callables or extend `metametric`/`anls_star` yourself. This is the central missing piece for business-cost-aligned IE evaluation.
2. **Schema-tuned canonicalization-before-scoring pipelines.** Generic normalizers (Whisper, jiwer transforms) are not schema-aware. Canonicalizing dates, currencies, units, enumerations, and entity aliases per your schema — and versioning that canonicalizer alongside the gold set — is bespoke work. Mis-tuned normalization silently inflates or deflates scores and, as the Indic-script data shows, can break non-English scripts catastrophically.
3. **Partial-credit nested-structure metrics with interpretable error attribution.** ANLS\\* gives a single number; metametric gives substructure F1; but a tool that decomposes a nested-JSON score into per-path/per-type TP/FP/FN with partial credit and confidence intervals is not off-the-shelf.
4. **Standardized cross-benchmark normalization.** Because benchmarks under-specify normalizers, comparable leaderboard numbers require re-running everyone's outputs through one shared canonicalizer — infrastructure teams must build.
5. **Graph-matching at scale.** Exact GED is NP-hard; approximations exist in research code (IPFP, GNCCP, bipartite) but are not packaged as maintained, well-documented Python libraries with cost-function APIs. networkx is the practical default but slow on non-trivial graphs.
6. **Slice-aware regression harness for structured IE.** Golden-set CI tools (DeepEval, promptfoo) score flat outputs well but don't natively compute TEDS/GriTS/tree-edit/ANLS\\* per slice with regression baselines; wiring structured metrics into CI gating is custom integration work.

## Recommendations (staged)

**Stage 0 — Foundations (week 1).** Pick output types per field. Stand up a **versioned canonicalizer** (Unicode NFC/NFKC decision, case, whitespace, number/date/currency folding) as a standalone, tested module. Adopt jiwer/torchmetrics for string fields and seqeval+nervaluate for slot fields. Benchmark threshold to change this: if normalization changes a field's score by >2–3 points, the normalizer is doing too much or too little — inspect. (The Indic-script evidence — up to 152% spurious WER reduction for Malayalam — is the cautionary tale: never reuse an English normalizer on another script without auditing Mark-class handling.)

**Stage 1 — Structured scoring (weeks 2–4).** For tables, adopt TEDS-Struct + GriTS (use the Table-Transformer GriTS code). For nested JSON, adopt `anls_star` as the baseline single-number metric and `metametric` for substructure F1 with error attribution. Establish a gold test set with double annotation + adjudication; report Krippendorff's α (target ≥0.8; below 0.67, fix the annotation guidelines before trusting any model score).

**Stage 2 — Cost alignment (month 2).** Where business cost is uneven, implement per-field/per-type weights: weighted F1 for flat fields, custom `zss`/networkx cost functions for graphs. Validate that the weighted metric ranks known-good vs known-bad extractions in the intended order on a hand-built diagnostic set.

**Stage 3 — Regression harness (ongoing).** Wire structured metrics into CI (DeepEval goldens or promptfoo YAML) with a small (20–50) slice-stratified golden set; fail builds on threshold regressions. Expand the golden set from production failures. Re-baseline thresholds ~2 weeks after any major model/prompt change.

**Triggers to revisit:** if a new maintained library ships cost-sensitive typed-graph scoring, replace your custom code; if you add a new language/script, re-audit the canonicalizer for Mark-class/diacritic damage; if leaderboard comparisons matter, re-run competitor outputs through your shared normalizer.

## Caveats
- Several findings rest on benchmark/library docs and arXiv preprints; preprint metrics (KIEval, TreeForm, ExStrucTiny, CDM) may change on peer review.
- Acceptable-agreement thresholds (κ/α) are conventions, not laws; calibrate to your domain.
- Some secondary sources mis-expand "CDM"; the primary paper (arXiv 2409.03643) defines it as Character Detection Matching.
- GED approximations trade accuracy for tractability; document which approximation and cost model you used, since results are not comparable across choices.
- promptfoo's OpenAI acquisition (March 2026) is per secondary reporting; governance/licensing could shift.
- LayoutLMv3 FUNSD/CORD numbers are from the model authors and HF model cards; exact figures vary slightly by fine-tuning run and reported version.

## REFERENCES
1. jiwer — [GitHub jitsi/jiwer](https://github.com/jitsi/jiwer); [docs](https://jitsi.github.io/jiwer/).
2. torchmetrics WER/CER — [WordErrorRate](https://lightning.ai/docs/torchmetrics/stable/text/word_error_rate.html); [PyTorch Ignite WER/CER feature issue #3634](https://github.com/pytorch/ignite/issues/3634).
3. Whisper normalizer — [whisper_normalizer PyPI](https://pypi.org/project/whisper-normalizer/); Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/pdf/2212.04356).
4. Normalization pitfalls — [What is lost in Normalization?](https://arxiv.org/html/2409.02449v4).
5. seqeval — [GitHub chakki-works/seqeval](https://github.com/chakki-works/seqeval).
6. nervaluate — [GitHub MantisAI/nervaluate](https://github.com/MantisAI/nervaluate); [PyPI](https://pypi.org/project/nervaluate/).
7. KIEval — [arXiv 2503.05488](https://arxiv.org/abs/2503.05488).
8. TreeForm — [arXiv 2402.05282](https://arxiv.org/pdf/2402.05282).
9. TEDS / PubTabNet — Zhong et al., [arXiv 1911.10683](https://arxiv.org/abs/1911.10683); [GitHub ibm-aur-nlp/PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet).
10. GriTS — Smock et al., [arXiv 2203.12555](https://arxiv.org/pdf/2203.12555); [GitHub microsoft/table-transformer](https://github.com/microsoft/table-transformer).
11. zss — [Zhang-Shasha docs](https://zhang-shasha.readthedocs.io/); [GitHub timtadh/zhang-shasha](https://github.com/timtadh/zhang-shasha).
12. networkx GED — [Similarity Measures](https://networkx.org/documentation/stable/reference/algorithms/similarity.html); [optimize_graph_edit_distance](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.optimize_graph_edit_distance.html); [Graph edit distance — Wikipedia](https://en.wikipedia.org/wiki/Graph_edit_distance).
13. ANLS\\* — Peer et al., [arXiv 2402.03848](https://arxiv.org/abs/2402.03848); [GitHub deepopinion/anls_star_metric](https://github.com/deepopinion/anls_star_metric).
14. metametric — Chen et al., [arXiv 2310.13793](https://arxiv.org/abs/2310.13793); [GitHub wanmok/metametric](https://github.com/wanmok/metametric); [PyPI](https://pypi.org/project/metametric/).
15. coreference scorers — [conll/reference-coreference-scorers](https://github.com/conll/reference-coreference-scorers); [corefud-scorer](https://github.com/ufal/corefud-scorer).
16. DocVQA / ANLS — [arXiv 2104.14336](https://arxiv.org/pdf/2104.14336).
17. DUE benchmark — Borchmann et al., [NeurIPS 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/069059b7ef840f0c74a814ec9237b6ec-Paper-round2.pdf).
18. OmniDocBench — Ouyang et al., [arXiv 2412.07626](https://arxiv.org/abs/2412.07626); [GitHub opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench).
19. CDM — Wang et al., [arXiv 2409.03643](https://arxiv.org/abs/2409.03643).
20. OCRBench v2 — Fu et al., [arXiv 2501.00321](https://arxiv.org/abs/2501.00321); [GitHub Yuliang-Liu/MultimodalOCR](https://github.com/yuliang-liu/multimodalocr).
21. Nougat — Blecher et al., [arXiv 2308.13418](https://arxiv.org/pdf/2308.13418); [test.py](https://github.com/facebookresearch/nougat/blob/main/test.py).
22. docTR — [GitHub mindee/doctr](https://github.com/mindee/doctr); [metrics docs](https://mindee.github.io/doctr/latest/modules/utils.html).
23. PaddleOCR — [ppstructure table README](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/table/README.md); [detection docs](https://www.paddleocr.ai/v2.10.0/en/ppocr/model_train/detection.html).
24. promptfoo — [docs](https://www.promptfoo.dev/docs/intro/); [assertions/metrics](https://www.promptfoo.dev/docs/configuration/expected-outputs/); [eval guides](https://www.promptfoo.dev/docs/guides/).
25. DeepEval — [evaluation datasets / goldens](https://deepeval.com/docs/evaluation-datasets); [Golden Synthesizer](https://deepeval.com/docs/golden-synthesizer).
26. IAA libraries — [agreement PyPI](https://pypi.org/project/agreement/); [disagree (TDS)](https://medium.com/data-science/assessing-annotator-disagreements-in-python-to-build-a-robust-dataset-for-machine-learning-16c74b49f043).
27. Field-extraction F1 critique — [Reading Order Independent Metrics](https://arxiv.org/pdf/2404.18664).
28. ExStrucTiny — [arXiv 2602.12203](https://arxiv.org/pdf/2602.12203).
29. LayoutLMv3 (FUNSD/CORD F1) — Huang et al., [arXiv 2204.08387](https://arxiv.org/abs/2204.08387).
30. LLM eval framework comparison — [inference.net guide](https://inference.net/content/llm-evaluation-tools-comparison/); [Comet blog](https://www.comet.com/site/blog/llm-evaluation-frameworks/).