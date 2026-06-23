# R1 — OCR Systems Capability Inventory

**Author:** Thor Whalen  
**Engine:** Claude Code (concrete repo/API/doc survey)  
**Date:** 2026-06-22  
**Companion:** *Information Extraction Evaluation — A Conceptual Map* (shared framing; §7 OCR-specific). See also the sibling report **R6 — Library Landscape & Integration Map**.

> **Scope & method.** For each of 15 OCR/VLM systems this inventory records *only evaluation-relevant capabilities*: the rawest structured output it returns, what confidence/likelihood it natively emits (and exactly how to read it), its built-in language/lexicon correction, its structured-layout output, and its deployment/license posture. Each profile was produced in two passes — a first-pass profile, then an adversarial verification pass that re-checked the riskiest claims (confidence-field access, token-logprob availability, license, math/handwriting support) against live repos/docs/API references in mid-2026. The unifying question throughout: *which systems hand us a usable quality-estimation (QE) signal natively, and which need an external confidence layer bolted on?*

---

## TL;DR

- RICHEST native QE signal (per-unit confidence + geometry, ready to feed selective prediction): Google Cloud Vision (per-symbol/word/para/block confidence + glyph-level bboxes) and Azure Document Intelligence (word + selection-mark + field/KV + table/row/cell confidence). Mathpix is the calibrated standout for STEM (dual confidence + confidence_rate at global/line/word + engine-side gating). Tesseract gives the richest LOCAL signal (per-symbol/word/line conf + GetChoiceIterator alternatives + bboxes at every level) — but uncalibrated.
- MIDDLE tier: PaddleOCR / RapidOCR / EasyOCR expose line-level recognition scores plus polygon geometry, but no per-char/word confidence by default and no logprobs — usable as features for an external calibration layer, not turnkey QE.
- MINIMAL / external-layer-required: OCR.space and Apple Vision (ocrmac) emit essentially no usable confidence (OCR.space none; Apple coarse, often quantized ~{0.3,0.5,1.0}). pix2tex emits nothing. These need a bolt-on confidence layer (lexicon agreement, geometry outliers, cross-engine disagreement).
- VLM/LLM extractors hinge on ONE question — logprobs: OpenAI GPT-4o/4.1 CAN return per-BPE-token logprobs (but flaky/-9999/empty on image inputs, empty under strict json_schema — load-test per snapshot). Claude Vision returns NO logprobs and no confidence at all (verified mid-2026) → zero intrinsic QE, must use sampling self-consistency / LLM-judge / external verifier. Mistral OCR is the VLM exception: opt-in word+page confidence (logprob-derived, exp(mean token logprob)).
- Specialists: math → pix2tex (local, no confidence), Mathpix (cloud, rich confidence), Mistral OCR (LaTeX + confidence); handwriting → TrOCR (local, recoverable per-token logprobs via compute_transition_scores, but NO geometry), plus Azure/Mathpix/Mistral.
- Local TOKEN-level logprobs only from generative locals: TrOCR (recoverable) and pix2tex (patch required). All classical locals are CTC → no logprobs by construction.

---

## Decision points — if you need X, reach for Y

- Rich PER-UNIT confidence for QE/selective prediction (cloud): Google Cloud Vision (per-symbol→block confidence) or Azure Document Intelligence (word/selection-mark/field/KV/table-cell confidence). AWS Textract gives per-block Confidence 0–100 (word/line/cell/KV) but no finer.
- Rich confidence for MATH/STEM: Mathpix (calibrated confidence + confidence_rate at global/line/word, plus engine-side thresholds) — best calibrated QE of the set.
- Per-TOKEN logprobs (the only path to intrinsic VLM confidence): OpenAI GPT-4o/4.1 (logprobs=True/top_logprobs — caveat: unreliable on images + empty under strict json_schema); TrOCR locally (recoverable via output_scores + compute_transition_scores); Mistral OCR (logprob-DERIVED word/page confidence, no raw logprobs). NOT Claude Vision (none).
- Rich LOCAL confidence + alternatives (air-gapped): Tesseract via tesserocr (per-symbol/word/line conf, GetChoiceIterator N-best, bbox every level) — uncalibrated, feed an external calibrator.
- Bbox provenance for HITL overlays: glyph/word geometry from Google Cloud Vision (per-symbol bbox), Tesseract (BoundingBox at every level), Azure/Textract (word polygons + normalized bboxes), PaddleOCR/RapidOCR/EasyOCR (polygon quads). AVOID for provenance: Mistral OCR (text locator is char-offset, bbox only for figures), Claude/OpenAI (model-guessed pixels, not provenance), TrOCR/pix2tex (no geometry).
- Table STRUCTURE (cells, not just text): Azure Document Intelligence (rowIndex/colIndex/rowSpan/colSpan), PaddleOCR PP-StructureV3 (pred_html, TEDS-ready), AWS Textract (TABLE/CELL/MERGED_CELL), Mathpix/Mistral (markdown/HTML tables). NOT Google Cloud Vision (TABLE = tag only, no cells).
- Math/formula extraction: pix2tex (local, free, MIT weights, no confidence), Mathpix (cloud, rich confidence + chemistry), Mistral OCR (LaTeX + confidence), PaddleOCR (PP-FormulaNet LaTeX).
- Handwriting: TrOCR (local, single-line, per-token logprobs), Azure (styles.isHandwritten flag), Mathpix, Mistral OCR.
- Fully LOCAL / private / air-gappable: Tesseract, EasyOCR, RapidOCR, PaddleOCR, ocrmac/Apple Vision (macOS-only), pix2tex, TrOCR. Self-host containers: Azure (disconnected), OCR.space Local. NOT self-hostable: Google Vision, Claude, OpenAI, Mistral OCR (cloud-only).
- LOWEST cost: the local-and-free stack (Tesseract / EasyOCR / RapidOCR / PaddleOCR / ocrmac / pix2tex / TrOCR — all Apache-2.0 or MIT, no per-call fee). Cheapest rich-confidence cloud: Mistral OCR (~$2/1k pages, ~$1/1k batch).

---

## Comparison matrix

Rows = systems (grouped by class). Columns distill the five evaluation dimensions plus a native-QE verdict. Full detail — exact API fields, output schemas, code — is in the per-system profiles below.

| System | Rawest output | Native confidence | Token logprobs | LM / lexicon prior | Structured output | Deployment & license | Native QE verdict |
|---|---|---|---|---|---|---|---|
| **Tesseract (+ tesserocr / pytesseract)**<br>*Local classical OCR — CTC-decoded LSTM line recognizer* | Hierarchical tree BLOCK>PARA>TEXTLINE>WORD>SYMBOL; per-level text+bbox+conf via ResultIterator/iterate_level; GetChoiceIterator alt glyphs. pytesseract image_to_data 12-col TSV; image_to_boxes per-char boxes (no conf). | Confidence(level) float 0..100 (uncalibrated, not a probability) at SYMBOL/WORD/LINE/PARA/BLOCK; AllWordConfidences()/MapWordConfidences()/MeanTextConf(); pytesseract conf col (=-1 on structural rows); GetChoiceIterator candidate confs. | None. CTC LSTM, not autoregressive. No logits/logprobs. Closest signal: 0..100 candidate confidences via GetChoiceIterator / GetBestLSTMSymbolChoices (prob 0..1 internally). | DAWG dictionaries baked into lang.traineddata (load_system_dawg, load_freq_dawg, load_punc/number/bigram_dawg); language_model_penalty_non_dict_word; --user-words/--user-patterns; char whitelist/blacklist. | bbox every level via BoundingBox(level); hOCR (x_wconf always, x_confs only with lstm_choice_mode=2), ALTO XML, TSV, searchable PDF. NO tables/KV/JSON-schema; axis-aligned rects only. | 100% local/on-device C++ binary, air-gappable, CPU-only, free. Wrappers: tesserocr (in-process Cython, rich iterators) + pytesseract (subprocess). ~100+ langs. — Engine Apache-2.0; pytesseract Apache-2.0; traineddata (tessdata/_best/_fast) Apache-2.0; tesserocr MIT (NOT Apache-2.0). No AGPL/non-commercial anywhere. | Rich per-unit conf + alternatives + geometry, but uncalibrated and no logprobs — feature source for an EXTERNAL calibration/selective-prediction layer, not turnkey QE. |
| **EasyOCR (JaidedAI)**<br>*Local neural OCR (CRAFT detector + CRNN/CTC recognizer); non-generative* | list of (bbox, text, confidence) tuples per text region; bbox = 4-pt polygon TL,TR,BR,BL; word/line granularity (not per-glyph); no hOCR/ALTO | 1 scalar/region (result[i][2] or dict key 'confident') | None (not token-generative); per-char pred_max_prob internal only | No default word-LM/dictionary; char-level allowlist/blocklist; optional wordbeamsearch (internal-corpus LM, not user dictionary) | 4-pt quad polygons + heuristic line/paragraph merge; no tables/KV/reading-order model; output_format dict/json keys boxes,text,confident; free_merge | Fully local on-device PyTorch; models auto-download once then offline; GPU optional; free — Apache-2.0 (code); CRAFT weights from Clova AI - verify weight redistribution; LICENSE template copyright unfilled (#1422) | Low: one uncalibrated geometric-mean recognition score per region; detector confidence not returned; no per-char/token signal without patching |
| **RapidOCR (rapidocr 3.8.4, 2026-06-15)**<br>*Classic CTC OCR (PaddleOCR PP-OCR port; multi-backend, fully local)* | Single RapidOCROutput dataclass: img, boxes (N,4,2 ndarray), txts, scores, word_results, elapse_list, elapse, viser. NO lang_rec field (draft was wrong). Per-line default; per-word/char via return_word_box=True. | Per-line result.scores[i] in [0,1] (always on); per-word/char = 2nd elem of word_results inner tuple (opt-in, rounded 5 dp). Det/cls confidences internal only, not surfaced. | None. CTC-decoded CRNN/SVTR, not autoregressive -> no token logprobs. Only softmax-style recognition confidence. | None. No LM/lexicon/spell-correct. Only knobs: rec_keys_path (keys.txt charset) + rec model variant. text_score/box_thresh/unclip_ratio are thresholds, not correction. | 4-pt polygon per line (+per word/char); to_json(), to_markdown(), vis(). No hOCR/ALTO, no KV/form fields. Table/layout/formula = separate packages (rapid_table, rapid_layout, rapid_undistorted, rapid_doc). | Fully local/offline; CPU default (ONNXRuntime); backends onnxruntime/openvino/paddlepaddle/pytorch(+MNN/TensorRT). Models lazy-downloaded+cached (not in wheel). Python 3.8-3.13. — Apache-2.0 (code; owner). PP-OCR weights (c) Baidu, Apache-family. No AGPL/non-commercial -> commercial-safe. | Moderate: 2 native confidence levels (line always, word/char opt-in) + geometry for provenance overlays. No logprobs, no LM-prior signal. text_score default silently filters low-conf lines -- read raw scores pre-filter. |
| **PaddleOCR (incl. PP-OCRv6/v5, PP-StructureV3, PaddleOCR-VL, PP-ChatOCRv4)**<br>*Local neural OCR + document-parsing toolkit; VLM tier (PaddleOCR-VL)* | predict() -> list of result objs; line-level dict: rec_texts, rec_scores, dt_polys, dt_scores, rec_polys, rec_boxes(int16 n×4). StructureV3/VL: parsing_res_list[block_bbox,block_label,block_content], layout_det_res. Tables: table_res_list[i]['pred_html']. No per-char output. | Line-level rec_scores[i] (0-1) recognition posterior aligned w/ rec_texts; dt_scores[i] box detection; layout 'score'. No per-char/per-word confidence. Thresholds: text_rec_score_thresh, text_det_thresh, text_det_box_thresh, drop_score. | None from packaged pipeline (classic or PaddleOCRVL.predict()). PaddleOCR-VL self-served on vLLM uses OpenAI-compatible /v1; logprobs/top_logprobs MAY be requestable via SamplingParams but UNDOCUMENTED for this VLM/image inputs — treat as unverified. | No built-in n-gram LM / spell-correct / allow-deny lists in classic API. Lexicon = per-lang model + rec_char_dict_path. Implicit LM only via ERNIE-4.5-0.3B in PaddleOCR-VL / ERNIE in PP-ChatOCRv4. | Geometry: dt_polys/rec_polys quads + rec_boxes AABB. Layout: ~20 block categories w/ bbox+label+score+reading order. Tables: pred_html (SLANeXt/SLANet) TEDS-ready. Formulas: LaTeX (PP-FormulaNet/UniMERNet). Exports: json/markdown/html/xlsx/word. NO hOCR/ALTO. | Fully local/offline; paddleocr 3.7.0 PyPI (2026-06-11, adds PP-OCRv6). CPU/GPU, mobile/edge models. PaddleOCR-VL servable via vLLM/SGLang/Transformers. PP-ChatOCRv4 KIE optionally calls hosted ERNIE (egress/cost) or local. — Apache-2.0 (toolkit + PaddleOCR-VL 0.9B & 1.6 + SLANeXt weights). Not AGPL / not non-commercial. ERNIE hosted-API terms separate (Baidu). | Medium. Ready QE signals: line rec_score, box det_score, region score. Missing for fine QE: per-char/word confidence, token logprobs. Selective prediction must layer external confidence on top of line-level posteriors. |
| **ocrmac / Apple Vision**<br>*On-device classical OCR wrapper (PyObjC)* | Flat list of tuples (str, float, [x,y,w,h]); detail=False -> strings. Vision=per-line; LiveText=per-char (unit='token', default) or per-line. No hierarchy/tables/KV/hOCR/ALTO. | Per-line float result.confidence() (0-1). QUANTIZED ~{0.3,0.5,1.0} on 'accurate'; varied on 'fast'. LiveText=hardcoded 1.0 (no signal). | None (non-generative). N-best topCandidates_ NOT surfaced -> no margin/entropy. | setRecognitionLanguages_ + setRecognitionLevel_ only. NO setUsesLanguageCorrection_ / setCustomWords_ (correction at Apple default, untunable; no custom lexicon). LiveText: setLocales_ only. | Normalized bboxes 0..1, bottom-left origin, axis-aligned rect (quad reduced). No tables/KV/JSON-schema/hOCR/ALTO. annotate_PIL/matplotlib overlays. | Fully local/on-device, no network/key/cost. macOS-only: Vision 10.15+, LiveText 14+. Behavior OS-version-dependent. — Wrapper MIT (c) 2022 M. Strauss. Apple Vision/VisionKit proprietary OS frameworks, no weights shipped, no AGPL/non-commercial. | Low: coarse line-level (often quantized) confidence only; no logprobs, no N-best, no word/char confidence. Word/char/margin/entropy QE needs forking to call topCandidates_/boundingBoxForRange_ or an external confidence layer. |
| **pix2tex / LaTeX-OCR**<br>*Specialized image-to-LaTeX (math formula) model; local PyTorch* | Single LaTeX str for whole image; no char/word/line/region units, no geometry | None exposed (no confidence field anywhere in return path) | No — token IDs only; x_transformers discards step logits; patch required | No NL LM/lexicon/dictionary; implicit decoder over fixed LaTeX-token vocab; only knob = temperature (0.25); no beam search (TODO), no top_k/top_p/rep_penalty | None — no bboxes, reading order, layout/region, tables, KV, hOCR/ALTO; LaTeX markup is content, not geometry | Local/self-hosted, on-device, no telemetry; CLI/GUI/FastAPI/Streamlit; PyPI 0.1.4 (2025-01-18), stable-but-unmaintained mid-2026 — MIT (code + weights); commercial OK; weights via GitHub release | Empty — bare generator; QE needs external confidence layer (re-score/logit-capture patch or sampling-agreement proxy) |
| **TrOCR (handwritten) — microsoft/trocr-large-handwritten**<br>*Local VLM/seq2seq line-OCR (HF VisionEncoderDecoderModel; BEiT/DeiT enc + RoBERTa dec)* | token-id LongTensor → str; richest = GenerateEncoderDecoderOutput (sequences/scores/logits/attentions). No geometry, no segmentation; single-line input | Derived only: exp(compute_transition_scores(...,normalize_logits=True)); per sub-word token. sequences_scores = length-penalized beam score (not calibrated). No native confidence field | Yes, recoverable (not default): generate(output_scores=True,return_dict_in_generate=True)+compute_transition_scores(normalize_logits=True); output_logits=True for raw logits. Local tensors only, no hosted API | Intrinsic RoBERTa-init decoder LM; no swappable LM/dictionary/lexicon. Only generic generate() controls (bad_words_ids, force_words_ids, prefix_allowed_tokens_fn) | None: no bbox/polygon/reading-order/table/KV/JSON-schema/hOCR/ALTO. Needs external detector for geometry/provenance | Fully local/self-hosted (transformers+PyTorch, ONNX-exportable); air-gappable; ~558M params; GPU recommended. English/IAM single-line; handwriting YES; math/formula NO — Large repo: NO license declared (HF /api/models + card). MIT by provenance (unilm MIT; base-handwritten tagged MIT). Confirm before commercial use. No AGPL/non-commercial | Medium: rich per-sub-word logprobs/posteriors for sequence- & token-level QE, but NO geometry (no provenance overlays), NO word/line confidence natively (must aggregate), single-line scope |
| **OCR.space API**<br>*Classical cloud OCR REST (a9t9); optional on-prem (OCR.space Local)* | JSON; rawest = WORD (isOverlayRequired=true): ParsedResults[]->TextOverlay.Lines[].Words[]{WordText,Left,Top,Height,Width}; no char/glyph level | NONE at any level; only coarse enums OCRExitCode(1-4) & FileParseExitCode | None (classical OCR, not a VLM; no tokens) | Not tunable; only `language` picks a built-in pack; no custom dict/allow-deny/hint/decoder | Axis-aligned word px boxes + line grouping; no block/para/column; tables=text/Markdown only (no cell coords); no KV; no hOCR/ALTO; no field extraction; searchable PDF export | Hosted cloud (EU servers, no doc storage) OR self-hosted on-prem offline via OCR.space Local / Enterprise (~$999+/mo) — Proprietary commercial (no AGPL/non-commercial trap); engine closed-source; free tier watermarks searchable PDFs | Very low intrinsic: no confidence/logprobs -> external QE required (lexicon agreement, bbox-geometry outliers, cross-engine 1/2/3 disagreement) |
| **Google Cloud Vision (TEXT_DETECTION / DOCUMENT_TEXT_DETECTION)**<br>*Cloud OCR API, discriminative recognizer* | fullTextAnnotation (TextAnnotation): pages>blocks>paragraphs>words>symbols (glyph-level); flat textAnnotations EntityAnnotation word list w/ geometry | `confidence` [0,1] at Page/Block/Paragraph/Word/Symbol; detectedLanguages[].confidence [0,1]; uncalibrated | None (no token logprobs; discriminative) | Proprietary internal LM/script model; user controls = languageHints[] (BCP-47) + Feature.model (builtin/stable\|latest\|weekly). No custom dict/allow-deny/decoder knobs | BoundingPoly vertices (int px, image) / normalizedVertices (0-1, PDF/TIFF); 4-vtx axis-aligned quads; blockType{UNKNOWN,TEXT,TABLE,PICTURE,RULER,BARCODE}; TABLE=tag only, no cells; no KV; no hOCR/ALTO/PAGE-XML; zero coords omitted | Hosted cloud only (vision.googleapis.com, REST+gRPC); no on-prem/air-gapped; PDF/TIFF async->GCS — Proprietary GCP ToS; closed weights; commercial OK; not AGPL/non-commercial | High per-level confidence (incl. per-glyph); no logprobs; good geometry for provenance overlays |
| **AWS Textract**<br>*Hosted OCR + document-analysis API (discriminative, non-generative)* | Flat array of typed Block objects (PAGE/LINE/WORD + analysis types), linked by Id/Relationships; finest unit = WORD (no glyph) | Per-block scalar block['Confidence'], Float 0–100, fuses text + geometry accuracy; per WORD/LINE/CELL/KV/SELECTION/QUERY_RESULT/SIGNATURE/PAGE | None — not generative; no token logprobs, no n-best/top-k, no per-char confidence | None configurable; no dictionary/lexicon/allow-deny/user-words; only Custom Queries adapters (AdaptersConfig) tune Queries, not base OCR | BoundingBox{Left,Top,Width,Height}+Polygon[{X,Y}] (norm 0–1)+RotationAngle on geometry; TABLE/CELL/MERGED_CELL, KEY_VALUE_SET, SELECTION_ELEMENT, QUERY, LAYOUT_*; no hOCR/ALTO/PAGE-XML; no schema-constrained output | Hosted-only AWS regional API; no on-prem/on-device; opt-out (AWS Orgs AI policy) stops cross-region service-improvement storage — Proprietary AWS API (Customer Agreement/Service Terms); no weights; response-parser libs Apache-2.0 | Moderate: word/line/field Confidence only; aggregate (min/mean) for QE; calibration undocumented; no logprobs/n-best to exploit |
| **Azure AI Document Intelligence (v4.0, 2024-11-30 GA; under Azure AI Foundry / Content Understanding)**<br>*Cloud OCR/document-IE API + self-hostable container (discriminative, not generative LM)* | Word is lowest unit (no glyph). analyzeResult.pages[].words[]{content,polygon,confidence,span}; lines[] (no confidence); selectionMarks[]; + paragraphs/tables/keyValuePairs/styles/languages/figures/sections/documents.fields; page.barcodes[]/formulas[] with add-ons | Multi-granular [0,1] floats: word, selectionMark, style/font, language, field (documents.fields), keyValuePairs, AND table/row/cell (custom, 2024-11-30 GA). NO line confidence. Formula confidence hard-coded; barcode hard-coded to 1 (sample output contradicts) -> not QE-usable | None. No token logprobs/logits/temperature; not autoregressive | No user LM/lexicon/dictionary/allow-deny/hotword/decoder param. Only: optional locale hint, detect-only languages add-on, post-OCR field normalization (ISO-8601/E.164/currency/ISO-3166) | polygon quads in boundingRegions; reading-order content + char-offset spans; tables[] structured JSON (rowIndex/colIndex/rowSpan/colSpan/kind); outputContentFormat=markdown renders tables as HTML (rowspan/colspan/caption); figures/sections; searchable-PDF overlay. NO hOCR/ALTO | Hosted Azure regional service; official Docker containers (Read, Layout, prebuilt, custom) incl. disconnected/air-gapped (license file, commitment-tier) — Microsoft proprietary/commercial (Product Terms); no source/weights; no AGPL/non-commercial; metered | High for OCR posteriors: word + selectionMark + field/KV + table/row/cell confidence are real signals; handwriting flag (styles.isHandwritten) + language detect add for triage. Watch-outs: no line confidence, formula/barcode confidence fake, no logprobs |
| **Mistral OCR (Document AI, mistral-ocr-latest / -2505 / -2512 'OCR 3')**<br>*Hosted VLM document-AI API (proprietary, closed weights)* | OCRResponse: pages[OCRPageObject{index,markdown,images,dimensions,tables?,hyperlinks?,header?,footer?,confidence_scores?}], model, usage_info, document_annotation?. Primary text = per-page GFM markdown (LaTeX math + ![id](id) refs). No glyph/word/line geometry. | Opt-in via request param confidence_scores_granularity='word'\|'page' (default None). Response: page.confidence_scores = OCRPageConfidenceScores{average_page_confidence_score, minimum_page_confidence_score, word_confidence_scores?[OCRConfidenceScore{text,confidence(0-1),start_index}]}. Tables: OCRTableObject.word_confidence_scores. NEW since 2025 launch. Azure: NOT supported (422). | No raw logprobs / no `logprobs` field / no top-k. confidence IS exp(logprob)-derived: word=exp(mean(token_logprobs)) (geometric mean over subword tokens); page avg/min from per-token exp(logprob). Only this aggregated, monotonic form is exposed. | None user-facing. No lexicon/dictionary/allow-deny list, no decoder controls (temperature/top-p/beam). Internal LMM correction only. Language steering only via annotation prompt (document_annotation_prompt) + JSON schemas. | Bboxes ONLY for extracted images/figures (OCRImageObject top_left/bottom_right, abs pixels vs OCRPageDimensions). NO text bbox; text span locator = start_index char offset into markdown. Tables (markdown\|html), header/footer, hyperlinks. KV extraction via JSON-schema Annotations (bbox_/document_annotation_format). hOCR/ALTO: NO. Doc annotation capped at 8 pages / first 8 image bboxes. | Hosted cloud only (la Plateforme client.ocr.process); also Azure AI Foundry (subset, no confidence). No public open-weights / on-device build; air-gap only via enterprise self-deploy. EU provider. — Proprietary model (closed weights, commercial ToS/Premier, pay-per-page). SDK mistralai = Apache-2.0 (client only). No AGPL / no non-commercial restriction on SDK. | Medium for QE: word+page confidence (logprob-derived) good for selective prediction; figure-level bbox provenance only — text-span provenance is char-offset, not geometric. Pricing (mid-2026): $2/1k pages, ~$1/1k batch (50% off), $3/1k annotations. Handwriting + LaTeX math supported. |
| **Mathpix (Convert API / MathpixOCR)**<br>*Proprietary cloud OCR API (math/STEM specialist; printed + handwritten)* | JSON line_data + word_data (finest grain = word; no char_data); top-level text = Mathpix Markdown | confidence + confidence_rate, both [0,1], at 3 grains: global, per-line, per-word | None exposed. Per-token OCR confidence is computed internally (folded into scalar `confidence`) but no logprob/top-k field. Not a generative VLM API. | Script-level only: alphabets_allowed (allow/deny by script) + detected_alphabets. No user dictionary / custom LM / word allow-deny list. Correction internal/opaque. | Geometry = cnt polygon (x,y px; [TL,TR,BR,BL] when axis-aligned); no bbox/{x,y,w,h} key. Math renderings in data[] (latex/asciimath/mathml/svg/tsv). Tables (mmd/HTML/TSV). SMILES chemistry. No hOCR/ALTO. No form KV. No JSON-schema-constrained output. | Cloud REST api.mathpix.com (v3/text sync, v3/pdf async, v3/strokes). On-prem via enterprise 'On-prem PDF Cloud' only. Client: mpxpy (MIT). — Proprietary SaaS engine/weights (ToS, paid per-usage). Official mpxpy client MIT (thin client, no local OCR). | High for an OCR engine: dual calibrated scalars (confidence + confidence_rate) at 3 grains + engine-side gating (confidence_threshold, confidence_rate_threshold default 0.75) + polygon provenance. No logprobs. |
| **Anthropic Claude Vision (VLM via Messages API)**<br>*Hosted VLM / LLM-as-extractor (not an OCR engine)* | Generic `Message`: content = blocks (TextBlock \| ToolUseBlock). No glyph/word/line layer; text or schema-JSON only. | None. No char/word/line/region confidence. `usage` = token counts only; `stop_reason`/`stop_details` are status, not likelihood. | No. No `logprobs`/`top_logprobs` request param and no logprob field in response (verified live, mid-2026). | No OCR LM/lexicon/allow-list. Claude's generative prior auto-corrects (can hallucinate). Tunables: temperature/top_p/top_k/stop_sequences + prompt only. | GA `output_config.format`={type:json_schema,schema} via constrained decoding (grammar); formerly `output_format`+header structured-outputs-2025-11-13. Strict tool use via strict:true. No per-field confidence. | Hosted-only, closed weights. Claude API / Bedrock / Vertex. Per-visual-token cost (28x28px patch=1 token; 1568 or 4784 token cap). No self-host. — Proprietary (Anthropic Commercial ToS); also Bedrock/Vertex terms. No OSS/weights. | Effectively zero native QE signal. No posterior/logprob/confidence. QE must be external: sampling self-consistency (T>0), LLM-judge, cross-model agreement, separate verifier. Bbox provenance is prompted+approximate (resized_size() remap; not on PDFs). |
| **OpenAI GPT-4o / 4.1 Vision**<br>*Hosted VLM (not an OCR engine)* | Generated text tokens -> string; optional JSON-Schema-shaped JSON. No glyph/char/word/line/block primitive, no native geometry. | Per-BPE-token logprob ONLY (natural log, max 0.0); no per-char/word/line/region posterior; no confidence on geometry. | Yes in principle. Chat: logprobs=True,top_logprobs=N -> choices[].logprobs.content[].{token,logprob,bytes,top_logprobs}. Responses: top_logprobs=N + include=['message.output_text.logprobs'] -> output[].content[].logprobs[]. CAVEAT: flaky/-9999/empty on gpt-4o image inputs; empty when json_schema strict enabled (confirmed GPT-5.1/5.2 Jan-2026, regression-prone). Load-test per snapshot. | No OCR lexicon/allow-deny list. Whole system is an LM -> silent autocorrect. Only logit_bias (token-id) + sampling knobs; structure constrained via json_schema, not vocabulary. | Structured Outputs (json_schema, strict:true) on gpt-4o-2024-08-06 / gpt-4o-mini and later. Gives KV/table/reading-order JSON as MODEL-INFERRED content. NO native bbox/polygon/ALTO/hOCR; schema bbox = model pixel guess, not provenance. | Hosted-only (OpenAI API or Azure OpenAI). Closed weights, no on-prem/on-device. API data not trained on by default; 30-day retention; ZDR for eligible endpoints. — Proprietary (OpenAI Terms / Azure OpenAI). Closed weights, no self-host. NOT AGPL, NOT non-commercial; vendor-locked. | Medium-low: single per-token logprob signal, reliability caveats with images + structured outputs; no geometry confidence; token->field mapping is integrator's job. |

---

## Per-system profiles

### Tesseract

**Category:** local-classical OCR — an LSTM line recognizer, CTC-decoded (not autoregressive/generative). Profiled with its two canonical Python wrappers: [tesserocr](https://github.com/sirfz/tesserocr) (in-process Cython binding to libtesseract, rich iterator API) and [pytesseract](https://github.com/madmaze/pytesseract) (subprocess wrapper parsing TSV/hOCR/ALTO stdout). **License is not uniform: the engine, pytesseract, and all official traineddata are Apache-2.0, but tesserocr is MIT** — both are permissive and commercial-friendly with no copyleft/AGPL/non-commercial restriction.

#### 1. Rawest output
Full hierarchical tree **BLOCK > PARA > TEXTLINE > WORD > SYMBOL(glyph)**, every level carrying a bbox and a confidence. Lowest level = per-symbol via `ResultIterator`, plus per-glyph alternatives via `GetChoiceIterator`. Verified against the official [C++ API example](https://tesseract-ocr.github.io/tessdoc/APIExample.html): `ri->GetUTF8Text(level)`, `ri->Confidence(level)` (float, printed `%.2f`), `ri->BoundingBox(level, &x1,&y1,&x2,&y2)`, and `ChoiceIterator ci(*ri)` with `ci.GetUTF8Text()` / `ci.Confidence()`.

```python
from tesserocr import PyTessBaseAPI, RIL, iterate_level
with PyTessBaseAPI() as api:
    api.SetImageFile('page.png'); api.Recognize()
    it = api.GetIterator()
    for sym in iterate_level(it, RIL.SYMBOL):
        sym.GetUTF8Text(RIL.SYMBOL)   # 'A'
        sym.Confidence(RIL.SYMBOL)    # 92.30  (0..100 float, uncalibrated)
        sym.BoundingBox(RIL.SYMBOL)   # (x1, y1, x2, y2) ints
        for ch in sym.GetChoiceIterator():
            ch.GetUTF8Text(), ch.Confidence()   # alt glyph + 0..100
```
pytesseract flat table (one row per layout element at every level), verified column set:
```python
import pytesseract; from pytesseract import Output
d = pytesseract.image_to_data(img, output_type=Output.DICT)
# 12 keys, exact order: 'level','page_num','block_num','par_num',
#   'line_num','word_num','left','top','width','height','conf','text'
# level: 1=page 2=block 3=para 4=line 5=word
# conf: 0..100 percentage on word (level-5) rows; conf == -1 marks a
#       structural/non-word row (text is empty there) -> filter conf != -1
```
`image_to_boxes(img)` gives per-**character** boxes in Tesseract `.box` format (`char x_bottom y_bottom x_top y_top page`, image-bottom-left origin), but this path carries **no** confidence. Note: pytesseract's own README does not enumerate the TSV columns — it defers to Tesseract's TSV documentation, so the schema is the engine's, not a pytesseract guarantee.

#### 2. Confidence (verified)
Confidence is a **0..100 float, higher = more confident, uncalibrated — NOT a probability and NOT a logprob.** Access points:
- tesserocr per-element: `Confidence(level)` for `level ∈ {RIL.SYMBOL, RIL.WORD, RIL.TEXTLINE, RIL.PARA, RIL.BLOCK}`.
- tesserocr per-word vectors: `AllWordConfidences()`, `MapWordConfidences()`, page mean `MeanTextConf()` (the README shows `AllWordConfidences`, `MeanTextConf`, `GetChoiceIterator`, `iterate_level`; treat exact signatures of the less-common ones as docstring-confirmed, not README-confirmed).
- tesserocr alternatives: `GetChoiceIterator()` over a symbol yields each candidate's `Confidence()` — the closest thing to a posterior over a glyph.
- pytesseract: the `conf` column from `image_to_data` (`-1` on structural rows).
- C++: `ResultIterator::Confidence()`, `TessBaseAPI::AllWordConfidences()`, `MeanTextConf()`. The raw LSTM symbol-choice probabilities are 0..1 internally (`GetBestLSTMSymbolChoices()`), surfaced integerized as 0..100.

For QE you will need an external calibration/aggregation layer (min/mean word conf, char-conf distribution features) on top.

#### 3. Token logprobs
**None.** Tesseract's LSTM is a CTC-decoded line recognizer, not autoregressive — there are no token logprobs/logits in the public API. The only intrinsic per-unit signals are the 0..100 `Confidence()` values and the per-candidate confidences from `GetChoiceIterator()` (enabled richly with `-c lstm_choice_mode=2`).

#### 4. LM / lexicon priors
Language-sensitive correction via DAWG dictionaries baked into each `lang.traineddata`, plus runtime user lexicons (all config-controllable via `config=` in pytesseract or `-c key=value` on CLI): `load_system_dawg`, `load_freq_dawg`, `load_unambig_dawg`, `load_punc_dawg`, `load_number_dawg`, `load_bigram_dawg`; decoder penalties `language_model_penalty_non_dict_word`, `language_model_penalty_non_freq_dict_word`; `--user-words` / `--user-patterns`; hard char constraints `tessedit_char_whitelist` / `_blacklist` / `_unblacklist`.

#### 5. Structured layout
**Exists:** bbox at every level via `BoundingBox(level)`; per-char boxes via `image_to_boxes`; reading order encoded by `block_num`/`par_num`/`line_num`/`word_num`; **hOCR** (`GetHOCRText` / `image_to_pdf_or_hocr`) with `bbox` and `x_wconf` (word conf) always, plus per-character `x_confs` **only when `-c lstm_choice_mode=2`** (verified — same flag that enables `GetChoiceIterator` alternatives); **ALTO XML** (`GetAltoText` / `image_to_alto_xml`); **TSV** (`GetTSVText` / `image_to_data`); searchable PDF; plain text/box.
**Does NOT exist natively:** no table-structure / cell-row-column model; no key-value / form-field extraction; no JSON-schema-constrained output; no polygon/curved-baseline regions (axis-aligned rectangles only). Orientation/script detection is separate (`image_to_osd` / `DetectOS`).

#### 6. Deployment, models, license
100% local/on-device C++ binary; no network, fully air-gappable (strong privacy posture); CPU-only, free. ~100+ languages across three Apache-2.0 traineddata repos: `tessdata` (legacy `--oem 0` + integerized-best LSTM `--oem 1`), `tessdata_best` (LSTM-only, float, retrainable, most accurate), `tessdata_fast` (LSTM-only, integerized, fastest). **`tessdata_best` and `tessdata_fast` are LSTM-only — `--oem 0`/`--oem 2` do not work with them; legacy `--oem 0` requires the main `tessdata` repo models** (verified).
**Handwriting:** not supported out of the box (printed-text models). **Math/formula:** no structural math (no LaTeX/MathML), only whatever glyphs the char model emits.
**License (corrected):** engine Apache-2.0, pytesseract Apache-2.0, all official traineddata Apache-2.0 — but **tesserocr is MIT, not Apache-2.0.** No AGPL or non-commercial restriction anywhere.

**QE implication:** rich per-unit confidence + alternatives + geometry, but uncalibrated and with no logprobs — a feature source for an **external** confidence/selective-prediction layer, not a turnkey QE signal.

**Sources:** [Tesseract C++ API examples (Confidence/BoundingBox/ChoiceIterator/GetBestLSTMSymbolChoices)](https://tesseract-ocr.github.io/tessdoc/APIExample.html); [pytesseract README (function list, Apache-2.0)](https://github.com/madmaze/pytesseract/blob/master/README.rst); [Tesseract TSV format — exact 12 columns, conf=-1 semantics](https://tomrochette.com/tesseract-tsv-format/); [tesserocr README (AllWordConfidences/MeanTextConf/GetChoiceIterator/iterate_level)](https://github.com/sirfz/tesserocr); [tesserocr LICENSE — MIT (not Apache-2.0)](https://github.com/sirfz/tesserocr/blob/master/LICENSE); [tessdata_best — Apache-2.0, LSTM-only](https://github.com/tesseract-ocr/tessdata_best); [tessdata (main) — Apache-2.0, --oem 0 legacy + --oem 1 LSTM](https://github.com/tesseract-ocr/tessdata); [tessdata_fast — Apache-2.0, LSTM-only, oem 0/2 unsupported](https://github.com/tesseract-ocr/tessdata_fast); [PR #1851 — accumulated glyph confidences / lstm_choice_mode / hOCR x_confs](https://github.com/tesseract-ocr/tesseract/pull/1851)

---

### EasyOCR

[EasyOCR](https://github.com/JaidedAI/EasyOCR) (JaidedAI) is a local-neural OCR pipeline: a [CRAFT](https://github.com/clovaai/CRAFT-pytorch) text detector feeding a CRNN (ResNet/VGG feature extractor → BiLSTM → CTC) recognizer. Apache-2.0, runs on-device via PyTorch. It is **not** a generative/VLM extractor — there are no token logprobs. Latest release is **1.7.2 (Sept 2024)**; no newer release as of mid-2026, so treat behavior below as stable.

#### 1. Rawest output
The lowest-level public result is **per-detected-region** (word/line/phrase after box-merging) — *not* per-glyph. The driving call:

```python
import easyocr
reader = easyocr.Reader(['en'])               # loads CRAFT + CRNN
result = reader.readtext(img, detail=1, output_format='standard')
# result -> list of (bbox, text, confidence):
# [([[189,75],[469,75],[469,165],[189,165]], '愚园路', 0.3754989504814148), ...]
```

`bbox` = 4-point polygon `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` in **TL, TR, BR, BL** order (confirmed in `utils.py`, where the box is built as `[[min_gx,min_gy],[max_gx,min_gy],[max_gx,max_gy],[min_gx,max_gy]]`). Each item is indexed `[0]=boxes, [1]=text, [2]=confidence`. Alternate shapes:

```python
reader.readtext(img, output_format='dict')   # [{'boxes':[...], 'text':..., 'confident':<float>}, ...]
reader.readtext(img, output_format='json')   # same keys, JSON string per line (boxes cast to int)
reader.readtext(img, output_format='free_merge')  # merge_to_free(result, free_list)
reader.readtext(img, detail=0)               # ['愚园路', ...]  ([item[1] for item in result])
reader.readtext(img, paragraph=True)         # (bbox, text)  -> confidence DROPPED
```

Note the key spelling is **`'confident'`** (not `'confidence'`); under `paragraph=True` the `'confident'` key is omitted. No per-character box/score is returned; no native hOCR/ALTO. ([easyocr.py](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py), [docs](https://www.jaided.ai/easyocr/documentation/))

#### 2. Confidence / likelihood
**One float per region**, in `result[i][2]` (or the `'confident'` key). Scale 0.0–1.0. It is a **recognition-only** score — explicitly *not* detector confidence and unrelated to the bbox (confirmed by the maintainer in [discussion #1097](https://github.com/JaidedAI/EasyOCR/discussions/1097), who describes it as an empirical geometric mean of the character-sequence probability). Computed in [`recognition.py`](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/recognition.py):

```python
preds_prob = F.softmax(preds, dim=2)              # per-step distribution
# (ignore_idx classes are zeroed BEFORE softmax-normalization and renormalized)
values  = preds_prob.max(axis=2)                  # per-step max prob
indices = preds_prob.argmax(axis=2)               # per-step argmax
max_probs = v[i != 0]                             # drop CTC blank (index 0) -> pred_max_prob
def custom_mean(x):
    return x.prod() ** (2.0 / np.sqrt(len(x)))
confidence_score = custom_mean(pred_max_prob)
```

Two distinct filters apply (the draft conflated them): an **`ignore_idx` zero-and-renormalize step before softmax**, and a separate **drop of the index-0 CTC blank** when collecting per-character max-probs. Because of the `2.0/sqrt(len)` exponent this is a **length-weighted geometric mean, not a calibrated posterior** — treat it as a monotone-ish score that needs external calibration. **No per-char confidence** (the `pred_max_prob` vector is internal), **no token logprobs**, and **no detector confidence** (CRAFT region/affinity heatmap scores governed by `text_threshold`, `low_text`, `link_threshold`, `bbox_min_score`) are surfaced through `readtext`. For QE you get exactly **one scalar per region**; any per-char or detector-side signal requires patching the library / calling `recognition.py`/`detection.py` internals.

#### 3. Language model / smoothing / lexicon
No built-in word-level LM or dictionary in the default path — recognition is CRNN + CTC. Controls are character- and decoder-level:
- `allowlist` (str) — restrict recognition to a **character** subset; `blocklist` (str) — exclude characters (ignored if `allowlist` is set). Both are character-level, not word/lexicon-level.
- `decoder` ∈ `'greedy'` (default) | `'beamsearch'` | `'wordbeamsearch'`, with `beamWidth=5`. `'wordbeamsearch'` is the only "lexicon-aware" mode — **but** its language model is derived from the recognizer's own character set / training corpus (`decode_wordbeamsearch` → `ctcBeamSearch`); you **cannot** plug in your own domain dictionary by default.
- Language priors come purely from which models you load via `Reader(['en','fr',...])`, plus `recog_network`/`user_network_directory` for a custom-trained recognizer.

There are **no domain-lexicon hooks, allow/deny WORD lists, or post-OCR spelling-correction** in the core API. Net: minimal language-side correction — character whitelisting and an optional internal word-beam decoder, nothing that rewrites toward a user-supplied dictionary by default.

#### 4. Structured layout / geometry
Every region carries a 4-point quadrilateral (axis-aligned-ish after merge; `rotation_info` enables rotated text). Reading order / grouping is **heuristic** line and paragraph merging via `slope_ths, ycenter_ths, height_ths, width_ths, add_margin` (line merge) and `x_ths, y_ths` with `paragraph=True` (paragraph grouping) — a rough top-to-bottom, left-to-right order, **no semantic layout model**. Serialization: `output_format='dict'/'json'` give `{boxes, text, confident}`; `'free_merge'` merges adjacent boxes. **No table structure, no key-value/form extraction, no JSON-schema-constrained output, no hOCR/ALTO, no region classification** (heading/paragraph/figure). The 4-point polygons are directly usable for provenance overlays; anything richer must be built externally.

#### 5. Deployment, license, scope
- **Deployment:** fully **local**, on-device Python (PyTorch). Models auto-download once (`download_enabled=True`, `model_storage_directory`) then run offline — good privacy posture, no data leaves the host. GPU optional (`gpu=True` default; CPU fallback). Free; cost is just compute.
- **Languages:** 80+ languages / all major scripts (Latin, Chinese sim/trad, Arabic, Devanagari, Cyrillic, Thai, Korean, Japanese, …) via `Reader(lang_list)`.
- **Handwriting:** **NOT supported** — listed under "What's coming next" in the README (planned, not shipped).
- **Math/formula:** **NOT supported** — no mention of LaTeX/equation/formula recognition anywhere in the repo or docs.
- **License:** codebase is **Apache-2.0** (commercial use OK). Caveat: the CRAFT detector weights derive from **Clova AI**'s CRAFT (the README uses their pretrained model) — verify weight redistribution separately if you ship the models. The repo's `LICENSE` is the standard Apache template with the copyright field left unfilled ([issue #1422](https://github.com/JaidedAI/EasyOCR/issues/1422)). Custom recognizers via `recog_network`/`user_network_directory`.

**QE takeaway:** EasyOCR exposes exactly one uncalibrated geometric-mean **recognition** confidence per region, plus 4-point geometry for provenance — and nothing else (no per-char, no detector, no logprobs) without patching internals. A selective-prediction layer must calibrate this single scalar (and likely supplement it with external signals) before using it to accept / flag / block.

**Sources:** [easyocr.py (readtext, output_format, dict keys) — master](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py); [recognition.py (custom_mean, pred_max_prob, softmax) — master](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/recognition.py); [utils.py (box assembly TL,TR,BR,BL; get_paragraph; merge) — master](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/utils.py); [Discussion #1097 — confidence is recognition-only, unrelated to bbox](https://github.com/JaidedAI/EasyOCR/discussions/1097); [README — license, CRAFT/Clova attribution, handwriting roadmap, languages](https://github.com/JaidedAI/EasyOCR/blob/master/README.md); [Issue #1422 — Apache-2.0 LICENSE copyright placeholder unfilled](https://github.com/JaidedAI/EasyOCR/issues/1422); [Jaided AI API documentation (decoder/beamWidth, allowlist/blocklist)](https://www.jaided.ai/easyocr/documentation/); [PyPI easyocr (latest 1.7.2, 2024-09-24)](https://pypi.org/project/easyocr/)

---

### RapidOCR

[RapidOCR](https://github.com/RapidAI/RapidOCR) is a multi-backend (ONNXRuntime / OpenVINO / PaddlePaddle / PyTorch, plus MNN / TensorRT) port of Baidu's PaddleOCR (PP-OCR) models. It runs the classic detect -> classify-orientation -> recognize (CTC) pipeline fully locally. Latest PyPI [`rapidocr` 3.8.4](https://pypi.org/project/rapidocr/) (2026-06-15, Python 3.8-3.13). Models are **not** shipped in the wheel; they lazy-download on first use and cache locally.

> **Default model — corrected.** As of the live `main` config ([`config.yaml`](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml) / [`default_models.yaml`](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml)), the **default is PP-OCRv4 mobile, Chinese** (`ch_PP-OCRv4_det_mobile`, `ch_ppocr_mobile_v2.0_cls_mobile`, `ch_PP-OCRv4_rec_mobile`). PP-OCRv5 models exist and are selectable (v3.8.0 added a PP-OCRv5 cls module), but PP-OCRv5 is **not** the out-of-the-box default. Set the model explicitly if you want v5.

#### 1. Rawest output
A single `RapidOCROutput` dataclass ([`python/rapidocr/utils/output.py`](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/utils/output.py)). Verbatim fields (in source order):

```
img: Optional[np.ndarray]                 # original image
boxes: Optional[np.ndarray]               # (N, 4, 2): 4 corner pts per line
txts: Optional[Tuple[str]]                # length N
scores: Optional[Tuple[float]]            # length N, per-line recog confidence
word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]]  # opt-in; default ((\"\",1.0,None),)
elapse_list: List[Union[float, None]]     # [det, cls, rec] times (s)
elapse: float                             # summed in __post_init__
viser: Optional[VisRes]                   # visualization handler
```

> **Correction:** the draft's `lang_rec` field does **not** exist in the live source (DeepWiki's field table is stale on this point). The visualization handle is `viser` (typed `Optional[VisRes]`). Methods: `to_json()`, `to_markdown()`, `vis(save_path)`, `__len__()`.

```python
from rapidocr import RapidOCR
engine = RapidOCR()
result = engine("img.jpg")

result.boxes        # np.ndarray (N, 4, 2): 4 corner pts per line
result.txts         # Tuple[str]   length N
result.scores       # Tuple[float] length N (per-line recog confidence)
result.elapse_list  # [det_t, cls_t, rec_t] seconds
result.to_json()    # [{box, text, score}, ...]
result.to_markdown()
result.vis("overlay.jpg")
```

Per-word / per-char geometry + score (opt-in):
```python
result = engine("img.jpg", return_word_box=True)
# Runtime semantics of result.word_results (per DeepWiki):
#   per-line  ->  per-word/char  ->  (text, confidence_0to1, [[x0,y0]..[x3,y3]])
# Note: the source's *literal* annotation is shallower —
#   Tuple[Tuple[str, float, Optional[List[List[int]]]]] — and the box element is Optional.
# return_single_char_box=True forces char-level boxes even for English words.
```

#### 2. Confidence / likelihood
- **Per-line**: `result.scores[i]` — a normalized recognition confidence in `[0,1]` (always available; primary QE signal). The `[0,1]` scale is consistent with the `text_score`/`box_thresh` thresholds; primary docs do not literally label it "softmax," so treat it as a normalized posterior rather than a doc-asserted softmax value.
- **Per-word / per-char**: 2nd element of each `word_results` inner tuple (`return_word_box=True`), rounded to 5 decimal places.
- **No token logprobs.** RapidOCR is a CTC-decoded CRNN/SVTR recognizer, not autoregressive, so there is nothing logprob-like to expose. Detection-box and orientation-classifier confidences exist internally (used by `box_thresh` / cls) but are **not** surfaced as output fields.
- WARNING: `text_score` (a threshold) silently drops low-confidence lines before they reach the output — a QE layer should read raw `scores` upstream of that filter.

#### 3. Language model / lexicon
**None.** No built-in LM, spell-corrector, n-gram smoothing, or allow/deny list — pure CTC decode over a fixed character dictionary (`keys.txt`). Only language-sensitive knobs: `rec_keys_path` (the keys.txt char set) and the recognition-model variant (model choice selects script/language coverage). Post-processing params (`text_score`, `box_thresh`, `unclip_ratio`, `use_det`/`use_cls`/`use_rec`) are thresholds/stage toggles, **not** linguistic correction. Any lexical or domain correction must be an **external** layer.

#### 4. Structured output
- 4-point polygon per line (and per word/char on opt-in); serializers `to_json()`, `to_markdown()`, and `vis()` overlay for provenance.
- Reading order = raw detector box order (roughly top-to-bottom / left-to-right); no explicit column/paragraph reading-order model in core.
- **No** native hOCR or ALTO XML export. **No** key-value / form-field extraction in core. **No** JSON-schema-constrained output (not LLM-based).
- Table structure, layout regions, document de-warping, and end-to-end document parsing are **separate** RapidAI packages — [`rapid_table`/RapidTable](https://github.com/RapidAI/RapidTable) (decouples OCR; requires a separate OCR engine), [`rapid_layout`/RapidLayout](https://github.com/RapidAI/RapidLayout), [`rapid_undistorted`/RapidUnDistort](https://github.com/RapidAI/RapidUnDistort), and the `rapid_doc` pipeline — **not** the default `RapidOCR()` output. **Formula recognition is not supported** by core RapidOCR (it lives in PaddleOCR's PP-StructureV3, not ported into the core pipeline).

#### 5. Handwriting / math
- **Math/formula:** not in core (see above).
- **Handwriting:** not a target; printed-text models. PP-OCRv5 improved some handwriting but it is not robust, and v5 is not the default anyway.

#### 6. Deployment / license
- **Deployment:** fully local / on-device, no inference-time network calls (after the one-time model download), no per-call API fee — cost is your own CPU/GPU. Default runtime is ONNXRuntime on CPU; pluggable backends: onnxruntime, openvino, paddlepaddle, pytorch (+ MNN/TensorRT per repo description). Pin a known-good version (the wheel-packaging path churned across 3.8.x point releases).
- **License:** Apache-2.0 for engineering code (repo owner). OCR model weights (c) Baidu, distributed under PaddleOCR's Apache-family license. No AGPL, no non-commercial clause — **commercial-safe**. (This covers the PP-OCR weights only; PaddleOCR's separate VL model carries its own license and is not used by core RapidOCR.)

**QE takeaway:** moderate native signal — two confidence levels (per-line always, per-word/char opt-in) plus geometry for overlays, but **no logprobs and no LM/lexicon prior**. Build selective-prediction on raw `scores` read before the `text_score` filter.

**Sources:** [RapidOCROutput source (live, main) — definitive field list](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/python/rapidocr/utils/output.py); [Default models config (PP-OCRv4 default)](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml); [Engine config.yaml (ocr_version: PP-OCRv4, mobile, ch)](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml); [DeepWiki — Python API (output object, params)](https://deepwiki.com/RapidAI/RapidOCR/5.1-python-api); [DeepWiki — Word-Level Information (word_results nesting, 5dp rounding)](https://deepwiki.com/RapidAI/RapidOCR/6.3-word-level-information); [PyPI rapidocr 3.8.4 (version, date, Python, backends)](https://pypi.org/project/rapidocr/); [RapidOCR README (license: Apache-2.0, weights (c) Baidu)](https://github.com/RapidAI/RapidOCR); [RapidTable (separate table package, OCR-decoupled)](https://github.com/RapidAI/RapidTable); [RapidLayout (separate layout package)](https://github.com/RapidAI/RapidLayout); [RapidUnDistort (separate de-warp package)](https://github.com/RapidAI/RapidUnDistort)

---

### PaddleOCR

Apache-2.0, fully local neural OCR + document-parsing toolkit (PaddlePaddle). As of mid-2026 the current PyPI release is **paddleocr 3.7.0** (2026-06-11), which adds **PP-OCRv6** (a single unified recognition model covering 50 languages) alongside PP-OCRv5, PP-StructureV3, PP-ChatOCRv4, and the **PaddleOCR-VL** vision-language pipeline (ERNIE-4.5-0.3B backbone, 0.9B total; a newer **PaddleOCR-VL-1.6** weight set was published 2026-05-28, still Apache-2.0). Primary sources: [GitHub repo](https://github.com/PaddlePaddle/PaddleOCR), [OCR usage docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html), [PaddleOCR-VL docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html), [PaddleOCR-VL on HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL), [table recognition docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/table_recognition_v2.html).

#### 1. Rawest output (line-level; no per-character boxes in the public API)
`PaddleOCR().predict(img)` returns a list of result objects (one per image/page). Each wraps a dict whose evaluation-relevant fields are **line-level** — PaddleOCR does NOT emit per-character/per-glyph boxes or confidences via the public API:

```python
{
  "dt_polys":  [array([[x0,y0],...,[x3,y3]]), ...],  # raw detection quads (4 vertices)
  "dt_scores": [0.98, ...],                          # per-box detection confidence
  "rec_texts": ["Hello", "World", ...],              # per-line recognized strings
  "rec_scores":[0.9985, 0.9421, ...],                # per-line recognition confidence (0-1)
  "rec_polys": [array([[...]]), ...],                # quads kept after score filter
  "rec_boxes": ndarray(shape=(n,4), dtype=int16),    # axis-aligned [x_min,y_min,x_max,y_max]
}
```
Granularity is per-text-line: `rec_texts[i]` pairs with `rec_scores[i]`, `rec_polys[i]`/`rec_boxes[i]`. Exporters: `save_to_json()`, `save_to_markdown()`, `save_to_html()`, `save_to_xlsx()`, `save_to_word()`. **No native hOCR or ALTO XML.**

PP-StructureV3 (`PPStructureV3().predict(...)`) and PaddleOCR-VL (`PaddleOCRVL().predict(...)`) return richer document objects: `layout_det_res` (boxes with `coordinate`/`label`/`cls_id`/`score`), `overall_ocr_res` (same rec_texts/rec_scores), LaTeX formulas, and `parsing_res_list` blocks with `block_bbox`, `block_label`, `block_content`, `block_id` (~20 layout categories with reading order).

#### 2. Confidence (the ready QE signal)
- **Recognition:** `res["rec_scores"][i]` — a 0-1 per-text-line posterior, aligned index-for-index with `rec_texts`. (The docs describe it as the recognition confidence for the line; they do not formally state it is the mean per-step softmax, so treat that as a plausible-but-unstated interpretation.)
- **Detection:** `res["dt_scores"][i]` — DB detector box score.
- **Layout-region (StructureV3/VL):** each layout box carries a 0-1 `score`.

Filtering knobs: `text_rec_score_thresh` (drops low-confidence lines from `rec_texts`/`rec_scores`), `text_det_thresh`, `text_det_box_thresh`, legacy `drop_score`.

**Not available:** per-character / per-glyph confidence, per-word sub-line confidence. For selective prediction the only ready signals are line-level `rec_score`, box `dt_score`, and region `score`; finer-grained QE needs an external confidence layer.

#### 3. Logprobs (VLM tier)
The packaged pipeline — classic `PaddleOCR` **and** `PaddleOCRVL().predict()` — does **not** surface token-level logprobs. The VL result object exposes only block/layout content and scores. PaddleOCR-VL is officially supported on **vLLM** (also SGLang/Transformers) and served behind an **OpenAI-compatible `/v1` endpoint**; in principle a client could request `logprobs`/`top_logprobs` there, but the official vLLM recipe does **not** document logprobs for this model, and there is no documented guarantee they behave correctly with image inputs and the structured doc-parsing prompts. **Verdict: not exposed by the packaged pipeline; possibly obtainable via self-serving but unverified for PaddleOCR-VL image inputs — do not assume it works.**

#### 4. Language / lexicon priors
No built-in word-level n-gram LM, spell-correction, or allow/deny lists in the classic API. The language prior is the per-language recognition model plus its `rec_char_dict_path` vocabulary (the CTC decoder's character set). `lang` selects the model/charset; `ocr_version` ("PP-OCRv6"/"v5"/"v4") and explicit `text_recognition_model_name`/`text_detection_model_name` select models. Implicit language modeling appears only via the **ERNIE-4.5** backbone in PaddleOCR-VL and PP-ChatOCRv4 (generative).

#### 5. Structured layout, tables, formulas
- **Geometry:** detection quads `dt_polys`/`rec_polys` (4 vertices) + axis-aligned `rec_boxes` ((n,4) int16). Strong provenance for overlays.
- **Layout & reading order:** PP-StructureV3 / PaddleOCR-VL emit ~20 block categories with `block_bbox`/`coordinate`, `block_label`/`label`, `score`, and reading order.
- **Tables:** the table pipeline (SLANet/SLANet_plus/**SLANeXt_wired**/**SLANeXt_wireless**) predicts an HTML structure sequence composed into **`pred_html`** — but note this field lives inside **`table_res_list[i]['pred_html']`** (table-recognition / StructureV3 path), not as a top-level result field. PaddleOCR-VL surfaces table content via `block_content` / `save_to_html()` rather than a `pred_html` field. The `pred_html` string is TEDS-evaluable.
- **Formulas:** LaTeX via PP-FormulaNet / UniMERNet. **Handwriting** (PP-OCRv5/VL) and **math/formula** are both supported.
- **KIE/SER:** PP-ChatOCRv4 via `visual_predict()` → `visual_info` (`normal_text_dict`, `table_text_list`, `table_html_list`), then `chat()` + ERNIE for key-value extraction. No fixed JSON-schema-constrained output mode in the classic path.

#### 6. Language coverage (mid-2026)
- **PP-OCRv5 single model** unifies **5 text types** in one model: Simplified Chinese, Traditional Chinese, Pinyin, English, Japanese. PP-OCRv5 overall supports **109 languages** via language-grouped models (Latin, Cyrillic, Arabic, Devanagari, …).
- **PP-OCRv6** (new in v3.7.0): a single unified model covers **50 languages** (Chinese, Traditional Chinese, English, Japanese + 46 Latin-script) with no model switching.
- **PaddleOCR-VL:** **109 languages**. (Total documented ecosystem coverage is ~109–111 languages.)

#### 7. Deployment & license
Fully local / on-device (CPU or GPU; edge-friendly mobile models); privacy-preserving for classic OCR / StructureV3 / PaddleOCR-VL. PaddleOCR-VL can be served via vLLM/SGLang/Transformers for throughput, still locally. Exception: PP-ChatOCRv4's KIE step can call a hosted ERNIE API (network + per-call cost + data egress) or run ERNIE locally.

**License: Apache-2.0** for the toolkit and the published PaddleOCR-VL (0.9B and 1.6) and SLANeXt model weights — permissive, commercial-friendly, **not AGPL or non-commercial**. ERNIE hosted-API usage in PP-ChatOCRv4 is governed separately by Baidu's API terms.

**QE bottom line:** ready signals are line-level `rec_score`, box `dt_score`, and region `score` (all 0-1). No per-character confidence and no exposed token logprobs from the packaged pipelines — fine-grained QE / selective prediction must add an external confidence layer.

**Sources:** [PaddleOCR GitHub repo (license, PP-OCRv6, language counts)](https://github.com/PaddlePaddle/PaddleOCR); [OCR pipeline usage docs (result dict fields, rec_scores, thresholds)](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html); [PaddleOCR-VL pipeline docs (parsing_res_list, no logprobs, vLLM/Transformers backends)](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html); [PaddleOCR-VL on HuggingFace (apache-2.0 badge, ERNIE-4.5-0.3B, 109 languages, handwriting/formulas)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL); [PaddleOCR-VL-1.6 on HuggingFace (newer weights, 2026-05-28, apache-2.0)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6); [Table recognition docs (pred_html in table_res_list, SLANeXt)](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/table_recognition_v2.html); [paddleocr on PyPI (v3.7.0, 2026-06-11, Apache-2.0)](https://pypi.org/project/paddleocr/); [vLLM PaddleOCR-VL recipe (OpenAI-compatible serving; logprobs undocumented)](https://github.com/vllm-project/recipes/blob/main/PaddlePaddle/PaddleOCR-VL.md); [DeepWiki model selection & language support (PP-OCRv5/v6 grouping)](https://deepwiki.com/PaddlePaddle/PaddleOCR/2.8-model-selection-and-language-support); [PP-OCRv5 announcement (5 text types in one model)](https://x.com/PaddlePaddle/status/1925212617288868156)

---

### ocrmac / Apple Vision

[ocrmac](https://github.com/straussmaximilian/ocrmac) (v1.0.1, last touched Oct 2025) is an **MIT-licensed** PyObjC wrapper around Apple's [`VNRecognizeTextRequest`](https://developer.apple.com/documentation/vision/vnrecognizetextrequest) (default `framework="vision"`) and the VisionKit **LiveText** analyzer (`framework="livetext"`, macOS Sonoma 14+). Fully **on-device, macOS-only, free** (no network, no API key, no model weights distributed). This is a *classical* recognizer, **not** a generative/VLM extractor — so there are **no token logprobs**, and the QE surface is thin.

> Source-citation note: an earlier profile cited specific line numbers (e.g. `ocrmac.py:169–175`, `260/272`). Current `main` was rewritten (now wraps the body in `objc.autorelease_pool()` with `pil2buf` + `initWithData_options_`), so those line numbers are stale. Claims below are cited by **function name**, verified against live `main`.

#### 1. Rawest output
The Vision path (`text_from_image`) iterates `req.results()` and returns a **flat list of per-line tuples** `(text, confidence, [x, y, w, h])` (or bare strings when `detail=False`):
```python
from ocrmac import ocrmac
ocrmac.OCR("test.png").recognize()
# [("GitHub: Let's build from here", 0.5, [0.16, 0.91, 0.17, 0.01]),
#  ("github.com",                    1.0, [0.174, 0.87, 0.06, 0.01]), ...]
```
The core loop (in `text_from_image`):
```python
for result in req.results():            # VNRecognizedTextObservation (~ a line)
    confidence = result.confidence()
    if confidence >= confidence_threshold:
        bbox = result.boundingBox()     # CGRect, normalized 0..1, bottom-left origin
        x, y = bbox.origin.x, bbox.origin.y
        w, h = bbox.size.width, bbox.size.height
        res.append((result.text(), confidence, [x, y, w, h]))
```
Granularity on the Vision path = **LINE only**. Note: current `main` reads the string/confidence via `result.text()` / `result.confidence()` directly on the observation — it does **not** use the canonical PyObjC pattern `observation.topCandidates_(1)[0].string()`, and **never calls `topCandidates_` anywhere** (verified: zero occurrences in source; no monkey-patch/category in `__init__.py`). Either way, **Apple's N-best candidate list is never accessed**, so candidate-margin / entropy QE signals are unavailable without patching the wrapper. Per-word/per-character boxes (`boundingBoxForRange:`) are likewise not surfaced.

The `framework="livetext"` path (`livetext_from_image`, macOS 14+) gives finer geometry: `unit="token"` (the **default**) emits one tuple per child glyph via `line.children()` + `char.quad().boundingBox()`; `unit="line"` emits one tuple per line. LiveText flips y (`y = 1 - y - h`) to match Vision's bottom-left convention. **But confidence on this path is a hard-coded literal `1.0`** in both the line and token branches — treat it as *no* usable confidence signal.

#### 2. Confidence / likelihood
- **Vision path:** a single `result.confidence()` float per observation (~line), mapping to Apple's `VNRecognizedText.confidence` (Float, **0.0–1.0**, higher = more certain), exposed as the 2nd tuple element and filterable via `confidence_threshold` (default `0.0`).
- **Caveat that matters for selective prediction:** Apple's confidence on `recognition_level="accurate"` is **heavily quantized** — developers report it clustering at roughly `{0.3, 0.5, 1.0}` (see [Apple Developer Forums](https://developer.apple.com/forums/thread/695693)). `recognition_level="fast"` yields more varied confidences but worse recognition. So a continuous accept/flag/block threshold has very few effective operating points on the accurate path; calibrate accordingly or expect step-function behavior.
- **No** per-word / per-character confidence, **no** N-best candidate confidences (would require calling `topCandidates_`), and **no** token logprobs (non-generative).
- **LiveText path:** confidence is constant `1.0` → effectively absent.

*Implication for QE:* coarse, often-quantized line-level confidence is the only intrinsic signal. Word/char-level acceptance, candidate-margin, or entropy-based selective prediction require (a) forking ocrmac to call `topCandidates_(_:)` + `boundingBoxForRange:`, or (b) an external confidence/calibration layer.

#### 3. LM / lexicon priors
Apple's `VNRecognizeTextRequest` exposes `recognitionLanguages`, `usesLanguageCorrection`, and `customWords`. **What ocrmac actually wires up:**
- `recognition_level` → `setRecognitionLevel_(0)` (accurate / neural) or `(1)` (fast / more glyph-by-glyph).
- `language_preference` → `setRecognitionLanguages_(...)`, validated against `supportedRecognitionLanguagesAndReturnError_(None)`.

**Confirmed absent in source** (no occurrences): `setUsesLanguageCorrection_`, `setCustomWords_`, `setMinimumTextHeight_`, `setAutomaticallyDetectsLanguage_`, `setRevision_`. So language correction runs at Apple's **default and is not user-tunable**, and there is **no custom lexicon / allow-list**. To bias domain terms or disable correction (e.g. to recover raw glyphs for QE), you must patch the wrapper to call the underlying setters. The LiveText path exposes only `setLocales_(...)`.

#### 4. Structured layout / provenance
- **Geometry:** normalized `[x, y, w, h]` floats in 0..1, **bottom-left origin** (Vision convention; LiveText flips y to match). Axis-aligned rects only — the four-corner quad is reduced to a rect on both paths. Helpers (`convert_coordinates_*`, `annotate_PIL()`, `annotate_matplotlib()`) support pixel conversion and provenance overlays.
- **Reading order:** Apple's returned order (roughly top-to-bottom line order); no paragraph/column/region segmentation, no `paragraphs` grouping.
- **NOT present:** table structure, key-value / form-field extraction, JSON-schema-constrained output, hOCR, ALTO, PAGE-XML, or a single nested block→line→word→char hierarchy. Output is a flat list either way. Any layout/table/KV reconstruction is downstream work on the bboxes.

#### 5. Deployment & license
- **Deployment:** fully local / on-device via PyObjC; strong privacy posture, free (compute only). README timings on M3 Max: fast ~131 ms, accurate ~207 ms, livetext ~174 ms per image.
- **Constraints:** **macOS-only** — Vision path needs macOS 10.15+, LiveText needs macOS 14+. Behavior is tied to the host OS's Vision build, so results vary across macOS releases (re-verify per OS). No Linux/Windows.
- **License:** wrapper **MIT** (© 2022 Maximilian Strauss). Underlying Apple Vision / VisionKit are **proprietary OS frameworks** (no shipped model weights, no separate API terms, **no AGPL / non-commercial** restriction) — usable on-device at no cost but closed-source and macOS-bound.

#### 6. Math / handwriting
- **Math/formula:** **not supported** — no LaTeX/equation output (plain recognized-text strings only).
- **Handwriting:** not documented or configurable by ocrmac; whatever the host-OS Vision build does (inconsistent, OS-version-dependent). Treat as uncontrolled for QE. Best fit: printed text — receipts, screenshots, signage, documents.

**Sources:** [ocrmac source (ocrmac/ocrmac.py, main)](https://github.com/straussmaximilian/ocrmac/blob/main/ocrmac/ocrmac.py); [ocrmac README (output format, livetext, timings, macOS reqs)](https://github.com/straussmaximilian/ocrmac/blob/main/README.md); [ocrmac LICENSE (MIT, (c) 2022 Maximilian Strauss)](https://github.com/straussmaximilian/ocrmac/blob/main/LICENSE); [ocrmac on PyPI (v1.0.1)](https://pypi.org/project/ocrmac/); [Apple: VNRecognizedTextObservation.topCandidates(_:)](https://developer.apple.com/documentation/vision/vnrecognizedtextobservation/3152637-topcandidates); [Apple: VNRecognizedText (confidence, string)](https://developer.apple.com/documentation/vision/vnrecognizedtext); [Apple Developer Forums: VNRecognizedText confidence quantization (0.3/0.5/1.0)](https://developer.apple.com/forums/thread/695693); [Apple: Recognizing Text in Images](https://developer.apple.com/documentation/vision/recognizing-text-in-images)

---

### pix2tex / LaTeX-OCR

[pix2tex (LaTeX-OCR)](https://github.com/lukas-blecher/LaTeX-OCR) is a **specialized image-to-LaTeX** model: a ViT encoder with a ResNet backbone plus a Transformer decoder ([README](https://github.com/lukas-blecher/LaTeX-OCR)). It converts a single cropped image of a math formula into a LaTeX string. For an evaluation/QE pipeline it is essentially a **bare generator** — rich on content, empty on intrinsic quality signals. Every claim below was verified against the live repo, x-transformers source, the LICENSE file, and PyPI in mid-2026.

#### 1. Rawest output
A single LaTeX `str` for the whole image. No char/word/line/region units, no geometry.

```python
from PIL import Image
from pix2tex.cli import LatexOCR
model = LatexOCR()
latex = model(Image.open('eq.png'))   # -> r'\frac{1}{2}\sigma^2'  (str)
```

Internally ([`models/utils.py`](https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/models/utils.py)) generation returns **token IDs only**, which `cli.py` detokenizes via `post_process(token2str(dec, self.tokenizer)[0])`:

```python
@torch.no_grad()
def generate(self, x, temperature: float = 0.25):
    return self.decoder.generate(            # x_transformers AR decoder -> token ids
        (torch.LongTensor([self.args.bos_token]*len(x))[:, None]).to(x.device),
        self.args.max_seq_len, eos_token=self.args.eos_token,
        context=self.encoder(x), temperature=temperature)
```

The [FastAPI server](https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/api/app.py) endpoints `POST /predict/` and `POST /bytes/` are both typed `-> str` — plain string, no JSON envelope, no confidence field. **Verified.**

#### 2. Confidence / likelihood
**None exposed — no documented field or attribute to read.** No per-char, per-token, per-line, or per-region confidence; **no logprobs**. The decoder is x-transformers' `AutoregressiveWrapper.generate()`: by default (`return_intermediates=False`) it returns only the generated token IDs (`out`); with `return_intermediates=True` it returns `(out, cache)`, where `cache` holds KV intermediates, **not** per-step logits/scores. There is **no `return_logits` / `output_scores` parameter** — the softmax is consumed internally for sampling and discarded ([autoregressive_wrapper.py](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/autoregressive_wrapper.py)). To get any signal you must **add an external confidence layer** (all require a code patch, not just an API arg): (a) re-score the produced sequence in teacher-forcing mode to recover per-token logprobs; (b) patch the decoder loop to capture logits per step; or (c) use sampling agreement at `temperature>0` (run *k* times, measure edit-distance variance) as an uncertainty proxy. None is surfaced as a documented field.

#### 3. Language model / smoothing / lexicon
No NL language model, dictionary, or allow/deny list. The implicit "LM" is the decoder over the fixed LaTeX-token vocabulary (`tokenizer.json`), which biases toward syntactically plausible LaTeX but is not a configurable lexicon. Only configurable knob: **`temperature`** (default `0.25`; via config, CLI `t=0.XX`, or `LatexOCR(arguments=...)`). **No beam search** (listed as unimplemented in the repo TODO), and no `top_k` / `top_p` / `repetition_penalty` exposed through pix2tex's public API. Post-processing is a deterministic `post_process()` that fixes LaTeX whitespace/spacing — **not** lexicon correction.

#### 4. Structured / layout output
**None.** No bounding boxes or polygons, no reading-order metadata, no layout/region segmentation, no table structure, no key-value extraction, no JSON-schema-constrained output, and no hOCR / ALTO export. The only "structure" is the LaTeX markup itself (`\frac{}{}`, sub/superscripts) — that is the content string, not a geometry-aware data structure. Detecting/cropping formulas from a full page (and thus any provenance overlay) is out of scope and must be done by an upstream layout/detector model.

#### 5. Deployment / license / scope
Local / self-hosted PyTorch model (`pip install pix2tex`, extras `[gui]`, `[api]`, `[train]`). Fully on-device — no network calls at inference, no telemetry; weights download once from **GitHub release assets** (`weights.pth` ~97 MB, optional `image_resizer.pth` ~18 MB). CLI `pix2tex`, GUI `latexocr`, optional FastAPI server (`pix2tex.api.app`) and Streamlit demo. Free / compute-only (CPU works, GPU optional). **License: MIT** for both code (LICENSE: "Copyright (c) 2021 Lukas Blecher"; PyPI classifier "License :: OSI Approved :: MIT License") and the released weights — **commercial use permitted, no AGPL, no non-commercial restriction**.

**Scope:** MATH/FORMULA ONLY — image-of-equation → LaTeX. Not a general-text OCR engine and not "multilingual" in the OCR sense (the relevant vocabulary is the LaTeX token set, not human languages). **Handwriting:** printed/rendered formulas are the supported path; the README TODO lists handwritten formulae as *"kinda done"* — i.e. an **experimental/partial** handwritten model exists (training colab) but it is not the default and is unreliable for production. No tables, no document layout. Reported quality (README): BLEU 0.88, normalized edit distance 0.10, token accuracy 0.60. **Version note:** current PyPI release is **0.1.4 (2025-01-18)**; no newer release as of mid-2026 — treat the project as **stable-but-unmaintained**.

**QE takeaway:** pix2tex emits no native quality signal of any kind — no confidence, no logprobs, no geometry, no structured envelope. A QE / selective-prediction layer over it must synthesize its own signal (logprob re-scoring via a decoder patch, or sampling-agreement proxies), and any provenance/bbox overlay must come from an upstream detector.

**Sources:** [pix2tex cli.py (LatexOCR.__call__)](https://raw.githubusercontent.com/lukas-blecher/LaTeX-OCR/main/pix2tex/cli.py); [pix2tex models/utils.py (generate)](https://raw.githubusercontent.com/lukas-blecher/LaTeX-OCR/main/pix2tex/models/utils.py); [pix2tex api/app.py (FastAPI endpoints)](https://raw.githubusercontent.com/lukas-blecher/LaTeX-OCR/main/pix2tex/api/app.py); [x-transformers autoregressive_wrapper.py (generate return behavior)](https://raw.githubusercontent.com/lucidrains/x-transformers/main/x_transformers/autoregressive_wrapper.py); [pix2tex LICENSE (MIT)](https://raw.githubusercontent.com/lukas-blecher/LaTeX-OCR/main/LICENSE); [pix2tex PyPI JSON (version 0.1.4, license classifier)](https://pypi.org/pypi/pix2tex/json); [LaTeX-OCR README (scope, TODO beam search/handwriting, metrics, weights)](https://github.com/lukas-blecher/LaTeX-OCR); [LaTeX-OCR releases (weights.pth, image_resizer.pth assets)](https://github.com/lukas-blecher/LaTeX-OCR/releases)

---

### TrOCR (handwritten)

`microsoft/trocr-large-handwritten` — a Hugging Face `transformers` [`VisionEncoderDecoderModel`](https://huggingface.co/docs/transformers/model_doc/trocr) (BEiT/DeiT image encoder + RoBERTa-initialized text decoder), wrapped by `TrOCRProcessor` (ViT feature extractor + RoBERTa BPE tokenizer). It is a **specialized, single-text-line handwriting recognizer**, not a document/layout engine. ([model card](https://huggingface.co/microsoft/trocr-large-handwritten), [TrOCR paper arXiv:2109.10282](https://arxiv.org/abs/2109.10282))

#### 1. Rawest output
A token-id tensor decoded to a `str`. **No geometry, no per-word/line/char segmentation** — the model card states it does OCR "on single text-line images," so the input is assumed to be a pre-cropped single line.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values  # (B,3,384,384)
generated_ids = model.generate(pixel_values)            # torch.LongTensor (B, seq_len)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]  # str
```

The richest raw object (since `config.is_encoder_decoder=True`) is `GenerateEncoderDecoderOutput` (or `GenerateBeamEncoderDecoderOutput` for beam search) via `return_dict_in_generate=True`:
- `sequences`: `LongTensor (batch*num_return_sequences, seq_len)` — token ids.
- `scores`: tuple (len = #generated tokens) of `FloatTensor (batch*num_beams, vocab_size)` — **processed** next-token scores per step (only with `output_scores=True`).
- `logits`: tuple of `FloatTensor (batch*num_beams, vocab_size)` — **unprocessed** raw LM-head logits per step (only with `output_logits=True`).
- `sequences_scores`: `(batch*num_return_sequences)` — final length-penalized beam score (beam search only).
- plus `encoder_attentions`, `cross_attentions`, `decoder_hidden_states`, `past_key_values`.

There is **no bbox, polygon, line index, hOCR, or ALTO** in any field. "Per-token" means per RoBERTa BPE **sub-word piece**, NOT per character or per word.

#### 2. Confidence (derived, not native)
There is **no native confidence/probability field** on the default output. Confidence is reconstructed from generation scores:

```python
out = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
transition_scores = model.compute_transition_scores(
    out.sequences, out.scores, normalize_logits=True)   # (batch, gen_len), natural-log probs
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]  # =1 here
gen_tokens = out.sequences[:, input_length:]
# per-token prob = np.exp(transition_scores); sequence log-prob = transition_scores.sum(axis=1)
```

Verified against the live HF docs (mid-2026): the signature is `compute_transition_scores(sequences, scores, beam_indices=None, normalize_logits=False)` returning a `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`. **`normalize_logits` defaults to `False`** ("the logits … for legacy reasons, may be unnormalized") — you **must** pass `normalize_logits=True` to get true natural-log log-softmax per token, where `np.exp(score)` is a probability in [0,1]. For beam search you pass `out.beam_indices` and typically `normalize_logits=False` to reconstruct `sequences_scores` (a length-penalized beam score, **not** a calibrated probability).

Granularity is **per sub-word token only**. There is no per-character, per-word, per-line, or per-region confidence natively — word-level QE requires external aggregation (mean/min/product over the sub-word tokens spanning a word after detokenization). There is **no hosted-API surface**; this is direct local tensor access.

#### 3. LM / lexicon
The "language model" is **intrinsic**: the autoregressive decoder is RoBERTa-initialized and learns a sub-word language prior during IAM fine-tuning. There is **no separate/swappable LM, no dictionary, no lexicon/allow-/deny-list, no domain-vocabulary config**. The only language-influencing controls are generic `generate()`/`GenerationConfig` params: `num_beams`, `do_sample`, `temperature`, `top_k`, `top_p`, `length_penalty`, `no_repeat_ngram_size`, `repetition_penalty`, `bad_words_ids`, `force_words_ids`, `prefix_allowed_tokens_fn` (constrained decoding — the closest thing to an allow-list, but you supply it), `max_length`/`max_new_tokens`, `early_stopping`.

The shipped `generation_config.json` is **minimal** (verbatim): it sets only `bos_token_id:0`, `decoder_start_token_id:2`, `eos_token_id:2`, `pad_token_id:1`, `use_cache:false` (plus `_from_model_config:true`, `transformers_version:"4.27.0.dev0"`). It does **NOT** pin `num_beams`/`max_length`/`length_penalty`, so beam search and lengths must be passed explicitly at call time (or fall back to `GenerationConfig` defaults).

#### 4. Structured layout
TrOCR emits **only a text string for a single line crop**: no bounding boxes/polygons, no reading order, no layout/table structure, no key-value extraction, no JSON-schema-constrained output, and no hOCR/ALTO export. For multi-line documents or provenance overlays you must run an **external** line/word detector to produce crops + geometry, then run TrOCR per crop; geometry comes entirely from that upstream detector, never from TrOCR. Cross/decoder attentions are exposed but are uncalibrated and have no documented attention→bbox mapping (single-line input makes this largely moot).

#### 5. Deployment, language, license
**Deployment:** fully local / self-hosted via `transformers` + PyTorch (ONNX-exportable). No vendor API; on-device / air-gappable, so private documents never leave the host. Cost = your own compute (no per-call fees); `trocr-large` is ~558M params, so a GPU is recommended for throughput (CPU works for low volume). Runs offline after weights download.

**Language / domain:** fine-tuned on the **IAM** handwriting dataset (English handwriting), single-text-line natural text. Handwriting: **YES** (this checkpoint's purpose). **Math/formula: NO** — no LaTeX/formula support is declared. Note: the card does **not** explicitly state a language restriction; "English-only" is an inference from IAM, not a card claim — treat as "effectively English (IAM-trained), no declared multilingual capability." Microsoft also ships printed and stage-1 variants; multilingual TrOCR is separate/community work, not this checkpoint.

**License (flag):** The `microsoft/trocr-large-handwritten` HF repo declares **NO `license` field** — neither in the model-card YAML/body nor in `/api/models/microsoft/trocr-large-handwritten` metadata (`cardData` carries only `tags` and `widget`). By contrast the sibling `microsoft/trocr-base-handwritten` **is** tagged `license: mit`, and upstream `github.com/microsoft/unilm` ships the **MIT License** ("Copyright (c) Microsoft Corporation"). So the large checkpoint is **MIT-by-provenance but UN-tagged on the Hub** — treat as "MIT (inherited from microsoft/unilm), not explicitly declared on the repo; confirm before commercial reliance." **No AGPL or non-commercial constraint found.**

> **QE takeaway:** rich per-sub-word logprobs/posteriors make sequence- and token-level confidence (and selective prediction) straightforward, but there is **no geometry** (so no provenance overlays without an external detector), **no native word/line confidence** (must aggregate sub-word scores), and the scope is single-line. Budget for an upstream detector + a confidence-aggregation layer.

**Sources:** [microsoft/trocr-large-handwritten model card](https://huggingface.co/microsoft/trocr-large-handwritten); [trocr-large-handwritten generation_config.json (raw)](https://huggingface.co/microsoft/trocr-large-handwritten/raw/main/generation_config.json); [trocr-large-handwritten /api/models metadata (no license key)](https://huggingface.co/api/models/microsoft/trocr-large-handwritten); [trocr-base-handwritten /api/models metadata (license: mit)](https://huggingface.co/api/models/microsoft/trocr-base-handwritten); [HF transformers text_generation docs — compute_transition_scores (normalize_logits defaults False)](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation); [microsoft/unilm LICENSE (MIT)](https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE); [TrOCR paper (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)

---

### OCR.space API

Hosted classical-OCR REST service (by a9t9), with an official **on-premise/offline** variant (*OCR.space Local* / Enterprise). Strong on cheap, easy text + word bounding boxes; **weak on intrinsic quality signals** — it emits no confidence anywhere, so any QE / selective-prediction layer must be built externally. Not a VLM, so no logprobs. *(Verified against live docs/forum, June 2026.)*

**Endpoint:** `POST https://api.ocr.space/parse/image` (multipart `file`, `base64Image`, or `url`); `apikey` in header. ([API docs](https://ocr.space/ocrapi))

#### 1. Rawest output
Granularity stops at the **word** (only when `isOverlayRequired=true`); there is no char/glyph level. Hierarchy is `ParsedResults[]` (one per page) → `ParsedText` (full string) + `TextOverlay.Lines[].Words[]`.

```json
{
  "ParsedResults": [{
    "TextOverlay": {
      "Lines": [{
        "Words": [
          {"WordText": "Word", "Left": 106, "Top": 91, "Height": 9, "Width": 11}
        ],
        "MaxHeight": 13,
        "MinTop": 90
      }],
      "HasOverlay": true,
      "Message": null
    },
    "FileParseExitCode": 1,
    "ParsedText": "full plain text...",
    "ErrorMessage": null,
    "ErrorDetails": null
  }],
  "OCRExitCode": 1,
  "IsErroredOnProcessing": false,
  "SearchablePDFURL": null,
  "ProcessingTimeInMilliseconds": "..."
}
```
Word boxes are **axis-aligned pixel rectangles** (`Left/Top/Width/Height`), not polygons. The docs type `OCRExitCode`/`FileParseExitCode` as integers; some clients surface them as strings, so parse defensively. ([API docs](https://ocr.space/ocrapi))

#### 2. Confidence / likelihood
**None — confirmed.** No per-char, per-word, per-line, or per-region confidence/score/probability field exists. The Word object is strictly `{WordText, Left, Top, Height, Width}`; `Line` adds only `{MaxHeight, MinTop}`. The only intrinsic signals are coarse status enums:
- `OCRExitCode`: 1 all-parsed · 2 partial · 3 all-failed · 4 fatal
- `FileParseExitCode`: 1 success · 0/-10/-20/-30/-99 failure modes

→ For selective prediction you **must** add an external confidence layer (lexicon agreement on `ParsedText`, bbox-geometry outliers from `Word.Height` vs `Line.MaxHeight`, or cross-engine disagreement by running Engine 1 vs 2 vs 3). ([API docs](https://ocr.space/ocrapi))

#### 3. Language model / smoothing / lexicon
**Not tunable — confirmed.** No user dictionary, allow/deny-list, custom lexicon, hint list, or decoder/beam params. The single language-sensitive knob is `language` (e.g. `eng`, `chs`, `jpn`, `ara`, or auto-detect on Engines 2/3), which just selects the engine's built-in trained pack. Internal smoothing is opaque and non-configurable. Any domain-lexicon QE must be layered on top of `ParsedText` externally. ([API docs](https://ocr.space/ocrapi))

#### 4. Structured output
| Feature | Status |
|---|---|
| Word bounding boxes | ✅ axis-aligned px (`Left/Top/Width/Height`) |
| Char/glyph boxes | ❌ |
| Line grouping | ✅ `Lines` (`MaxHeight`, `MinTop`) |
| Block / paragraph / column model | ❌ |
| Reading order | implicit (array order) only |
| Tables | text-only (`isTable=true` → line-by-line; Engine 3 → Markdown in `ParsedText`); **no cell coordinates** |
| Key-value / field extraction | ❌ |
| JSON-schema structured extraction | ❌ |
| hOCR / ALTO | ❌ |
| Searchable PDF | ✅ `isCreateSearchablePdf` → `SearchablePDFURL` (rendered PDF, watermarked on free tier) |

#### 5. Engines, handwriting, math
- **Engines:** `OCREngine` = **1, 2, 3** only. ⚠️ **Engine 5 is gone** — the official team merged its features into Engine 2 (forum, Oct 2025) and declared it deprecated, to be replaced by Engine 3 (Jan 2026). Do **not** rely on Engine 5 for new work. ([forum](https://forum.ocr.space/t/is-engine-5-still-available/28772), [engine selection](https://forum.ocr.space/t/ocr-engine-selection/29564))
- **Engine 1:** many languages incl. CJK. **Engine 2:** Western Latin + Chinese, auto-detect. **Engine 3:** 200+ languages, auto-detection, **strong handwriting recognition**.
- **Handwriting:** ✅ Engine 3. **Math / formula / LaTeX:** ❌ none documented for any engine.
- All engines share the **same response schema** (the schema above); they differ in accuracy/coverage, not output shape.

#### 6. Deployment, pricing & license
- **Deployment is NOT hosted-only.** Default is the hosted cloud REST API (servers in France/Germany/Finland; docs/images **not stored**). But an official **self-hosted offline** product exists — *OCR.space Local* / Enterprise — that "runs 100% local and offline... never contacts the Internet," with the same API parameters and docs as the cloud PRO PDF plan (Engines 1+2 available offline; Engine 3 offline availability not clearly documented). This makes air-gapped / sensitive-document pipelines feasible. ([OCR.space Local](https://ocr.space/blog/ocr.space-local/))
- **Pricing/quotas (June 2026, verify on live order page):** Free = 25,000 req/mo (Engine 1/2) **+ a separate 2,500/mo Engine-3 quota**, 500 req/day/IP, 1 MB file, 3 PDF pages. PRO ≈ $30/mo (300k req + 30k Engine-3, 5 MB). PRO PDF (100 MB+ files, 999+ pages). Enterprise ≈ $999+/mo (incl. on-premise install). The PRO dollar figure is approximate — re-confirm on the live pricing page as tiers shift. ([API docs](https://ocr.space/ocrapi))
- **License:** Proprietary commercial; engine/service closed-source. **No AGPL / non-commercial trap.** Free-tier searchable PDFs are watermarked. Third-party client wrappers on GitHub are MIT but unofficial. ([API docs](https://ocr.space/ocrapi))

**Sources:** [OCR.space official API reference (response schema, parameters, engines, pricing, handwriting, on-prem mention)](https://ocr.space/ocrapi); [OCR.space FAQ (data not stored; EU servers; PRO SLA)](https://ocr.space/faq); [Official forum — 'Is Engine 5 still available?' (Engine 5 merged into Engine 2, Oct 2025)](https://forum.ocr.space/t/is-engine-5-still-available/28772); [Official forum — 'OCR Engine Selection' (Engine 5 deprecated Jan 2026; Engine 3 beta→PRO Feb 2026)](https://forum.ocr.space/t/ocr-engine-selection/29564); [OCR.space Local — on-premise/offline self-hosted OCR server](https://ocr.space/blog/ocr.space-local/); [OCR.space updated on-premise (offline) OCR engine blog](https://ocr.space/blog/offline-ocr-e2/); [Third-party code example (Zaargh) referencing response shape](https://github.com/Zaargh/ocr.space_code_example)

---

### Google Cloud Vision

Google Cloud Vision exposes two OCR features — [`TEXT_DETECTION`](https://docs.cloud.google.com/vision/docs/ocr) (short text in natural images) and `DOCUMENT_TEXT_DETECTION` (dense text/documents, and the handwriting path). For evaluation/QE it is a *discriminative recognizer with rich per-level confidence* — there are **no token logprobs**, because nothing is autoregressively generated. All claims below were re-verified against the live v1 API reference and pricing page (mid-2026).

#### 1. Rawest structured output
The lowest-level result is `fullTextAnnotation` (a `TextAnnotation`), a full hierarchy down to the **glyph** (`Symbol`). It is produced by `DOCUMENT_TEXT_DETECTION` and is also populated by `TEXT_DETECTION`:

`TextAnnotation.pages[] → Page.blocks[] → Block.paragraphs[] → Paragraph.words[] → Word.symbols[] → Symbol`, where `Symbol.text` is a single character.

```jsonc
// response.full_text_annotation
{
  "text": "full UTF-8 text",
  "pages": [{
    "confidence": 0.97,            // number, Range [0,1]
    "width": 1024, "height": 768,
    "blocks": [{
      "blockType": "TEXT",        // {UNKNOWN, TEXT, TABLE, PICTURE, RULER, BARCODE}
      "confidence": 0.96,
      "boundingBox": { "vertices": [{ "x": 12, "y": 34 }, /* …4 pts */] },
      "paragraphs": [{
        "confidence": 0.95,
        "words": [{
          "confidence": 0.94,
          "symbols": [{
            "text": "H",
            "confidence": 0.93,
            "property": { "detectedBreak": { "type": "SPACE", "isPrefix": false } }
          }]
        }]
      }]
    }]
  }]
}
```

A second, flatter carrier is `textAnnotations` — an array of `EntityAnnotation`. Element `[0]` is the whole-image text blob (`description`, `locale`, `boundingPoly`); elements `[1..]` are per-WORD entries (`description`, `boundingPoly.vertices`, `locale`). **Caveat (corrected):** `EntityAnnotation` does define `score` and a `confidence` field, but `confidence` is documented as **deprecated** ("use score instead"), and for OCR word entries these label-scoring fields are not populated as a recognition posterior. So for usable per-word recognition confidence you must read `fullTextAnnotation`, not `textAnnotations`.

REST call: `POST https://vision.googleapis.com/v1/images:annotate` with `{"requests":[{"image":{...},"features":[{"type":"DOCUMENT_TEXT_DETECTION"}],"imageContext":{"languageHints":[...]}}]}`. Python: `client.document_text_detection(image=image)` → `response.full_text_annotation`. PDF/TIFF go through `files:asyncBatchAnnotate` (async, GCS output).

#### 2. Confidence (the primary QE signal)
A `confidence` field of type **number, Range [0,1]** is present at **every** hierarchy node — `Page`, `Block`, `Paragraph`, `Word`, **and `Symbol`** (per-glyph confidence is genuinely exposed, contrary to a common assumption). Language-detection confidence is additionally available via `TextProperty.detectedLanguages[].confidence` ([0,1]) at each level.

Read paths (Python): `…pages[i].blocks[j].paragraphs[k].words[w].confidence` and `…words[w].symbols[s].confidence`. Google publishes no calibration semantics, so treat these as **uncalibrated heuristic posteriors** — fine as a QE feature, but calibrate against reference data before setting accept/flag/block thresholds for selective prediction.

#### 3. Logprobs
None. `AnnotateImageResponse` contains no logprob/logits field anywhere. Vision OCR is discriminative, not an autoregressive LM, so the only intrinsic signals are the `confidence` posteriors above (plus `detectedLanguages[].confidence`).

#### 4. LM / lexicon controls
The engine has a proprietary, non-inspectable internal language/script model that drives language-sensitive recognition and break inference, but it is **not a user-editable lexicon**. The only language/version controls are:
- `imageContext.languageHints[]` — BCP-47 codes (e.g. `"en"`, `"zh"`, or `"en-t-i0-handwrit"` to force handwriting mode). Docs recommend leaving this empty for best results.
- `Feature.model` — model-version selector: `"builtin/stable"` (default), `"builtin/latest"`, and (TEXT/DOCUMENT_TEXT only) `"builtin/weekly"`. This is a version knob, **not** a decoder/lexicon knob.

There are **no** user dictionaries, allow/deny lists, custom vocab, regex constraints, or beam/temperature parameters. `detectedBreak` (BreakType: `UNKNOWN`, `SPACE`, `SURE_SPACE`, `EOL_SURE_SPACE`, `HYPHEN`, `LINE_BREAK`, plus `isPrefix`) is the engine's own whitespace/line inference — usable for reconstructing reading order, not tunable. For custom-lexicon/form extraction Google steers users to **Document AI** (separate product/SKU).

#### 5. Structured layout / geometry
- **Geometry:** `boundingBox` is a `BoundingPoly` with `vertices` (`Vertex {x,y}` integer pixels) for image inputs, and `normalizedVertices` (floats 0–1) for PDF/TIFF file inputs. Boxes are 4-vertex **axis-aligned quads** — no rotation angle, no free-form polygons. **Zero-valued coordinates are omitted from the JSON** (parsing gotcha). `EntityAnnotation` uses `boundingPoly.vertices`; `Page` carries `width`/`height`.
- **Reading order:** implied by array order (pages > blocks > paragraphs > words > symbols) plus `Symbol.property.detectedBreak`; no explicit index.
- **Tables:** only a coarse `Block.blockType = TABLE` tag — **no** row/column/cell structure.
- **Key-value / form parsing:** none (Document AI's domain).
- **hOCR / ALTO / PAGE-XML:** not emitted; output is Google's own JSON only. Conversion is your responsibility.

#### 6. Deployment, license, cost
- **Deployment:** hosted cloud API only (`vision.googleapis.com`, REST + gRPC, official clients). No on-prem / air-gapped option for the public Vision OCR endpoint; images leave the device. Async PDF/TIFF results land in your GCS bucket.
- **License:** proprietary, under Google Cloud Platform Terms of Service; closed model weights. **Not AGPL, not non-commercial** — commercial use permitted under standard GCP per-unit billing.
- **Cost (corrected):** per-unit (per image, or per PDF/TIFF page). First **1,000 units/feature/month free**; **1,001–5,000,000: $1.50 per 1,000**; **5,000,001+: $0.60 per 1,000** (the draft's $1.00 high-tier figure was wrong). `TEXT_DETECTION` and `DOCUMENT_TEXT_DETECTION` are separate features, each with its own free 1,000.
- **Handwriting (corrected):** enabled via `DOCUMENT_TEXT_DETECTION` (optionally `languageHints:["xx-t-i0-handwrit"]`). The live table lists **3 Supported** scripts — Japanese (`Jpan`), Korean (`Kore`), Latin (`Latn`) — and **6 Experimental** — Bengali (`Beng`), Cyrillic (`Cyrl`), Devanagari (`Deva`), Greek (`Grek`), Chinese (`Hani`), and **Vietnamese (`vi`)**. Note Vietnamese is **Experimental, not Supported** (the draft mis-listed it among the supported four).
- **Math/formula:** not supported — no LaTeX/equation recognition. Use a specialized engine for math.

**Sources:** [v1 AnnotateImageResponse / TextAnnotation reference (Page/Block/Paragraph/Word/Symbol confidence, BlockType, BreakType, EntityAnnotation)](https://docs.cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageResponse); [v1 Feature reference (model: builtin/stable|latest|weekly)](https://docs.cloud.google.com/vision/docs/reference/rest/v1/Feature); [Cloud Vision pricing (Text Detection / Document Text Detection tiers)](https://cloud.google.com/vision/pricing); [OCR language support (handwriting Supported vs Experimental scripts)](https://docs.cloud.google.com/vision/docs/languages); [Detect handwriting (DOCUMENT_TEXT_DETECTION + handwriting languageHint format)](https://docs.cloud.google.com/vision/docs/handwriting)

---

### AWS Textract

Hosted, proprietary document-OCR + document-analysis API. It is a **discriminative extractor, not a generative/VLM model**: it returns a flat list of typed `Block` objects, each with one scalar `Confidence` and geometry. QE signal is limited to word/line/field-level confidence — **no token logprobs, no per-character confidence, no n-best/alternatives**, and **no user-configurable LM/lexicon**.

#### 1. Rawest output
Finest granularity is the **WORD** block — defined as "one or more ISO basic Latin script characters that aren't separated by spaces"; there is no per-glyph/character output. Blocks are a flat array linked by `Id`/`Relationships[].Ids` (PAGE → LINE → WORD via CHILD). [`DetectDocumentText`](https://docs.aws.amazon.com/textract/latest/dg/API_DetectDocumentText.html) (sync) / `StartDocumentTextDetection`+`GetDocumentTextDetection` (async) yield only PAGE/LINE/WORD; [`AnalyzeDocument`](https://docs.aws.amazon.com/textract/latest/dg/API_AnalyzeDocument.html) adds TABLE/CELL/MERGED_CELL/KEY_VALUE_SET/SELECTION_ELEMENT/SIGNATURE/QUERY/QUERY_RESULT/LAYOUT_*.

```python
resp = textract.detect_document_text(Document={"Bytes": img_bytes})
blocks = resp["Blocks"]   # flat list of Block dicts
```
```json
{
  "BlockType": "WORD",
  "Text": "Hello,",
  "TextType": "PRINTED",
  "Confidence": 99.74746704101562,
  "Geometry": {
    "BoundingBox": {"Width": 0.0, "Height": 0.0, "Left": 0.0, "Top": 0.0},
    "Polygon": [{"X": 0.0, "Y": 0.0}, ...],
    "RotationAngle": 0
  },
  "Id": "7f97e2ca-063e-47a8-981c-8beee31afc01"
}
```
Coordinates are **normalized ratios (0–1)** of page width/height. Geometry carries `BoundingBox` (axis-aligned), a fine-grained `Polygon`, and a `RotationAngle` (0/90/180/270). See the [`Block`](https://docs.aws.amazon.com/textract/latest/dg/API_Block.html) and [`Geometry`](https://docs.aws.amazon.com/textract/latest/dg/API_Geometry.html) references.

#### 2. Confidence / likelihood
Every block exposes one `Confidence` field — **Type: Float, Valid Range 0 to 100** ([`Block.Confidence`](https://docs.aws.amazon.com/textract/latest/dg/API_Block.html)). The docs define it verbatim as "the confidence score that Amazon Textract has in the accuracy of the recognized text **and** the accuracy of the geometry points around the recognized text" — i.e. a **fused text+geometry scalar**, not a pure character/text-recognition posterior. Available per-WORD, -LINE, -CELL, -KEY_VALUE_SET (separate KEY and VALUE), -SELECTION_ELEMENT, -QUERY_RESULT, -SIGNATURE, -PAGE.

```python
conf = block["Confidence"]   # e.g. 99.51  (0–100 scale, NOT 0–1)
```
**No per-character confidence, no token logprobs, no alternative hypotheses / n-best.** Textract exposes no sampling/decoding surface at all, so QE must aggregate WORD confidences (min/mean) up to line/field level yourself. AWS does not document calibration of this score; treat it as an ordinal, uncalibrated posterior.

#### 3. Language-model / lexicon
**None configurable.** There is no dictionary, allow/deny list, `user-words`/`user-patterns`, or decoder parameter anywhere in the API. The only steering surfaces are QUERIES (`QueriesConfig.Queries[].Text/Alias/Pages`) and **Custom Queries adapters** ([`AdaptersConfig`](https://docs.aws.amazon.com/textract/latest/dg/API_AdaptersConfig.html) → `AdapterId`+`AdapterVersion`, max one adapter per page) — these fine-tune the **Queries** extractor on your annotated docs and do **not** alter base-OCR decoding or provide a general lexicon/LM. Any internal language smoothing is opaque and non-configurable.

#### 4. Structured output
- **Geometry** (every block): `BoundingBox` {Left, Top, Width, Height} + fine-grained `Polygon` [{X, Y}, ...] + `RotationAngle`, normalized 0–1 ([`Geometry`](https://docs.aws.amazon.com/textract/latest/dg/API_Geometry.html)).
- **Tables**: TABLE → CELL (`RowIndex`/`ColumnIndex` 1-based, `RowSpan`/`ColumnSpan`), MERGED_CELL, TABLE_TITLE/TABLE_FOOTER; `EntityTypes` include COLUMN_HEADER, TABLE_TITLE, TABLE_FOOTER, **TABLE_SECTION_TITLE, TABLE_SUMMARY**, STRUCTURED_TABLE, SEMI_STRUCTURED_TABLE.
- **Key-value (FORMS)**: KEY_VALUE_SET blocks with `EntityTypes` KEY|VALUE.
- **Selection**: SELECTION_ELEMENT + `SelectionStatus` (SELECTED|NOT_SELECTED).
- **Queries**: QUERY/QUERY_RESULT pairs (alias-linked).
- **Reading order / layout**: the LAYOUT feature returns reading-order-aware semantic blocks LAYOUT_TITLE/HEADER/FOOTER/SECTION_HEADER/PAGE_NUMBER/LIST/FIGURE/TABLE/KEY_VALUE/TEXT (base detection order is not guaranteed semantic).
- **No native hOCR/ALTO/PAGE-XML export** (the open-source `amazon-textract-response-parser` / `textractor` libs post-process). **No JSON-schema-constrained / extract-to-your-schema output** — that is Bedrock Data Automation territory, not Textract. Output is always the Textract Block JSON.

#### 5. Deployment, license, coverage
- **Deployment**: hosted-only managed AWS service (regional endpoints); **no on-prem/on-device/self-host**. Processed content is encrypted and stored at rest in the Region of use; separately, **unless you set an AWS Organizations AI services opt-out policy**, some portion of inputs "may be stored in another AWS region" for service improvement (opting out also deletes prior shared content). See the [FAQ](https://aws.amazon.com/textract/faqs/) and [AI opt-out policy](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_policies_ai-opt-out.html).
- **License**: proprietary AWS API (AWS Customer Agreement / Service Terms) — not OSS, no model weights exposed. The official `amazon-textract-response-parser` / Textractor libraries are Apache-2.0, but the engine is closed.
- **Languages**: printed text + FORMS/TABLES in **English, German, French, Spanish, Italian, Portuguese** (6). **Handwriting, Queries, AnalyzeExpense (invoices/receipts), and AnalyzeID (identity) are English-only**; handwriting is limited to the Standard English alphabet + ASCII symbols.
- **Math/formula**: **not supported** (no equation/LaTeX extraction anywhere in the docs).

**QE takeaway:** usable signal is the per-block `Confidence` (0–100) at word/line/field granularity plus bbox/polygon geometry for provenance overlays. There is no logprob, no n-best, and no per-character confidence to mine, and the score conflates localization with recognition — so selective-prediction thresholds should be fit empirically per field type and treated as uncalibrated.

**Sources:** [Amazon Textract API reference — Block (Confidence, BlockType, EntityTypes, TextType)](https://docs.aws.amazon.com/textract/latest/dg/API_Block.html); [Amazon Textract API reference — Geometry (BoundingBox, Polygon, RotationAngle)](https://docs.aws.amazon.com/textract/latest/dg/API_Geometry.html); [Amazon Textract API reference — AdaptersConfig](https://docs.aws.amazon.com/textract/latest/dg/API_AdaptersConfig.html); [Customizing your Queries Responses (Custom Queries adapters)](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.html); [Amazon Textract FAQs (languages, data storage, opt-out, no math)](https://aws.amazon.com/textract/faqs/); [Amazon Textract recognizes handwriting and adds five new languages (handwriting = Standard English alphabet + ASCII)](https://aws.amazon.com/blogs/machine-learning/amazon-textract-recognizes-handwriting-and-adds-five-new-languages/); [AWS Organizations — AI services opt-out policies (cross-region service-improvement storage)](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_policies_ai-opt-out.html)

---

### Azure AI Document Intelligence

Hosted document-IE API (formerly *Form Recognizer*; mid-2026 surfaced under [Azure AI Foundry / Content Understanding](https://azure.microsoft.com/en-us/products/ai-foundry/tools/document-intelligence)). Current GA API is **v4.0 = `2024-11-30`**. Model families: `prebuilt-read` (OCR), `prebuilt-layout` (structure + tables + KV + markdown), domain prebuilts (invoice/receipt/ID/W-2/health), and custom (template/neural/generative) models. A **discriminative** OCR/IE engine — **not** an autoregressive LLM — so it emits real per-element confidence posteriors but **no token logprobs**.

> Verification note (mid-2026, against the live v4.0 docs/REST reference): claims below were re-checked against primary sources. Two draft claims were corrected — **handwriting now covers 12 languages, not 9**, and **tables are HTML only in the *markdown* output, not in the `tables[]` JSON** — and one omission was added: **table/row/cell confidence**.

#### 1. Rawest output
Lowest unit is the **word** (no per-glyph/char). Async call:
```
POST {endpoint}/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30
  → 202 + Operation-Location
GET  …/analyzeResults/{resultId}  → analyzeResult JSON
```
```json
{ "analyzeResult": {
  "content": "While healthcare …",            // full reading-order text
  "pages": [ { "pageNumber": 1, "angle": 0, "width": 915, "height": 1190, "unit": "pixel",
    "words": [
      { "content": "While",
        "polygon": [x1,y1,x2,y2,x3,y3,x4,y4],  // 4-pt quad; px (image) / inch (PDF)
        "confidence": 0.997,
        "span": { "offset": 0, "length": 5 } } // offset into top-level content
    ],
    "lines":  [ { "content": "...", "polygon": [...], "spans": [ {"offset":0,"length":5} ] } ], // NO confidence on lines
    "selectionMarks": [ { "state": "selected", "polygon": [...], "confidence": 0.91, "span": {...} } ],
    "spans": [...] } ],
  "paragraphs": [ { "role": "title", "content": "...", "boundingRegions": [...], "spans": [...] } ],
  "tables": [ { "rowCount": 3, "columnCount": 4,
                "cells": [ { "rowIndex":0,"columnIndex":0,"rowSpan":1,"columnSpan":1,
                             "kind":"columnHeader","content":"...","boundingRegions":[...],"spans":[...] } ] } ],
  "keyValuePairs": [ { "key": {...}, "value": {...}, "confidence": 0.88 } ],
  "styles": [ { "isHandwritten": true, "confidence": 0.95, "spans": [...] } ],
  "languages": [ { "locale": "en", "confidence": 0.7, "spans": [...] } ],
  "documents": [ { "fields": { "InvoiceTotal": { "type": "currency", "valueCurrency": {...}, "confidence": 0.945 } } } ]
}}
```
SDK (`azure-ai-documentintelligence`):
```python
poller = DocumentIntelligenceClient(...).begin_analyze_document(
    "prebuilt-read", AnalyzeDocumentRequest(url_source=url))
result = poller.result()              # AnalyzeResult
for page in result.pages:
    for w in page.words:
        w.content, w.confidence, w.polygon   # word.confidence is the OCR posterior
```
Geometry: every element has a `polygon` (4-vertex quad, clockwise from top-left; px for images, inches for PDF) inside page-scoped `boundingRegions`. Reading order: top-level `content`; all elements locate via char-offset `spans` (caveat: no cross-page reading order; selection marks not positioned among words). **Not supported:** hOCR and ALTO XML — no native export.

#### 2. Confidence (the QE-load-bearing part)
All confidences are floats in **[0, 1]** ("an estimated probability between 0 and 1 that the prediction is correct"). Granularities that carry a *real* posterior:
- **Word** — `result.pages[*].words[*].confidence` — the primary OCR posterior.
- **Selection mark** — `selectionMarks[*].confidence` (+ `.state`). (Markdown renderer drops checkboxes below 0.1 confidence.)
- **Style / handwriting & font add-on** — `styles[*].confidence` paired with `isHandwritten:true` — a per-span handwriting-likelihood signal.
- **Language add-on** — `languages[*].confidence` per detected text-line locale (detection only).
- **Document field** — `documents[*].fields[name].confidence` — field/value posterior for prebuilt + custom + queryFields. (Docs caveat: *not all fields* return a confidence.)
- **Key-value pair** — `keyValuePairs[*].confidence`.
- **Table / row / cell** — *added in 2024-11-30 GA (custom models)*: confidence at table, row, and cell level. Merged/missing cells get lower confidence; recommended usage is top-down (table → row → cell). **This is a usable QE signal for tabular extraction** and was missing from the first-pass profile.

**NOT trustworthy / absent:**
- **Lines have NO confidence** — to score a line, aggregate its word confidences.
- **No per-character/glyph confidence.**
- **No token logprobs** of any kind (no logit/temperature/sampling surface).
- **Formula confidence is hard-coded** (docs verbatim: *"The confidence score is hard-coded"*) and **barcode confidence is hard-coded to 1** (docs: *"The confidence is hard-coded for as 1"* — and the sample output even prints 0.95/0.98, so treat it as meaningless either way). Neither is a real posterior.

Net for QE: trustworthy intrinsic signals are **word**, **selection-mark**, **field/KV**, and **table/row/cell** confidence (plus the handwriting flag and language-detect for triage).

#### 3. Language model / lexicon
**No** user-facing language model, lexicon, dictionary, allow/deny list, hotword, or decoder/beam parameter. Any internal language-sensitive correction is opaque and non-configurable. The only language-aware knobs: (1) optional `locale` hint (docs *warn against* setting it unless you are certain — wrong locale can truncate output); (2) `languages` add-on, which only **detects** (locale + confidence per line) and does not correct; (3) post-OCR field **normalization** on prebuilt fields (date→ISO-8601, phone→E.164, currency→{amount,currencySymbol}, countryRegion→ISO-3166) — value normalization, not character-level lexical correction; (4) `queryFields`/custom schemas, which constrain *field extraction* by name but not OCR decoding. For QE: treat the text as a black box whose only lexical-correction trace is the per-word confidence drop.

#### 4. Structured layout & rendered formats
- **Tables (JSON):** `tables[].cells[]` with `rowIndex/columnIndex/rowSpan/columnSpan` and `kind` (columnHeader/rowHeader/stubHead/description) — **structured JSON, not HTML.**
- **Markdown output:** set `outputContentFormat=markdown` (Layout, 2024-11-30 GA). *In the markdown rendering only*, tables are emitted as **HTML** (`<table>/<tr>/<th>/<td>` with `rowspan`/`colspan` and `<caption>`) to preserve merged cells / multi-row headers; formulas as `$…$`/`$$…$$`; selection marks as ☒/☐; barcodes as image syntax; page headers/footers/numbers as HTML comments; `<!-- PageBreak -->` delimits pages. This is the canonical RAG-chunking output.
- **Paragraph roles:** `paragraphs[].role` (title, sectionHeading, pageHeader/Footer, pageNumber, footnote); Layout adds `figures[]` (croppable via `output=figures`) and hierarchical `sections[]`.
- **Searchable PDF:** `output=pdf` on `prebuilt-read` overlays an OCR text layer (free; `GET …/analyzeResults/{id}/pdf`).
- **Add-ons** (`features=…`): `ocrHighResolution`, `formulas` (LaTeX in `formula.value`, kind inline/display — *hard-coded confidence*), `styleFont`, `barcodes` (*hard-coded confidence*), `languages` (detect), `keyValuePairs` (free on Layout), `queryFields` (premium; ≤20 fields/request).

#### 5. Deployment & license
- **Hosted** as an Azure AI service (regional; data-residency commitments; customer data not used to train Microsoft models). 
- **Self-host** via official **Docker containers** (Read, Layout, prebuilt, custom), including **disconnected / air-gapped** containers (require a license file; sold via commitment-tier pricing). The v4.0 Read container (with searchable PDF) and the Layout container are available — the closest to a private/offline deployment.
- **License:** Microsoft **proprietary/commercial** (Product Terms / Azure terms) — no source, no weights, **no AGPL or non-commercial restriction**; usage is metered.

#### 6. Language coverage, math, handwriting
- **Printed OCR:** hundreds of languages/locales (Latin, Cyrillic, Arabic, Devanagari, CJK, etc.).
- **Handwriting (v4.0): 12 languages** — English, Chinese Simplified, French, German, Italian, Japanese, Korean, Portuguese, Spanish, **Russian, Arabic, Thai** *(corrected: the earlier 9-language list was v3.1)*.
- **Math/formula:** `formulas` add-on returns LaTeX in `formula.value` (kind inline/display) — usable for extraction, **but confidence is hard-coded**, so not a QE signal.

**Sources:** [AnalyzeResult REST reference (v4.0, 2024-11-30): word/line/selection-mark/field schema + confidence](https://learn.microsoft.com/en-us/rest/api/aiservices/document-models/get-analyze-result?view=rest-aiservices-v4.0+(2024-11-30)); [Add-on capabilities: formula 'confidence is hard-coded', barcode 'hard-coded for as 1', languages detect-only, queryFields, formula/barcode JSON shapes](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/add-on-capabilities?view=doc-intel-4.0.0); [Accuracy & confidence: 0-1 scale, words/KV/selection-marks/fields confidence, table/row/cell confidence in 2024-11-30 GA](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/accuracy-confidence?view=doc-intel-4.0.0); [Supported Markdown elements: outputContentFormat=markdown, tables rendered as HTML (rowspan/colspan/caption), selection-mark 0.1 filter](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/markdown-elements?view=doc-intel-4.0.0); [Language & locale support (OCR): printed (hundreds) + handwriting 12 languages incl. Russian/Arabic/Thai in v4.0](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/language-support/ocr?view=doc-intel-4.0.0); [Disconnected (air-gapped) containers: Read/Layout, license file, commitment-tier](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/disconnected?view=doc-intel-4.0.0); [azure-ai-documentintelligence (PyPI) — DocumentIntelligenceClient.begin_analyze_document](https://pypi.org/project/azure-ai-documentintelligence/)

---

### Mistral OCR (Document AI — `mistral-ocr-latest` / `mistral-ocr-2505` / `mistral-ocr-2512` "OCR 3")

Hosted document-AI API from Mistral. A proprietary vision LMM that returns per-page **markdown** (GitHub-flavored, with LaTeX math and inline `![id](id)` image refs) rather than glyph-level layout, plus extracted-image bounding boxes, tables, and — new since the 2025 launch — **opt-in per-word / per-page confidence scores derived from token logprobs**. No self-hostable weights. Every field below is verified against the live SDK source and official docs (mid-2026). ([launch](https://mistral.ai/news/mistral-ocr/), [OCR 3](https://mistral.ai/news/mistral-ocr-3/), [OCR docs](https://docs.mistral.ai/studio-api/document-processing/basic_ocr), [SDK models](https://github.com/mistralai/client-python/tree/main/src/mistralai/client/models)).

#### 1. Rawest output
The lowest-level result is **per-page markdown plus figure-level geometry only** — there is *no* per-glyph/word/line text bounding box.

```python
from mistralai import Mistral
client = Mistral(api_key=API_KEY)
resp = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "document_url", "document_url": "https://arxiv.org/pdf/2201.04234"},
    confidence_scores_granularity="word",   # opt-in, see §2
)
```

Verified response shape (from `mistralai.client.models`):

```python
OCRResponse{ pages: list[OCRPageObject], model: str, usage_info: OCRUsageInfo,
             document_annotation: str | None }   # JSON string, if document_annotation_format given
OCRPageObject{ index: int, markdown: str,        # primary text output
               images: list[OCRImageObject],
               dimensions: OCRPageDimensions | None,
               tables: list[OCRTableObject] | None,
               hyperlinks: list[str] | None,
               header: str | None, footer: str | None,
               confidence_scores: OCRPageConfidenceScores | None }
OCRImageObject{ id, top_left_x|y, bottom_right_x|y: int | None,   # absolute pixels (figures/charts only)
                image_base64: str | None, image_annotation: str | None }
OCRPageDimensions{ dpi: int, height: int, width: int }            # page-screenshot pixels
OCRTableObject{ id: str, content: str, format: "markdown"|"html", # Python attr is `format_`, JSON key `format`
                word_confidence_scores: list[OCRConfidenceScore] | None }
```

There is **no bounding box for text spans/words/lines** — text is delivered only as the per-page `markdown` string; the only word-level locator is `OCRConfidenceScore.start_index`, a character offset into that markdown.

#### 2. Confidence (the QE signal)
Opt-in request param `confidence_scores_granularity: "word" | "page"` (default `None` → field omitted, to keep payloads small). When set, each page carries:

```python
OCRPageConfidenceScores{ average_page_confidence_score: float,
                         minimum_page_confidence_score: float,
                         word_confidence_scores: list[OCRConfidenceScore] | None }
OCRConfidenceScore{ text: str, confidence: float (0–1), start_index: int }
```

**Semantics (verbatim from the SDK `OCRPageConfidenceScores` docstring — load-bearing):** for `page` granularity, page average/minimum are computed from per-token `exp(logprob)`; for `word` granularity, each word's `confidence` is `exp(mean(token_logprobs))` — a geometric mean over the word's subword tokens. So the underlying signal *is* token logprobs, surfaced as monotone 0–1 confidences. Tables also expose `OCRTableObject.word_confidence_scores` when `word` granularity is set. Read via `page.confidence_scores.word_confidence_scores[i].confidence`, `.average_page_confidence_score`, `.minimum_page_confidence_score`.

**Caveat:** Azure AI Foundry's hosted Mistral OCR does **not** support this param — it returns HTTP **422** (`extra_forbidden`). Confidence scores are **native-platform (la Plateforme) only**.

#### 3. Logprobs
**No raw token logprobs are exposed.** There is no `logprobs` field, no per-token array, no top-k alternatives. The intrinsic logprob signal is available only in the aggregated, `exp(logprob)`-transformed confidence form above. For QE this is a derived, monotonic surrogate — fine for thresholding/selective prediction, but you cannot recover entropy or distributional spread per token.

#### 4. LM / lexicon
No user-facing language model, lexicon, dictionary, allow/deny list, or decoder controls (no temperature/top-p/beam on the OCR endpoint). Any language-sensitive correction is internal and not configurable. The only language-steering surfaces are in the *annotation* layer (which routes OCR output through a vision LLM): `document_annotation_prompt` (free text) plus the JSON schemas passed via `bbox_annotation_format` / `document_annotation_format`. `table_format`, `extract_header`/`extract_footer`, `image_limit`/`image_min_size`, and `pages` control layout extraction, not linguistic correction.

#### 5. Structured layout & provenance
Bounding boxes: **yes, but only for extracted images/figures/charts** (`OCRImageObject` absolute-pixel corners relative to `OCRPageDimensions`), **not** for text. Reading order is implicit in the markdown sequence (no explicit indices). Tables come as markdown or HTML (HTML preserves merged cells/headers in OCR 3); headers/footers via `extract_header`/`extract_footer`; hyperlinks via `hyperlinks`. Key-value / JSON-schema output is available via **Annotations** — `bbox_annotation_format` annotates each extracted image region (→ `OCRImageObject.image_annotation`, a JSON string) and `document_annotation_format` returns whole-doc JSON in `OCRResponse.document_annotation`; only `json_schema` response formats are valid. **Document annotation is limited to 8 pages and the first 8 image bounding boxes.** **No hOCR / ALTO XML.** Net for QE provenance overlays: figure bboxes + page pixel dimensions are geometric, but text-span provenance is only a character `start_index` into the markdown.

#### 6. Deployment, license, cost
Hosted cloud API only (la Plateforme `client.ocr.process`); also on Azure AI Foundry (subset — no confidence scores). No public open-weights / on-device build, so no air-gapped option except enterprise self-deployment arrangements; EU-based provider, API-terms-governed privacy. **License:** the model is proprietary/closed-weights under Mistral's commercial ToS (pay-per-page); the `mistralai` Python SDK is **Apache-2.0** (Speakeasy-generated) and licenses only the client — **no AGPL or non-commercial restriction**. **Pricing (mid-2026, live page):** $2 / 1000 pages OCR, ~$1 / 1000 via the 50%-off Batch API, $3 / 1000 for Annotations (the March-2025 launch was $1/1000; OCR 3 raised it). **Handwriting:** yes (cursive, mixed/handwritten-over-printed). **Math/formula:** yes (LaTeX in markdown). Models: `mistral-ocr-latest` (alias), `mistral-ocr-2505`, `mistral-ocr-2512` (OCR 3).

**Sources:** [OCR request model source (confidence_scores_granularity, all params)](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrrequest.py); [OCRPageConfidenceScores source (exp(logprob) / geometric-mean docstring)](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrpageconfidencescores.py); [OCRConfidenceScore source (text/confidence 0-1/start_index)](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrconfidencescore.py); [OCRPageObject source](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrpageobject.py); [OCRImageObject source (figure-only bboxes)](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrimageobject.py); [OCRTableObject source (format_ alias, word_confidence_scores)](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrtableobject.py); [OCRResponse source](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/ocrresponse.py); [Official OCR processor docs (confidence_scores_granularity)](https://docs.mistral.ai/studio-api/document-processing/basic_ocr); [Official Document Annotations docs (8-page / first-8-bbox limit, json_schema only)](https://docs.mistral.ai/capabilities/document_ai/annotations); [Azure 422 extra_forbidden for confidence_scores_granularity (Microsoft Q&A)](https://learn.microsoft.com/en-us/answers/questions/5890356/azure-mistral-ocr-api-does-not-support-confidence); [Live pricing page ($2/1000 OCR, $3/1000 annotations)](https://mistral.ai/pricing/); [OCR 3 announcement (handwriting, HTML tables, model 2512)](https://mistral.ai/news/mistral-ocr-3/); [client-python repo (SDK Apache-2.0)](https://github.com/mistralai/client-python)

---

### Mathpix

Mathpix (MathpixOCR / "Convert API") is a hosted, **proprietary** OCR API specialized for **math/STEM, tables, chemistry, and handwriting**, returning [Mathpix Markdown](https://docs.mathpix.com/) plus structured LaTeX/MathML. It is **not** a generative VLM — it emits calibrated confidence scalars instead of token logprobs, which makes it unusually QE-friendly for an OCR engine. (All field names and prices below were re-verified against the live v3/text reference and pricing page in mid-2026.)

#### 1. Rawest structured output
`POST v3/text` with `include_line_data: true` and `include_word_data: true` returns **line-level and word-level** segments with polygon geometry. **The finest grain is `word_data` — there is no per-character (`char_data`) output** in the documented schema.

```json
{
  "request_id": "...",
  "text": "...",               // Mathpix Markdown (reading order)
  "confidence": 0.97,           // global, number in [0,1]
  "confidence_rate": 0.99,      // global, number in [0,1]
  "is_printed": true, "is_handwritten": false,
  "data": [{"type": "latex", "value": "x^2+y^2=z^2"}],
  "line_data": [{
    "id": "...", "parent_id": "...", "children_ids": ["..."],
    "type": "math", "subtype": "...",
    "cnt": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],   // polygon (px); [TL,TR,BR,BL] if axis-aligned
    "included": true, "conversion_output": true,
    "is_printed": true, "is_handwritten": false,
    "error_id": null, "text": "...",
    "confidence": 0.98, "confidence_rate": 0.99,
    "after_hyphen": false, "html": "...",
    "data": [{"type": "latex", "value": "..."}]
  }],
  "word_data": [{
    "type": "text", "subtype": "...",
    "cnt": [[x,y], ...],
    "text": "...", "latex": "...",
    "confidence": 0.97, "confidence_rate": 0.99
  }]
}
```
**Verified field lists.** LineData: `id, parent_id, children_ids, type, subtype, cnt, included, conversion_output, is_printed, is_handwritten, error_id, text, confidence, confidence_rate, after_hyphen, html, data`. WordData: `type, subtype, cnt, text, latex, confidence, confidence_rate` — **WordData carries its own `confidence_rate`, not just `confidence`.**

**Geometry primitive is `cnt`**: documented as a "list of (x,y) pixel coordinate pairs." It is a 4-point quad only in the axis-aligned case (vertices then ordered `[TL, TR, BR, BL]`, clockwise from top-left); it is a general polygon, not constrained to 4 points. **There is no `bbox` / `{x,y,w,h}` key** in v3/text line/word objects — geometry *is* `cnt`. Math renderings (`latex` / `asciimath` / `mathml` / `svg` / `tsv`) are selected via `data_options`. ([v3/text reference](https://docs.mathpix.com/reference/post-v3-text))

#### 2. Confidence / likelihood (native QE)
**Rich for an OCR engine.** Two float fields, both `number in [0,1]`, present at **three grains**: global (top level), per-line (`line_data[i]`), and per-word (`word_data[i]`).

Documented semantics (verbatim from the v3/text reference, which is terser than community lore):
- `confidence` — **"Estimated probability 100% correct"**, described as the *product of per-token OCR confidence*. This is an all-or-nothing-style accept probability that degrades sharply on long spans.
- `confidence_rate` — **"Estimated confidence of output quality."** It degrades more gracefully than `confidence` and is the better-behaved continuous signal for selective-prediction gating. (The exact aggregation — often described as a per-symbol/geometric-mean rate — is **not** spelled out verbatim in the docs; treat the precise formula as unconfirmed.)

The engine can self-reject via request parameters: **`confidence_threshold`** (number in [0,1]) and **`confidence_rate_threshold`** (number in [0,1], **default 0.75**).

**No token logprobs are exposed.** Mathpix *does* compute per-token OCR confidences internally (that is literally how `confidence` is defined), but the API surfaces only the aggregated scalars — there is **no** logprob / top-k token-distribution field, and no per-character confidence.

#### 3. Token logprobs
**Not exposed.** v3/text is not a token-generative LLM/VLM API; the v3/text reference mentions no `logprob` / token-distribution output. Intrinsic QE = the `confidence` / `confidence_rate` scalars (global / line / word) only.

#### 4. LM / lexicon control
Script-level, not word-level:
- **`alphabets_allowed`** (AlphabetsAllowed object) — allow/deny symbols by script (e.g. Latin vs Cyrillic vs CJK) to disambiguate visually-identical glyphs. This is the main lexicon/script-prior knob.
- **`include_detected_alphabets`** → `detected_alphabets` response object (per-script booleans: en, ru, zh, ja, ko, hi, …).

**No** user-supplied custom dictionary, **no** word-level allow/deny list, **no** pluggable external LM. Recognition + correction models are bundled and opaque. Math/correction toggles exist (`numbers_default_to_math`, `rm_spaces`, `idiomatic_eqn_arrays`, `include_smiles` for chemistry, `enable_tables_fallback`, etc.) but none is a lexicon.

#### 5. Structured layout / provenance
- **Geometry:** `cnt` polygons on every line and word element (image pixels) — usable for provenance overlays. No rectangular bbox key.
- **Reading order / regions:** top-level `text` is Mathpix Markdown in document order; element `type`/`subtype` classify regions (text, math, table, diagram, chart, equation_number, …).
- **Tables:** emitted as Mathpix-Markdown / HTML tables; TSV via `data_options.include_tsv`; `enable_tables_fallback` (default true).
- **Math / chemistry:** structured renderings in `data[]` (latex / asciimath / mathml / svg); `include_smiles` for chemistry diagrams (RDKit-normalized SMILES).
- **Key-value / form extraction:** not a native primitive. **JSON-schema-constrained output:** not offered (this is OCR, not a structured-extraction LLM).
- **hOCR / ALTO XML: NOT supported.** Output is JSON + Mathpix Markdown + Convert document formats (PDF→Markdown/DOCX/HTML/LaTeX; image→LaTeX/SMILES; table→CSV/LaTeX/Markdown).

#### 6. Deployment, license, pricing
- **Deployment:** hosted cloud REST at `api.mathpix.com` (`v3/text` sync images, `v3/pdf` async documents, `v3/strokes` digital ink). On-prem/air-gapped is available only as an enterprise **"On-prem PDF Cloud"** offering (not self-serve). Default is cloud-hosted, **data sent to Mathpix servers.**
- **Client / license:** the OCR engine and model weights are **proprietary** (ToS-governed, paid per-usage). The official Python client **`mpxpy` is MIT-licensed** but is a thin client to the hosted API — it performs **no local OCR** and requires Mathpix Console credentials.
- **Pricing (pay-as-you-go, verified mid-2026):** **$19.99** one-time setup fee; **$29** testing credit on signup; **Image** (v3/text) **$0.002/image** (0–1M), $0.0015 (1M+); **PDF** (v3/pdf) **$0.005/page** (0–1M), $0.0035 (1M+); **Strokes** free <1K sessions, then $0.01 / $0.008 / $0.005 per session by tier. ([API pricing](https://mathpix.com/pricing/api))
- **Capabilities:** strong multilingual OCR (Latin, Cyrillic, CJK, Indic, Arabic, etc.); **printed AND handwritten** (`is_printed` / `is_handwritten` flags at result and line grain); **best-in-class math/equation OCR** to LaTeX/MathML/ASCIIMath plus chemistry (SMILES) and tables.

**QE takeaway:** Mathpix gives dual calibrated confidence scalars at three grains plus engine-side thresholding and polygon provenance — strong for a non-generative OCR engine — but offers **no token logprobs, no per-character signal, and no user lexicon control**, so any richer QE must be built on the scalar confidences and geometry it does expose.

**Sources:** [Mathpix v3/text API reference (response schema, confidence/confidence_rate, line_data/word_data, cnt, alphabets_allowed, thresholds)](https://docs.mathpix.com/reference/post-v3-text); [Mathpix Convert API pricing (2026)](https://mathpix.com/pricing/api); [Mathpix Convert API overview (output/conversion formats, on-prem PDF Cloud)](https://mathpix.com/docs/convert/overview); [mpxpy on PyPI (official Python client, MIT, calls api.mathpix.com)](https://pypi.org/project/mpxpy/); [Mathpix v3/pdf reference (async document processing)](https://docs.mathpix.com/reference/post-v3-pdf)

---

### Anthropic Claude Vision

Claude Vision is **not an OCR engine** — it is a closed, hosted vision-language model (VLM) accessed through the [Messages API](https://platform.claude.com/docs/en/api/messages) with image content blocks. For information extraction it is the canonical **"LLM-as-extractor"** case: it returns whatever the prompt asks for (a transcription, or schema-constrained JSON) and exposes **no intrinsic confidence signal of any kind**, so any quality estimation must be layered externally. All claims below were verified against live primary docs (platform.claude.com), mid-2026.

#### 1. Rawest output
No glyph/word/line/block layer and no native geometry. The lowest-level result is a generic `Message` whose `content` is an array of content blocks (`type:"text"` → `TextBlock`; tool/function calls → `ToolUseBlock`). The image is supplied as a user block `{"type":"image","source":{...}}` with `source.type` of `base64` | `url` | `file` (Files API). Supported media: `image/jpeg|png|gif|webp`; max 8000×8000 px, 10 MB/image (5 MB on Bedrock/Vertex); up to 100 images/request (200k-context models) or 600 otherwise.

```json
// request
{ "model":"claude-opus-4-8","max_tokens":1024,
  "messages":[{"role":"user","content":[
    {"type":"image","source":{"type":"base64","media_type":"image/png","data":"<b64>"}},
    {"type":"text","text":"Transcribe all text; return each line's box as [x1,y1,x2,y2] in pixels."}]}]}
// response (Message) — per /docs/en/api/messages
{ "id":"msg_...","type":"message","role":"assistant",
  "content":[{"type":"text","text":"...transcription / JSON...","citations":null}],
  "model":"claude-opus-4-8",
  "stop_reason":"end_turn","stop_sequence":null,"stop_details":null,
  "usage":{"input_tokens":1600,"output_tokens":120,
           "cache_creation_input_tokens":0,"cache_read_input_tokens":0} }
```

Any per-line/word segmentation or boxes appear **only because the prompt asked** and the model textually produced them; they are approximate and expressed against the **resized** image Claude sees. Claude views images in **28×28-pixel patches** (each patch = one *visual token*); use the docs' [`resized_size()`](https://platform.claude.com/docs/en/build-with-claude/vision) helper (alongside `count_image_tokens()`) to recover the dimensions Claude saw and map coordinates back to original pixels. **PDF pages are rasterized server-side at dimensions you don't control**, so their coordinates can't be reliably remapped — the doc's explicit advice is to rasterize PDFs yourself first.

#### 2. Confidence / likelihood
**None — at any granularity.** Verified on the live Messages API reference: there is **no `logprobs`/`top_logprobs` request parameter** and **no logprob/probability/confidence field anywhere in the response**. The only quantitative response field is `usage`, which holds token **counts** only (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens_details.thinking_tokens`, plus `cache_creation.*` and `server_tool_use.*`) — counts, not likelihoods. The nearest things to status flags are `stop_reason` (`end_turn|max_tokens|stop_sequence|tool_use|pause_turn|refusal`) and `stop_details` (a refusal-details object) — neither is a posterior. There is no per-char, per-word, per-line, or per-region confidence to read.

**QE consequence:** this is the canonical *no-native-QE* extractor. A reference-free quality layer must be built entirely externally — e.g. self-consistency/sampling agreement at `temperature>0`, an LLM-judge faithfulness check, cross-model agreement, or a separate verifier. Token logprobs remain a long-standing unimplemented community request.

#### 3. Structured output
**Native (GA): JSON-schema-constrained output** via `output_config.format` = `{"type":"json_schema","schema":{...}}`. This uses **constrained decoding** (the schema is compiled into a grammar that constrains generation), guaranteeing schema-valid JSON. It is now GA on Opus 4.5/4.6/4.7/4.8, Sonnet 4.5/4.6, Haiku 4.5, Fable 5, Mythos 5 (and on Bedrock/Vertex; beta on Microsoft Foundry). It supersedes the prior beta `output_format` parameter + `structured-outputs-2025-11-13` header, which still work during a transition period. Strict tool use (`strict:true` on tool defs) similarly constrains `tool_use` inputs. This yields reliable key-value / table-as-JSON shapes — but the **content is model-generated, not layout-derived, and carries NO per-field confidence**.

**Prompted-only (approximate, no guarantee):** bounding boxes/points (`[x1,y1,x2,y2]` absolute pixels vs the resized image — normalized 0–1000 coords work poorly), reading order, table structure, region labels.

**Not available at all:** native hOCR, ALTO, PAGE-XML, polygon masks, glyph/word geometry, engine-side reading order, or any provenance overlay tied to detected layout. Bbox provenance must be prompted and rescaled via `resized_size()`; accuracy is explicitly "approximate" and should be spot-checked.

#### 4. Language model / lexicon
No OCR-style language model, dictionary, allow/deny list, or lexicon config exists. The "language model" is Claude itself — its generative prior implicitly corrects/normalizes text, a double-edged property for OCR faithfulness (it may silently "fix" or hallucinate text not present; the doc warns it "may hallucinate or make mistakes when interpreting low-quality, rotated, or very small images under 200 pixels," and gives only approximate object **counts**). The only tunables affecting this language-sensitive behavior are sampling params: `temperature` (0.0–1.0), `top_p`, `top_k`, and `stop_sequences`. There are **no** domain/vocabulary-restriction parameters; `output_config.effort` (`low|medium|high|xhigh|max`) and `thinking` affect reasoning depth, not lexical priors. To bias toward verbatim output you must instruct it in the prompt ("transcribe exactly, do not correct spelling").

#### 5. Deployment & license
**Hosted-only, closed-weights, proprietary** (Claude API direct, plus Amazon Bedrock and Google Vertex AI). No local/on-device/self-host option. Licensing is Anthropic's **Commercial Terms of Service** (no OSS license, no weights), with Bedrock/Vertex resale terms. On **data use**: by default Anthropic does **not** train on commercial/API inputs or outputs; note this is the default-commercial posture rather than a blanket "deleted immediately" guarantee — standard usage retains data for a limited period for abuse-monitoring, and truly no-storage behavior requires a **Zero Data Retention** agreement for qualifying enterprise customers.

**Cost model:** per visual token — an image costs `⌈width/28⌉ × ⌈height/28⌉` tokens, capped by model: **1568 tokens / 1568 px long-edge** on most models, **4784 tokens / 2576 px long-edge** on Opus 4.7/4.8, Fable 5, Mythos 5. (E.g. a 4K image is downscaled to 2576×1449 = 4784 tokens.)

#### 6. Modality notes
- **Handwriting**: supported only via general vision/transcription — **no dedicated mode**, subject to the hallucination caveat above.
- **Math/formula**: can emit LaTeX/structured math, but only because the model generates it — **generative, not engine-verified**; no dedicated formula mode is documented.
- **Languages**: broad multilingual coverage (an LLM, not an enumerated OCR language list); no published language count.
- **Other constraints**: refuses to identify/name real people; not for diagnostic medical imaging; coordinate/spatial outputs are explicitly approximate.

**Bottom line for QE/selective prediction:** Claude Vision supplies essentially **zero native QE signal** (no logprobs, no confidence, no layer geometry). Accept/flag/block decisions must rest entirely on an external QE stack (sampling self-consistency, LLM-judge, cross-model agreement, or a verifier), with prompted-and-rescaled bounding boxes as the only — approximate, non-PDF — provenance handle.

**Sources:** [Messages API reference (request params, Message/usage/stop_reason fields)](https://platform.claude.com/docs/en/api/messages); [Structured outputs (output_config.format, constrained decoding, GA models, prior beta)](https://platform.claude.com/docs/en/build-with-claude/structured-outputs); [Vision guide (28px patches/visual tokens, resized_size(), bbox/coords, PDF caveat, hallucination/counting limits, resolution caps)](https://platform.claude.com/docs/en/build-with-claude/vision); [Anthropic Privacy Center — is my data used for model training (commercial default + ZDR)](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training)

---

### OpenAI GPT-4o / 4.1 Vision

A hosted vision-language model, **not an OCR engine**. It reads images and emits *text tokens*; any structure (fields, tables, even bounding boxes) exists only because you ask for it in the prompt or a JSON Schema. The single intrinsic confidence signal is **per-token logprobs** — usable for QE, but with reliability caveats that I could confirm are live as of mid-2026. (See [Images & vision guide](https://platform.openai.com/docs/guides/vision), [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), [logprobs cookbook](https://cookbook.openai.com/examples/using_logprobs).)

#### 1. Rawest output
Token stream -> string, optionally JSON-Schema-constrained. No glyph/char/word/line/block primitive, no native geometry.

```python
resp = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role":"user","content":[
        {"type":"text","text":"Transcribe verbatim."},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,...","detail":"high"}}]}],
    logprobs=True, top_logprobs=5)
```
Returned: `choices[0].message.content` (string); `choices[0].logprobs.content[]` is a list of items each with `token`, `logprob`, `bytes`, and `top_logprobs[]` (each with `token`, `logprob`, `bytes`). Responses API: image part is `{"type":"input_image",...}`; text comes back in `output[].content[]`.

#### 2. Confidence / likelihood
**Per-token logprobs only**, natural log, max 0.0 (0.0 ≈ certainty); `exp(logprob)` gives linear probability. No per-char/word/line/region posterior and **no confidence on geometry**; you must map BPE tokens to words and fields yourself.
- Chat: `logprobs=True`, `top_logprobs=N` -> `resp.choices[0].logprobs.content[i].logprob` plus `.top_logprobs[j]`. **`top_logprobs` is documented 0–20 for current Chat Completions** (older docs/SDKs and the Responses/Azure surfaces still show 0–5 in places — pin and verify per endpoint).
- Responses: `top_logprobs=N` plus `include=["message.output_text.logprobs"]` -> `response.output[k].content[m].logprobs[i]`.

**Reliability caveats (confirmed live, do NOT skip):**
1. **Flaky/empty on gpt-4o, especially with images.** Live bug reports show the top-token logprob returned as `-9999.0`, `top_logprobs` misaligned with the sampled token, intermittent `logprobs=None`, and **empty logprobs arrays with image input** — tracked in [openai-python #2257](https://github.com/openai/openai-python/issues/2257) and community Bugs threads ([flaky logprobs gpt-4o](https://community.openai.com/t/flaky-logprobs-with-gpt-4o/1152027), [-9999.0](https://community.openai.com/t/logprobs-returning-9999-0/1146499), [incorrect/off](https://community.openai.com/t/gpt4o-logprobs-return-incorrect-outputs-completely-off/1147333)).
2. **Empty when structured outputs are also on.** With `json_schema` (strict) enabled, `message.output_text.logprobs` returns `[]` on current snapshots — [confirmed for GPT-5.1/5.2, Jan 2026](https://community.openai.com/t/gpt-5-1-5-2-message-output-text-logprobs-is-empty-when-structured-outputs-json-schema-is-enabled-in-responses-api/1371927), with a Mar 2026 follow-up suggesting ongoing regression. GPT-4.1 was reported still working in that thread, but this is **regression-prone and under-documented**.

**Verdict for QE: load-test logprob population on your EXACT snapshot, with images, and with your schema — do not assume the signal is present.**

#### 3. Language model / smoothing / lexicon
The whole system *is* an LM, so it **silently autocorrects** read text (a verbatim-transcription hazard — it may "fix" a genuine misspelling on the source). No OCR dictionary, allow/deny list, or domain-lexicon config. Only generic decoder knobs: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `logit_bias` (token-id -> bias map; the only thing resembling vocabulary control), `max_tokens`/`max_output_tokens`, `seed` (best-effort), `stop`. `response_format` json_schema (Chat) / `text.format` (Responses) constrain output *structure*, not an allowed-word vocabulary. No per-language model selection; only in-context prompt examples can nudge domain vocabulary.

#### 4. Structured output
Structured Outputs via JSON Schema (`strict: true`) is the main lever — on `gpt-4o-2024-08-06`, `gpt-4o-mini`, and later (docs now steer new projects to gpt-5.x). Gives key-value extraction, table-as-JSON, reading-order text, arbitrary typed records — all **model-inferred** content. It does **NOT** give measured bounding boxes/polygons, ALTO/hOCR, or any provenance geometry. A `bbox`/`quad` field in your schema is a **model pixel-space GUESS** (frequently inaccurate/hallucinated), not a detector output — unsuitable as ground-truth provenance for overlays without independent verification. No layout-segmentation primitives, no line/region objects, no confidence-per-field; reading order is whatever the model emits.

#### 5. Math / handwriting
Both **best-effort, not SLA**: can read cursive/handwriting and can emit LaTeX/MathML for formulas when prompted, but neither is a dedicated recognition engine and accuracy varies with script, density, and image quality.

#### 6. Deployment & license
Hosted-only, closed-weights — **OpenAI API or Azure OpenAI are the only hosting paths; no local/on-device weights**. By default API inputs/outputs are **not** used for training; default 30-day retention for abuse monitoring; **Zero Data Retention** available for eligible endpoints/qualifying use-cases ([enterprise privacy](https://openai.com/enterprise-privacy/)). Still server-side inference — unsuitable where data cannot leave premises. Image inputs billed as resolution/detail-derived tokens, so dense-page OCR can be costly vs a local engine. **License: proprietary** (OpenAI Terms of Use / Azure OpenAI); closed weights, no self-host. **Not AGPL, not non-commercial — but fully vendor-locked.** Flag: snapshots, `detail` defaults, and `top_logprobs` caps have shifted through 2025–2026 (docs now front gpt-5.x); pin a snapshot and re-verify.

**Sources:** [OpenAI Cookbook — Using logprobs (field shapes, top_logprobs)](https://cookbook.openai.com/examples/using_logprobs); [Responses API reference (create) — include 'message.output_text.logprobs'](https://developers.openai.com/api/reference/resources/responses/methods/create); [openai-python #2257 — flaky logprobs with gpt-4o](https://github.com/openai/openai-python/issues/2257); [Community — flaky logprobs with gpt-4o](https://community.openai.com/t/flaky-logprobs-with-gpt-4o/1152027); [Community — Logprobs returning -9999.0](https://community.openai.com/t/logprobs-returning-9999-0/1146499); [Community — GPT4o logprobs return incorrect outputs](https://community.openai.com/t/gpt4o-logprobs-return-incorrect-outputs-completely-off/1147333); [Community (Jan 2026) — logprobs empty when json_schema enabled (GPT-5.1/5.2)](https://community.openai.com/t/gpt-5-1-5-2-message-output-text-logprobs-is-empty-when-structured-outputs-json-schema-is-enabled-in-responses-api/1371927); [OpenAI — Introducing Structured Outputs (gpt-4o-2024-08-06 / gpt-4o-mini support)](https://openai.com/index/introducing-structured-outputs-in-the-api/); [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs); [OpenAI — Enterprise privacy / data retention / ZDR](https://openai.com/enterprise-privacy/); [OpenAI — Services Agreement (license/terms)](https://openai.com/policies/services-agreement/)

---

## Synthesis — native QE richness across the field

Native QE richness splits the field cleanly. The systems that hand us usable confidence out of the box are the discriminative cloud recognizers — Google Cloud Vision (per-symbol through per-block posteriors with glyph-level geometry) and Azure Document Intelligence (word, selection-mark, field/KV, and table/row/cell confidence) — plus Mathpix, whose dual confidence + confidence_rate across three grains is the only signal in the set that is presented as calibrated. AWS Textract sits just below with per-block (word/line/cell/KV) Confidence. These four can drive accept/flag/block thresholds with minimal external machinery, though calibration is undocumented for Google, Azure, and Textract.

A second tier — Tesseract (rich per-unit conf + N-best alternatives + full geometry, but explicitly uncalibrated and not a probability), PaddleOCR, RapidOCR, and EasyOCR — emits line-level recognition scores and polygon provenance but no per-char/word confidence by default and no logprobs; treat these as feature sources for an EXTERNAL calibration/selective-prediction layer rather than turnkey QE. The minimal tier — OCR.space (no confidence at all), Apple Vision/ocrmac (coarse, often quantized), and pix2tex (nothing) — requires a fully external confidence layer built from lexicon agreement, geometry outliers, or cross-engine disagreement.

The VLM/LLM verdict turns entirely on logprobs, since that is the only intrinsic-confidence channel a generative extractor has. OpenAI GPT-4o/4.1 exposes per-BPE-token logprobs in principle but they are flaky/-9999/empty on image inputs and empty under strict json_schema, so any QE built on them must be load-tested per model snapshot. Mistral OCR is the pragmatic VLM choice: it converts internal logprobs into opt-in word- and page-level confidence. TrOCR (and, with a patch, pix2tex) recovers genuine per-token logprobs locally via compute_transition_scores — but with no geometry for provenance. Claude Vision is the outlier with zero native QE signal — no logprobs, no confidence (verified mid-2026) — so for Claude, selective prediction must come entirely from sampling self-consistency, an LLM-judge, cross-model agreement, or a separate verifier.
