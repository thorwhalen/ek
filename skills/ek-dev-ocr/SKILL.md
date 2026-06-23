---
name: ek-dev-ocr
description: "Build and maintain ek's OCR evaluation layer on top of the ocracy package \u2014 the concrete instance of ek's reference-based score() and reference-free estimate_quality() facades applied to image->OcrResult engines. Use when working on the ek[ocr] optional extra, wiring or running the OCR offline benchmark (CER/WER/ANLS per-slice over a gold corpus), mapping OcrResult/TextBlock into AnnotatedExtraction/FieldEstimate, deciding which ocracy backend to evaluate, handling null-safe VLM engines (Claude/GPT-4o/Mistral/Mathpix returning text-only, blocks=[], confidence=None), reading per-engine capability profiles (who emits real vs model-guessed geometry, who is calibrated, who emits TEDS-able tables), gating remote engines that make paid API calls, dealing with the missing normalized table type, or anything touching ek<->ocracy. Enforces the dependency-direction rule: ek depends on ocracy via an extra; never ocracy->ek; ek core depends only on the OcrResult shape so it evaluates any image->OcrResult callable."
metadata:
  audience: developers
---

# ek-dev-ocr — the OCR concrete instance of ek

You are building/maintaining ek's OCR evaluation: the first concrete instance of
the generic engine (`score`, `estimate_quality`, the two-layer data model) wired
to real OCR engines via **ocracy**. OCR is the noisiest special case of
information extraction — the proving ground for the whole framework.

ocracy source (read on demand): the locally-installed `ocracy` package (`python -c "import ocracy, os; print(os.path.dirname(ocracy.__file__))"`), or https://github.com/thorwhalen/ocracy
Capability profiles (cite by relative path): `misc/docs/ek_01 --  ocr-systems-capability-inventory.md`

---

## RULE 0 — Dependency direction (non-negotiable)

```
ek  ->  ocracy        (optional extra: ek[ocr])
ek  -X  ocracy        is FORBIDDEN as a hard dep of ek core
ocracy -X-> ek        NEVER. ocracy must not know ek exists.
```

- `pyproject.toml`: `[project.optional-dependencies] ocr = ["ocracy>=..."]`.
- ek **core** depends only on the **`OcrResult` SHAPE**, never on importing
  ocracy. That shape: `text: str`; `blocks: list[TextBlock(text, bbox,
  confidence, level, language, meta)]`; `raw`; `meta`; `confidence` normalized
  to `[0,1]`. Because core only knows the shape, ek can evaluate **any**
  `image -> OcrResult` callable — ocracy is just the best-stocked supplier.
- Import ocracy **lazily, inside functions**, behind `@requires_extra("ocr")`
  so `import ek` works with the lean core installed. A missing extra must raise
  an actionable install hint (`pip install ek[ocr]`), not an `ImportError`.
- Anything that needs ocracy types at typing time: `if TYPE_CHECKING:` import
  only, or duck-type on the shape. Do not add a runtime top-level
  `import ocracy` anywhere in `ek/` core.

Self-check before committing: `rg -n "import ocracy" ek/` — every hit must be
inside a function/method or a `TYPE_CHECKING` block.

---

## ocracy's API surface (what you call)

```python
import ocracy
res = ocracy.ocr(image, *, backend=None, **kwargs)  # -> OcrResult
ocracy.read_text(image, *, backend=None, **kwargs)   # -> str (text only)
```

`image` is an `ImageInput`: path, `http(s)` URL, `bytes`, PIL image, or numpy
array (decoded lazily — importing ocracy needs neither Pillow nor numpy).
`backend=None` picks the first *installed implemented* backend.

`OcrResult` (from `ocracy.base`), the SSOT shape ek evaluates:

```python
res.text                  # full text in reading order (the headline payload)
str(res)                  # == res.text
for block in res: ...     # iterates res.blocks (TextBlocks)
res.blocks                # list[TextBlock]; may be [] for VLM/markdown engines
res.at_level("word")      # res.words / res.lines / res.paragraphs
res.mean_confidence       # mean over blocks reporting one, else None
res.markdown              # res.meta.get("markdown") — set by VLM engines, else None
res.filter_confidence(t)  # copy keeping blocks with confidence >= t (drops None-conf)
res.backend, res.raw, res.meta
```

`TextBlock`: `text, bbox: BBox|None, confidence: float|None (∈[0,1]), level
(one of LEVELS = page>block>paragraph>line>word>char), language, meta`.
`BBox`: `x0,y0,x1,y1` (always populated) + optional `polygon`; props
`width/height/area/xywh/as_tuple`.

### The ledger (choose engines with eyes open)
```python
ocracy.catalog                                # 64-entry Catalog (data/backends.json)
ocracy.find(is_local=True, open_source=True)  # filter -> new Catalog
ocracy.find(implemented=True)                 # only the ~15 runnable today
ocracy.catalog.compare(["tesseract","mathpix"])   # side-by-side fields
ocracy.catalog["mathpix"].price_note          # surface this before billing!
```
The catalog is a research ledger of **64 entries**; only ~15 have a working
adapter (`google_vision, aws_textract, azure_document_intelligence, tesseract,
easyocr, paddleocr, rapidocr, ocrmac, mathpix, mistral_ocr, claude_vision,
gpt_4o_vision, ocr_space, pix2tex_latex_ocr, trocr_handwritten`). Filter by
`bounding_boxes`, `confidence_scores`, `tables`, `pricing_model`, `is_local`,
`handwriting`, etc. — these fields drive which ek evaluation paths apply.

---

## RULE 1 — Null-safe ingestion (the #1 OCR landmine)

VLM / markdown engines (Claude Vision, GPT-4o Vision, Mistral OCR, Mathpix)
return **text only**: `blocks == []`, every `confidence is None`, geometry
absent. Verified in the adapters — e.g. `claude_vision` and `gpt_4o_vision` call
`OcrResult.from_text(text, ..., markdown=text)`; Mistral joins per-page markdown;
Mathpix is text-oriented and stashes a **document-level** score at
`res.meta['confidence']` (NOT per-block).

Therefore, in **every** ek adapter that maps `OcrResult -> AnnotatedExtraction`:

- Never assume `res.blocks` is non-empty. Branch: text-only vs structured.
- Never call `block.confidence` arithmetic without a `None` guard.
- `res.mean_confidence` returns `None` when nothing reports confidence — handle
  it; do not coerce to 0.0 (that silently fabricates a calibration signal).
- For Mathpix specifically, read `res.meta.get('confidence')` as a coarse
  doc-level signal; do not expect per-word.
- When `blocks == []` and you need geometry, you cannot manufacture it — record
  `Provenance(bbox=None)` and let downstream validators/policies know geometry
  is unavailable rather than inventing boxes.

A text-only `OcrResult` still maps cleanly to a flat-text node in the grammar
(one node, a text field), so offline CER/WER/ANLS still work — only the
geometry-aware uncertainty mapping is N/A. Make that the explicit fallback.

---

## RULE 2 — Per-engine capability profiles (route by what's REAL)

Read `misc/docs/ek_01 --  ocr-systems-capability-inventory.md` before trusting
any signal. The catalog flags are coarse; ek_01 is the truth about *quality*:

- **Real per-unit confidence + geometry (turnkey QE):** Google Vision
  (per-symbol→block conf + glyph bboxes), Azure Document Intelligence
  (word/KV/table-cell conf). **Tesseract** = richest *local* signal (per-symbol/
  word/line conf + N-best) **but uncalibrated**.
- **Calibrated (the only one):** **Mathpix** — `confidence` + `confidence_rate`
  at global/line/word, with engine-side gating. Treat as the calibration
  reference; everything else feeds an *external* calibrator.
- **Middle (line score + polygon, no per-char, no logprobs):** PaddleOCR,
  RapidOCR, EasyOCR — features for an external calibration layer, not turnkey.
- **Model-GUESSED geometry — never use as provenance:** Claude Vision, GPT-4o
  Vision (any bbox is a pixel guess, not a measurement); Mistral text locator is
  a char offset, not a box. ek_01 line 29 is explicit. If an adapter surfaces
  these boxes, tag provenance as `engine="vlm"` / unreliable so the
  geometry-aware uncertainty mapping and HITL overlays don't trust them.
- **TEDS-able table STRUCTURE (real cells):** Azure DI
  (rowIndex/colIndex/span), PaddleOCR PP-StructureV3 (`pred_html`), AWS Textract
  (TABLE/CELL/MERGED_CELL), Mathpix/Mistral (markdown/HTML tables). **NOT**
  Google Vision (TABLE = tag only).
- **No usable confidence at all:** OCR.space (none), ocrmac/Apple Vision (coarse,
  quantized ~{0.3,0.5,1.0}), pix2tex (nothing). These *require* a bolt-on
  confidence layer (lexicon agreement, geometry outliers, cross-engine
  disagreement) before `estimate_quality` produces anything meaningful.

Practical rule: pick the **smallest reliable unit** an engine actually reports
(ek_01 "rawest output" column), and don't synthesize signal an engine doesn't
have. Don't let `estimate_quality` emit a confident `decision=accept` off a
fabricated confidence.

---

## RULE 3 — The table GAP

ocracy has **no normalized table type** — `TextBlock.level` stops at `block`,
and there is no cell/row/col model. Engines that *do* emit table structure bury
it in `res.raw` (Textract `Block` objects, Paddle `table_res_list[i]['pred_html']`,
Azure `tables[]`, Mathpix/Mistral markdown/HTML). So a TEDS metric in ek must
either (a) dig into `res.raw` per backend (brittle, engine-specific) or
(b) parse the HTML/markdown table from `res.markdown`. **Prefer (b)** for the
markdown engines (one path, engine-agnostic). For the structured engines,
isolate the raw-digging in a clearly-named per-backend extractor and consider
**pushing a normalized table type upstream into ocracy** (it belongs there, not
in ek) — file it, don't quietly fork. Until then, TEDS is best-effort and must
degrade to text-only CER/WER when no table can be recovered.

---

## RULE 4 — Billing & remote engines (real money)

`google_vision, aws_textract, azure_document_intelligence, mathpix, mistral_ocr,
claude_vision, gpt_4o_vision, ocr_space` make **real, paid API calls**. The
offline benchmark fans out over a corpus × engines — that multiplies cost.

- Remote engines must be **explicit opt-in**, never a default in a benchmark run
  (a no-arg benchmark must hit only free/local engines or a tiny smoke sample).
- Before running a remote engine, surface its
  `ocracy.catalog[backend].price_note` (and `pricing_model`) to the user/log.
- Gate with a keyword-only flag (e.g. `allow_paid=False`) that the caller must
  flip; pair it with a corpus-size guard. No magic — make the cost visible.
- Cache results in the dol store so a re-run of the benchmark does not re-bill
  (use `dol.cache_this`; key by image + backend + options).

---

## RULE 5 — Mapping OcrResult -> ek's two-layer model

Layer A (`GraphGrammar`) defines the schema/importance/cost SSOT; Layer B
(`AnnotatedExtraction(grammar, estimates: Mapping[NodePath, FieldEstimate])`)
holds the run. The OCR adapter is a pure function `OcrResult -> AnnotatedExtraction`:

- Flat-text OCR -> a single text node; `FieldEstimate(value=res.text, ...)`.
- Structured OCR -> one node per block/region; carry `block.confidence` into
  `FieldEstimate.confidence` (or `raw_signals` if uncalibrated), and
  `block.bbox` + `res.backend` into `Provenance(engine=res.backend,
  bbox=..., raw_transcripts=...)`.
- `decision ∈ {accept, flag, block}` and `Finding`s come from the
  validators/policy in `estimate_quality`, not from the adapter — the adapter
  only *ingests* signal, it doesn't *judge*.

Keep the adapter null-safe (RULE 1) and provenance-honest (RULE 2): an estimate
with no real confidence must say so (e.g. `confidence=None`, signal in
`raw_signals`), not pretend.

---

## The flagship deliverable: the OCR offline benchmark

The first thing this instance proves end-to-end. Reference-based / offline:

1. Load a **gold corpus** from the dol store (`get_artifact_dir("gold")`).
2. For each (image, engine) — engines chosen via `ocracy.find(...)`, paid ones
   gated (RULE 4) — run `ocracy.ocr(image, backend=...)` and map to Layer B.
3. Score with the `score(pred, gold, *, grammar=None, metric=None, ...)` facade.
   Metric dispatched by output type: **CER / WER** for text, **ANLS** for
   normalized-string fields, the cost-weighted typed-graph distance for
   structured graphs (the flagship offline metric). TEDS where tables recover
   (RULE 3).
4. Report **per-slice** (by engine, by document class, by language, by
   difficulty) — slice-awareness is a must-build, not an afterthought.
5. **dol-persist** results: `dol.Jsons(get_artifact_dir("results"))`, grouped in
   the mall; cache engine outputs so re-runs don't re-bill.

Use `rapidfuzz` for edit-distance metrics — **never** `Levenshtein` /
`python-Levenshtein` (GPL; see the license gate). Yield rows
(`Iterable[Report]`) rather than building giant lists.

---

## Persistence (same as core)

`config2py.AppData("ek")` -> `~/.local/share/ek/`;
`dol.Jsons(get_artifact_dir(kind))` is a JSON `MutableMapping` per kind
(`gold/`, `results/`, `calibrators/`, `corrections/`, `runs/`), grouped as a
dol mall. Stores are `MutableMapping` facades, never god-classes.
`dol.cache_this` for persisted baselines and engine outputs.

---

## Conventions (ruff-enforced)

- Every `.py` needs a module docstring (D100).
- Public OCR API surfaces through `ek/__init__.py` (or a clearly named subpkg).
- Functional over OOP; `dataclasses` + `collections.abc`; keyword-only beyond
  the 3rd arg; no magic numbers (kw-only / external config; open-closed).
- Progressive disclosure: the simple call works
  (`benchmark_ocr(corpus)` over free/local engines), everything tunable via
  kw-only args (`backends=`, `metric=`, `allow_paid=`, `slices=`, `calibrator=`).
- Generators (`Iterable[T]`) over list-building.
- CLI via argh (`_dispatch_funcs` SSOT + `__main__.py` `dispatch_with_namespaces`).
- `@requires_extra("ocr")` on every function that imports ocracy.

## Quick self-audit before you commit
- `rg -n "import ocracy" ek/` — all hits lazy or `TYPE_CHECKING`.
- No `Levenshtein`/`python-Levenshtein` import anywhere (use `rapidfuzz`).
- No code path arithmetic on `block.confidence` / `mean_confidence` without a
  `None` guard.
- No remote backend runnable without an explicit opt-in flag; `price_note`
  surfaced.
- New per-backend `raw`-digging isolated in a named extractor (table gap), or an
  upstream ocracy issue filed instead of a fork.
