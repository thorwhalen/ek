# R6 — Library Landscape & Integration Map

**Author:** Thor Whalen  
**Engine:** Claude Code (concrete library/PyPI/GitHub survey + integration design)  
**Date:** 2026-06-22  
**Companion:** *Information Extraction Evaluation — A Conceptual Map* (shared framing); **R1 — OCR Systems Capability Inventory** (R1's 15 profiled engines, which this map puts behind a uniform interface — plus two strong *local* libraries surfaced by this survey but not in R1's set: `python-doctr` and `surya-ocr`).

> **What this is.** A concrete, *implementable* plan: which existing libraries to pool to build information-extraction (IE) evaluation tooling — for both the **offline / reference-based** need (metrics over structured outputs) and the **online / reference-free** need (confidence → calibration → conformal/selective prediction → accept/flag/block), plus validation/correction, OCR front-ends, and human-in-the-loop review. Every library below was verified against **live PyPI + GitHub** in mid-2026 (license, last release, maintenance, the exact API we would call). The design then composes them behind a thin facade with pluggable, function-injected backends, tiered by dependency weight and license, consistent with a permissive-license downstream goal.

---

## TL;DR — decision points first

- **The metrics, calibration, validation, and HITL tiers are almost entirely permissive and well-maintained.** A clean default stack exists with *zero* copyleft exposure: `rapidfuzz` + `jiwer` + `sacrebleu` (surface) · `apted` + `zss` + `networkx` + `nervaluate` + `table-recognition-metric` (structured/graph) · `netcal` + `MAPIE` + `crepes` + `puncc` (calibration & conformal) · `pydantic` + `jsonschema` + `pandera` + `outlines`/`xgrammar` (validation & constrained generation) · `label-studio` + `cvat` (HITL).
- **Six license landmines** — all verified, all avoidable: **`Levenshtein` / `python-Levenshtein` (GPL-2.0)** → replace wholesale with `rapidfuzz`; **`TorchCP` (LGPL-3.0)** → the *only* library with graph/sequence conformal, quarantine as an optional import-only plugin with a permissive fallback; **`surya-ocr`** → Apache *code* but **non-commercial RAIL-M model weights** (the PyPI "Apache-2.0" classifier is a trap); **`Potato` (GPL-3.0)** and **`Prodigy` (commercial)** → HTTP-only or design-reference, never linked.
- **The biggest *build* is the same one R1 implied: cost-sensitive, type-aware metrics over a typed graph.** No library ships per-type / importance-weighted edit costs — but `apted` (`Config.rename/delete/insert`), `zss` (`label_dist` + cost callables) and `networkx.graph_edit_distance` (`node_subst_cost`/`edge_subst_cost` callables) all expose the cost hooks. We supply cost *functions*; we do not build a distance engine.
- **The architecture is a thin facade over function-injected strategies**, with the conceptual map's **two-layer split kept first-class**: a **graph grammar / schema** layer (the SSOT for *what we extract and what matters* — types and their importance weights) separate from an **extraction-and-verification metadata** layer (per-value provenance, confidence signals, validator findings, calibrated scores, accept/flag/block decisions). The same schema object is scored offline (vs gold) and estimated online (vs consensus/expectation).
- **R1's cloud/VLM engines are first-class, permissive *clients* over proprietary *services*.** Every official SDK for R1's richest-QE engines is permissive and actively maintained — `google-cloud-vision` (Apache-2.0), `boto3` + `amazon-textract-response-parser`/`-textractor` (Apache-2.0), `azure-ai-documentintelligence` (MIT, the package that replaced `azure-ai-formrecognizer`), `mistralai` (Apache-2.0), `mpxpy` (Mathpix, MIT), `openai` (Apache-2.0), `anthropic` (MIT) — so shipping the wrappers is license-clean; only the *services* are pay-per-call. Two scanner traps to note: `mistralai` and `mistralai`-style packages declare their license by file only (PyPI metadata `license=None`), and `mistralai` is a **namespace package** — import `from mistralai.client import Mistral`, not `from mistralai import Mistral`.
- **The other big *build* is R1's only confidence source for the no-confidence engines: cross-engine agreement.** R1 found Claude Vision, OCR.space, and pix2tex emit *no* usable confidence — agreement is the only signal. Verified result: there is **no maintained, permissive, pip-installable ROVER** (NIST SCTK is C/GPL-flavored), so ROVER-style N-way alignment+voting is a small clean build on `rapidfuzz`/`jiwer`. For the LLM/VLM half, **`uqlm`** (Apache-2.0) gives black-box self-consistency confidence out of the box (sample N → semantic-entropy/non-contradiction → [0,1]); `lm-polygraph` (MIT) covers white-box logit uncertainty for self-hosted VLMs.
- **Reuse > wrap > build.** Reuse ~30 libraries as-is (incl. the seven cloud SDKs + `uqlm`); wrap the OCR engines and calibration/constrained-gen backends behind small adapters; build only the connective tissue that no library owns: the two-layer eval object, cost-weighted graph metrics, field/graph-level calibration mapping, cross-field/cross-source consistency validators, the **ROVER engine + a geometry-aware mapping of whole-response uncertainty onto per-unit `OcrResult.confidence`**, provenance linking, and the harness (stratified sampling, per-slice + golden-set regression).

---

## Dependency tiers (`extras_require`)

Tiering is by **dependency weight AND license**, so the common case installs light and permissive, heavy/optional/ license-encumbered backends are opt-in, and nothing copyleft or non-commercial is ever a default. `pip install ie-eval` pulls only `core`; everything else is an explicit extra (open–closed: new backends are added as extras, never by editing callers).

| Extra | Libraries (verified mid-2026) | License posture | Weight / maintenance risk | Role |
|---|---|---|---|---|
| `core` *(always)* | `rapidfuzz` (MIT) · `jiwer` (Apache-2.0) · `pydantic` (MIT) · `numpy` · `networkx` (BSD-3) | All permissive | Light; all active | Surface metrics + alignment, the typed-payload schema spine, graph container. |
| `metrics` | `sacrebleu` (Apache-2.0) · `apted` (MIT) · `zss` (BSD-3) · `nervaluate` (MIT) · `table-recognition-metric` (Apache-2.0) · *opt:* `seqeval`/`textdistance` | All permissive | Light; `apted`/`zss` frozen (2017/2018) but algorithmically stable — low risk | Structured/field/graph scoring: chrF/TER, tree-edit (TEDS), span-F1, table TEDS. Cost-callable hooks for weighted variants. |
| `calibration` | `netcal` (Apache-2.0) · `scikit-learn` (BSD-3) · `MAPIE` (BSD-3) · `crepes` (BSD-3) · `puncc` (MIT) | All permissive | Medium; `netcal` recently revived after ~20-mo dormancy — monitor | Temperature/Platt/isotonic + ECE/reliability; split/CRC conformal; risk-coverage; online drift martingales (`crepes`). |
| `calibration-torch` *(opt, heavy)* | `torch-uncertainty` (Apache-2.0)  — pulls **PyTorch** | Permissive | Heavy (torch); pre-1.0 | Deep-model temperature/vector scaling + APS/RAPS conformal when a torch pipeline already exists. |
| `calibration-graph` *(opt, LICENSE-GATED)* | **`TorchCP` (LGPL-3.0)**  — pulls torch | ⚠️ **weak-copyleft (LGPL-3.0)** — isolated, import-only | Heavy; ~8-mo since release | The *only* graph-node & LLM-sequence conformal. Quarantined: never vendored/patched; permissive fallback = `calibration[-torch]`. |
| `validation` | `jsonschema` (MIT) · `fastjsonschema` (BSD-3) · `pandera` (MIT) · `pydantic-extra-types` (MIT) *(pydantic is core)* | All permissive | Light; `fastjsonschema` ~10-mo (watch) | Declarative type/range/enum/regex + business-rule validation; per-field defect streams; aggregate frame reporting. |
| `constrained` *(opt, heavy)* | `outlines` (Apache-2.0) · `xgrammar` (Apache-2.0)  — pull model/inference stacks | Permissive *code* (model weights/API licensed separately) | Heavy; pre-1.0 / post-rewrite API churn — pin | Validation-on-generate: force structured outputs to a pydantic/JSON-Schema by construction. |
| `ocr-local` *(opt, heavy)* | `tesserocr` (MIT)/`pytesseract` (Apache-2.0) · `rapidocr` (Apache-2.0) · `python-doctr` (Apache-2.0) · `easyocr` (Apache-2.0) · `paddleocr` (Apache-2.0, heaviest) · *macOS:* `ocrmac` (MIT) | Permissive incl. standard weights | Heavy; weights download at first use; `tesserocr` needs system libs | Local OCR front-ends behind a uniform `OcrResult` (text + geometry + confidence). |
| `ocr-cloud` *(opt)* | `google-cloud-vision` (Apache-2.0) · `boto3`+`amazon-textract-response-parser`/`-textractor` (Apache-2.0) · `azure-ai-documentintelligence` (MIT) · `mistralai` (Apache-2.0) · `mpxpy` Mathpix (MIT) · `openai` (Apache-2.0) · `anthropic` (MIT) | Permissive **clients** (all verified); **proprietary hosted services** (egress + per-call cost) | Light clients, all active; `trp` parser slower (2024) | R1's cloud OCR/VLM behind the same `OcrResult`; richest native confidence (GCV/Azure/Textract/Mathpix) + the logprob path (OpenAI) and the no-signal case (Claude). |
| `agreement` *(opt)* | `uqlm` (Apache-2.0) · `lm-polygraph` (MIT, heavy) · *primitives:* `rapidfuzz`/`jiwer` (core) | All permissive (vet downloaded NLI/embedding model weights; avoid `lm-polygraph`'s non-commercial COMET extra) | `uqlm` active; `lm-polygraph` heavy (torch) | Reference-free agreement confidence for R1's no-confidence engines: LLM self-consistency (`uqlm`) + the ROVER engine we build on the primitives. |
| `hitl` *(opt)* | `label-studio-sdk` (Apache-2.0) · `argilla` (Apache-2.0) · `cvat-sdk` (MIT) | Permissive clients; servers run **out-of-process** | `label-studio`/`cvat` active; `argilla` feature-frozen | Push uncertain items + model confidence; pull human corrections as new gold; review-queue triage. |
| **Excluded by default** *(license)* | **`surya-ocr`** (non-commercial weights) · **`Levenshtein`/`python-Levenshtein`** (GPL) · **`Potato`** (GPL) · **`Prodigy`** (commercial) | ❌ Not permissive — see license register | — | Not packaged as installable extras. `surya-ocr` only behind an explicit, labelled non-permissive flag for internal research; the GPL fuzzy-distance libs are fully replaced by `rapidfuzz`. |

> A CI **license gate** (e.g. `pip-licenses` / `reuse`) should fail the build if anything in `core`–`hitl` resolves to GPL/AGPL/non-commercial — `python-Levenshtein` is a classic transitive contaminant via legacy `fuzzywuzzy` code, and `TorchCP`/`surya-ocr` carry their copyleft/non-commercial terms only in repo files (not PyPI metadata), so scanners alone will miss them.

---

## License register — the six things to keep out of the permissive build

All verified against primary sources (repo `LICENSE` files / PyPI license expressions) in mid-2026.

| Library | Tier | License | Why it's a problem | What to do instead |
|---|---|---|---|---|
| `Levenshtein` | surface | **GPL-2.0-or-later** | Program copyleft; disqualifying for permissive redistribution. | `rapidfuzz.distance.Levenshtein` (MIT) — identical `distance`/`ratio`/`editops`/`median`, faster. |
| `python-Levenshtein` | surface | **GPL-2.0-or-later** | Same; *and* pulled in transitively by legacy `fuzzywuzzy`. | `rapidfuzz` + a CI license gate to block the transitive pull. |
| `TorchCP` | calibration-graph | **LGPL-3.0** (repo only; PyPI metadata blank) | Weak copyleft: fine imported/dynamically linked, but **patching or static-bundling triggers copyleft**. | Quarantine in the optional `calibration-graph` extra, import-only, never vendor/modify. Permissive fallback: `torch-uncertainty` / `MAPIE` / `crepes` for non-graph paths. |
| `surya-ocr` | ocr *(R6-surveyed; not in R1's 15)* | code Apache-2.0; **weights modified AI-Pubs Open RAIL-M** | Highest-accuracy OCR, but **weights are non-commercial** above a $5M funding/revenue threshold. PyPI's "Apache-2.0" classifier covers only the code. (This flag originates in R6's survey — R1 does not profile surya.) | Keep out of `ocr-local`; expose only behind an explicit, labelled non-permissive engine flag for internal research/benchmarking under the threshold. |
| `Potato` | hitl | **GPL-3.0-or-later** (relicensed in v2.6.0) | Strong copyleft would impose GPL on our distribution if linked. | Run as a separately-deployed standalone app over HTTP only (no code linkage), or prefer Label Studio. |
| `Prodigy` | hitl | **Commercial / proprietary** (pay-once, closed source) | No open-source tier; cannot ship as a dependency. | Study its `prefer_uncertain` active-learning sorter as a *design reference*; build the equivalent triage on Label Studio / our own queue. |

Two **metadata caveats** that fool automated scanners but are *not* blockers: `zss` is **BSD-3-Clause** (not MIT; PyPI omits the classifier — read the vendored `LICENSE`), and the PubTabNet repo's top-level `LICENSE.md` is **CDLA-Permissive-1.0** while only its `src/metric.py` carries the **Apache-2.0** header we actually rely on — cite the file header, not the repo license. `nonconformist` is genuinely **MIT** per its repo despite a blank PyPI license field (but it's ~9 years stale — avoid on maintenance grounds, not license).

---

## Architecture sketch

A thin **facade** (`ie_eval`) over **pluggable, function-injected strategies**, organized as a **strategy/plugin registry**. Favoring functional composition over inheritance: metrics, normalizers, confidence sources, calibrators, and validators are *callables* conforming to `typing.Protocol` signatures, injected as keyword-only parameters with smart defaults (progressive disclosure). Data is carried in `dataclasses`; containers use `collections.abc` interfaces. The conceptual map's two-layer separation is structural, not incidental.

### Layer A — graph grammar / schema (SSOT: *what we extract and what matters*)

The typed-graph target: node/edge/attribute **types**, their value **domains**, and — critically — the **importance weights** that make metrics cost-sensitive. This is the single source of truth shared by the offline and online paths and consumed by constrained decoders (`model_json_schema()` → `outlines`/`xgrammar`).

```python
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any

@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str                       # 'string'|'number'|'date'|'currency'|'enum'|...
    importance: float = 1.0         # ATTRIBUTE-level cost weight (how much an error here costs)
    domain: tuple[Any, ...] = ()    # enum members / (lo, hi) range / regex — read by validators
    normalizer: str | None = None   # registry key: canonicalize before comparison

@dataclass(frozen=True)
class NodeType:
    name: str
    fields: Mapping[str, FieldSpec]
    importance: float = 1.0         # NODE-level cost weight

@dataclass(frozen=True)
class EdgeType:
    name: str
    src: str                        # source NodeType name
    dst: str                        # target NodeType name
    importance: float = 1.0         # EDGE-level cost weight

@dataclass(frozen=True)
class GraphGrammar:                 # the schema (SSOT); cost weights live ON the types
    node_types: Mapping[str, NodeType]
    edge_types: Mapping[str, EdgeType]
    # kind-aware default weight lookups for cost-sensitive metrics:
    def node_cost(self, name: str) -> float: ...
    def edge_cost(self, name: str) -> float: ...
    def field_cost(self, node: str, field: str) -> float: ...
```

### Layer B — extraction-and-verification metadata (rides alongside values; never pollutes the schema)

Per-extraction runtime annotations: provenance for HITL drill-down, raw and calibrated confidence, validator findings, and the selective-prediction decision. The frozen `GraphGrammar` is **referenced, never mutated** — a separate `AnnotatedExtraction` container keys per-value metadata by graph **path**, so the schema stays the SSOT while provenance/confidence/findings/decisions ride strictly *alongside* it. The same (grammar, estimates) pair is used whether benchmarking (vs gold) or estimating quality (vs consensus/expectation).

```python
@dataclass(frozen=True)
class NodePath:                    # addresses a node (and optional field) in an extracted graph
    node_id: str
    node_type: str                 # a key into GraphGrammar.node_types
    field: str | None = None       # a key into NodeType.fields, or None for the whole node

@dataclass
class Provenance:
    engine: str
    source_span: tuple[int, int] | None = None     # char offsets into the raw text
    bbox: Any = None                               # geometry for the image overlay (R1)
    raw_transcripts: Sequence[str] = ()            # multiple raw OCR outputs for adjudication

@dataclass
class FieldEstimate:                # one extracted value + its verification metadata
    value: Any
    raw_signals: dict[str, float] = field(default_factory=dict)  # intrinsic conf, logprob, agreement...
    confidence: float | None = None                # calibrated P(correct) once through a Calibrator
    findings: tuple["Finding", ...] = ()           # validator outputs (flag vs correct)
    provenance: Provenance | None = None
    decision: str | None = None                    # 'accept' | 'flag' | 'block' (selective prediction)

@dataclass
class AnnotatedExtraction:          # Layer B rides ALONGSIDE Layer A — never inside it
    grammar: GraphGrammar                          # the unmutated SSOT schema, by reference
    estimates: Mapping[NodePath, FieldEstimate]    # metadata keyed by graph path
```

### Pluggable strategies (functions-as-parameters, typed via `Protocol`)

Every swappable behavior is a callable. Defaults are resolved from a registry by name; third parties register new backends via entry points (open–closed). A missing optional extra raises an *actionable* error ("install `ie-eval[ocr-local]`"), separated from core logic via a small decorator.

```python
from typing import Protocol, runtime_checkable
from collections.abc import Callable, Iterable

@dataclass(frozen=True)
class TypeRef:                          # identifies what a cost applies to, so the
    kind: str                           #   injected cost fn can read the SSOT's weights:
    name: str                           #   'node'|'edge' type name, or 'field's owning node
    field: str | None = None            #   field name when kind == 'field'

@runtime_checkable
class Metric(Protocol):                 # reference-based: compare pred vs gold
    # grammar carries the cost weights; field/string metrics may ignore it
    def __call__(self, pred: Any, gold: Any, *, grammar: GraphGrammar | None = None) -> float: ...

@runtime_checkable
class Validator(Protocol):              # reference-free: emit findings for a value
    def __call__(self, value: Any, *, spec: FieldSpec) -> Iterable["Finding"]: ...

Normalizer       = Callable[[str], str]                              # canonicalize before compare
ConfidenceSource = Callable[[FieldEstimate], Mapping[str, float]]    # reference-free signal(s)
# the cost fn reads the SSOT: a typed reference into the grammar -> weight
CostWeight       = Callable[[GraphGrammar, TypeRef], float]          # default: read *.importance

@runtime_checkable
class Calibrator(Protocol):
    def fit(self, scores: Sequence[float], correct: Sequence[bool]) -> "Calibrator": ...
    def __call__(self, raw_score: float) -> float: ...               # -> calibrated P(correct)

@runtime_checkable
class SelectivePolicy(Protocol):        # risk-coverage operating point -> a decision
    def __call__(self, confidence: float) -> str: ...                # 'accept'|'flag'|'block'
```

### The facade — progressive disclosure (one-liner defaults; full control underneath)

```python
def score(pred, gold, *, grammar: GraphGrammar | None = None,
          metric: Metric | str | None = None,
          normalize: Normalizer | None = None,
          weights: CostWeight | None = None) -> "Report":
    """Reference-based. Picks the metric by output-object type unless one is given:
    str -> CER/WER (jiwer) or chrF (sacrebleu); fields -> nervaluate span-F1;
    table -> TEDS (table-recognition-metric); graph -> cost-weighted GED (networkx)."""
    metric = _resolve_metric(metric, pred, gold, grammar)     # SSOT registry dispatch
    ...

def estimate_quality(extraction, *, sources: Sequence[ConfidenceSource] = (),
                     calibrator: Calibrator | None = None,
                     validators: Sequence[Validator] = (),
                     policy: SelectivePolicy | None = None) -> "QualityReport":
    """Reference-free. Gather signals -> calibrate -> validate -> decide accept/flag/block."""
    ...
```

`score(pred, gold)` with no other arguments Just Works; an advanced caller passes an explicit cost-weighted graph metric, a domain normalizer, a fitted calibrator, and a conformal policy. Both entry points operate on the *same* Layer-A object.

### How a metric, a confidence source, and a validator plug in

- **A metric** is resolved by object type and wraps a borrowed engine. For the typed graph it is the one genuine *build*: a cost-weighted GED that supplies `networkx.graph_edit_distance` with `node_subst_cost`/`edge_subst_cost` callables reading `NodeType.importance` / `EdgeType.importance` (and `FieldSpec.importance` for attribute-level substitutions), via the injected `CostWeight(grammar, TypeRef)` — so "two extra digits on a monetary amount" outweighs "a misspelled city." Tables route to `apted` (override `Config.rename` / `PerEditOperationConfig(del,ins,ren)` for per-cell-type weights); strings to `rapidfuzz`/`jiwer`/`sacrebleu`; fields to `nervaluate` (aggregate its per-type P/R/F1 with our weights).
- **A confidence source** maps an extractor's native signal (R1) into `FieldEstimate.raw_signals`: per-word confidence (Tesseract/GCV/Azure/Textract/Mathpix), token logprobs (`compute_transition_scores` for TrOCR; OpenAI logprobs with the R1 reliability caveats; Mistral's logprob-derived word confidence), or — for engines that emit nothing (Claude, OCR.space, pix2tex) — a *built* external layer: lexicon agreement, geometry outliers, and **ROVER-style cross-engine/self-consistency agreement**. Those raw signals pass through a `Calibrator` (`netcal` temperature scaling) and a `SelectivePolicy` (`MAPIE`/`crepes` conformal) to a calibrated `confidence` and a `decision`.
- **A validator** reads `FieldSpec` and emits `Finding`s, distinguishing *correct* (coerce: pydantic/`pydantic-extra-types`) from *flag* (pydantic/`jsonschema`/`pandera`). The reference-free **cross-field/cross-source consistency** validators (totals reconcile, dates ordered, referential integrity, cross-source triangulation) are a build — declarative and composable on top of the schema. Constrained generation (`outlines`/`xgrammar`) is validation-on-*generate*: the same schema, enforced by construction.

### The OCR facade and the R1 → confidence-source mapping

OCR is one (noisy) front-end among many (PDF, DOCX, XLSX, DB). The uniform `OcrResult` normalizes R1's 15 engines + the two local additions (`python-doctr`, `surya-ocr`) into one object — carrying the *union* of granularities R1 documented (per-symbol/word/line/region), a single geometry convention (R1 found engines disagree: axis-aligned rect vs 4-point polygon, pixel vs normalized 0–1, top-left vs ocrmac's bottom-left origin), and an explicit **provenance-capability flag** because R1 showed three engines cannot give real provenance (Mistral's text locator is a char offset, Claude/OpenAI geometry is model-*guessed*).

```python
@dataclass
class OcrUnit:
    text: str
    bbox: Any = None                 # NORMALIZED to one convention (top-left origin, 0..1) in the adapter
    confidence: float | None = None  # raw, engine-native scale (None if the engine emits none)
    logprob: float | None = None     # generative engines only (TrOCR/OpenAI; Mistral pre-derived)
    level: str = "word"              # 'symbol'|'word'|'line'|'region' — engines differ (R1)

@dataclass
class OcrResult:
    text: str
    units: Sequence[OcrUnit]
    engine: str
    confidence_grain: str            # finest native grain available, from R1: e.g. 'symbol'|'word'|'line'|'none'
    has_real_provenance: bool        # False for Mistral text spans / Claude / OpenAI (model-guessed)
    calibrated: bool = False         # True only for Mathpix (R1: the sole calibrated emitter)

OcrBackend = Callable[[bytes], OcrResult]   # tesseract, rapidocr, doctr, gcv, textract, ... all conform
```

The table below is the concrete R1→R6 bridge: it routes each engine's native signal to a `ConfidenceSource` path and a calibration treatment. Three paths emerge — **(i) intrinsic confidence → calibrate**, **(ii) token logprobs → recover then calibrate**, **(iii) no signal → built agreement layer** (`uqlm` self-consistency for VLMs; the ROVER engine for multi-OCR voting).

| R1 engine | Finest native confidence | Logprobs | Real provenance (geometry) | Table structure | → Confidence-source path |
|---|---|---|---|---|---|
| Google Cloud Vision | per-symbol→block `[0,1]` (uncalibrated) | — | yes — bbox/normalized vertices per symbol | tag only (no cells) | **(i)** calibrate (`netcal`); rich provenance overlay |
| Azure Document Intelligence | word/cell/field/KV `[0,1]` (no line; uncalibrated) | — | yes — polygon per word | yes — rows/cols/spans JSON | **(i)** calibrate; table→TEDS/GriTS |
| AWS Textract | per-block `0–100` (word/line/cell/KV; uncalibrated) | — | yes — bbox + polygon (norm) | yes — TABLE/CELL/MERGED_CELL | **(i)** calibrate; table→TEDS |
| Mathpix | `confidence`+`confidence_rate` `[0,1]` global/line/word — **calibrated** | — | yes — `cnt` polygon | yes — mmd/HTML/TSV | **(i)** use directly + engine gating; no calibration needed |
| Tesseract | per-symbol/word/line `0–100` (uncalibrated) + N-best | — | yes — bbox every level (axis-aligned) | no | **(i)** calibrate + aggregate; alternatives as extra signal |
| PaddleOCR | per-line `rec_score` + region (uncalibrated) | — | yes — quads + AABB; reading order | yes — `pred_html` (TEDS-ready) | **(i)** calibrate; table→TEDS |
| RapidOCR | per-line `[0,1]` (+per-word opt-in; uncalibrated) | — | yes — 4-pt polygon | via add-on pkgs | **(i)** calibrate |
| EasyOCR | per-region scalar (uncalibrated) | — | yes — 4-pt polygon | no | **(i)** calibrate (weak signal) |
| ocrmac / Apple Vision | per-line (often **quantized** ~{.3,.5,1}) | — | yes — normalized bbox (bottom-left) | no | **(iii)** quantized → lean on agreement/external layer |
| TrOCR | none native; per-token recoverable | **yes** — `compute_transition_scores` | no — single-line crop | no | **(ii)** recover logprobs → calibrate |
| Mistral OCR | opt-in word/page (logprob-derived, monotonic) | no raw | **limited** — image bbox only; text = char offset | yes — markdown/HTML | **(ii)** use derived confidence → calibrate; weak provenance |
| OpenAI GPT-4o/4.1 | per-token logprob (flaky/empty on **image inputs**; json_schema-strict empties them on the GPT-5.x Responses API — R1) | **yes** (caveated) | no — model-guessed pixels | model-inferred JSON | **(ii)** logprobs *if reliable* else **(iii)** `uqlm` self-consistency |
| Claude Vision | **none** (no logprobs, no confidence) | **no** | no — model-guessed | model-inferred JSON | **(iii)** `uqlm` self-consistency / LLM-judge / cross-model / verifier |
| OCR.space | **none** | — | yes — word px boxes | text/markdown only | **(iii)** ROVER agreement + geometry/lexicon outliers |
| pix2tex / LaTeX-OCR | **none** | no (patch) | no | no (LaTeX content) | **(iii)** sampling agreement / external re-scorer |

This makes the online-QE half concrete: paths (i) and (ii) reuse the calibration/conformal tier (`netcal` → `MAPIE`/`crepes`); path (iii) is the one genuine build — `uqlm` (borrow) for VLM self-consistency, and a small ROVER engine (build, on `rapidfuzz`/`jiwer` alignment) for multi-OCR voting — plus a geometry-aware adapter that projects a whole-response `[0,1]` uncertainty back onto per-unit `OcrResult.confidence` (no library is geometry-aware).

### System dependencies

Heavy/native backends are guided dynamically via a `check_requirements(engine=...)` helper: the `tesseract` binary + leptonica for `tesserocr`/`pytesseract`, platform-gating for `ocrmac` (macOS-only), and credential/endpoint checks for each cloud SDK — each raising an actionable message (install command / link / `export` hint) rather than an opaque `ImportError`.

---

## Prioritized build / borrow / wrap list

### Reuse as-is (≈30 libraries — import and call)
`rapidfuzz` (edit distance, fuzzy, weighted Levenshtein, `cdist`), `jiwer` (CER/WER + alignment + normalization transforms), `sacrebleu` (chrF/TER + bootstrap significance), `nervaluate` (span/partial-overlap field P/R/F1), `apted` & `zss` (tree edit distance with cost hooks), `networkx` (graph edit distance with cost callables), `table-recognition-metric` (TEDS), `pydantic` + `pydantic-extra-types` (schema spine + typed fields), `jsonschema` / `fastjsonschema` (declarative validation), `pandera` (aggregate frame checks), `netcal` (temperature scaling + ECE + reliability diagrams), `scikit-learn` (Platt/isotonic baseline), `MAPIE` + `crepes` + `puncc` (conformal / risk-control / drift martingales), `python-doctr` (richest native OCR output object — also a model for our payload schema), `tesserocr`, `rapidocr`, the seven **cloud/VLM SDKs** (`google-cloud-vision`, `boto3`+`amazon-textract-response-parser`/`-textractor`, `azure-ai-documentintelligence`, `mistralai`, `mpxpy`, `openai`, `anthropic` — all permissive, behind the `OcrResult` facade), **`uqlm`** (black-box LLM self-consistency confidence), `label-studio`(+SDK) & `cvat` (HITL).

### Wrap behind a thin adapter (own the seam, swap the backend)
- **OCR engines → one `OcrResult`** (`text` + per-unit `geometry` + `confidence | None` + `logprob`), normalizing R1's 15 engines (local + cloud) — plus `python-doctr` and `surya-ocr` — and the coordinate conventions (normalized vs pixel, top-left vs ocrmac's bottom-left). Per-engine adapter gotchas to encapsulate (from the verified survey): GCV `word` has no `.text` (reassemble from `symbols`); Textract is cleanest off the **raw `boto3` `['Blocks']`** (the `trp` parser is a 2024 convenience); `azure-ai-documentintelligence` is the package that **replaced `azure-ai-formrecognizer`**; `mistralai` is a **namespace package** (`from mistralai.client import Mistral`); Claude's adapter must hard-set `confidence=None, geometry=None`.
- **Calibration/conformal → a `Calibrator` / `SelectivePolicy` facade** over `netcal` + `MAPIE`/`crepes`, so the copyleft `TorchCP` graph-conformal path is an isolated, swappable plugin and never a hard dependency.
- **Constrained generation → a `ConstrainedDecoder`** over `outlines`/`xgrammar`, fed the pydantic/JSON-Schema SSOT, so the volatile pre-1.0 APIs are pinned behind one seam.
- **Agreement/uncertainty → a `ConfidenceSource`** over `uqlm` (black-box self-consistency for VLMs) and optionally `lm-polygraph` (white-box logit UE for self-hosted HF VLMs — heavy; never enable its non-commercial COMET extra), so R1's no-confidence engines get a `[0,1]` signal behind the same seam.
- **HITL → a `ReviewQueue` adapter** over Label Studio's `create_predictions`/export/webhooks (and Argilla's `Suggestion(score=)`/`Response` model as the design template), so the push-uncertain / pull-as-gold loop is decoupled from any one tool.

### Build ourselves (the connective tissue no library owns)
1. **The two-layer eval object** — `GraphGrammar` (schema + importance weights) and the `FieldEstimate`/`Provenance` metadata layer. The integrating abstraction; everything else plugs into it.
2. **Cost-sensitive, type-aware metrics** — supply cost functions to `apted`/`zss`/`networkx`; aggregate `nervaluate` per-type outputs with weights; a normalized, partial-credit graph-similarity wrapper. *(The headline metric gap — confirmed unbuilt across the whole survey.)*
3. **The ROVER engine + the geometry-aware uncertainty mapping** — *verified gap:* there is **no maintained permissive pip ROVER** (NIST SCTK is C/GPL), so N-way alignment+voting + per-token agreement confidence is a small build on `rapidfuzz`/`jiwer` editops; and since `uqlm`/`lm-polygraph` score *whole responses* and are geometry-blind, the adapter that projects their `[0,1]` onto per-unit `OcrResult.confidence` is also ours.
4. **Field/graph-level calibration & conformal mapping** — cast the structured score into the per-field probability/score vectors `netcal`/`MAPIE` consume; risk–coverage operating-point selection over the graph object (the libraries calibrate classifiers, not typed graphs).
5. **Cross-field / cross-source consistency validators** — reference-free integrity constraints (totals reconcile, dates ordered, referential integrity) and cross-source triangulation; declarative and composable on the schema.
6. **Provenance linking & review-queue triage** — value → source span/bbox/raw-transcript wiring for the HITL overlay (honoring `OcrResult.has_real_provenance` — disabled for Mistral/Claude/OpenAI), and ordering the human queue by (cost × uncertainty).
7. **The harness** — stratified sampling/slicing, per-slice reporting, golden-set regression, double-annotation + IAA — orchestration is ours; the metrics inside it are borrowed.
8. **A canonicalization registry** — number/date/unicode folding before scoring; reuse `jiwer.transforms` + `ftfy` + `dateparser` as the primitives, but the registry and the per-type normalizer dispatch are ours.

---

## Detailed tier inventory (verified)

Each tier below is the verified library survey — license, last release, maintenance, the exact API surface, and reuse/wrap/build call — with inline links. License/version/maintenance claims were re-checked against live PyPI + GitHub primary sources in mid-2026.

### Surface & Edit-Distance Metrics + Normalization

This tier covers character/word-level surface scoring of structured-field string values and the normalization pipeline that precedes it. All license/version/API claims below were verified against live PyPI and GitHub in June 2026.

**Adopt (permissive, maintained):**
- **rapidfuzz** (MIT, v3.14.5 Apr 2026) — the workhorse. Fast C++ backend, `process.cdist` for vectorized candidate→reference score matrices, and `distance.Levenshtein.distance(weights=(ins,del,sub))` for cost-weighted edit distance (the only seed lib giving this out of the box). Canonical base for the tier. **reuse-as-is.**
- **jiwer** (Apache-2.0, v4.0.0 Jun 2025, repo jitsi/jiwer) — cleanest off-the-shelf WER/CER + alignment + normalization pipeline. `process_words`/`process_characters` return substitution/insertion/deletion breakdowns and alignment chunks for error-typing; `transforms.Compose` doubles as a reusable normalization harness. Depends only on rapidfuzz (MIT). Note kwargs renamed to `reference_transform`/`hypothesis_transform` in 3.x+. **reuse-as-is.**
- **sacrebleu** (Apache-2.0, v2.6.0 Jan 2026) — adopt specifically for **chrF/chrF++** (character-n-gram F-score, robust to OCR-ish noise) and **TER**, plus bootstrap significance testing for benchmark A/B. BLEU itself is marginal for IE field values. **wrap.**

**LICENSE-DISQUALIFIED — do NOT ship in a permissive-downstream product:**
- **Levenshtein** and **python-Levenshtein** — both **GPL-2.0-or-later** (VERIFIED on live PyPI License Expression metadata; v0.27.3 Nov 2025). Precise license is GPLv2+ (program copyleft), not AGPL, but it is equally disqualifying. `python-Levenshtein` is a classic transitive contaminant via legacy `fuzzywuzzy` code. Every function they provide (distance, ratio, editops, median) is available under MIT via **rapidfuzz** — there is zero reason to take the GPL dependency. Add a CI license-gate to block both. **avoid.**

**Conditional / niche:**
- **torchmetrics** (Apache-2.0, v1.9.0 Mar 2026) — unified `torchmetrics.text` CER/WER/EditDistance with op-weights, but pulls in PyTorch. Only adopt if torch is already a dependency. **wrap, else avoid for this tier.**
- **editdistance** (MIT, v0.8.1 Feb 2024) — minimal edit distance over arbitrary token/object sequences. >2yr stale but trivial scope; largely redundant with rapidfuzz. **wrap (niche) / build-ourselves.**
- **textdistance** (MIT, v4.6.3 Jul 2024) — ~30 algorithms under one normalized API; handy buffet for metric experimentation, slow pure-Python by default. ~23mo stale flag. **wrap (optional, non-hot-path).**

**Avoid (indirection or abandonment):**
- **evaluate** (Apache-2.0, v0.4.6 Sep 2025) — thin Hub-fetch wrappers over jiwer/sacrebleu; adds runtime network dependency + arbitrary-code-execution surface and heavy deps. Import the underlying libs directly. **avoid.**
- **fastwer** (MIT, v0.1.3 **Mar 2020** — CORRECTED from draft's 2024) — ~6 years stale, single-author, reported pip-install/build failures. jiwer covers the same WER/CER with alignment and active maintenance. **avoid.**

**Bottom line:** rapidfuzz + jiwer + sacrebleu(chrF/TER) cover the entire tier under permissive licenses with no GPL drag and no heavy deps. The two GPL `*Levenshtein` packages are the only landmines and are fully replaceable by rapidfuzz.

### Structured / Field / Graph Metrics

Tools for scoring structured outputs against references: tree-edit-distance (TEDS/GriTS) for tables, graph-edit-distance for typed graphs, and span/entity F1 for fields/slots. **All libraries in this tier are permissive (Apache-2.0 / MIT / BSD-3-Clause) — no GPL/AGPL/non-commercial/paid traps were found.** Two verification corrections were applied: **zss is BSD-3-Clause, not MIT** (still permissive), and the **PubTabNet repo's top-level LICENSE.md is CDLA-Permissive-1.0** while only its `src/metric.py` code carries the Apache-2.0 header we rely on.

**Cost-weighting is the recurring gap.** None of these metrics ship per-type / importance-weighted edit costs out of the box, but three give first-class extension hooks we can supply cost functions to:
- **apted** (MIT, pure-Python): subclass `Config.rename/delete/insert` — the supported hook for per-type weighted *tree* edit costs. Our primary weighted-tree build-block. Stale (2017) but algorithmically frozen, low risk.
- **zss** (BSD-3-Clause): lighter alternative with `label_dist` + `insert/remove` cost callables — simpler ceremony than APTED, fallback/cross-check. Slower on large trees. Stale (2018).
- **networkx** (BSD-3-Clause, active 3.6.1/2025-12): `graph_edit_distance(..., node_subst_cost=, edge_subst_cost=, ...)` callables encode type-aware costs directly — the closest off-the-shelf cost-weighted *graph* metric. Caveat: GED is NP-hard; rely on `timeout` + `optimize_graph_edit_distance` approximations. We build normalization, large-graph matching, and partial-overlap semantics on top.

**Table metrics:** `table-recognition-metric` (Apache-2.0, active 0.0.6/2025-12) is the clean pip path to **TEDS** — reuse-as-is, pin the pre-1.0 version. The canonical **PubTabNet** `metric.py` (Apache-2.0 code; `TEDS(structure_only, n_jobs, ignore_nodes).evaluate(pred, true)`) is worth vendoring only to customize the cost model. **GriTS** (microsoft/table-transformer, MIT) is the better grid-topology table metric but is repo-only and coupled to training loaders — extract `grits.py`. All three use uniform costs; per-cell-type weighting is a build-on-top.

**Field/slot metrics:** `nervaluate` (MIT, active 1.2.1/2026-03) is the reuse pick for partial/span-overlap P-R-F1 with COR/INC/PAR/MIS/SPU buckets — but it weights every entity equally (aggregate its per-type outputs with our weights). `seqeval` (MIT, stale 2020) covers strict-boundary BIO/IOBES NER F1 only. `spaCy.scorer.Scorer` (MIT library; note per-model license caveats) is worth pulling in only if a spaCy pipeline is already a front-end.

All of these operate on table-/sequence-/graph-shaped sub-payloads, not on a unified typed-graph object — the type-aware, cost-weighted aggregation across sub-payloads is ours to build, with apted/zss/networkx cost hooks as the engine.

### Calibration, Conformal & Selective Prediction

This tier covers the two evaluation paths: post-hoc **calibration** (turning raw confidences into trustworthy probabilities) and **conformal / selective prediction** (turning calibrated scores into coverage-guaranteed accept/flag/block decisions). All license claims below were re-verified against live PyPI metadata and GitHub LICENSE files in mid-2026.

**Calibration**
- **scikit-learn** (`scikit-learn`, BSD-3-Clause, permissive; latest 1.9.0) — the floor, already a transitive dep. Use `CalibratedClassifierCV` (Platt/isotonic) and `calibration_curve` only. No temperature scaling, no ECE, no conformal. **reuse-as-is.**
- **netcal** (`netcal`, Apache-2.0, permissive; 1.4.0 2026-04-16) — most complete pure-calibration toolkit (TemperatureScaling, ECE/ACE/MCE/NLL/QCE, reliability diagrams). **wrap** as the default calibration backend. **Maintenance caveat:** 1.4.0 ended a ~20-month dormancy (prior 1.3.6 was 2024-08) and its changelog literally says it "makes netcal usable again." Recently revived, single-maintainer — monitor.

**Conformal / Selective Prediction (permissive core)**
- **MAPIE** (`mapie`, BSD-3-Clause, permissive; 1.4.1 2026-06-08) — most mature; split/cross/CQR/EnbPI plus RCPS/CRC risk control that maps directly onto field-level accept/flag/block. **reuse-as-is / wrap.** Pin: the 0.x→1.x API break is real.
- **crepes** (`crepes`, BSD-3-Clause, permissive; 0.9.1 2026-06-12) — complements MAPIE with conformal predictive systems (CDFs), Mondrian/class-conditional coverage, and conformal **test martingales for online drift monitoring**. **wrap.** Pre-1.0, pin.
- **torch-uncertainty** (`torch-uncertainty`, Apache-2.0, permissive; 0.12.1 2026-06-17) — PyTorch-native temperature/vector/matrix scaling + APS/RAPS conformal + coverage metrics in one permissive package; the copyleft-free deep-model alternative to TorchCP. **wrap (selectively).** Pre-1.0.
- **puncc** (`puncc`, MIT via PyPI OSI classifier, permissive; 0.9.2 2026-06-20) — overlaps MAPIE/crepes; distinctive for conformal **anomaly detection** (SplitCAD) and **object-detection** (SplitBoxWise), useful for noisy OCR/detection front-ends. Import path `deel.puncc`. **reuse/wrap (optional).**

**Conformal for structured/graph outputs — license-flagged**
- **TorchCP** (`torchcp`, **LGPL-3.0, weak-copyleft**; 1.2.1 **2025-10-14**) — the ONLY surveyed library with first-class **graph-node and LLM-sequence conformal prediction**, the most direct fit for our typed-graph eval object. **wrap-with-caution.** Two flags: (1) **LGPL-3.0** — fine imported/dynamically linked, but patching the source or static bundling triggers copyleft; isolate as an optional import-only plugin, never vendor/modify. (2) The LGPL declaration lives only in the repo LICENSE/LICENSE.GPL files (GitHub detects LGPL-3.0/GPL-3.0); **PyPI metadata carries no license field**, so scanners may flag it "unknown." Latest release is ~8 months old (not a 2026 release as an earlier draft implied) — maintained but slowing; keep MAPIE/crepes/torch-uncertainty as the copyleft-free fallback for non-graph paths.

**Avoid**
- **nonconformist** (`nonconformist`; repo LICENSE = MIT, but PyPI metadata = UNKNOWN; last release 2.1.0 **2017-06-20**) — the historical original, now ~9 years stale and superseded on every axis by MAPIE/crepes/puncc. The license is genuinely MIT per the repo (the "unknown" is just a packaging-metadata gap), but the ~9-year staleness alone disqualifies it. **avoid** — reference only.

*Net:* permissive default stack = **netcal** (calibration) + **MAPIE/crepes/puncc** (conformal) + **torch-uncertainty** (PyTorch deep-model calibration/conformal). **TorchCP** is the unique graph/sequence-conformal capability but must be quarantined as an optional LGPL plugin with a permissive fallback path.

### Validation, Constrained Generation & Post-OCR Correction

This tier covers three jobs: (a) **validating** typed structured payloads / graph nodes against a contract, (b) **constraining generation** so structured outputs are valid by construction, and (c) **correcting** what can be safely coerced while **flagging** the rest for human-in-the-loop review. Licensing across the whole tier is clean — every library below is **permissive** (MIT / BSD / ISC / Apache-2.0), with **no GPL, AGPL, non-commercial, or paid-only traps** in the library code. (The one standing caveat: `outlines` and `xgrammar` are permissive *as code*, but the **model weights / hosted APIs** they drive carry their own separate licenses — keep those out of the library license accounting.)

**Schema / validation core (the SSOT bridge).** `pydantic` (MIT, 2.13.4 — May 6 2026) is the spine: `model_json_schema()` is the single source of truth fed to every constrained decoder, and `ValidationError.errors()` is the per-field defect stream that drives flag-vs-correct routing. `jsonschema` (MIT, 4.26.0 — Jan 7 2026) covers the contract-as-JSON-Schema case with path-addressable `iter_errors`; `fastjsonschema` (BSD-3, 2.21.2 — Aug 14 2025) is the throughput drop-in for batch eval loops (the closest to the staleness line — ~10 months old, re-check before adoption). `pandera` (MIT, 0.32.0 — Jun 19 2026) handles the flatten-to-frames aggregate reporting view with `lazy=True` / `failure_cases`. `pydantic-extra-types` (MIT, 2.11.1 — Mar 16 2026) supplies cheap type-aware validators (phones, country/currency codes, coordinates) that map directly onto graph-node typing.

**Heavyweight / overlap (use sparingly).** `great-expectations` (Apache-2.0, 1.18.1 — Jun 11 2026) is clean on license but heavy and opinionated — pull in only for its Data Docs / run-history UI, otherwise `pandera` + custom reporting is lighter. `cerberus` (**ISC**, **1.3.8 — Nov 6 2025**; the draft's "1.3.7, Apr 2025, unconfirmed" was wrong) and `voluptuous` (BSD-3, 0.16.0 — Dec 18 2025) both work and both do light correction via `normalized()`/`Coerce(...)`, but overlap fully with pydantic — **avoid as new dependencies** unless pure-dict schemas are specifically wanted.

**Constrained generation (validation-on-generate).** `outlines` (Apache-2.0, 1.3.0 — May 13 2026) is the primary engine: `outlines.Generator(model, output_type)` (verified to exist in the v1 rewrite) forces output to a pydantic model / `outlines.types.JsonSchema` / `Regex` / `CFG` by construction. Note the v1 wrapper is `JsonSchema`, not a `Json(...)` wrapper — pydantic models are passed directly as `output_type`. `xgrammar` (Apache-2.0, 0.2.2 — Jun 11 2026; XGrammar-2 blogged May 4 2026) is the fastest token-masking backend: compile via `GrammarCompiler.compile_json_schema/compile_regex/compile_grammar/compile_builtin_json_grammar`, then mask with `GrammarMatcher.fill_next_token_bitmask` + `apply_token_bitmask_inplace`. **API correction:** the `from_json_schema`/`from_ebnf` factories belong to the `Grammar` class, **not** `CompiledGrammar` as the draft claimed.

Both constrained-gen engines are **evolving (pre-1.0 / post-rewrite)** — pin versions. Everything else in the tier is stable or stable-core. Net: a license-safe, well-maintained tier with no copyleft exposure; the only real risks are API churn in `outlines`/`xgrammar` and mild staleness in `fastjsonschema`.

### HITL Review & Annotation Tooling

Human-in-the-loop review tooling for the confidence -> review -> correct -> pull-as-gold loop. License is the decisive axis: three permissive options anchor the tier (Label Studio, CVAT, INCEpTION — all verified Apache-2.0 or MIT against primary sources), while doccano/doccano-client are MIT but going stale, Argilla is permissive but feature-frozen, Potato is GPL-3.0 (avoid as a linked dependency), and Prodigy is commercial-paid (cannot ship).

**Permissive, actively maintained, recommended:**

- **Label Studio (Community)** — `pip install label-studio` + `label-studio-sdk`. Apache-2.0 (verified PyPI classifier + repo; the SDK PyPI page omits the license field but the repo is Apache-2.0). label-studio v1.23.0 (2026-03-13), SDK v2.0.23 (2026-05-26). Best overall fit for OCR/document review: image+bbox overlay, transcript-span review, and the SDK's `create_predictions` (push model output + confidence) / export (pull corrections as gold) / webhooks (active-learning loop) map directly onto our flow. Reuse-as-is behind a thin adapter. Caution: the restrictive terms belong to the separate, proprietary **label-studio-enterprise** and **Starter Cloud SaaS** — do not depend on those.

- **CVAT** — self-hosted web app + `pip install cvat-sdk` (v2.68.0, 2026-06-10). MIT Community core (LICENSE file verified). Best-in-class image+bbox+polygon overlay with built-in QA/review stages and honeypot ground-truth jobs — strong for OCR-layout verification. License caution: keep to the MIT core; the serverless AI-assist nuclio assets and the Enterprise edition are licensed separately. Weak for structured KV/table/graph payload review.

- **INCEpTION** — Java/Spring web app (no pip package), Apache-2.0 (verified). v40.6 (2026-06-09), active. Strongest here for relation / knowledge-base-linking review and recommender-driven active learning — relevant if our graph payload includes entity linking. Integrate over its REST API or UIMA CAS / WebAnno-TSV files (e.g. dkpro-cassis); heavyweight to deploy and text-centric.

**Permissive but flagged:**

- **Argilla** (formerly Rubrix) — `pip install argilla`, Apache-2.0 (verified). v2.8.0 (2025-03-11); **feature-frozen** — the repo explicitly states no new features, bug fixes/patches only (core team moved to Hugging Face). No release in ~15 months; last commit 2025-08-05. Its `Suggestion(value, score=...)` + `Response` record model is the cleanest match in this tier for confidence-driven triage and pull-as-gold — but wrap behind an adapter given the longevity risk. Text/LLM-centric; weak for image+bbox.

- **doccano** — `pip install doccano` (+ `doccano-client`), MIT (both verified). **Maintenance flag + a corrected fact:** the newest GitHub tag is **v1.8.5 dated 2025-01-11** (the first-pass draft's '2026-01-11' was a year error), and the published PyPI wheel is still **v1.8.4 (2023-07-20)**; doccano-client is **v1.2.8 (2023-06-13)**. So the newest code is ~17 months old and the wheels ~3 years old. TEXT-ONLY (no image/bbox/OCR overlay, no KV/table correction). Wrap only for the transcript-span slice, and verify the wheel-vs-tag gap before pinning.

**Avoid as dependencies:**

- **Potato** — `pip install potato-annotation`. **GPL-3.0-or-later** (verified; v2.6.0 explicitly relicensed to GPL-3.0-or-later). v2.6.0 (2026-06-19), active. Strong copyleft would impose GPL on our distribution if linked — acceptable only as a separately-deployed standalone app over HTTP, with no code-level integration. Research-grade, not image/OCR-focused; little reason to take the risk.

- **Prodigy** — **commercial / proprietary, pay-once lifetime license** (verified on prodi.gy), no open-source tier (free interim research license for degree-granting academic institutions only). Closed-source, license-gated wheel. Cannot be a shipped dependency under a permissive-downstream goal. Its `prefer_uncertain` / `prefer_high_scores` active-learning sorter API is the cleanest in the tier — study it as a design reference only.

Net: anchor on Label Studio (document/OCR review) and CVAT (image-layout review) for permissive, maintained coverage; mine Argilla's and Prodigy's data/active-learning models as design references behind adapters; treat doccano as a stale text-only fallback; keep Potato and Prodigy out of the linked dependency graph for license reasons.

### OCR Engines Behind a Uniform Interface

OCR is one (noisy) front-end among many; this tier collects engines we would normalize behind a single facade that emits a common `(text, geometry, confidence)` record. All claims below were verified live against PyPI and the source repos (mid-2026).

**License bottom line.** Every engine here is permissive (Apache-2.0 / MIT) for both code AND standard weights **except one**: `surya-ocr`. Surya's code is Apache-2.0 but its **model weights are non-commercial** (modified AI Pubs Open RAIL-M; free only for research, personal use, or orgs under $5M funding/revenue). The PyPI `Apache-2.0` classifier reflects only the code and is a trap — flag `surya-ocr` loudly and keep it out of any permissively-licensed shipped product. `paddleocr`'s standard PP-OCR weights are Apache-2.0, but its bundled VLM (PaddleOCR-VL) weights should be license-checked separately before commercial use.

**Tesseract family (permissive, baseline).**
- `pytesseract` (Apache-2.0, 0.3.13 / 2024-08-16): thin CLI shim; `image_to_data(..., output_type=Output.DICT)` yields per-word text+bbox+conf. Maintenance is *slow* (~22 months, normal for a stable shim). Requires the external `tesseract` binary — declare as a system dependency. **wrap.**
- `tesserocr` (MIT binding / Apache-2.0 engine, 2.10.0 / 2026-02-12): in-process Cython binding, GIL released during recognition, finest iterator-level conf+bbox of the Tesseract wrappers. Harder C build (use conda-forge). **wrap** — preferred when throughput matters.

**Neural, fully permissive (incl. weights).**
- `easyocr` (Apache-2.0 incl. weights, 1.7.2 / 2024-09-24): clean `readtext` returning polygon+text+conf. Maintenance *slow* (~21 months — flag). Weights download on first use (offline-cache concern for reproducible eval). **wrap.**
- `rapidocr` (Apache-2.0 incl. ONNX weights, 3.8.4 / 2026-06-15): PP-OCR accuracy without the Paddle dependency; multi-backend (`with_onnx/with_openvino/with_paddle/with_torch`), `RapidOCROutput` with `.boxes/.txts/.scores`. The most license-clean neural engine in tier. v1→v3 API churn — pin. **wrap** (prefer over legacy `rapidocr-onnxruntime`).
- `rapidocr-onnxruntime` (Apache-2.0, 1.4.4 / 2025-01-17): legacy ONNX-only build, superseded. **avoid** unless pinning the old 1.x API.
- `paddleocr` (Apache-2.0 incl. standard weights, 3.7.0 / 2026-06-11): top-tier accuracy, active, but heavy/quirky `paddlepaddle` dependency and 2.x→3.x API churn (`ocr()` vs `predict()`). **wrap**, isolated behind the facade.
- `python-doctr` (Apache-2.0 incl. weights, 1.0.1 / 2026-02-04): richest native structured output — hierarchical `Document` (block>line>word, each with confidence+geometry) and `.export()` to JSON. A strong model to study for our own typed-graph eval payload. **reuse-as-is.**
- `ocrmac` (MIT, 1.0.1 / 2026-01-08): Apple Vision wrapper, zero-cost on macOS dev machines; returns text+conf+normalized bbox. macOS-only — **wrap, platform-gated** (not available on Linux eval servers).

**Pipeline framework (overlaps our own goal).**
- `paddlex` (Apache-2.0, 3.7.1 / 2026-06-11): low-code pipeline layer over PaddlePaddle; conceptually a competitor to the facade we're building and pulls the full Paddle stack. **build-ourselves** — mine for pipeline ideas, reuse `paddleocr` directly.

**SOTA but license-disqualified.**
- `surya-ocr` (code Apache-2.0; **weights non-commercial** RAIL-M, 0.20.0 / 2026-05-27): highest accuracy in tier (single 650M-param VLM in v2; `RecognitionPredictor` now returns `PageOCRResult` blocks with label/html/polygon/bbox/confidence). **AVOID for permissive downstream** — usable only for internal research/benchmarking under the $5M threshold; cannot ship without a paid Datalab license. If used, gate behind an explicit non-permissive engine flag.

### OCR Cloud / VLM Client SDKs — R1's engines behind the uniform interface

The official client SDKs for R1's cloud/VLM engines. **Every one is permissive** (Apache-2.0 / MIT) and actively maintained; the *hosted services* they call are proprietary, pay-per-call, network-only (a runtime/cost concern, not a license one). Verified live, mid-2026.

| SDK | Install | License (SDK) | Last release | Maint. | Rec. |
|---|---|---|---|---|---|
| `google-cloud-vision` | `google-cloud-vision` | Apache-2.0 | v3.14.0 · 2026-05-07 | active | reuse-as-is |
| `AWS Textract` | `boto3 amazon-textract-response-parser amazon-textract-textractor` | Apache-2.0 | v1.43.34 · 2026-06-19 | active | reuse-as-is |
| `azure-ai-documentintelligence` | `azure-ai-documentintelligence` | MIT | v1.0.2 · 2025-03-27 | active | reuse-as-is |
| `mistralai` | `mistralai` | Apache-2.0 | v2.4.13 · 2026-06-19 | active | reuse-as-is |
| `mpxpy` | `mpxpy` | MIT | v0.0.20 · 2026-01-26 | active | wrap |
| `openai` | `openai` | Apache-2.0 | v2.43.0 · 2026-06-17 | active | wrap |
| `anthropic` | `anthropic` | MIT | v0.111.0 · 2026-06-18 | active | reuse-as-is |

**Signal access (verified) — how each feeds `OcrResult`:**
- **`google-cloud-vision`** (Apache-2.0, 3.14.0): `document_text_detection()` → `full_text_annotation.pages[].blocks[].paragraphs[].words[].symbols[]`, each with `.confidence [0,1]` and `.bounding_box.(normalized_)vertices`. `word` has **no** `.text` — reassemble from `symbols[].text` honoring `detected_break`. Use `DOCUMENT_TEXT_DETECTION` (dense), not `TEXT_DETECTION`.
- **AWS Textract** (`boto3` Apache-2.0, near-daily; `amazon-textract-response-parser`/`trp` 2024, `-textractor` 2025): adapt off the **raw `['Blocks']`** — `block['Confidence']` (0–100) + `block['Geometry']{BoundingBox, Polygon}`, `BlockType` in PAGE/LINE/WORD/TABLE/CELL/MERGED_CELL/KEY_VALUE_SET/SELECTION_ELEMENT. The parsers are optional convenience.
- **`azure-ai-documentintelligence`** (MIT, 1.0.2 GA — **replaces `azure-ai-formrecognizer`**): `begin_analyze_document().result()` → `pages[].words[].{content,polygon,confidence}`, plus `tables`, `key_value_pairs`, `documents[].fields`; `output_content_format='markdown'`.
- **`mistralai`** (Apache-2.0, 2.4.13): `client.ocr.process(...)` → per-page `markdown`; opt-in `confidence_scores_granularity='word'|'page'`. **Namespace package** — `from mistralai.client import Mistral` (the bare `from mistralai import Mistral` raises `ImportError`); PyPI `license=None` (file-only) is a scanner trap.
- **`mpxpy`** (Mathpix, MIT, 0.0.20): `MathpixClient(...).image_new(..., include_line_data=True)` → `lines_json()` (`cnt` polygons + per-line `confidence`/`confidence_rate`), `mmd()`. The only **calibrated** confidence in R1.
- **`openai`** (Apache-2.0, 2.43.0): the stable path is `chat.completions.create(..., logprobs=True, top_logprobs=N)` → `choices[0].logprobs.content[].logprob`. Caveats (R1): logprobs are **flaky/empty on image inputs**, and on the **Responses API** they come back empty when strict `json_schema` Structured Outputs is enabled (observed on GPT-5.x; GPT-4.1 reported unaffected) — gate on reliability, else fall back to `uqlm` self-consistency.
- **`anthropic`** (MIT, 0.111.0): `messages.create(...)` → text blocks only. **The load-bearing negative result:** no logprobs, no confidence — the adapter sets `confidence=None, geometry=None` and confidence comes entirely from `uqlm` / agreement.

### Cross-Engine Agreement & LLM Self-Consistency — confidence for R1's no-signal engines

R1 found three engines (Claude Vision, OCR.space, pix2tex) emit **no usable confidence**, so agreement is the only signal. The verified verdict: borrow `uqlm` for the LLM half, build ROVER for the multi-OCR half.

| Library | Install | License (SDK) | Last release | Maint. | Rec. |
|---|---|---|---|---|---|
| `uqlm` | `uqlm` | Apache-2.0 | v0.6.1 · 2026-06-08 | active | reuse-as-is |
| `lm-polygraph` | `lm-polygraph` | MIT | v0.7.0 · 2026-05-04 | active | wrap |
| `jiwer` | `jiwer` | Apache-2.0 | v4.0.0 · 2025-06-19 | active | build-ourselves |
| `rapidfuzz` | `rapidfuzz` | MIT | v3.14.5 · 2026-04-07 | active | reuse-as-is |

**Gap (verified):** CONFIRMED CENTRAL GAP — there is NO maintained, permissive, pip-installable ROVER (Recognizer Output Voting Error Reduction) library. Adversarial PyPI/GitHub search verified this: the only PyPI hits for 'rover' (the-rover, rover, roverio, roverpro) are robotics/CLI packages, entirely unrelated to ASR/OCR hypothesis voting. The canonical ROVER implementation remains NIST SCTK (`sclite`/`rover`) — written in C, GPL-flavored tooling, NOT a pip library and not suitable as a permissive Python dependency. `asr-evaluation` and `jiwer` compute WER/alignment but neither does multi-system word-transition-network voting. CONCLUSION: ROVER-style cross-engine alignment+voting MUST BE BUILT. The build is small and clean: use rapidfuzz (MIT) or jiwer (Apache-2.0, itself rapidfuzz-backed) for the pairwise DP alignment primitive, then write (1) incremental WTN construction across N hypotheses and (2) the per-slot voting + per-token agreement-ratio confidence. No existing permissive lib gives the N-way lattice or the voting layer. SECONDARY GAP — there is no library that fuses geometry-aware OCR token confidence with LLM whole-response self-consistency; uqlm and lm-polygraph operate on whole strings only and are blind to per-unit OCR geometry, so the bridge from their [0,1] response-level score into a per-unit OcrResult.confidence is also our code.

**Build vs borrow:** BORROW (reuse-as-is): (1) uqlm (Apache-2.0) for the LLM/VLM self-consistency half — sample a no-confidence VLM (Claude Vision, OCR.space, pix2tex) N times, get a [0,1] confidence per extraction via black-box semantic-entropy / non-contradiction / match scorers; this is the highest-leverage borrow and directly fills R1's 'no native confidence' gap for LLM extractors. (2) rapidfuzz (MIT) as the alignment primitive — also the mandatory permissive replacement for GPL python-Levenshtein. BORROW (wrap, secondary): lm-polygraph (MIT) for white-box logit-based UE on self-hosted HF VLMs and as a cross-check to uqlm; keep it an offline eval dep and do NOT enable its non-commercial COMET extra. BORROW as alignment scaffold but BUILD the algorithm on top: jiwer (Apache-2.0) gives pairwise alignment but ROVER's N-way WTN + voting is ours. BUILD: (a) the ROVER engine itself — incremental multi-hypothesis WTN construction + per-slot majority/confidence voting + per-token agreement-ratio confidence (on rapidfuzz/jiwer); no maintained permissive pip ROVER exists. (b) the adapter that maps uqlm/lm-polygraph whole-response [0,1] scores onto per-unit OcrResult.confidence, since neither tool is geometry-aware. ALL four surveyed libraries are permissive (2x Apache-2.0, 2x MIT) — zero GPL/AGPL/non-commercial in the declared trees; the only license traps are downloaded MODEL WEIGHTS (vet NLI/embedding model licenses for uqlm; avoid lm-polygraph's COMET extra).

---

## Primary sources

Per-tier verification links (PyPI project pages, repo `LICENSE` files, API docs).

**Surface & Edit-Distance Metrics + Normalization**  
[Levenshtein on PyPI (GPL-2.0-or-later, v0.27.3)](https://pypi.org/project/Levenshtein/) · [python-Levenshtein on PyPI (GPL-2.0-or-later)](https://pypi.org/project/python-Levenshtein/) · [rapidfuzz on PyPI (MIT, v3.14.5)](https://pypi.org/project/rapidfuzz/) · [jiwer on PyPI (Apache-2.0, v4.0.0)](https://pypi.org/project/jiwer/) · [jitsi/jiwer GitHub repo](https://github.com/jitsi/jiwer) · [torchmetrics on PyPI (Apache-2.0, v1.9.0)](https://pypi.org/project/torchmetrics/) · [evaluate on PyPI (Apache-2.0, v0.4.6)](https://pypi.org/project/evaluate/) · [sacrebleu on PyPI (Apache-2.0, v2.6.0)](https://pypi.org/project/sacrebleu/) · [editdistance on PyPI (MIT, v0.8.1)](https://pypi.org/project/editdistance/) · [textdistance on PyPI (MIT, v4.6.3)](https://pypi.org/project/textdistance/) · [fastwer on libraries.io (MIT, v0.1.3 Mar 2020)](https://libraries.io/pypi/fastwer) · [fastwer on Snyk Advisor (maintenance signal)](https://snyk.io/advisor/python/fastwer) · [kahne/fastwer GitHub repo](https://github.com/kahne/fastwer)

**Structured / Field / Graph Metrics**  
[table-recognition-metric on PyPI (Apache-2.0, 0.0.6, 2025-12-02)](https://pypi.org/project/table-recognition-metric/) · [PubTabNet repo top-level LICENSE.md (CDLA-Permissive-1.0)](https://raw.githubusercontent.com/ibm-aur-nlp/PubTabNet/master/LICENSE.md) · [PubTabNet src/metric.py (Apache-2.0 header; TEDS.evaluate signature)](https://raw.githubusercontent.com/ibm-aur-nlp/PubTabNet/master/src/metric.py) · [microsoft/table-transformer LICENSE (MIT)](https://github.com/microsoft/table-transformer/blob/main/LICENSE) · [apted on PyPI (MIT, 1.0.3, 2017-11-08)](https://pypi.org/project/apted/) · [apted GitHub (no releases, ~101 stars, MIT)](https://github.com/JoaoFelipe/apted) · [zss LICENSE — BSD-3-Clause (corrects MIT claim)](https://raw.githubusercontent.com/timtadh/zhang-shasha/master/LICENSE) · [zss on PyPI (1.2.0, 2018-03-12, classifier omitted)](https://pypi.org/project/zss/) · [networkx on PyPI (BSD-3-Clause, 3.6.1, 2025-12-08)](https://pypi.org/project/networkx/) · [nervaluate on PyPI (MIT, 1.2.1, 2026-03-12)](https://pypi.org/project/nervaluate/) · [nervaluate GitHub (Evaluator API: tags/loader)](https://github.com/MantisAI/nervaluate) · [seqeval GitHub (MIT, v1.2.2 2020-10-23)](https://github.com/chakki-works/seqeval) · [spaCy on PyPI (MIT, 3.8.14, 2026-03-29)](https://pypi.org/project/spacy/) · [spaCy LICENSE (MIT)](https://github.com/explosion/spaCy/blob/master/LICENSE)

**Calibration, Conformal & Selective Prediction**  
[scikit-learn on PyPI (JSON, version + BSD-3-Clause)](https://pypi.org/pypi/scikit-learn/json) · [netcal on PyPI (1.4.0 2026-04-16, Apache-2.0, release history showing 2024-08 gap)](https://pypi.org/project/netcal/) · [MAPIE on PyPI (1.4.1 2026-06-08, BSD-3-Clause)](https://pypi.org/project/MAPIE/) · [crepes on PyPI (0.9.1 2026-06-12, BSD-3-Clause)](https://pypi.org/project/crepes/) · [TorchCP on PyPI (JSON: 1.2.1 2025-10-14, license field null, empty classifiers)](https://pypi.org/pypi/torchcp/json) · [TorchCP GitHub repo (LGPL-3.0 + GPL-3.0 detected; LICENSE/LICENSE.GPL files)](https://github.com/ml-stat-Sustech/TorchCP) · [torch-uncertainty on PyPI (JSON: 0.12.1 2026-06-17, Apache-2.0)](https://pypi.org/pypi/torch-uncertainty/json) · [puncc on PyPI (JSON: 0.9.2 2026-06-20, MIT OSI classifier)](https://pypi.org/pypi/puncc/json) · [nonconformist on PyPI (JSON: 2.1.0 2017-06-20, license UNKNOWN)](https://pypi.org/pypi/nonconformist/json) · [nonconformist GitHub repo (MIT license detected)](https://github.com/donlnz/nonconformist)

**Validation, Constrained Generation & Post-OCR Correction**  
[pydantic — PyPI JSON metadata (2.13.4, May 6 2026, MIT)](https://pypi.org/pypi/pydantic/json) · [pydantic — repo LICENSE (MIT, verified)](https://github.com/pydantic/pydantic/blob/main/LICENSE) · [jsonschema — PyPI project page (4.26.0, Jan 7 2026, MIT)](https://pypi.org/project/jsonschema/) · [fastjsonschema — PyPI JSON metadata (2.21.2, Aug 14 2025, BSD)](https://pypi.org/pypi/fastjsonschema/json) · [pandera — PyPI project page (0.32.0, Jun 19 2026, MIT)](https://pypi.org/project/pandera/) · [great-expectations — PyPI project page (1.18.1, Jun 11 2026, Apache-2.0)](https://pypi.org/project/great-expectations/) · [pydantic-extra-types — PyPI JSON metadata (2.11.1, Mar 16 2026, MIT)](https://pypi.org/pypi/pydantic-extra-types/json) · [cerberus — PyPI JSON metadata (1.3.8, Nov 6 2025, ISC)](https://pypi.org/pypi/cerberus/json) · [cerberus — GitHub repo (ISC, active, not archived)](https://github.com/pyeve/cerberus) · [voluptuous — PyPI JSON metadata (0.16.0, Dec 18 2025, BSD-3)](https://pypi.org/pypi/voluptuous/json) · [outlines — GitHub repo (Apache-2.0, 1.3.0, ~14k stars)](https://github.com/dottxt-ai/outlines) · [outlines — Generator API docs (Generator(model, output_type) verified)](https://dottxt-ai.github.io/outlines/latest/features/core/generator/) · [outlines — Output Types docs (JsonSchema, Regex, CFG)](https://dottxt-ai.github.io/outlines/latest/features/core/output_types/) · [xgrammar — PyPI project page (0.2.2, Jun 11 2026; version history)](https://pypi.org/project/xgrammar/) · [xgrammar — repo LICENSE (Apache-2.0, verified)](https://github.com/mlc-ai/xgrammar/blob/main/LICENSE) · [xgrammar — Workflow docs (GrammarCompiler.compile_* / Grammar.from_* API)](https://xgrammar.mlc.ai/docs/tutorials/workflow_of_xgrammar.html) · [XGrammar-2 announcement blog (May 4 2026)](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)

**HITL Review & Annotation Tooling**  
[PyPI label-studio](https://pypi.org/project/label-studio/) · [PyPI label-studio-sdk](https://pypi.org/project/label-studio-sdk/) · [GitHub HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) · [PyPI argilla](https://pypi.org/project/argilla/) · [GitHub argilla-io/argilla (maintenance notice)](https://github.com/argilla-io/argilla) · [Argilla last commit (GitHub API)](https://api.github.com/repos/argilla-io/argilla/commits?per_page=1) · [PyPI doccano](https://pypi.org/project/doccano/) · [GitHub doccano/doccano releases (v1.8.5 = 2025-01-11)](https://github.com/doccano/doccano/releases) · [GitHub doccano/doccano LICENSE (MIT)](https://github.com/doccano/doccano/blob/master/LICENSE) · [PyPI doccano-client JSON (v1.2.8, MIT, 2023-06-13)](https://pypi.org/pypi/doccano-client/json) · [PyPI cvat-sdk](https://pypi.org/project/cvat-sdk/) · [GitHub cvat-ai/cvat LICENSE (MIT)](https://github.com/cvat-ai/cvat/blob/develop/LICENSE) · [Prodigy (prodi.gy)](https://prodi.gy/) · [PyPI potato-annotation](https://pypi.org/project/potato-annotation/) · [Potato latest release (GitHub API, v2.6.0 = 2026-06-19)](https://api.github.com/repos/davidjurgens/potato/releases/latest) · [GitHub inception-project/inception (Apache-2.0, v40.6)](https://github.com/inception-project/inception)

**OCR Engines Behind a Uniform Interface**  
[pytesseract on PyPI (0.3.13, Apache-2.0)](https://pypi.org/project/pytesseract/) · [tesserocr on PyPI (2.10.0, MIT)](https://pypi.org/project/tesserocr/) · [easyocr on PyPI (1.7.2, Apache-2.0)](https://pypi.org/project/easyocr/) · [rapidocr on PyPI (3.8.4, Apache-2.0)](https://pypi.org/project/rapidocr/) · [rapidocr-onnxruntime on PyPI (1.4.4, Apache-2.0)](https://pypi.org/project/rapidocr-onnxruntime/) · [paddleocr on PyPI (3.7.0, Apache-2.0)](https://pypi.org/project/paddleocr/) · [PaddleOCR repo (Apache 2.0 license statement)](https://github.com/PaddlePaddle/PaddleOCR) · [paddlex on PyPI (3.7.1, Apache-2.0)](https://pypi.org/project/paddlex/) · [ocrmac on PyPI (1.0.1, MIT)](https://pypi.org/project/ocrmac/) · [python-doctr on PyPI (1.0.1, Apache-2.0)](https://pypi.org/project/python-doctr/) · [surya-ocr on PyPI (0.20.0; classifier Apache-2.0 = CODE only)](https://pypi.org/project/surya-ocr/) · [surya repo (code Apache-2.0; weights modified AI Pubs Open RAIL-M, $5M threshold)](https://github.com/datalab-to/surya) · [RapidOCR repo (v3 result object: boxes/txts/scores, multi-engine)](https://github.com/RapidAI/RapidOCR)

**OCR Cloud / VLM Client SDKs**  
[PyPI JSON (version 3.14.0, license Apache 2.0, requires-python >=3.9, upload 2026-05-07T08:03:19Z, Production/Stable) — fetched live mid-2026](https://pypi.org/pypi/google-cloud-vision/json) · [PyPI project page](https://pypi.org/project/google-cloud-vision/) · [Cloud Vision v1 text_annotation.proto (verified: confidence at Page/Block/Paragraph/Word/Symbol, bounding_box on Block/Paragraph/Word/Symbol, Page width/height no bbox, Word no text, Symbol text, detected_break in TextProperty, DetectedBreak enum)](https://raw.githubusercontent.com/googleapis/googleapis/master/google/cloud/vision/v1/text_annotation.proto) · [Dense document text detection tutorial (document_text_detection + pages→blocks→paragraphs→words→symbols traversal) — now served at docs.cloud.google.com](https://docs.cloud.google.com/vision/docs/fulltext-annotations) · [Python client types.text_annotation source (proto-plus message defs)](https://googleapis.dev/python/vision/latest/_modules/google/cloud/vision_v1/types/text_annotation.html) · [google-cloud-python monorepo (google-cloud-vision package, maintainer)](https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-vision) · [boto3 on PyPI JSON (verified 1.43.34, license Apache-2.0, uploaded 2026-06-19)](https://pypi.org/pypi/boto3/json) · [amazon-textract-response-parser on PyPI JSON (verified trp 1.0.3, Apache Software License, uploaded 2024-06-13)](https://pypi.org/pypi/amazon-textract-response-parser/json) · [amazon-textract-textractor on PyPI JSON (verified 1.9.2, Apache 2.0, uploaded 2025-04-24)](https://pypi.org/pypi/amazon-textract-textractor/json) · [amazon-textract-textractor Word entity source (confirmed self._confidence = confidence / 100)](https://github.com/aws-samples/amazon-textract-textractor/blob/master/textractor/entities/word.py) · [amazon-textract-textractor DocumentEntity.confidence property (docstring: 'between 0 and 1')](https://github.com/aws-samples/amazon-textract-textractor/blob/master/textractor/entities/document_entity.py) · [amazon-textract-response-parser trp source (BaseBlock .text/.confidence/.geometry; Line.words; Form.fields; trp2/t_pipeline)](https://github.com/aws-samples/amazon-textract-response-parser/tree/master/src-python/trp) · [AWS Textract Block API reference (BlockType/Text/Confidence 0-100/Geometry.BoundingBox/Polygon, textract-2018-06-27)](https://docs.aws.amazon.com/textract/latest/dg/API_Block.html) · [PyPI JSON (verified: version 1.0.2 latest, released 2025-03-27; MIT trove classifier; release history 1.0.0 2024-12-18 / 1.0.1 2025-03-13)](https://pypi.org/pypi/azure-ai-documentintelligence/json) · [DocumentWord class reference (verified: content/polygon/span/confidence attributes + Required status)](https://learn.microsoft.com/en-us/python/api/azure-ai-documentintelligence/azure.ai.documentintelligence.models.documentword) · [DocumentTableCell class reference (verified: NO confidence attribute — refutes R1)](https://learn.microsoft.com/en-us/python/api/azure-ai-documentintelligence/azure.ai.documentintelligence.models.documenttablecell) · [AnalyzeResult class reference (verified: content/pages/tables/key_value_pairs/documents/content_format/paragraphs/styles)](https://learn.microsoft.com/en-us/python/api/azure-ai-documentintelligence/azure.ai.documentintelligence.models.analyzeresult) · [Azure SDK for Python source (azure-ai-documentintelligence)](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/documentintelligence/azure-ai-documentintelligence) · [PyPI mistralai JSON (version 2.4.13, upload 2026-06-19T11:52:32Z; license=None, no SPDX expression, no License classifier; requires_python>=3.10; 2.0.0 dated 2026-03-10)](https://pypi.org/pypi/mistralai/json) · [GitHub client-python LICENSE (verified Apache License 2.0)](https://raw.githubusercontent.com/mistralai/client-python/main/LICENSE) · [GitHub client-python README (verified: official examples use `from mistralai.client import Mistral`; references v1->v2 migration guide)](https://raw.githubusercontent.com/mistralai/client-python/main/README.md) · [OCR SDK docs: ocr.process signature + confidence_scores_granularity ('word'|'page')](https://github.com/mistralai/client-python/blob/main/docs/sdks/ocr/README.md) · [Installed 2.4.13 wheel source: model fields + confidence docstring verified in mistralai/client/models/ (OCRResponse/OCRPageObject/OCRImageObject/OCRPageConfidenceScores/OCRConfidenceScore/OCRTableObject); no top-level mistralai/__init__.py in dist-info/RECORD (namespace package)](https://github.com/mistralai/client-python/tree/main/src/mistralai) · [PyPI mpxpy JSON API (verified: version 0.0.20, upload 2026-01-26, License MIT License, requires_python >=3.8, summary 'Official Mathpix client for Python')](https://pypi.org/pypi/mpxpy/json) · [GitHub Mathpix/mpxpy mpxpy/image.py (verified Image methods: results, wait_until_complete, lines_json, mmd, latex_styled, html; lines_json requires include_line_data)](https://github.com/Mathpix/mpxpy/blob/main/mpxpy/image.py) · [GitHub Mathpix/mpxpy mpxpy/pdf.py (verified Pdf methods: pdf_new result, wait_until_complete, to_md_text, to_mmd_text, to_lines_json, to_lines_mmd_json, to_mmd_zip_*, pdf_status)](https://github.com/Mathpix/mpxpy/blob/main/mpxpy/pdf.py) · [mpxpy docs — Processing Images (MathpixClient, image_new, mmd(), lines_json() keys: type/text/confidence/confidence_rate/cnt)](https://mathpix.com/docs/mpxpy/images) · [mpxpy docs — Processing PDFs (pdf_new, wait_until_complete, to_md_text, to_lines_json, to_lines_mmd_json)](https://mathpix.com/docs/mpxpy/pdf) · [Mathpix API v3/text reference (verified top-level confidence, confidence_rate, is_printed, is_handwritten)](https://docs.mathpix.com/reference/image-results) · [PyPI JSON (version 2.43.0, license Apache-2.0, requires_python >=3.9) — live-verified](https://pypi.org/pypi/openai/json) · [GitHub releases (2.43.0 2026-06-17 + 2.40.0–2.42.0 June dates) — live-verified](https://github.com/openai/openai-python/releases) · [OpenAI Cookbook: Using logprobs — confirms choices[0].logprobs.content with .token/.logprob/.bytes/.top_logprobs](https://developers.openai.com/cookbook/examples/using_logprobs) · [Responses API reference (create, include=['message.output_text.logprobs'], top_logprobs 0..5, output_text)](https://developers.openai.com/api/reference/resources/responses/methods/create) · [Bug (live-verified): logprobs EMPTY under json_schema Structured Outputs on GPT-5.1/5.2; GPT-4.1 unaffected (Responses API)](https://community.openai.com/t/gpt-5-1-5-2-message-output-text-logprobs-is-empty-when-structured-outputs-json-schema-is-enabled-in-responses-api/1371927) · [Why doesn't the Responses API support logprobs / image-input flakiness](https://community.openai.com/t/why-doesnt-the-responses-api-support-logprobs/1148097) · [Structured Outputs intro (vision compatibility, schema deser.)](https://openai.com/index/introducing-structured-outputs-in-the-api/) · [PyPI JSON API (anthropic 0.111.0, MIT, 2026-06-18) — verified live](https://pypi.org/pypi/anthropic/json) · [PyPI project page](https://pypi.org/project/anthropic/) · [GitHub anthropic-sdk-python repo + api.md (Message/TextBlock types, no logprob field)](https://github.com/anthropics/anthropic-sdk-python) · [Messages API reference (response fields id/content/usage..., no logprobs param/field; usage = token counts only)](https://platform.claude.com/docs/en/api/messages) · [Structured outputs docs (output_config.format json_schema; beta header no longer required; messages.parse -> parsed_output)](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

**Cross-Engine Agreement & LLM Self-Consistency**  
[PyPI uqlm project page (verified 0.6.1 / 2026-06-08)](https://pypi.org/project/uqlm/) · [GitHub cvs-health/uqlm releases](https://github.com/cvs-health/uqlm/releases) · [PyPI lm-polygraph (verified 0.7.0 / 2026-05-04, MIT, Alpha)](https://pypi.org/project/lm-polygraph/) · [GitHub IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph) · [PyPI jiwer (verified 4.0.0 / 2025-06-19, Apache-2.0)](https://pypi.org/project/jiwer/) · [jiwer process/alignments reference](https://jitsi.github.io/jiwer/reference/process/) · [PyPI rapidfuzz (verified 3.14.5 / 2026-04-07, MIT)](https://pypi.org/project/rapidfuzz/) · [GitHub rapidfuzz/RapidFuzz releases](https://github.com/rapidfuzz/RapidFuzz/releases) · [ROVER original NIST paper (Fiscus 1997) — establishes the algorithm that has no pip impl](https://www.nist.gov/publications/post-processing-system-yield-reduced-word-error-rates-recognizer-output-voting-error) · [LV-ROVER arXiv 1707.07432](https://arxiv.org/abs/1707.07432)
