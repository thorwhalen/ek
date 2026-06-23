---
name: ek-dev-architecture
description: "Master orientation map for developers BUILDING ek (a framework for building Knowledge Evaluation systems that evaluate information-extraction outputs; OCR is the noisiest special case). Read this FIRST before touching any ek component. Covers the two-need 2x2 (reference-based vs reference-free, instance vs system), the two-layer data model (Layer A GraphGrammar schema/cost SSOT + Layer B AnnotatedExtraction metadata), the facades score()/evaluate()/estimate_quality(), the strategy-Protocol + registry + dependency-injection + requires_extra open-closed pattern, the module layout (base, registry, stores, metrics, canonicalize, qe/rover, validate, harness, ocr, tools, __main__), the config2py.AppData + dol persistence model, the must-builds (two-layer object, cost-weighted typed-graph distance, ROVER), and a which-report-for-which-component table. Use when deciding where code goes, what a component is named, how facades share the Layer-A object, how strategies plug in, or which ek_0N report to read."
metadata:
  audience: developers
---

# ek Architecture — the orientation map

**Read this first.** `ek` is a framework for **building Knowledge Evaluation systems**: tools that evaluate the outputs of information-extraction (IE) systems. OCR is treated as the *noisiest special case* of a general problem, not the problem itself — everything except the image-to-text error model is **source-agnostic** (PDF, DOCX, XLSX, tables, DB responses, LLM extractors).

This skill is the **index**. It tells you what the pieces are called, how they fit, and **which research report to read** before building each one. The reports live under `misc/docs/` (cite by relative path; read them on demand — do not inline them).

Audience: developers building `ek`. Not `ek`'s end users.

---

## 1. The two needs (the 2x2 that organizes everything)

Evaluation splits along two independent axes:

- **Reference availability** — is there a gold answer (*reference-based* / offline) or not (*reference-free* / online)?
- **Granularity** — scoring one item (*instance-level*) or aggregating a corpus (*system-level*)?

|                       | **Reference-based** (gold available)                  | **Reference-free** (no gold at inference)                       |
|-----------------------|-------------------------------------------------------|----------------------------------------------------------------|
| **System / corpus**   | **Offline benchmarking** — pick the system/params     | Production monitoring & drift detection                        |
| **Instance**          | Per-item scoring; spot-checks; error analysis         | **Online QE -> triage & HITL** (accept/flag/block)             |

Two facades cover the two columns: `score()` (reference-based) and `estimate_quality()` (reference-free). Both operate on the **same Layer-A object** (see section 3). The unifying frame for the reference-free column is *reference-free QE feeding a selective-prediction policy* (signal -> calibrate -> validate -> decide).

Conceptual grounding: `misc/docs/information-extraction-evaluation-conceptual-map.md` (the map into all six reports; read its sections 9-10 for the cross-cutting findings and reading guide).

---

## 2. Three load-bearing ideas

1. **Pick the right object and comparison function.** The big lever is *not* a better string metric — it is moving evaluation down to the **structured payload** you consume downstream, and weighting errors by **real cost** (a misspelled city != two extra digits on a donation amount). Defer the boolean: keep numeric scores, threshold late.
2. **Separate the front-end from the rest.** OCR is one error-prone transducer. `ek` core depends only on the `OcrResult` *shape*, so it evaluates *any* `image -> OcrResult` callable. Direction is strict: `ek -> ocracy` (optional extra `ek[ocr]`), **never** `ocracy -> ek`.
3. **Calibration is the hidden prerequisite, and the best comparator does not exist off-the-shelf.** Almost every raw signal (OCR confidence, LLM logprob, validator firing) is uncalibrated. The cost-weighted, type-aware typed-graph distance is the principal **build**, not a config knob.

---

## 3. The two-layer data model (the SSOT every component plugs into)

This is the integrating abstraction. Everything else plugs into it. Defined in `ek/base.py`. Use `dataclasses` + `collections.abc` interfaces; favor functional composition over inheritance.

### Layer A — `GraphGrammar` (frozen schema SSOT: *what we extract and what matters*)

Carries types **and** importance/cost weights. Shared by the offline path, the online path, and constrained decoders. Cost weights live **on** the types.

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str                       # 'string'|'number'|'date'|'currency'|'enum'|...
    importance: float = 1.0         # attribute-level cost weight
    domain: tuple[Any, ...] = ()    # enum members / (lo, hi) range / regex — read by validators
    normalizer: str | None = None   # registry key: canonicalize before comparison

@dataclass(frozen=True)
class NodeType:
    name: str
    fields: Mapping[str, FieldSpec]
    importance: float = 1.0         # node-level cost weight

@dataclass(frozen=True)
class EdgeType:
    name: str
    src: str                        # source NodeType name
    dst: str                        # target NodeType name
    importance: float = 1.0         # edge-level cost weight

@dataclass(frozen=True)
class GraphGrammar:                  # the schema (SSOT)
    node_types: Mapping[str, NodeType]
    edge_types: Mapping[str, EdgeType]
    def node_cost(self, name: str) -> float: ...
    def edge_cost(self, name: str) -> float: ...
    def field_cost(self, node: str, field: str) -> float: ...
```

### Layer B — `AnnotatedExtraction` (verification metadata, rides *alongside* Layer A; never inside it)

The frozen `GraphGrammar` is **referenced, never mutated**. Per-value metadata is keyed by graph **path**.

```python
@dataclass(frozen=True)
class NodePath:                     # addresses a node (and optional field) in an extracted graph
    node_id: str
    node_type: str                  # key into GraphGrammar.node_types
    field: str | None = None        # key into NodeType.fields, or None for the whole node

@dataclass
class Provenance:
    engine: str
    source_span: tuple[int, int] | None = None  # char offsets into raw text
    bbox: Any = None                            # geometry for the image overlay
    raw_transcripts: Sequence[str] = ()         # multiple raw OCR outputs for adjudication

@dataclass
class FieldEstimate:                # one extracted value + its verification metadata
    value: Any
    raw_signals: dict[str, float] = field(default_factory=dict)  # intrinsic conf, logprob, agreement...
    confidence: float | None = None              # calibrated P(correct), once through a Calibrator
    findings: tuple["Finding", ...] = ()         # validator outputs
    provenance: Provenance | None = None
    decision: str | None = None                  # 'accept' | 'flag' | 'block'

@dataclass
class Finding:
    field: str
    layer: str                                   # which validation layer fired
    severity: str                                # 'correct' | 'flag'
    message: str
    suggestion: Any = None

@dataclass
class AnnotatedExtraction:          # Layer B alongside Layer A, never inside it
    grammar: GraphGrammar
    estimates: Mapping[NodePath, FieldEstimate]
```

**Why two layers:** the schema stays the SSOT (also feeds constrained decoders via `model_json_schema()`); provenance/confidence/findings/decisions ride strictly alongside. The same `(grammar, estimates)` pair is scored offline (vs gold) and estimated online (vs consensus/expectation).

Detail: `misc/docs/ek_06 -- library-landscape-and-integration-map.md` (section "Architecture sketch").

---

## 4. The two facades (both share the Layer-A object)

```python
def score(pred, gold, *, grammar=None, metric=None, normalize=None, weights=None) -> Score:
    """Reference-based / offline, ONE comparison. Metric dispatched by output type
    unless given: str -> CER/WER (jiwer); record/dict -> field-F1; (extras add
    chrF/sacrebleu, span-F1/nervaluate, TEDS, cost-weighted GED/networkx)."""

def evaluate(cases, *, metric=None, grammar=None, normalize=None) -> Report:
    """Reference-based / offline, a CORPUS of (pred, gold[, slice]) cases. Aggregates
    via the metric's own aggregator (global CER/WER accumulation, micro-F1) -- never
    a naive mean -- with optional per-slice cuts. The harness/benchmark build on this."""

def estimate_quality(extraction, *, sources=(), calibrator=None, validators=(), policy=None) -> QualityReport:
    """Reference-free / online. Gather signals -> calibrate -> validate -> decide accept/flag/block."""
```

A `Metric` is the atomic strategy returning a `Score`; `score()` is the single-pair
facade (returns `Score`); `evaluate()` is the corpus facade (returns `Report`).
`score(pred, gold)` with nothing else **Just Works** (progressive disclosure). An
advanced caller passes an explicit cost-weighted graph metric, a domain normalizer,
a fitted calibrator, a conformal policy. Public API is re-exported in `ek/__init__.py`.

---

## 5. Strategy protocols + registry + DI + requires_extra (the open-closed pattern)

Every swappable behavior is a **callable conforming to a `typing.Protocol`**, resolved from a **registry by name**, **injected keyword-only with smart defaults**. Third parties register new backends via **entry points** — never by editing callers. Lives in `ek/registry.py`.

```python
@runtime_checkable
class Metric(Protocol):
    def __call__(self, pred, gold, *, grammar: GraphGrammar | None = None) -> float: ...

@runtime_checkable
class Validator(Protocol):
    def __call__(self, value, *, spec: FieldSpec) -> Iterable[Finding]: ...

@runtime_checkable
class Calibrator(Protocol):
    def fit(self, scores: Sequence[float], correct: Sequence[bool]) -> "Calibrator": ...
    def __call__(self, raw_score: float) -> float: ...   # -> calibrated P(correct)

@runtime_checkable
class SelectivePolicy(Protocol):                          # a.k.a. DecisionPolicy
    def __call__(self, confidence: float) -> str: ...     # 'accept'|'flag'|'block'

# Callable aliases:
Normalizer       = Callable[[str], str]
ConfidenceSource = Callable[[FieldEstimate], Mapping[str, float]]   # a.k.a. Signal
CostWeight       = Callable[[GraphGrammar, TypeRef], float]          # default: read *.importance
OcrBackend       = Callable[[bytes], "OcrResult"]
```

Core protocols: **Metric, Validator, Calibrator, DecisionPolicy/SelectivePolicy, Signal**. Aliases: **Normalizer, ConfidenceSource, CostWeight, OcrBackend**.

**`@requires_extra`** decorator: a strategy from an optional extra raises an *actionable* install hint ("install `ek[ocr]`") instead of an opaque `ImportError`. Keep error-raising separate from core logic. Heavy/native backends are guided dynamically via a `check_requirements(...)` helper (binaries, platform gates, credential checks).

---

## 6. Module layout

| Module | Responsibility |
|---|---|
| `ek/__init__.py` | Public API surface (every name a user imports). |
| `ek/base.py` | The two-layer data model: `GraphGrammar`, `FieldSpec`, `NodeType`, `EdgeType`, `AnnotatedExtraction`, `FieldEstimate`, `Provenance`, `NodePath`, `Finding`. The SSOT. |
| `ek/registry.py` | Strategy `Protocol`s, the registry, DI resolution, entry-point discovery, `@requires_extra`, `check_requirements`. |
| `ek/stores.py` | Persistence: `config2py.AppData("ek")` + `dol` JSON stores grouped as a mall. |
| `ek/metrics/` | Reference-based metrics; the cost-weighted typed-graph GED (the flagship build) + wrappers over `rapidfuzz`/`jiwer`/`sacrebleu`/`nervaluate`/`apted`/TEDS. Houses `score()`. |
| `ek/canonicalize.py` | The versioned canonicalization registry; number/date/Unicode folding before scoring (primitives: `jiwer.transforms`, `ftfy`, `dateparser`). |
| `ek/qe/` | Reference-free QE; houses `estimate_quality()`, calibration/conformal mapping, geometry-aware uncertainty mapping. |
| `ek/qe/rover.py` | The ROVER engine: N-way align + per-slot vote + per-position agreement confidence (the second must-build). |
| `ek/validate.py` | Validators: cross-field/cross-source consistency, schema/range/enum, constrained-generation seam. |
| `ek/harness.py` | The slice-aware regression harness: stratified sampling, per-slice reporting, golden-set regression, IAA. |
| `ek/ocr/` | The `OcrResult` shape + adapters; the optional `ek[ocr]` integration with `ocracy`. |
| `ek/tools.py` | Shared helpers (`_helper` for same-module-only; no underscore for cross-module reuse). |
| `ek/__main__.py` | CLI via `argh`: `_dispatch_funcs` SSOT + `dispatch_with_namespaces`. |

`ocr/` depends on the `OcrResult` shape only: `OcrResult(text; blocks=[TextBlock(text, bbox, confidence, level, language, meta)]; raw; meta)` with `confidence` normalized to `[0,1]`. This is what makes `ek` able to evaluate *any* `image -> OcrResult` callable, and what keeps `ek -> ocracy` one-directional.

Every `.py` needs a module docstring (ruff D100 enforced). Keyword-only beyond the 3rd arg; no magic numbers (kw-only args / external config; open-closed); generators (`Iterable[T]`) over list-building.

---

## 7. Persistence model

```python
config2py.AppData("ek")            # resolves ~/.local/share/ek/
dol.Jsons(get_artifact_dir(kind))  # a JSON MutableMapping store per kind
```

Stores per kind: `gold/`, `results/`, `calibrators/`, `corrections/`, `runs/`. Group them as a **dol mall**. Use `dol.cache_this` for persisted baselines. Stores are **`MutableMapping` facades, never god-classes**. Lives in `ek/stores.py`.

---

## 8. The must-builds (no off-the-shelf equivalent)

Confirmed across the whole survey — these are the connective tissue no library owns:

1. **The two-layer eval object** (`ek/base.py`) — section 3. Everything plugs into it.
2. **Cost-weighted, type-aware typed-graph distance** (`ek/metrics/`) — the flagship offline metric. Supplies `networkx.graph_edit_distance` with `node_subst_cost`/`edge_subst_cost` callables reading `*.importance` via the injected `CostWeight(grammar, TypeRef)`.
3. **The ROVER engine** (`ek/qe/rover.py`) — N-way align + per-slot vote + per-position agreement confidence, on `rapidfuzz`/`jiwer` editops. No maintained permissive pip ROVER exists.
4. **Geometry-aware uncertainty mapping** — projects a whole-response `[0,1]` uncertainty back onto per-unit `OcrResult` confidence (no library is geometry-aware).
5. **Field/graph calibration & conformal mapping** (`ek/qe/`) — cast the structured score into the per-field probability vectors `netcal`/`MAPIE` consume.
6. **Cross-field/cross-source consistency validators** (`ek/validate.py`) — totals reconcile, dates ordered, referential integrity, cross-source triangulation.
7. **The slice-aware regression harness** (`ek/harness.py`).
8. **The versioned canonicalization registry** (`ek/canonicalize.py`).

---

## 9. Licensing posture (build-time discipline)

Lean **permissive core**; everything copyleft/non-commercial is opt-in behind extras + a **CI license gate**. Landmines — some hide in repo files, invisible to PyPI scanners:

- `Levenshtein` / `python-Levenshtein` (GPL) -> use **`rapidfuzz`** (MIT).
- `TorchCP` (LGPL) -> import-only, **quarantine**, never vendor/patch.
- `surya-ocr` (non-commercial RAIL-M weights behind an Apache PyPI classifier) -> keep out of default `ocr` extras.
- `Potato` (GPL), `Prodigy` (proprietary) -> never a dependency.

`ek -> ocracy` is the **only** allowed direction; `ek[ocr]` is optional.

Detail: `misc/docs/ek_06 -- library-landscape-and-integration-map.md` (sections "Dependency tiers", "License register").

---

## 10. WHICH REPORT TO READ FOR WHICH COMPONENT

All under `misc/docs/`. Read on demand; cite by relative path.

| ek component / phase | Module(s) | Read this report |
|---|---|---|
| Whole-system orientation, the 2x2, cross-cutting findings | (all) | `information-extraction-evaluation-conceptual-map.md` |
| Choosing/characterizing an OCR/VLM front-end; native confidence, provenance, table structure | `ek/ocr/` | `ek_01 --  ocr-systems-capability-inventory.md` |
| `score()`, all reference-based metrics, cost-weighted typed-graph GED, the harness | `ek/metrics/`, `ek/harness.py`, `ek/canonicalize.py` | `ek_02 -- Reference-Based (Offline) Evaluation of Information-Extraction Systems.md` |
| `estimate_quality()`, confidence sources, calibration, conformal, selective prediction, ROVER | `ek/qe/`, `ek/qe/rover.py` | `ek_03 -- Reference-Free Quality Estimation, Confidence, Calibration & Selective Prediction.md` |
| Validators, the FLAG-vs-CORRECT layers, constrained generation, post-OCR correction | `ek/validate.py` | `ek_04 -- Post-Extraction Validation & Correction (incl. Post-OCR).md` |
| Provenance/drill-down, review-queue triage, active learning, drift/production monitoring | `ek/qe/` (decision/provenance), HITL adapters | `ek_05 -- HITL Review UX, Active Learning & Production Monitoring for Document-Understanding Systems.md` |
| Build/borrow/wrap decisions, the dependency graph, the facade architecture, license gate | `ek/registry.py`, `ek/stores.py`, `pyproject.toml` extras | `ek_06 -- library-landscape-and-integration-map.md` |

When in doubt about a placement, naming, or facade-sharing decision, re-read sections 3-6 here, then the matching report row above.
