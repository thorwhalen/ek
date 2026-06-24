---
name: ek-dev-add-metric
description: "How to add a reference-based (offline) metric to ek's score() side \u2014 the output-type to metric to library decision table (CER/WER via jiwer, ANLS*/anls_star, span-F1 via seqeval/nervaluate, TEDS/GriTS for tables, cost-weighted typed-graph GED via networkx/apted/zss, coref via metametric), the Metric Protocol __call__(pred, gold, *, grammar=None) -> float, the Report/Score decomposition object, how to wrap a library behind the Protocol and register it, and the flagship cost-weighted typed-graph must-build whose weights come from Layer-A importance. Use when implementing, wrapping, registering, choosing, or debugging an offline/reference-based metric, scoring pred vs gold, slot/entity F1, table TEDS, graph edit distance, ANLS, WER/CER, canonicalization-before-scoring, match schemes, or per-slice/per-path TP/FP/FN. For online/reference-free confidence, calibration, and decisions use the estimate_quality side instead."
metadata:
  audience: developers
---

Authoritative spec for **adding a reference-based (offline) metric** to `ek` — the `score(pred, gold, ...)` side. You are building `ek` itself, not using it. Match the names and decisions in this file and in `misc/docs/ek_02 -- Reference-Based (Offline) Evaluation of Information-Extraction Systems.md` exactly. The flagship typed-graph metric is also covered in `misc/docs/ek_06 -- library-landscape-and-integration-map.md` (the architecture report).

A reference-based metric compares a prediction against a known-good gold value and returns a score. It does NOT touch confidence, calibration, or accept/flag/block decisions — that is the reference-free `estimate_quality()` side. Keep the two facades cleanly separated.

## The contract: the `Metric` Protocol

Every metric is a `typing.Protocol` callable, registry-resolved, injected keyword-only with smart defaults. The signature is fixed:

```python
@runtime_checkable
class Metric(Protocol):                 # reference-based: compare pred vs gold
    # grammar carries the cost weights; field/string metrics may ignore it
    def __call__(self, pred: Any, gold: Any, *, grammar: GraphGrammar | None = None) -> float: ...
```

Rules:
- `pred` and `gold` are the SAME output-object type (both strings, both field dicts, both typed graphs).
- `grammar` is optional. String/field metrics may ignore it; the typed-graph metric REQUIRES it (that is where cost weights live).
- Return a single float in `[0, 1]` where higher is better (a *similarity*, not a distance). Convert error rates: `score = max(0.0, 1.0 - error_rate)`. Convert edit distances: `score = 1 - dist / max(|a|, |b|)`.
- The bare float is the public contract. The richer decomposition (P/R/F1, per-slice, per-path TP/FP/FN, alignment) lives on the `Score`/`Report` object the `score()` facade assembles — see below. A metric may return a `Score` subtype that *is* a float (or the facade wraps the float); never break the `-> float` Protocol.

## The facade you plug into

```python
# A Metric is the atomic strategy: __call__(pred, gold, *, grammar=None) -> Score.
# score() is the single-pair facade (returns Score); evaluate() is the corpus
# facade (returns Report, aggregating via the metric's own .aggregate()).

def score(pred, gold, *, grammar: GraphGrammar | None = None,
          metric: Metric | str | None = None,
          normalize: Normalizer | None = None,
          weights: CostWeight | None = None) -> "Score":
    """str -> CER/WER (jiwer); record/dict -> field-F1; (extras add chrF/sacrebleu,
    span-F1/nervaluate, TEDS, cost-weighted GED/networkx)."""
    canon = resolve_canonicalizer(normalize)
    m = _resolve_metric(metric, pred, gold, canon)            # SSOT registry dispatch
    return m(pred, gold, grammar=grammar)

def evaluate(cases, *, metric=None, grammar=None, normalize=None) -> "Report":
    """Corpus of (pred, gold[, slice]) -> Report; aggregate via metric.aggregate()."""
```

`score(pred, gold)` with no other args Just Works (progressive disclosure): `_resolve_metric` dispatches by output-object type. An advanced caller passes an explicit `metric`, a `normalize` Normalizer, or `weights` CostWeight. Your new metric must be reachable BOTH ways: by type-dispatch default AND by name via the registry. **Corpus correctness:** put raw counts (edits/ref_len, or tp/fp/fn) in `Score.detail` and implement `Metric.aggregate(scores)` so `evaluate()` aggregates globally (CER/WER, micro-F1) -- never average per-item scores.

## Decision table — output type -> metric -> library

Pick the metric by the **object type**, never by habit. This is the authoritative routing (`ek_02` §Decision Table):

| Output type | Metric | Library (reuse-as-is) | Notes |
|---|---|---|---|
| **String** (free text, OCR/ASR transcript, one field value) | CER, WER, normalized edit similarity, chrF | `jiwer` (CER/WER + alignment + transforms), `sacrebleu` (chrF/TER), `rapidfuzz` | Normalize first. jiwer >=4.0 scores empty references (hallucination on silence). |
| **Single short answer vs. acceptable variants** | ANLS (~0.5 threshold) | `anls_star` (back-compatible with classic ANLS) | Tolerates minor OCR/spelling diffs. |
| **Fields / slots** (key-value, typed entities, spans) | entity/field P/R/F1 (exact AND partial) | `seqeval` (CoNLL/IOB strict), `nervaluate` (SemEval-2013: strict/exact/partial/type) | Choose match scheme EXPLICITLY (see gotcha). |
| **Fields with nesting/grouping** | group-aware F1, ANLS* over dict/list trees | `anls_star`, `metametric` | Flat entity F1 overstates KIE quality when grouping matters. |
| **Tables** (HTML/grid, merged cells) | TEDS, TEDS-Struct (structure-only), GriTS | **built clean-room** in `ek/metrics/tables.py`: TEDS on `apted` (MIT); GriTS in pure Python | TEDS-Struct ignores cell text to isolate structure. GriTS yields P/R and unifies topology/content/location. **License trap (verified mid-2026):** `table-recognition-metric`'s own code is Apache-2.0 but it hard-requires GPL `Levenshtein` (`from Levenshtein import distance`) — a scanner-invisible copyleft landmine — so it is NOT used; `microsoft/table-transformer` (MIT) ships GriTS only as research code, not a pip lib. Both were reimplemented (cf. the Krippendorff precedent). |
| **Nested objects / typed graphs** (JSON, relation/event graphs, line items) | cost-weighted GED, normalized tree-edit, ANLS* | `networkx` GED, `apted`/`zss` (tree, with cost hooks), `anls_star`, `metametric` | The flagship build — see below. GED is NP-hard. |
| **Coreference / clusters** | MUC, B3, CEAF, CoNLL avg | `metametric` (`coref_suite`), conll reference-coreference-scorers | CoNLL score = unweighted mean of MUC/B3/CEAFe F1. |

**Never reach for Levenshtein / python-Levenshtein (GPL).** Use `rapidfuzz` (MIT) — it gives weighted edit distance `distance.Levenshtein.distance(weights=(ins,del,sub))` out of the box. This is a license landmine, not a style preference.

## How to wrap a library behind the Protocol and register it

Five steps. Wrap thin; own the seam, not the engine.

1. **Write the metric in `ek/metrics/` (or the metrics module).** Add a module docstring (ruff D100 enforced). The function is a `Metric`: `(pred, gold, *, grammar=None) -> float`. Convert the library's native output to a `[0,1]` higher-is-better similarity inside the wrapper.

2. **Gate optional deps with `@requires_extra`.** If the engine is behind an extra (e.g. `ek[tables]`, `ek[graph]`), decorate so a missing import raises an *actionable* install hint ("install `ek[tables]`"), not an opaque `ImportError`. Separate the error-raising concern from the metric logic via the decorator.

3. **Register it** under a stable name so `_resolve_metric(metric="my_name", ...)` and entry-point discovery both find it. Third parties register via entry points (open-closed). Wire the type-dispatch default in `_resolve_metric` so the bare `score(pred, gold)` call routes to your metric for its object type.

4. **Read cost weights from the SSOT, never hardcode.** Importance/cost weights live on Layer-A (`FieldSpec.importance`, `NodeType.importance`, `EdgeType.importance`). Pull them via the injected `CostWeight` (default reads `*.importance`); no magic numbers, keyword-only beyond the 3rd arg.

5. **Canonicalize before scoring** (see gotcha). Apply the `normalize` Normalizer / per-type `FieldSpec.normalizer` from the versioned canonicalization registry before you compare. Do not bake normalization into the metric; it is a separate, versioned, swappable component.

Example wrapper shape (string CER via jiwer):

```python
"""String-level reference metrics (CER/WER) wrapping jiwer."""
import jiwer

def cer(pred: str, gold: str, *, grammar=None) -> float:
    """Character similarity = 1 - CER. Canonicalize before calling."""
    err = jiwer.cer(reference=gold, hypothesis=pred)   # 0 perfect; can exceed 1.0
    return max(0.0, 1.0 - err)
```

## The flagship build: cost-weighted, type-aware typed-graph GED

This is the one genuine build — confirmed unbuilt across the whole survey (`ek_02` §6, `ek_06`). No maintained library takes a typed schema + cost matrix and returns an importance-weighted typed-graph distance. You supply cost *functions*; you do NOT build a distance engine.

- **Engine:** `networkx.graph_edit_distance(G1, G2, node_subst_cost=, edge_subst_cost=, node_del_cost=, node_ins_cost=, ...)`. The callables encode type-aware costs directly.
- **Where weights come from:** Layer-A importance. The `node_subst_cost`/`edge_subst_cost` callables read `NodeType.importance` / `EdgeType.importance` (and `FieldSpec.importance` for attribute-level substitutions), via the injected `CostWeight(grammar, TypeRef)`. So "two extra digits on a monetary amount" outweighs "a misspelled city." The grammar is the single source of truth — the cost fn just looks weights up:

```python
CostWeight = Callable[[GraphGrammar, TypeRef], float]   # default: read *.importance
# TypeRef(kind='node'|'edge'|'field', name=<type name>, field=<field or None>)
```

- **Tables route to `apted`** (override `Config.rename` / `PerEditOperationConfig(del, ins, ren)` for per-cell-type weights) or `zss` (`zss.distance` with `insert_cost`/`remove_cost`/`update_cost` callables).
- **GED IS NP-HARD (and APX-hard).** Exact computation is feasible only for tiny graphs. You MUST: (a) use an approximation — `networkx.optimize_graph_edit_distance` is a generator of successively better upper bounds; add a `timeout`; (b) DOCUMENT which approximation and cost model you used and the bound returned, because results are NOT comparable across approximation/cost choices. Normalize the distance to a `[0,1]` similarity and build partial-overlap semantics + large-graph matching on top (those are also yours to build).

## Critical gotchas — get these wrong and your scores are bogus

1. **Never average WER/CER per-batch.** Naively averaging batch-level WER is mathematically wrong because sequences have variable lengths. ACCUMULATE the global edit counts (sub+ins+del) and the global reference length across the whole corpus, THEN divide once. For in-training logging use `torchmetrics` with global accumulation across the epoch — not a mean of per-batch rates.

2. **Never present a single F1 without naming the match scheme.** seqeval (entity-level, CoNLL) and nervaluate (SemEval-2013: strict / exact / partial / type) DIVERGE WIDELY, and they even disagree with each other (seqeval ignores other-type tags at tag level; nervaluate includes them). A model that finds "Electric" instead of "General Electric" gets F1=0 under exact-span but F1~=0.667 under token-level. Every reported F1 MUST carry its `MatchScheme` (strict/exact/partial/type, token vs span, type-aware vs type-agnostic) in the `Report`. The Protocol returns one float, but the facade must surface the scheme.

3. **GED is NP-hard** — see flagship section. Use approximation + document the bound. Never present an exact-GED number you didn't actually compute exactly.

4. **Canonicalize before scoring — the single biggest source of bogus scores.** Unicode form (NFC vs NFKC), case, whitespace, punctuation, number/date/currency folding can swing scores by tens of points. Whisper reports WER drops of up to 50% from a single normalization quirk (contractions separated by whitespace). And normalization tuned for English silently corrupts non-English: up to ~152% spurious WER reduction for Malayalam (Mark-class/diacritic stripping). So: build/version a schema-specific canonicalizer as a first-class, tested, versioned component; apply it before every comparison; re-audit it whenever you add a language/script. If normalization changes a field's score by more than ~2-3 points, inspect — the normalizer is doing too much or too little.

## The `Report` / `Score` decomposition object

The metric returns a float; the `score()` facade returns a `Report` that carries the **scalar PLUS the decomposition** a human or the regression harness needs:

- **scalar** — the headline `[0,1]` similarity.
- **P / R / F1** — for field/slot/table/coref metrics, with the **named match scheme**.
- **per-slice** — scores stratified by document type, language, layout, vendor, field rarity (a single aggregate hides failure modes; the harness slices).
- **per-path TP / FP / FN** — keyed by graph path (`NodePath`) so errors are attributable to specific fields/nodes/edges, with partial credit where the scheme allows.
- **alignment** — the edit-op alignment (jiwer `process_words`/`process_characters`; the GED edit path; the Hungarian leaf matching) for HITL drill-down. Note: neural GED regressors predict a similarity but CANNOT return an edit path — only use cost-callable GED when you need the alignment.

Keep `Report` a dataclass (collections.abc + dataclasses); never a god-class. The metric stays a pure `(pred, gold, *, grammar) -> float`; the facade assembles the `Report` around it.

## Conventions checklist (ruff/CLAUDE.md enforced)

- Module docstring on every `.py` (D100).
- Public API surfaced in `ek/__init__.py`.
- Favor functional over OOP; dataclasses + `collections.abc`; keyword-only beyond the 3rd arg.
- No magic numbers — thresholds/weights are kw-only args or come from Layer-A importance / external config (open-closed).
- Prefer `Iterable[T]` generators over building lists (per-slice/per-path streams).
- Optional engine -> extra + `@requires_extra` actionable hint; license gate for any non-permissive dep (Levenshtein GPL -> rapidfuzz; surya non-commercial; TorchCP LGPL import-only).
- Cite `ek_02` (decision table, gotchas, gold-standard harness) and `ek_06` (the typed-graph must-build) in docstrings/PRs by relative path under `misc/docs/`.
