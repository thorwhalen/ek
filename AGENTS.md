# AGENTS.md — building `ek`

`ek` is a **framework for building Knowledge Evaluation systems**: tools that
evaluate the outputs of information-extraction (IE) systems. OCR is treated as the
*noisiest special case* of a general problem — the core is **source-agnostic**
(PDF, DOCX, tables, DB responses, LLM extractors); OCR specifics live in `ek.ocr`.

This file is the **map**. Detail lives in the dev skills (`skills/`) and the
research reports (`misc/docs/`); read those on demand. Don't inline them here.

---

## Architecture in one screen

Two facades cover the two halves of evaluation; both operate on the same typed
schema (the two-layer data model in `ek/base.py`):

```python
# Reference-based (offline)
score(pred, gold, *, grammar=None, metric=None, normalize=None, weights=None) -> Score   # one comparison
evaluate(cases, *, metric=None, grammar=None, normalize=None) -> Report                   # corpus (global accumulation, per-slice)
# Reference-free (online)
estimate_quality(extraction, *, sources=(), calibrator=None, validators=(), policy=None) -> QualityReport
```

- **Layer A — `GraphGrammar`** (frozen schema SSOT): `FieldSpec / NodeType /
  EdgeType` carry *types* **and** *importance/cost weights*. The lever for
  cost-sensitive metrics; also feeds constrained decoders.
- **Layer B — `AnnotatedExtraction`**: `FieldEstimate(value, raw_signals,
  confidence, findings, provenance, decision)` keyed by `NodePath`, riding
  *alongside* the grammar, never mutating it.
- A `Metric` is the atomic strategy (`(pred, gold, *, grammar=None) -> Score`);
  `score()` returns a `Score`, `evaluate()` returns a `Report` aggregated by the
  metric's own `.aggregate()` (global CER/WER, micro-F1 — **never** a naive mean).
- Everything swappable is a `typing.Protocol` resolved from `ek.registry` and
  injected keyword-only with smart defaults (open-closed). Missing optional deps
  raise `@requires_extra("…")` with an actionable `pip install ek[…]` hint.

**Persistence:** `config2py.AppData("ek")` → `~/.local/share/ek/`; `dol.Jsons`
stores per kind (`gold/ results/ calibrators/ corrections/ runs/`), grouped as a
`mall`. See `ek/stores.py`. Tests/ephemeral use pass `rootdir=` (or `EK_DATA_HOME`).

**Dependency direction (hard rule):** `ek -> ocracy` via the `ek[ocr]` extra;
**never** `ocracy -> ek`. `ek` core depends only on the `OcrResult` *shape*
(`.text`, `.blocks`, `.mean_confidence`), so it evaluates any `image -> OcrResult`
callable.

### Module map

| Module | Role |
|---|---|
| `ek/base.py` | the two-layer data model + strategy Protocols + type aliases (SSOT) |
| `ek/registry.py` | strategy registry, `@requires_extra`, `check_requirements` |
| `ek/stores.py` | `dol` + `AppData` persistence (JSON `MutableMapping` stores, mall) |
| `ek/canonicalize.py` | versioned, composable canonicalizers (normalize before scoring) |
| `ek/metrics/` | reference-based metrics (`StringMetric` CER/WER, `FieldMetric`) |
| `ek/facade.py` | `score`, `evaluate`, `estimate_quality` |
| `ek/ocr/` | the OCR instance: ocracy bridge, capability profiles, benchmark |
| `ek/tools.py`, `ek/__main__.py` | CLI (`argh`, `_dispatch_funcs` SSOT) |

Not yet built (roadmap): `ek/qe/` (signals incl. `rover.py`, calibrate, decide),
`ek/validate.py`, `ek/harness.py`, `ek/review.py`, `ek/monitor.py`. The flagship
must-builds are the **cost-weighted typed-graph distance** and the **ROVER** engine.

---

## Dev skills (`skills/`, surfaced to Claude via `.claude/skills/` symlinks)

Read the relevant one before working on its area:

- **`ek-dev-architecture`** — the orientation map; read FIRST. Includes the
  which-report-for-which-component table.
- **`ek-dev-licensing`** — consult before adding ANY dependency. The permissive
  core, the extras tiers, the 6 license landmines, and the CI license gate.
- **`ek-dev-add-metric`** — adding a reference-based metric (`score()` side).
- **`ek-dev-add-signal`** — adding a reference-free signal/calibrator/policy
  (`estimate_quality()` side).
- **`ek-dev-ocr`** — the OCR instance on top of `ocracy`.

Dev skills are living artifacts: revise them as the code changes; mark stale ones
with `metadata.delete-after: <milestone>` rather than deleting.

## Research reports (`misc/docs/`, read on demand)

`information-extraction-evaluation-conceptual-map.md` is the map into six reports:
`ek_01` OCR systems inventory · `ek_02` reference-based/offline eval · `ek_03`
reference-free QE/calibration/selective prediction · `ek_04` post-extraction
validation & correction · `ek_05` HITL/active-learning/monitoring · `ek_06` library
landscape & integration map (the architecture report).

---

## Conventions (enforced)

- **Module docstrings on every `.py`** (ruff `D100` is the active lint rule).
- Favor functional over OOP; `dataclasses` + `collections.abc`; keyword-only args
  beyond the 3rd; no magic numbers (keyword-only args / config; open-closed).
- Progressive disclosure: the simple call Just Works, everything is tunable.
- Storage as `MutableMapping` facades (`dol`), never god-classes. Prefer generators
  (`Iterable[T]`) over building lists.
- New deps go in `pyproject.toml`: lean permissive **core**, everything else an
  **extra**. Nothing copyleft/non-commercial is ever a default (see
  `ek-dev-licensing`).

## Run it

```bash
pip install -e ".[dev,ocr]"          # editable install with test + OCR deps
python -m pytest tests ek --doctest-modules -q
python -m ruff check ek
python -m ek cer "hello wrld" "hello world"   # CLI smoke
```

## Work tracking (GitHub as memory)

Track work in issues (the EPIC + sub-issues); journal decisions on the issue as you
go; record design rationale in Discussions; capture discovered work as stub issues
("Discovered from #N"). **Privacy:** never put absolute local paths, hostnames, or
secrets in issues/PRs/commits/committed files — they may be public. Session
handoffs go in the gitignored `.claude/handoffs/`.
