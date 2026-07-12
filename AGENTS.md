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
estimate_quality(extraction, *, sources=(), signals=(), calibrator=None, validators=(), policy=None, agreement=True, assume_calibrated=False) -> QualityReport
```

`sources` are *additional hypotheses* of the same content (strings / `OcrResult`-shaped
objects) that ROVER fuses into an agreement signal; `signals` are explicit `Signal`
callables. `extraction` may be a value, a `FieldEstimate`, or a whole
`AnnotatedExtraction` (scored per field). Calibration is non-optional before gating
(Hard Rule 1): a `policy` with no `calibrator` warns unless `assume_calibrated=True`.

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
| `ek/metrics/` | reference-based metrics (`StringMetric` CER/WER, `FieldMetric`, typed-graph) |
| `ek/qe/` | reference-free QE: `rover.py` (ROVER agreement), `verifiers.py`, `signals.py`, `calibrate.py`, `decide.py` |
| `ek/facade.py` | `score`, `evaluate`, `estimate_quality` |
| `ek/harness.py` | offline harness: `evaluate_store`, regression gate, baselines, IAA |
| `ek/ocr/` | the OCR instance: ocracy bridge, capability profiles, benchmark |
| `ek/agents/` | the **agent instance**: episodes, pass^k, cost-per-success, tool-call/trajectory metrics, judge, agent harness |
| `ek/tools.py`, `ek/__main__.py` | CLI (`argh`, `_dispatch_funcs` SSOT) |

### The agent instance (`ek/agents/`) — cost per successful task

The second concrete instance (mirrors `ek/ocr/`). Agent evaluation is the **same 2×2** on a
different object: the evaluated thing is an **episode** (tool calls + observations ending in a
final state) and the unit is **cost per successfully completed task**, not cost per token.
Layer A becomes a **task/tool grammar** (`ToolSpec`/`TaskSpec` build a `GraphGrammar`;
`FieldSpec.importance` = the cost of a wrong argument, so a wrong argument to a *destructive*
tool is not one unit of error); Layer B becomes the `Episode` (`Trajectory`, `Cost`,
`RunProvenance`). **`ek.agents` adds zero dependencies** — it is pure-python, and the bridges
duck-type, so ek scores an Inspect/DeepEval run without importing either.

Four design rules that are easy to get wrong (they were caught in review — do not "simplify" them back):

1. **`pass^k` is not a `Metric`.** It is cross-task (k trials per task); `Metric.__call__(pred, gold)`
   cannot express it. It lives in the harness as pure functions + a `ReliabilityReport`.
2. **`TrajectoryMetric` must not use the GED engine.** `networkx.graph_edit_distance` is an
   isomorphism search: it **ignores step order**, caps at `max_nodes=60`, and is timeout-nondeterministic.
   Trajectories are linear → a Needleman–Wunsch sequence edit distance that reuses the *cost model*.
3. **Tool calls need a multiset matcher.** `FieldMetric` keys by field name; two `search(q=…)` calls
   collide. Match first (`match_calls`), *then* reuse `FieldMetric`'s TP/FP/FN counting.
4. **Only a reference-free, criteria-only judge is a `Signal`.** A reference-based judge is a `Metric`;
   pairwise judging needs two outputs (a helper). And **Hard Rule 1 applies to judges** — never gate an
   uncalibrated judge score.

Plus: agent metrics are **stochastic**, so the scalar `regression_gate` is unsound for them — use
`agent_regression_gate`, which tests the **difference** `current - baseline` with both runs'
uncertainty folded in (a Newcombe interval on the success rate; a combined-SE bound on `pass^k`).
Two tempting shortcuts are *both* wrong and both were shipped-then-fixed here: comparing our
interval to the baseline's **point** (the baseline is itself noisy → one flake reads as a
regression), and asking whether two 95% CIs **overlap** (that is a ~0.5% test, not a 5% one → a
15-point drop goes invisible). Record `RunProvenance` (seed, model, **user-simulator**, suite
version, scaffold); the gate *refuses* to compare runs whose simulator or suite version changed,
and refuses an empty run **or an empty baseline** (which would be a permanent free pass).

Both flagship must-builds are now built: the **cost-weighted typed-graph distance**
(`ek/metrics/graphs.py`) and the **ROVER** engine (`ek/qe/rover.py`). The reference-free
QE pipeline (signal → calibrate → validate → decide) ships pure-Python by default, with
library backends (netcal/sklearn/MAPIE/crepes, uqlm) opt-in behind extras. Not yet built
(roadmap): `ek/validate.py` (the six-layer flag-vs-correct pipeline, #7), `ek/review.py`,
`ek/monitor.py`.

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
- **`ek-dev-agents`** — the agent instance: episodes, `pass^k`, cost-per-successful-task,
  tool-call/trajectory metrics, the judge, and the variance-aware gate.

Dev skills are living artifacts: revise them as the code changes; mark stale ones
with `metadata.delete-after: <milestone>` rather than deleting.

## Research reports (`misc/docs/`, read on demand)

**Information extraction (`ek_01`–`ek_06`).**
`information-extraction-evaluation-conceptual-map.md` is the map into six reports:
`ek_01` OCR systems inventory · `ek_02` reference-based/offline eval · `ek_03`
reference-free QE/calibration/selective prediction · `ek_04` post-extraction
validation & correction · `ek_05` HITL/active-learning/monitoring · `ek_06` library
landscape & integration map (the architecture report).

**Agents & assistants (`ek_07`–`ek_12`).** `ek_07` is the map into five more:
`ek_07` conceptual map (the 2×2 lifted onto an episode; pass@k vs pass^k) · `ek_08`
task-success & outcome-based eval (state-based oracles, BFCL, contamination) · `ek_09`
LLM-as-judge & reference-free agent QE (biases, judge validation, RAG faithfulness) ·
`ek_10` trajectory/tool-use/multi-turn (+ memory, self-reflection, safety) · `ek_11`
cost per successful task (Cost-of-Pass, routing/cascades, error bars) · `ek_12` agent-eval
library landscape & integration map (**read before writing agent code**).

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
