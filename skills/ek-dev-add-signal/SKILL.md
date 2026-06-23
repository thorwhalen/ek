---
name: ek-dev-add-signal
description: "How to add a reference-free quality-estimation component to ek's online estimate_quality() side: a new Signal (deterministic verifier / intrinsic-confidence / logprob / agreement-ROVER / auxiliary-QE / LLM-judge), a Calibrator (Platt/Temperature/Isotonic), or a DecisionPolicy/SelectivePolicy (ConformalGate / RiskControlGate / CostSensitiveGate). Use when building or extending the signal->calibrate->decide pipeline, wiring a Strategy Protocol into the registry, adding confidence/uncertainty/agreement signals, ROVER voting, conformal prediction (MAPIE/crepes/TorchCP), calibration (netcal/sklearn), ECE/D-ECE, risk-coverage curves, or selective prediction / accept-flag-block gating for extraction or OCR quality. Not for offline reference-based metrics (that is the score()/add-metric side)."
metadata:
  audience: developers
---

# Add a reference-free QE component to `ek` (the `estimate_quality()` side)

This skill is for developers **building `ek`**. It tells you how to add a new
**Signal**, **Calibrator**, or **DecisionPolicy** to the reference-free (online,
no-gold) quality-estimation pipeline behind the `estimate_quality()` facade.

If you are adding an **offline, reference-based metric** (you have gold), you want
the `score()` side instead — wrong skill, stop here.

Authoritative background: `misc/docs/ek_03 -- Reference-Free Quality Estimation,
Confidence, Calibration & Selective Prediction.md` (this is THE report for this
skill) and `misc/docs/ek_04 -- Post-Extraction Validation & Correction (incl.
Post-OCR).md` (verifier-layer detail). Read them on demand — do not paste them.

## The pipeline you are plugging into

`estimate_quality()` runs a strict three-stage pipeline. Every component you add
belongs to exactly one stage.

```
signal  ──►  calibrate  ──►  validate  ──►  decide
(raw score)  (-> probability)  (cross-checks)  (accept/flag/block)
```

```python
def estimate_quality(
    extraction: AnnotatedExtraction,
    *,
    sources=(),            # extra OcrResult-shaped inputs (for agreement signals)
    calibrator=None,       # Calibrator; default Platt (smart default)
    validators=(),         # Validator strategies (cross-field/cross-source)
    policy=None,           # DecisionPolicy / SelectivePolicy; default a CostSensitiveGate
) -> QualityReport: ...
```

It consumes a Layer-B `AnnotatedExtraction(grammar, estimates)` and produces a
`QualityReport`. Per field it fills a `FieldEstimate(value, raw_signals,
confidence, findings, provenance, decision in {accept, flag, block})`. Your Signal
populates `raw_signals`; the Calibrator turns those into `confidence`; Validators
append `Finding`s; the DecisionPolicy sets `decision`.

### Three HARD RULES (violating any of these is a bug, not a style choice)

1. **Calibration is non-optional. Never gate raw posteriors or raw logprobs.** A
   raw score of 0.9 does not mean "correct 90% of the time" — RLHF'd LLMs are
   systematically overconfident, Tesseract conf is "locally informative but
   globally meaningless." A Signal emits a raw score; a Calibrator must run before
   any DecisionPolicy reads it. (ek_03 §2, Caveats.)
2. **Conformal guarantees are MARGINAL, not conditional.** Distribution-free
   per-field-type coverage is provably impossible (Barber et al. 2019). If you
   want per-field-type validity, use **Mondrian / class-conditional** calibration
   (calibrate separately per `NodeType`/`FieldSpec`). State which guarantee you
   ship. (ek_03 §3, Caveats.)
3. **Calibrate at the granularity of the DECISION.** If you gate on whole fields,
   build a binary "field-correct?" target and calibrate the aggregated *field*
   score against it. Do **not** calibrate per-token and hope it composes. (ek_03 §2.)

## Strategy Protocols and the registry

Every pluggable component is a `typing.Protocol` callable, registry-resolved, and
injected keyword-only with a smart default (open-closed; registered via entry
points; missing optional deps raise actionable hints via `@requires_extra`).

The relevant protocols: `Signal`, `Calibrator`, `DecisionPolicy`/`SelectivePolicy`,
`Validator`, plus the aliases `ConfidenceSource` and `OcrBackend`.

Pattern for adding any one of them:

1. Write the callable matching the protocol (module-level function or small
   dataclass with `__call__`; favor functional over OOP).
2. Give the module a docstring (ruff D100 is enforced — every `.py` needs one).
3. Register it under a name in the relevant registry; expose it via an entry point
   so third parties get it open-closed.
4. If it needs a non-core dependency, guard the import behind `@requires_extra`
   (e.g. `@requires_extra("conformal")`) so the error tells the user exactly what
   to `pip install`.
5. Export anything public from `ek/__init__.py`.
6. No magic numbers: thresholds, `alpha`, `T`, the `rho` ratio, etc. are
   keyword-only args with documented defaults or come from external config.

Smart-default discipline: the simple call `estimate_quality(extraction)` must work
with zero plugins specified. Tunability is all opt-in keyword args (progressive
disclosure).

## Choosing where your component goes

### 1. Adding a Signal — respect the strict COST ORDER

Signals are tried cheapest-first; cheaper reliable signals make expensive ones
unnecessary. The families, in the order they must run (ek_03 §1, §5):

| Order | Family | Cost | What it is |
|------|--------|------|------------|
| 1 (always first) | `VerifierSignal` (deterministic) | ~free | schema/type, regex/format, checksums (Luhn, IBAN, ISBN via `python-stdnum`), cross-field totals, dictionary/gazetteer, n-gram LM perplexity |
| 2 | `IntrinsicConfidenceSignal` / `LogprobSignal` | free (already computed) | per-unit posteriors, or token logprobs aggregated to field level |
| 3 | `AgreementSignal` | N× (engines or samples) | ROVER multi-engine vote (must-build) + uqlm self-consistency |
| 4 | `AuxiliaryQESignal` | one model load | trained reference-free estimator (CometKiwi idea ported to IE; ConfBERT-style "field-correct" classifier) |
| 5 (last resort) | `LlmJudgeSignal` | K× / $$$ | LLM self-consistency vote, semantic-entropy clustering, or LLM-as-judge |

**The deterministic `VerifierSignal` layer runs first and always** — it is free and
catches confident-but-wrong errors (a confidently misread but plausible digit)
that no confidence signal will. See ek_04 §Layers 0–3 for the verifier menu
(canonicalize, type/range/enum/regex, cross-field, lexicon via `rapidfuzz`/
`symspellpy`, KenLM/PLL perplexity). Use **rapidfuzz, not python-Levenshtein**
(GPL).

A Signal returns raw evidence into `FieldEstimate.raw_signals`; it must NOT emit a
final probability or a decision.

```python
from typing import Protocol, Mapping
from collections.abc import Iterable

class Signal(Protocol):
    def __call__(
        self, extraction: 'AnnotatedExtraction', *, sources: tuple = ()
    ) -> Mapping['NodePath', float]:
        """Map each field NodePath to a RAW (uncalibrated) score. No decisions."""
```

#### LogprobSignal: aggregation is a pluggable `Aggregator`

Raw token-logprob sum is length-biased. Aggregation is a modeling decision — keep
it pluggable, never hard-code one. Provide these `Aggregator` strategies:

- `geo_mean` — `exp(mean(log p))`; overall field plausibility (the family Mistral
  OCR reports).
- `length_normalized` — `exp(Σ log p / T**alpha)`, default `alpha≈0.6` (Wu et al.
  found 0.6–0.7 best); `alpha` is a keyword arg, not a literal.
- `min` — weakest token; use to catch a single transposed digit in an amount.
- `mean`.

```python
def field_score(token_logps, *, aggregator=geo_mean, alpha=0.6) -> float:
    ...  # dispatch on the injected aggregator; alpha only used by length_normalized
```

#### AgreementSignal: the ROVER must-build

ROVER (Fiscus 1997) is a flagship must-build — there is no off-the-shelf
equivalent we want to depend on. Build it as an N-way aligner + per-slot voter
that doubles as a confidence source (positions where engines disagree are exactly
where to flag). Interface:

```python
def rover(
    hypotheses: 'Iterable[OcrResult]',   # ek depends only on the OcrResult SHAPE
    *,
    use_confidence: bool = True,         # confidence-weighted vote vs frequency-only
) -> 'RoverConsensus':
    """N-way align hypotheses into a word transition network (iterative DP
    alignment), pick each slot by (optionally confidence-weighted) majority vote,
    and emit per-position agreement as a confidence signal.

    Returns consensus tokens, per-slot vote share, and per-position agreement in
    [0,1] suitable as a raw Signal score. Cost ~ O(N·l·L·L'), so designed for a
    handful of engines."""
```

ek depends only on the `OcrResult` shape (`text`; `blocks=[TextBlock(text, bbox,
confidence, level, language, meta)]`; `raw`; `meta`; confidence normalized to
[0,1]) so ROVER can fuse **any** image→OcrResult callable. The optional OCR
engines live behind the `ek[ocr]` extra (`ek -> ocracy`, never the reverse).
`AgreementSignal` also wraps `uqlm` self-consistency for stochastic LLM extractors.

### 2. Adding a Calibrator

Maps a raw scalar score → a meaningful probability. Wraps `netcal` / `sklearn`;
never reinvent these.

```python
class Calibrator(Protocol):
    def fit(self, scores, correct) -> 'Calibrator': ...   # correct: bool "field-correct?"
    def __call__(self, scores): ...                       # -> calibrated prob in [0,1]
```

Provide three, by what input you have (ek_03 §2):

- **Platt** (logistic on any scalar) — **the default** for aggregated OCR conf or
  aggregated logprobs (no logits needed). `sklearn` `CalibratedClassifierCV(method=
  "sigmoid")`.
- **Temperature scaling** — single scalar `T` on logits; only when you have logits.
  Does not change the argmax. `netcal.scaling.TemperatureScaling`.
- **Isotonic** — non-parametric monotonic; more flexible, needs more calibration
  data, can overfit small sets.

Measure calibration with **ECE** (and **D-ECE** for localized/bbox OCR outputs) +
a reliability diagram via `netcal`. For per-field-type validity, fit
class-conditional (Mondrian) — one calibrator per `FieldSpec`/`NodeType`. Persist
fitted calibrators in the `calibrators/` store (`dol.Jsons(get_artifact_dir(
"calibrators"))`; `dol.cache_this` for baselines). Calibration is dataset- and
model-specific and decays — support re-fit on drift.

### 3. Adding a DecisionPolicy / SelectivePolicy

Turns calibrated probabilities into `accept` / `flag` / `block`, with a guarantee.
Reads ONLY calibrated scores (Hard Rule 1).

```python
class DecisionPolicy(Protocol):
    def __call__(
        self, calibrated: Mapping['NodePath', float], *, grammar=None
    ) -> Mapping['NodePath', str]:   # -> {"accept","flag","block"}
        ...
```

Provide (ek_03 §3, §4):

- **`ConformalGate`** — split conformal: per-instance conformal p-value / prediction
  set with finite-sample marginal coverage `≥ 1−alpha`. Wrap **MAPIE** or
  **crepes** (use crepes for Mondrian/class-conditional, i.e. per-field-type). For
  PyTorch on-device, **TorchCP** — but it is **LGPL: import-only / quarantine,
  never vendor** (license gate in CI).
- **`RiskControlGate`** — Conformal Risk Control: bound a monotone loss (FNR,
  token-F1, graph distance). "Flagged-as-OK set has FNR ≤ 5%" is a CRC statement.
- **`CostSensitiveGate`** — the **default**. Pick the accept threshold from the cost
  ratio `rho = c_FN / c_FP` on the *calibrated* probability. `rho` is a keyword
  arg / config value, never a literal.

Expose the **risk–coverage curve** (selective risk among accepted vs coverage =
fraction auto-acted-on) so stakeholders choose the operating point explicitly. Map
the chosen point to the three `decision` states. Track realized coverage vs target
as a production metric and re-calibrate on drift (exchangeability is the
load-bearing assumption; it breaks under shift and is violated by autoregressive
LLM generation — ship LLM conformal guarantees as approximate, at claim/field
level).

## Persistence and CLI

- Calibration sets, fitted calibrators, and runs are `dol` MutableMapping stores
  (`gold/`, `calibrators/`, `runs/`), grouped as a dol mall via
  `config2py.AppData("ek")` → `~/.local/share/ek/`. Stores are facades, never
  god-classes (see the `python-storage` skill).
- New user-facing functions: add to the `_dispatch_funcs` SSOT and they surface in
  the CLI via `argh` (`__main__.py` `dispatch_with_namespaces`).

## Checklist before you finish

- [ ] Component matches exactly one protocol (`Signal` / `Calibrator` /
      `DecisionPolicy`) and one pipeline stage.
- [ ] Signal placed at the correct cost tier; the deterministic verifier tier still
      runs first.
- [ ] No raw score reaches a DecisionPolicy uncalibrated (Hard Rule 1).
- [ ] Conformal guarantee documented as marginal; Mondrian used if per-field-type
      claimed (Hard Rule 2).
- [ ] Calibration target is at decision granularity (Hard Rule 3).
- [ ] Aggregation / `alpha` / `T` / `rho` / thresholds are keyword args or config,
      not magic numbers.
- [ ] Registered + entry point; optional deps behind `@requires_extra`; copyleft/
      non-commercial deps behind an extra + the CI license gate (rapidfuzz not
      Levenshtein; TorchCP import-only).
- [ ] Module docstring present (D100); public API exported from `ek/__init__.py`.
- [ ] Collection-returning helpers are generators (`Iterable[T]`), keyword-only
      beyond the 3rd arg.
