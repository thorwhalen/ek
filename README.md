# ek

**ek (Evaluation Kit) — a framework for building Knowledge Evaluation systems**,
evaluating the outputs of information-extraction systems. OCR is treated as the noisiest *special case* of
a general problem, so the core is source-agnostic and the OCR pieces are optional.

```python
import ek

ek.score("hello wrld", "hello world")          # -> Score(value=0.0909..., metric='cer')
ek.score("hello wrld", "hello world", metric="wer").value   # 0.5
ek.evaluate([("ct", "cat"), ("dg", "dog")], metric="cer").aggregate   # 0.333... (global CER)
```

## What it does

Evaluating an extraction splits along two axes — is there a gold answer
(*reference-based*) or not (*reference-free*), and are we scoring one item or a whole
corpus. `ek` gives you both halves through two facades over one shared typed schema:

- **`score()` / `evaluate()`** — *reference-based*: compare against gold, one item or
  a corpus, the metric chosen by output type (string → CER/WER, record → field-F1),
  aggregated *correctly* (global error-rate accumulation, micro-F1; never a naive
  mean) with optional per-slice cuts.
- **`estimate_quality()`** — *reference-free*: gather signals → calibrate → validate →
  decide accept/flag/block, with no gold answer.

Everything swappable is a strategy injected with a smart default, so the simple call
works out of the box and every layer stays replaceable.

## Evaluate an OCR engine

The first concrete instance: measure OCR accuracy over a gold corpus. `ek` consumes
[`ocracy`](https://github.com/thorwhalen/ocracy)'s normalized `OcrResult`, so it can
benchmark any of its ~16 engines — or any `image -> OcrResult` callable of your own.

```python
import ek.ocr

gold = {"inv-1": {"image": "scan.png", "reference_text": "INVOICE 2024", "slice": "invoices"}}
report = ek.ocr.evaluate_ocr(
    "ocrmac", gold, metric="cer", normalize=["lower", "collapse_whitespace"], persist=True,
)
report.aggregate          # corpus CER
report.per_slice          # CER per document slice
report.detail["per_item"] # prediction, reference, score, confidence per document
```

Gold corpora, results, and runs persist to local `dol` stores under
`~/.local/share/ek/`.

## Evaluate an AI agent — in cost per successful task

The second instance. Agent evaluation is the *same* 2×2, but the evaluated object is an
**episode** (tool calls + observations ending in a final state) and the unit that matters is
**cost per successfully completed task**, not cost per token — because tokens spent on a failed
episode are pure waste.

```python
from ek.agents import TaskSpec, run_suite, per_million

tasks = [TaskSpec("t1", input="2+2", gold="4", slice="easy"),
         TaskSpec("t2", input="17*23", gold="391", slice="hard")]

report = run_suite(my_agent, tasks, k=8, price=per_million(3.0, 15.0))

report.pass_hat_k              # reliability: succeeds on ALL 8 trials (the production number)
report.pass_at_k               # capability: succeeds on ANY of 8 trials
report.success_ci              # a Wilson interval — a point estimate is not a result
report.cost["cost_per_success"]  # Cost-of-Pass: dollars per *successful* task (inf if none)
report.per_slice               # ...cut by difficulty
```

Two numbers, not one: an agent that "usually works" is not shippable, and `pass^k` is what
exposes it (a 90%-reliable agent is only ~43% reliable across 8 tries). Agent scores are
*stochastic*, so the regression gate compares **intervals, not points**, and refuses to compare
runs whose user-simulator or suite version changed:

```python
from ek.agents import agent_regression_gate, save_agent_baseline
save_agent_baseline(report, "v1")
assert agent_regression_gate(new_report, "v1")   # fails only on a *real* regression, not noise
```

Also included: BFCL-style tool-call correctness (cost-weighted by the tool grammar, so a wrong
argument to a *destructive* tool costs more), an order-sensitive trajectory distance, and an
LLM-as-judge signal — with `judge_validation()`, because an unvalidated judge is a liability, not
a metric. **None of this needs an extra**: `ek.agents` is pure-python, and the bridges duck-type,
so it scores an Inspect/DeepEval run without importing either.

## Install

```bash
pip install ek            # lean, permissive core (dol, config2py, jiwer, rapidfuzz)
pip install "ek[ocr]"     # + the ocracy OCR fleet (install engines via ocracy extras)
pip install "ek[agents]"  # + external agent harnesses to *run* (inspect-ai, deepeval, ragas)
pip install "ek[all]"     # + the permissive capability tiers (metrics, calibration, ...)
```

Heavier or copyleft/non-commercial libraries are never installed by default, and a CI license
gate enforces it; see the extras in `pyproject.toml`.

## CLI

```bash
ek cer "hello wrld" "hello world"     # character error rate
ek wer "hello wrld" "hello world"     # word error rate
ek pass-k 10 9 --k 8                  # pass@k (capability) vs pass^k (reliability)
ek cost-per-success 12.50 5           # dollars per successfully completed task
ek where                              # the local data folder
ek check tesseract                    # what an OCR engine needs to run
```

## For contributors

The architecture, conventions, and the research behind the design are documented for
agents and humans in **[AGENTS.md](AGENTS.md)**, the dev skills under `skills/`, and
the research reports under `misc/docs/`.
