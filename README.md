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

## Install

```bash
pip install ek            # lean, permissive core (dol, config2py, jiwer, rapidfuzz)
pip install "ek[ocr]"     # + the ocracy OCR fleet (install engines via ocracy extras)
pip install "ek[all]"     # + the permissive capability tiers (metrics, calibration, ...)
```

Heavier or copyleft/non-commercial libraries are never installed by default; see the
extras in `pyproject.toml`. Some capabilities (e.g. the cost-weighted typed-graph
metric and the ROVER consensus engine) are on the roadmap — see the tracking issue.

## CLI

```bash
ek cer "hello wrld" "hello world"     # character error rate
ek wer "hello wrld" "hello world"     # word error rate
ek where                              # the local data folder
ek check tesseract                    # what an OCR engine needs to run
```

## For contributors

The architecture, conventions, and the research behind the design are documented for
agents and humans in **[AGENTS.md](AGENTS.md)**, the dev skills under `skills/`, and
the research reports under `misc/docs/`.
