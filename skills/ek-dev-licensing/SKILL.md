---
name: ek-dev-licensing
description: "Licensing guardrail for ek developers: consult BEFORE adding ANY dependency. Covers the license-tiering policy (lean permissive core vs opt-in extras), the register of 6 license landmines (GPL Levenshtein/python-Levenshtein, LGPL TorchCP, non-commercial surya-ocr, GPL Potato, proprietary Prodigy) with their permissive replacements, scanner-invisible traps where license terms live in repo files not PyPI metadata (zss BSD-3, mistralai namespace pkg, PubTabNet CDLA), the rule that nothing copyleft or non-commercial is ever a default, and the CI license gate (pip-licenses/reuse) that fails the build on GPL/AGPL/non-commercial in core..hitl. Use when picking a metric/calibration/validation/OCR/agreement/HITL backend, adding to pyproject extras, choosing an edit-distance or conformal or annotation library, hitting a \"GPL\"/\"non-commercial\"/\"RAIL\"/\"copyleft\" license, deciding which extra a dep belongs in, or whether a dep needs import-only quarantine."
metadata:
  audience: developers
---

Read this BEFORE adding ANY dependency to ek. ek ships with a lean permissive core and a downstream-permissive promise: a consumer must be able to `pip install ek` and redistribute commercially with zero copyleft or non-commercial exposure. Everything that breaks that promise is opt-in behind an extra, quarantined, or excluded entirely.

Authoritative source: `misc/docs/ek_06 -- library-landscape-and-integration-map.md` (every license/version claim verified against live PyPI + GitHub mid-2026). Cite it, and re-verify primary sources before trusting any claim here.

## The one rule

**Nothing copyleft (GPL/AGPL/LGPL) or non-commercial is EVER a default.** It is opt-in behind an extra, isolated, or out. The common case (`pip install ek`, no extras) resolves to MIT/BSD/Apache-2.0 only. Adding a backend is open-closed: it goes in an extra, you never edit a caller to take a new hard dep.

## Decision procedure: "I want to add lib X"

1. **Find the real license.** PyPI metadata lies (see traps below). Read the repo `LICENSE` file AND scan the code headers. For model-driven libs, check the WEIGHTS/API license separately from the code license.
2. **Classify the license:**
   - Permissive (MIT/BSD-2/BSD-3/ISC/Apache-2.0): eligible for an extra, possibly core.
   - Weak copyleft (LGPL): eligible ONLY as an import-only quarantined plugin with a permissive fallback. Never vendor, never patch, never static-bundle.
   - Strong copyleft (GPL/AGPL): NOT a Python dependency. HTTP/out-of-process only, or design-reference only.
   - Non-commercial / RAIL / proprietary: NOT shippable. Excluded; usable only behind an explicit, loudly-labelled non-permissive flag for internal research.
3. **Pick the tier/extra** (table below). Core only if it's genuinely needed everywhere AND light AND permissive.
4. **Does it need quarantine?** Only LGPL (e.g. TorchCP) goes import-only with a permissive fallback path.
5. **Add it as an extra in `pyproject.toml`**, behind `@requires_extra` for runtime-actionable install hints, and confirm the CI license gate still passes.

If X is one of the six landmines, stop and use the replacement.

## The tier map (extras = dependency weight AND license)

`pip install ek` pulls only `core`. Everything else is an explicit extra.

| Extra | What goes in it | License posture |
|---|---|---|
| `core` *(always)* | `rapidfuzz` (MIT), `jiwer` (Apache-2.0), `pydantic` (MIT), `numpy`, `networkx` (BSD-3), `dol`, `config2py` | permissive only |
| `metrics` | `sacrebleu` (Apache-2.0), `apted` (MIT), `zss` (**BSD-3**), `nervaluate` (MIT), `table-recognition-metric` (Apache-2.0) | permissive |
| `calibration` | `netcal` (Apache-2.0), `scikit-learn` (BSD-3), `MAPIE` (BSD-3), `crepes` (BSD-3), `puncc` (MIT) | permissive |
| `calibration-torch` *(heavy)* | `torch-uncertainty` (Apache-2.0) — pulls torch | permissive |
| `calibration-graph` *(LICENSE-GATED, quarantine)* | **`TorchCP` (LGPL-3.0)** — import-only | weak copyleft, isolated |
| `validation` | `jsonschema` (MIT), `fastjsonschema` (BSD-3), `pandera` (MIT), `pydantic-extra-types` (MIT) | permissive |
| `constrained` *(heavy)* | `outlines` (Apache-2.0), `xgrammar` (Apache-2.0) | permissive *code* (weights/API licensed separately) |
| `ocr` / `ocr-local` *(heavy)* | `tesserocr`/`pytesseract`, `rapidocr`, `python-doctr`, `easyocr`, `paddleocr`, macOS `ocrmac` (all Apache-2.0/MIT incl. standard weights) | permissive |
| `ocr-cloud` | `google-cloud-vision` (Apache-2.0), `boto3`+`amazon-textract-response-parser`/`-textractor` (Apache-2.0), `azure-ai-documentintelligence` (MIT), `mistralai` (Apache-2.0), `mpxpy` (MIT), `openai` (Apache-2.0), `anthropic` (MIT) | permissive **clients**; services are pay-per-call (a cost concern, not a license one) |
| `agreement` | `uqlm` (Apache-2.0), `lm-polygraph` (MIT, heavy); ROVER builds on core `rapidfuzz`/`jiwer` | permissive (vet downloaded NLI/embedding weights; do NOT enable `lm-polygraph`'s non-commercial COMET extra) |
| `hitl` | `label-studio-sdk` (Apache-2.0), `argilla` (Apache-2.0), `cvat-sdk` (MIT) | permissive clients; servers run out-of-process |
| **Excluded** *(license)* | `surya-ocr`, `Levenshtein`/`python-Levenshtein`, `Potato`, `Prodigy` | NOT permissive — not installable extras |

The dependency direction `ek -> ocracy` is the `ek[ocr]` extra; never `ocracy -> ek`. ek core depends only on the `OcrResult` SHAPE, so it stays out of every OCR engine's license footprint.

## The six landmines — keep these out of the permissive build

All verified against repo `LICENSE` files / PyPI license expressions in mid-2026.

| Library | Tier | License | Problem | Do instead |
|---|---|---|---|---|
| `Levenshtein` | surface | **GPL-2.0-or-later** | program copyleft; disqualifying for permissive redistribution | `rapidfuzz.distance.Levenshtein` (MIT) — identical `distance`/`ratio`/`editops`/`median`, faster |
| `python-Levenshtein` | surface | **GPL-2.0-or-later** | same; ALSO a transitive contaminant via legacy `fuzzywuzzy` code | `rapidfuzz` + CI gate to block the transitive pull |
| `TorchCP` | calibration-graph | **LGPL-3.0** (repo only; PyPI metadata blank) | weak copyleft — fine imported/dynamically linked, but patching or static-bundling triggers copyleft. The ONLY lib with graph-node & LLM-sequence conformal | quarantine in `calibration-graph`, import-only, never vendor/modify. Permissive fallback: `torch-uncertainty`/`MAPIE`/`crepes` for non-graph paths |
| `surya-ocr` | ocr | code Apache-2.0; **weights modified AI-Pubs Open RAIL-M (non-commercial above $5M funding/revenue)** | highest-accuracy OCR but weights are non-commercial. The PyPI "Apache-2.0" classifier covers ONLY the code — a trap | keep out of `ocr-local`; expose only behind an explicit, labelled non-permissive engine flag for internal research under the threshold |
| `Potato` | hitl | **GPL-3.0-or-later** (relicensed in v2.6.0) | strong copyleft would impose GPL on our distribution if linked | run as a separately-deployed standalone app over HTTP only (no code linkage), or prefer Label Studio |
| `Prodigy` | hitl | **commercial / proprietary** (pay-once, closed source) | no open-source tier; cannot ship as a dependency | study its `prefer_uncertain` active-learning sorter as a design reference; build the equivalent on Label Studio / our own queue |

## Scanner-invisible traps (the dangerous part)

License scanners read PyPI metadata. Some real terms live ONLY in repo files or weights — invisible to `pip-licenses`. These four are why a human + the CI gate must both check:

- **`TorchCP`** — LGPL-3.0/GPL-3.0 detected by GitHub, but **PyPI metadata carries no license field** → scanners flag "unknown," not "copyleft."
- **`surya-ocr`** — PyPI `Apache-2.0` classifier reflects the CODE only; the non-commercial RAIL-M terms live with the **weights**.
- **`zss`** — actually **BSD-3-Clause, NOT MIT** (PyPI omits the classifier; read the vendored `LICENSE`). Still permissive — a correction, not a blocker.
- **PubTabNet** — repo top-level `LICENSE.md` is **CDLA-Permissive-1.0**, but only `src/metric.py` carries the **Apache-2.0** header we actually rely on. Cite the FILE header, not the repo license.
- **`mistralai`** — PyPI `license=None` (file-only Apache-2.0); also a **namespace package** → import `from mistralai.client import Mistral`, not `from mistralai import Mistral`. License-clean once you read the repo `LICENSE`.
- **`nonconformist`** — genuinely MIT per its repo despite a blank PyPI license field; avoid on staleness grounds (~9 yrs), not license.

Rule of thumb: **PyPI "unknown"/`license=None` ≠ permissive.** Open the repo `LICENSE`, and for any model-driven lib, the weights/API license too.

## The CI license gate (must exist, must fail the build)

A CI gate (`pip-licenses` and/or `reuse`) MUST fail the build if anything resolving in `core` through `hitl` (i.e. the shippable extras) is GPL/AGPL/non-commercial. It is the safety net for two failure modes:

1. **Transitive contamination** — `python-Levenshtein` sneaks in via legacy `fuzzywuzzy`-style code.
2. **Metadata-blind copyleft** — `TorchCP`/`surya-ocr` carry their terms only in repo files, so a pure-metadata scan misses them; the gate must treat "unknown" license as a failure (or maintain an explicit allowlist) and must scope-exclude only the explicitly-quarantined `calibration-graph` extra and the labelled non-permissive research flags.

The gate is necessary because scanners alone will miss the repo-file traps above — never rely on a green scanner as proof of compliance.

### Two resolved gate findings (audited decisions, #10)

- **`krippendorff` was GPL-3.0** and was the *only* dep behind the (now-removed) `harness` extra. Rather than quarantine it, Krippendorff's α was **reimplemented in pure Python** in `ek/harness.py` (coincidence-matrix method; nominal/ordinal/interval/ratio; missing data) — verified to <1e-9 vs the reference package. IAA is now permissive **core**, no extra. Prefer a clean-room reimplementation over quarantine when the algorithm is small and well-specified.
- **`nvidia-*` CUDA runtime wheels** (proprietary) are pulled transitively by the permissive `torch` (e.g. `uqlm` in `[agreement]`). They are NVIDIA's *redistributable GPU runtime* (driver-layer libs the user installs for acceleration; absent on a CPU-only install), not a library ek ships — so they are cleared by an audited **`nvidia-` name-prefix allowlist** in `check_licenses.py`, NOT by relaxing the GPL/non-commercial rules. This is the one standing prefix override; keep it justified and narrow.

## Runtime hints

Optional backends raise actionable install hints via the `@requires_extra` decorator (registry-resolved strategy protocols, injected keyword-only with smart defaults; open-closed via entry points). A missing extra tells the user exactly which `pip install ek[...]` to run. The non-permissive research flags (surya, etc.) must additionally warn loudly at the call site that the chosen engine is non-shippable.

## When in doubt

If you cannot determine a library's true license from primary sources, treat it as non-permissive and do not add it to a shippable extra. Flag it for the maintainer. A wrong permissive assumption poisons every downstream consumer of ek.
