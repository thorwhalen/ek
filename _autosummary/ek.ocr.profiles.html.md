# ek.ocr.profiles

Per-engine OCR capability profiles (what QE signal each engine actually emits).

Distilled from the 15-engine inventory in `misc/docs/ek_01`. These describe what
an engine *claims* to surface; `ek` exists in part to verify such claims
empirically, so treat them as defaults/priors, not ground truth. The fields that
matter for evaluation:

- `confidence_grain` – finest confidence granularity emitted: `"symbol"` >
  `"word"` > `"line"` > `"none"`. Genuine per-symbol confidence exists only
  in Google Cloud Vision and Tesseract; many VLMs emit nothing usable.
- `has_real_provenance` – whether bounding boxes are real geometry (vs
  model-guessed pixels from Claude/OpenAI, or char-offset only from Mistral).
- `calibrated` – whether the confidence is calibrated (Mathpix is the lone
  calibrated emitter).
- `tables` – whether the engine emits scoreable table cell structure.

### Example

```pycon
>>> profile("tesseract")["confidence_grain"]
'symbol'
>>> profile("claude-vision")["has_real_provenance"]
False
>>> profile("google_vision")["confidence_grain"]   # underscore form also accepted
'symbol'
>>> profile("totally-unknown-engine")["confidence_grain"]
'unknown'
```

### Module Attributes

| [`ENGINE_PROFILES`](#ek.ocr.profiles.ENGINE_PROFILES)   | Engine id -> capability profile.          |
|--------------------------------------------------------------------|-------------------------------------------|
| [`DEFAULT_PROFILE`](#ek.ocr.profiles.DEFAULT_PROFILE)   | Returned for any engine not in the table. |

### Functions

| [`profile`](#ek.ocr.profiles.profile)(engine)   | Capability profile for an engine id (a safe default if unknown).   |
|--------------------------------------------------------------------|--------------------------------------------------------------------|

### ek.ocr.profiles.DEFAULT_PROFILE *= {'calibrated': False, 'confidence_grain': 'unknown', 'has_real_provenance': False, 'tables': False}*

Returned for any engine not in the table.

### ek.ocr.profiles.ENGINE_PROFILES *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)]* *= {'aws-textract': {'calibrated': False, 'confidence_grain': 'word', 'has_real_provenance': True, 'tables': True}, 'azure-document-intelligence': {'calibrated': False, 'confidence_grain': 'word', 'has_real_provenance': True, 'tables': True}, 'claude-vision': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}, 'doctr': {'calibrated': False, 'confidence_grain': 'word', 'has_real_provenance': True, 'tables': False}, 'easyocr': {'calibrated': False, 'confidence_grain': 'line', 'has_real_provenance': True, 'tables': False}, 'google-vision': {'calibrated': False, 'confidence_grain': 'symbol', 'has_real_provenance': True, 'tables': False}, 'gpt-4o-vision': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}, 'mathpix': {'calibrated': True, 'confidence_grain': 'word', 'has_real_provenance': True, 'tables': False}, 'mistral-ocr': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}, 'ocr-space': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}, 'ocrmac': {'calibrated': False, 'confidence_grain': 'line', 'has_real_provenance': True, 'tables': False}, 'paddleocr': {'calibrated': False, 'confidence_grain': 'line', 'has_real_provenance': True, 'tables': True}, 'pix2tex-latex-ocr': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}, 'rapidocr': {'calibrated': False, 'confidence_grain': 'line', 'has_real_provenance': True, 'tables': False}, 'tesseract': {'calibrated': False, 'confidence_grain': 'symbol', 'has_real_provenance': True, 'tables': False}, 'trocr': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}, 'trocr-handwritten': {'calibrated': False, 'confidence_grain': 'none', 'has_real_provenance': False, 'tables': False}}*

Engine id -> capability profile. Keys are `ocracy`’s canonical (hyphenated)
backend ids; [`profile()`](#ek.ocr.profiles.profile) also accepts underscore variants.

### ek.ocr.profiles.profile(engine)

Capability profile for an engine id (a safe default if unknown).

Accepts `ocracy`’s canonical hyphenated ids and their underscore variants.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
