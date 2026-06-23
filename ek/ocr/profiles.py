"""Per-engine OCR capability profiles (what QE signal each engine actually emits).

Distilled from the 15-engine inventory in ``misc/docs/ek_01``. These describe what
an engine *claims* to surface; ``ek`` exists in part to verify such claims
empirically, so treat them as defaults/priors, not ground truth. The fields that
matter for evaluation:

- ``confidence_grain`` -- finest confidence granularity emitted: ``"symbol"`` >
  ``"word"`` > ``"line"`` > ``"none"``. Genuine per-symbol confidence exists only
  in Google Cloud Vision and Tesseract; many VLMs emit nothing usable.
- ``has_real_provenance`` -- whether bounding boxes are real geometry (vs
  model-guessed pixels from Claude/OpenAI, or char-offset only from Mistral).
- ``calibrated`` -- whether the confidence is calibrated (Mathpix is the lone
  calibrated emitter).
- ``tables`` -- whether the engine emits scoreable table cell structure.

Example:
    >>> profile("tesseract")["confidence_grain"]
    'symbol'
    >>> profile("claude-vision")["has_real_provenance"]
    False
    >>> profile("google_vision")["confidence_grain"]   # underscore form also accepted
    'symbol'
    >>> profile("totally-unknown-engine")["confidence_grain"]
    'unknown'
"""

from __future__ import annotations


def _p(confidence_grain: str, has_real_provenance: bool, calibrated: bool, tables: bool) -> dict:
    return {
        "confidence_grain": confidence_grain,
        "has_real_provenance": has_real_provenance,
        "calibrated": calibrated,
        "tables": tables,
    }


#: Engine id -> capability profile. Keys are ``ocracy``'s canonical (hyphenated)
#: backend ids; :func:`profile` also accepts underscore variants.
ENGINE_PROFILES: dict[str, dict] = {
    "tesseract": _p("symbol", True, False, False),
    "google-vision": _p("symbol", True, False, False),
    "aws-textract": _p("word", True, False, True),
    "azure-document-intelligence": _p("word", True, False, True),
    "mathpix": _p("word", True, True, False),  # the lone calibrated emitter
    "paddleocr": _p("line", True, False, True),
    "easyocr": _p("line", True, False, False),
    "rapidocr": _p("line", True, False, False),
    "ocrmac": _p("line", True, False, False),
    "doctr": _p("word", True, False, False),
    "ocr-space": _p("none", False, False, False),
    "gpt-4o-vision": _p("none", False, False, False),  # model-guessed geometry
    "claude-vision": _p("none", False, False, False),  # emits no confidence/geometry
    "mistral-ocr": _p("none", False, False, False),  # char-offset only, no real bbox
    "pix2tex-latex-ocr": _p("none", False, False, False),
    "trocr": _p("none", False, False, False),
    "trocr-handwritten": _p("none", False, False, False),
}

#: Returned for any engine not in the table.
DEFAULT_PROFILE = _p("unknown", False, False, False)

# Index by a normalized key so both "google-vision" and "google_vision" resolve.
_NORMALIZED = {k.replace("-", "_"): v for k, v in ENGINE_PROFILES.items()}


def profile(engine: str) -> dict:
    """Capability profile for an engine id (a safe default if unknown).

    Accepts ``ocracy``'s canonical hyphenated ids and their underscore variants.
    """
    if engine in ENGINE_PROFILES:
        return ENGINE_PROFILES[engine]
    return _NORMALIZED.get(engine.lower().replace("-", "_"), DEFAULT_PROFILE)
