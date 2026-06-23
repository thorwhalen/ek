"""OCR evaluation -- ``ek``'s first concrete instance, built on ``ocracy``.

OCR is the noisiest special case of information extraction. This subpackage adds
the OCR-specific pieces on top of ``ek``'s source-agnostic core:

- :func:`ocracy_backend` / :func:`read_text` -- a thin bridge that turns
  ``ocracy``'s 16-engine fleet into an ``image -> OcrResult`` callable. The
  dependency arrow is strict: ``ek -> ocracy`` (via the ``ek[ocr]`` extra), never
  the reverse. ``ek`` core depends only on the ``OcrResult`` *shape* (``.text`` /
  ``.blocks`` / ``.mean_confidence``), so the benchmark below evaluates *any*
  ``image -> OcrResult`` callable -- ``ocracy`` or your own.
- :func:`profile` -- per-engine capability profiles (who emits real vs
  model-guessed geometry, who is calibrated, who yields table structure), distilled
  from ``misc/docs/ek_01``. ``ek`` treats these as claims to verify empirically.
- :func:`evaluate_ocr` / :func:`add_gold_item` -- run a fleet over a gold corpus,
  score with CER/WER (globally accumulated), slice the report, and persist gold +
  results to the ``dol`` stores under ``~/.local/share/ek/``.

Heavy engine SDKs and credentials are never required to import this module; they
are checked at call time via :func:`ek.registry.check_requirements`.
"""

from __future__ import annotations

from .benchmark import add_gold_item, evaluate_ocr
from .bridge import ocracy_backend, read_text
from .profiles import ENGINE_PROFILES, profile

__all__ = [
    "ocracy_backend",
    "read_text",
    "evaluate_ocr",
    "add_gold_item",
    "profile",
    "ENGINE_PROFILES",
]
