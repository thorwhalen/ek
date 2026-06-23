"""String metrics: Character/Word Error Rate (CER/WER).

CER and WER are graded edit-distance metrics -- the right baseline for OCR/ASR and
free-text fields, far less brittle than exact match. Two correctness rules from the
research drive the design:

1. **Accumulate globally.** A corpus error rate is ``sum(edits) / sum(reference
   length)`` over the whole corpus -- never the mean of per-item rates (variable
   lengths make that mathematically wrong). So each :class:`~ek.base.Score` carries
   raw ``edits`` and ``ref_len`` in its ``detail``, and :meth:`StringMetric.aggregate`
   sums them. :func:`~ek.facade.evaluate` uses this.
2. **CER/WER are error rates, not bounded similarities.** A hypothesis longer than
   the reference pushes the rate above 1.0 -- itself a useful hallucination signal.
   ``Score.value`` is the error rate (lower is better; ``detail["higher_is_better"]
   is False``).

The canonical backend is ``jiwer`` (Apache-2.0, ``rapidfuzz`` C++ core), which also
yields the substitution/deletion/insertion/hit breakdown. ``ek`` prefers it,
falls back to ``rapidfuzz`` directly, then to a pure-Python edit distance, so the
metric works even in a bare environment.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..base import GraphGrammar, Score

# -- pick an edit-distance backend once, at import --------------------------
try:  # preferred: full alignment + S/D/I/H counts
    import jiwer as _jiwer

    _BACKEND = "jiwer"
except Exception:  # pragma: no cover - exercised only without jiwer
    _jiwer = None
    try:
        from rapidfuzz.distance import Levenshtein as _RFLev

        _BACKEND = "rapidfuzz"
    except Exception:
        _RFLev = None
        _BACKEND = "python"


_MISSING = object()


def _as_text(obj: Any) -> str:
    """Coerce a prediction/reference to text (duck-types ``OcrResult.text``).

    Null-safe: ``None`` (or an object whose ``.text`` is ``None`` -- the VLM/markdown
    OCR case) becomes the empty string, never the literal ``"None"``.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    text = getattr(obj, "text", _MISSING)
    if text is _MISSING:  # not an OcrResult-shaped object; last resort
        return str(obj)
    return text if isinstance(text, str) else ""


def _py_levenshtein(a: Sequence, b: Sequence) -> int:
    """Pure-Python Levenshtein distance over any two sequences (fallback only)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _edit_counts(ref: str, hyp: str, *, level: str) -> tuple[int, int, dict]:
    """Return ``(edits, ref_len, extra)`` for CER (``level='cer'``) or WER."""
    if level == "wer":
        ref_units: Sequence = ref.split()
        hyp_units: Sequence = hyp.split()
    else:
        ref_units, hyp_units = ref, hyp
    ref_len = len(ref_units)

    # Empty reference: edits = pure insertions; avoid backend edge cases.
    if ref_len == 0:
        return len(hyp_units), 0, {}

    if _BACKEND == "jiwer":
        proc = _jiwer.process_words if level == "wer" else _jiwer.process_characters
        out = proc(ref, hyp)
        edits = out.substitutions + out.deletions + out.insertions
        extra = {
            "substitutions": out.substitutions,
            "deletions": out.deletions,
            "insertions": out.insertions,
            "hits": out.hits,
        }
        return edits, ref_len, extra
    if _BACKEND == "rapidfuzz":
        return _RFLev.distance(ref_units, hyp_units), ref_len, {}
    return _py_levenshtein(ref_units, hyp_units), ref_len, {}


class StringMetric:
    """CER (``mode='cer'``) or WER (``mode='wer'``) as a :class:`~ek.base.Metric`.

    Args:
        mode: ``"cer"`` (character) or ``"wer"`` (word) error rate.
        canonicalizer: Optional ``str -> str`` applied to both sides before scoring
            (see :mod:`ek.canonicalize`). Usually supplied by the facade instead.
    """

    def __init__(self, mode: str = "cer", *, canonicalizer=None):
        if mode not in ("cer", "wer"):
            raise ValueError(f"mode must be 'cer' or 'wer', got {mode!r}")
        self.mode = mode
        self.canonicalizer = canonicalizer

    @property
    def name(self) -> str:
        return self.mode

    def __call__(self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None) -> Score:
        ref, hyp = _as_text(gold), _as_text(pred)
        if self.canonicalizer is not None:
            ref, hyp = self.canonicalizer(ref), self.canonicalizer(hyp)
        edits, ref_len, extra = _edit_counts(ref, hyp, level=self.mode)
        value = (edits / ref_len) if ref_len else float(edits)
        return Score(
            value=value,
            metric=self.mode,
            detail={"edits": edits, "ref_len": ref_len, "higher_is_better": False, **extra},
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Corpus error rate = total edits / total reference length (the correct way)."""
        total_edits = sum(s.detail.get("edits", 0) for s in scores)
        total_ref = sum(s.detail.get("ref_len", 0) for s in scores)
        return (total_edits / total_ref) if total_ref else float("nan")
