"""End-to-end OCR offline benchmark: run a fleet over gold, score, persist.

This is ``ek``'s first out-of-the-box experience -- "how accurate is my OCR?". It
pairs each gold item with the engine's output, scores text with a globally
accumulated CER/WER (the correct corpus aggregation), optionally slices the report,
and persists gold + per-item results + the run aggregate to the ``dol`` stores
under ``~/.local/share/ek/``.

The runner is deliberately source-agnostic: ``engine`` is any
``image -> OcrResult`` callable (or an ``ocracy`` backend id, bridged for you), so
the benchmark runs -- and is fully testable -- without any real OCR engine or cloud
credentials.

Example:
    >>> import tempfile; root = tempfile.mkdtemp()
    >>> gold = {
    ...     "d1": {"image": "ignored", "reference_text": "cat", "slice": "easy"},
    ...     "d2": {"image": "ignored", "reference_text": "dog", "slice": "easy"},
    ... }
    >>> # a stub engine that always reads "cat" (perfect for d1, wrong for d2):
    >>> engine = lambda image: type("R", (), {"text": "cat"})()
    >>> report = evaluate_ocr(engine, gold, metric="cer", rootdir=root)
    >>> report.n
    2
    >>> sorted(report.per_slice)
    ['easy']
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

from ..facade import evaluate
from ..stores import json_store


def _as_engine(engine: Union[str, Callable]) -> Callable[[Any], Any]:
    """A callable engine, bridging an ``ocracy`` backend id when given a string."""
    if callable(engine):
        return engine
    from .bridge import ocracy_backend

    return ocracy_backend(engine)


def add_gold_item(
    key: str,
    image: Any,
    reference_text: str,
    *,
    slice: Optional[str] = None,
    rootdir: Optional[str] = None,
    **extra: Any,
) -> None:
    """Add one gold example to the ``gold`` store.

    Args:
        key: Unique id for the document.
        image: An image reference the engine can read (path, URL, ...). Stored as a
            reference, not bytes -- gold stores hold references + the expected text.
        reference_text: The ground-truth transcription.
        slice: Optional stratification label (doc type, language, scan quality, ...).
        rootdir: Override the data root (tests/ephemeral use).
        **extra: Any additional metadata to store on the record.
    """
    store = json_store("gold", rootdir=rootdir)
    record = {"image": image, "reference_text": reference_text, **extra}
    if slice is not None:
        record["slice"] = slice
    store[key] = record


def evaluate_ocr(
    engine: Union[str, Callable],
    gold: Mapping,
    *,
    metric: str = "cer",
    normalize: Any = None,
    grammar: Any = None,
    image_key: str = "image",
    reference_key: str = "reference_text",
    slice_key: str = "slice",
    persist: bool = False,
    run_id: Optional[str] = None,
    rootdir: Optional[str] = None,
):
    """Benchmark an OCR engine against a gold corpus.

    Args:
        engine: An ``image -> OcrResult`` callable, or an ``ocracy`` backend id.
        gold: A mapping ``key -> {image, reference_text, [slice], ...}`` -- a plain
            dict or a ``ek`` gold store.
        metric: ``"cer"`` (default) or ``"wer"`` (or any registered string metric).
        normalize: Optional canonicalizer applied to both sides before scoring.
        grammar: Optional Layer-A grammar (forwarded to the metric).
        image_key, reference_key, slice_key: Field names within each gold record.
        persist: If true, write per-item results and the run aggregate to the
            ``results`` / ``runs`` stores.
        run_id: Identifier for this run (auto-generated from the clock if omitted).
        rootdir: Override the data root (tests/ephemeral use).

    Returns:
        A :class:`~ek.base.Report` whose ``detail['per_item']`` maps each gold key to
        its prediction, reference, slice, score, and engine confidence/backend.
    """
    eng = _as_engine(engine)
    items = list(gold.items()) if isinstance(gold, Mapping) else list(gold)

    cases = []
    per_item: dict = {}
    for key, rec in items:
        image = rec[image_key]
        reference = rec[reference_key]
        slice_label = rec.get(slice_key)
        result = eng(image)
        # Null-safe: an OcrResult whose .text is None (VLM/markdown engines) must
        # read as empty text, not the literal "None".
        if isinstance(result, str):
            pred_text = result
        else:
            text = getattr(result, "text", None)
            pred_text = text if isinstance(text, str) else ""
        cases.append((pred_text, reference, slice_label))
        per_item[key] = {
            "prediction": pred_text,
            "reference": reference,
            "slice": slice_label,
            "mean_confidence": getattr(result, "mean_confidence", None),
            "backend": getattr(result, "backend", None),
        }

    report = evaluate(cases, metric=metric, normalize=normalize, grammar=grammar)

    for key, sc in zip(per_item, report.scores):
        per_item[key]["score"] = sc.value
    report.detail["per_item"] = per_item

    if persist:
        _persist(report, per_item, run_id=run_id, rootdir=rootdir)
    return report


def _persist(report, per_item, *, run_id: Optional[str], rootdir: Optional[str]) -> None:
    if run_id is None:
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("ocr-%Y%m%dT%H%M%S")
    results = json_store("results", rootdir=rootdir)
    results[run_id] = {
        "metric": report.metric,
        "aggregate": report.aggregate,
        "n": report.n,
        "per_slice": report.per_slice,
        "per_item": per_item,
    }
    runs = json_store("runs", rootdir=rootdir)
    runs[run_id] = {
        "metric": report.metric,
        "aggregate": report.aggregate,
        "n": report.n,
        "per_slice": report.per_slice,
    }
