"""The ``ocracy`` bridge: turn the OCR fleet into an ``image -> OcrResult`` callable.

``ocracy`` is a deliberately dependency-free facade over ~16 OCR engines whose
single entry point ``ocracy.ocr(image, *, backend=None, **kwargs)`` returns a
normalized ``OcrResult`` (``.text`` plus ``.blocks`` carrying ``[0, 1]``-normalized
confidence and geometry). ``ek`` consumes that shape rather than reinventing engine
integration. The bridge is intentionally tiny -- it only adapts ``ocracy``'s call
into the :data:`~ek.base.OcrBackend` contract and guards the optional dependency
with an actionable install hint.

Null-safety reminder (see ``misc/docs/ek_01``): VLM/markdown engines (Claude,
GPT-4o, Mistral OCR) return text only -- ``blocks == []`` and ``confidence is
None``. Any code reading confidence/geometry off the result must tolerate ``None``.
"""

from __future__ import annotations

from typing import Any, Callable

from ..registry import requires_extra


@requires_extra("ocr", packages=["ocracy"])
def ocracy_backend(backend: str = None, **default_kwargs: Any) -> Callable[[Any], Any]:
    """Return an ``image -> OcrResult`` callable bound to one ``ocracy`` backend.

    Args:
        backend: An ``ocracy`` backend id (e.g. ``"tesseract"``, ``"google_vision"``);
            ``None`` uses ``ocracy``'s first installed backend.
        **default_kwargs: Options forwarded to ``ocracy.ocr`` on every call
            (e.g. ``languages=["en"]``); per-call kwargs override these.

    Returns:
        A callable suitable for :func:`ek.ocr.evaluate_ocr`.
    """
    import ocracy  # imported lazily; guarded by @requires_extra

    def run(image: Any, **kwargs: Any):
        return ocracy.ocr(image, backend=backend, **{**default_kwargs, **kwargs})

    run.__name__ = f"ocracy[{backend or 'default'}]"
    return run


@requires_extra("ocr", packages=["ocracy"])
def read_text(image: Any, *, backend: str = None, **kwargs: Any) -> str:
    """Convenience: OCR an image and return just the recognized text string."""
    import ocracy

    return ocracy.read_text(image, backend=backend, **kwargs)
