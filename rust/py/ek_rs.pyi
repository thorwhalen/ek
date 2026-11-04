"""Stub for the native module ``ek_rs`` (hand-written; checked with ``mypy.stubtest``).

Private kernels behind ek's backend seams; the facade (``ek.qe.rover._align``) is
the API. Do not call these directly.
"""

__all__ = ["DEBUG_BUILD", "align"]

DEBUG_BUILD: bool

def align(spine: list[int], hyp: list[int], /) -> list[tuple[str, int, int]]:
    """Aligns two sequences of interned token ids, `_align`'s operations out."""
