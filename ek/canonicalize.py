"""Versioned, composable canonicalization applied *before* scoring.

Under-specified normalization is the single biggest source of bogus evaluation
scores -- it can swing results by tens of points, and a reused English normalizer
catastrophically corrupts Indic/abugida scripts by stripping Unicode Mark-class
characters. So canonicalization is a first-class, *versioned* component here, not
a metric detail: a :class:`Canonicalizer` is an ordered pipeline of small named
steps with a stable ``version`` you can persist alongside a gold set.

Steps are plain ``str -> str`` functions registered under the ``"normalizers"``
namespace, so a schema can name a per-field normalizer (via
:attr:`~ek.base.FieldSpec.normalizer`) and third parties can add their own.

Example:
    >>> canon = Canonicalizer(["nfc", "lower", "collapse_whitespace"])
    >>> canon("  Héllo   WORLD ")
    'héllo world'
    >>> canon.version  # stable hash of the step names
    'nfc+lower+collapse_whitespace'

Always audit before folding a non-Latin script -- :func:`mark_class_count` reports
how many combining marks a string carries, the signal that aggressive folding
would destroy meaning:
    >>> mark_class_count("e\\u0301")  # 'e' + combining acute
    1
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Callable, Iterable, Sequence, Union

from .registry import get, register

_WS_RE = re.compile(r"\s+")


@register("normalizers", "nfc")
def nfc(s: str) -> str:
    """Unicode NFC normalization (compose); the safe default."""
    return unicodedata.normalize("NFC", s)


@register("normalizers", "nfkc")
def nfkc(s: str) -> str:
    """Unicode NFKC normalization (compatibility compose); lossy -- use with care."""
    return unicodedata.normalize("NFKC", s)


@register("normalizers", "lower")
def lower(s: str) -> str:
    """Casefold to lowercase (full Unicode casefold)."""
    return s.casefold()


@register("normalizers", "strip")
def strip(s: str) -> str:
    """Strip leading/trailing whitespace."""
    return s.strip()


@register("normalizers", "collapse_whitespace")
def collapse_whitespace(s: str) -> str:
    """Collapse runs of whitespace to a single space and strip the ends."""
    return _WS_RE.sub(" ", s).strip()


@register("normalizers", "strip_punctuation")
def strip_punctuation(s: str) -> str:
    """Remove Unicode punctuation characters (category starting with ``P``)."""
    return "".join(c for c in s if not unicodedata.category(c).startswith("P"))


def mark_class_count(s: str) -> int:
    """Number of Unicode Mark-class (combining) characters in ``s``.

    A non-zero count on a non-Latin script is a warning sign that case/diacritic
    folding will corrupt meaning (the Indic-script catastrophe). Use it to gate or
    audit a canonicalizer before trusting cross-script scores.
    """
    return sum(1 for c in s if unicodedata.category(c).startswith("M"))


class Canonicalizer:
    """An ordered, named, versioned pipeline of ``str -> str`` steps.

    Args:
        steps: Step references -- names registered under ``"normalizers"`` or
            plain callables. Applied left to right.
        version: Override the auto-derived version string (the ``+``-joined names).
        audit_marks: If true, warn when a step would run on text carrying combining
            marks (the cross-script damage guard).
    """

    def __init__(
        self,
        steps: Sequence[Union[str, Callable[[str], str]]] = (),
        *,
        version: str = "",
        audit_marks: bool = False,
    ):
        self._refs = list(steps)
        self._funcs = [s if callable(s) else get("normalizers", s) for s in steps]
        self._names = [
            getattr(s, "__name__", str(s)) if callable(s) else s for s in steps
        ]
        self._version = version or "+".join(self._names)
        self.audit_marks = audit_marks

    @property
    def version(self) -> str:
        """Stable identifier of this pipeline (persist it next to a gold set)."""
        return self._version

    @property
    def steps(self) -> list:
        """The resolved step names, in order."""
        return list(self._names)

    def __call__(self, s: str) -> str:
        if self.audit_marks and mark_class_count(s) > 0:
            warnings.warn(
                "Canonicalizing text with Unicode combining marks; verify the "
                "pipeline does not strip script-meaningful diacritics.",
                stacklevel=2,
            )
        for fn in self._funcs:
            s = fn(s)
        return s

    def __repr__(self) -> str:
        return f"Canonicalizer({self._names!r}, version={self._version!r})"


def default_canonicalizer() -> Canonicalizer:
    """A conservative default: NFC -> casefold -> collapse whitespace."""
    return Canonicalizer(["nfc", "lower", "collapse_whitespace"])


def resolve_canonicalizer(
    normalize: Union[None, str, Callable[[str], str], Iterable, Canonicalizer],
) -> Union[Canonicalizer, Callable[[str], str], None]:
    """Coerce a ``normalize`` argument to a callable canonicalizer (or ``None``).

    Accepts ``None``, a registered normalizer name, a callable, a sequence of
    steps, or an existing :class:`Canonicalizer`.
    """
    if normalize is None or isinstance(normalize, Canonicalizer):
        return normalize
    if isinstance(normalize, str):
        return get("normalizers", normalize)
    if callable(normalize):
        return normalize
    return Canonicalizer(list(normalize))
