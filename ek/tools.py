"""Small, CLI-friendly functions over the ``ek`` core (the dispatch surface).

These are thin, string-in/value-out wrappers designed for triple dispatch (CLI via
``argh``, and later HTTP/UI). The single ``_dispatch_funcs`` list is the SSOT of
what :mod:`ek.__main__` exposes, so there is no duplicated command registration.

Run them from the shell::

    python -m ek cer "hello wrld" "hello world"
    python -m ek where
    python -m ek check tesseract
"""

from __future__ import annotations

from typing import Optional

from .facade import score
from .registry import check_requirements
from .stores import app_folder


def cer(prediction: str, gold: str, *, normalize: Optional[str] = None) -> float:
    """Character Error Rate between a prediction and a gold string (lower is better)."""
    return score(prediction, gold, metric="cer", normalize=normalize).value


def wer(prediction: str, gold: str, *, normalize: Optional[str] = None) -> float:
    """Word Error Rate between a prediction and a gold string (lower is better)."""
    return score(prediction, gold, metric="wer", normalize=normalize).value


def rover(*hypotheses: str, confidence: bool = True) -> dict:
    """ROVER-fuse 2+ transcriptions into a consensus + mean agreement (reference-free).

    Example::

        python -m ek rover "the cat sat" "the cat sit" "the bat sat"
    """
    from .qe.rover import rover as _rover

    cons = _rover(list(hypotheses), use_confidence=confidence)
    return {
        "consensus": cons.text,
        "mean_agreement": round(cons.mean_agreement, 4),
        "agreement": [round(a, 4) for a in cons.agreement],
        "n_engines": cons.n_engines,
    }


def where() -> str:
    """Print the local data folder where ``ek`` persists gold/results/runs."""
    return str(app_folder())


def check(engine: str) -> dict:
    """Report what an OCR engine needs to run (delegates to ocracy when installed)."""
    return check_requirements(engine=engine)


def engines() -> list:
    """List installed OCR backends (requires the ``ek[ocr]`` extra)."""
    try:
        import ocracy

        return list(ocracy.available_backends())
    except ImportError:
        return ["(install ek[ocr] to list OCR backends)"]


def version() -> str:
    """Print the installed ``ek`` version."""
    from . import __version__

    return __version__


#: SSOT of the functions exposed by the ``ek`` CLI.
_dispatch_funcs = [cer, wer, rover, where, check, engines, version]
