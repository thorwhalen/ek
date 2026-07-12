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


# --- agent & assistant evaluation (cost per successful task) ---------------------


def pass_k(n: int, c: int, k: int = 1) -> dict:
    """pass@k (capability) and pass^k (reliability) from ``c`` successes in ``n`` trials.

    ``pass@k`` asks "can it *ever* do this?"; ``pass^k`` asks "does it do this *every* time?"
    -- and only the second is a production number.

    Example::

        python -m ek pass-k 10 9 --k 8      # a 90%-reliable agent, judged over 8 trials
    """
    from .agents import pass_at_k, pass_hat_k

    # This module is the string-in/value-out dispatch layer (the CLI hands us strings).
    n, c, k = int(n), int(c), int(k)
    return {
        "n": n,
        "c": c,
        "k": k,
        "pass_at_k": round(pass_at_k(n=n, c=c, k=k), 4),
        "pass_hat_k": round(pass_hat_k(n=n, c=c, k=k), 4),
    }


def cost_per_success(
    total_dollars: float, successes: int, *, attempts: Optional[int] = None
) -> dict:
    """Cost-of-Pass: dollars per *successfully completed* task (``inf`` if none succeeded).

    The unit that matters for an agent -- tokens spent on a failed episode are pure waste.

    Example::

        python -m ek cost-per-success 12.50 5 --attempts 20
    """
    from .agents import cost_of_pass

    total_dollars, successes = float(total_dollars), int(successes)
    out = {
        "total_dollars": total_dollars,
        "successes": successes,
        "cost_per_success": cost_of_pass(total_dollars, successes),
    }
    if attempts:
        out["success_rate"] = round(successes / int(attempts), 4)
    return out


#: SSOT of the functions exposed by the ``ek`` CLI.
_dispatch_funcs = [
    cer,
    wer,
    rover,
    pass_k,
    cost_per_success,
    where,
    check,
    engines,
    version,
]
