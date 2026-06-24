"""Intrinsic confidence signals: token logprobs and per-unit posteriors (cost tier 2).

When the extractor already emits something usable -- token log-probabilities
(generative locals like TrOCR/pix2tex, GPT-4o, Mistral OCR) or per-unit posteriors
(Google Cloud Vision, Azure, Tesseract) -- the cheapest reliable signal is *free*:
it was computed during inference. This module turns those raw numbers into a single
field-level score (``misc/docs/ek_03`` §1a). Two modelling facts drive the design:

1. **Raw summed logprob is length-biased** -- longer fields accumulate more negative
   terms and look worse. The fix is length normalisation; the aggregation is a
   *modelling choice to validate on held-out data*, so it stays **pluggable** (no
   hardcoded pooling). Use ``min`` to catch the single weakest token (a transposed
   digit in an amount); use the geometric mean for overall field plausibility (the
   family Mistral OCR reports as ``average_page_confidence_score``).
2. **It is uncalibrated** -- and for RLHF'd LLMs *systematically overconfident*.
   Every signal here is a raw score; a :class:`~ek.base.Calibrator` must run before
   any gate reads it.

Aggregators are registered under the ``aggregators`` namespace so a caller (or a
third party) can swap pooling by name.

Example:
    >>> import math
    >>> logps = [math.log(0.9), math.log(0.8), math.log(0.95)]
    >>> round(geo_mean(logps), 4)            # exp(mean log p)
    0.8811
    >>> round(min_prob(logps), 4)            # weakest token
    0.8
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..registry import register

#: Length-penalty exponent for :func:`length_normalized` (Wu et al. 2016 found
#: ``alpha`` in ``[0.6, 0.7]`` best); a keyword default, never a magic constant.
DEFAULT_LENGTH_ALPHA = 0.6

_SENTINEL = object()

#: Smallest probability treated as non-zero. ``log(p=0) = -inf`` clamps to
#: ``log(DEFAULT_PROB_FLOOR)`` (mirrors the confidence floor) so a p=0 token still
#: drives min_prob/geo_mean toward 0 instead of being silently discarded.
DEFAULT_PROB_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Logprob aggregators: Sequence[log p] -> field probability score in [0, 1]
# ---------------------------------------------------------------------------


def _finite_logps(
    logps: Sequence[float], *, prob_floor: float = DEFAULT_PROB_FLOOR
) -> list:
    """Sanitize log-probabilities for aggregation.

    A valid ``log p`` is ``<= 0`` (``p <= 1``). ``-inf`` is the *legitimate* log of a
    probability-0 token (an impossible/OOV unit) -- the single most damning piece of
    evidence -- so it is **clamped** to ``log(prob_floor)`` (kept, not dropped): the
    weakest-token and geometric-mean safety signals must reflect it (otherwise
    ``min_prob`` returns the second-weakest token and ``geo_mean`` of an all-zero
    field returns a *maximal* 1.0). Only ``+inf``/``NaN`` -- which cannot be a real
    ``log p`` and would poison an aggregate -- are dropped.
    """
    floor = math.log(prob_floor)
    out = []
    for lp in logps:
        lp = float(lp)
        if math.isnan(lp) or lp == math.inf:
            continue
        out.append(floor if lp == -math.inf else min(lp, 0.0))
    return out


@register("aggregators", "geo_mean")
def geo_mean(logps: Sequence[float]) -> float:
    """Geometric mean of token probabilities, ``exp(mean log p)`` (perplexity-inverse)."""
    logps = _finite_logps(logps)
    if not logps:
        return 1.0
    return math.exp(sum(logps) / len(logps))


@register("aggregators", "length_normalized")
def length_normalized(
    logps: Sequence[float], *, alpha: float = DEFAULT_LENGTH_ALPHA
) -> float:
    """Length-normalised score ``exp(Σ log p / T**alpha)`` (tunable length penalty)."""
    logps = _finite_logps(logps)
    t = len(logps)
    if t == 0:
        return 1.0
    return math.exp(sum(logps) / (t**alpha))


@register("aggregators", "min")
def min_prob(logps: Sequence[float]) -> float:
    """Weakest-token probability ``exp(min log p)`` -- catches one bad character."""
    logps = _finite_logps(logps)
    if not logps:
        return 1.0
    return math.exp(min(logps))


@register("aggregators", "mean")
def mean_prob(logps: Sequence[float]) -> float:
    """Arithmetic mean of token probabilities ``mean(exp(log p))``."""
    logps = _finite_logps(logps)
    if not logps:
        return 1.0
    return sum(math.exp(lp) for lp in logps) / len(logps)


@dataclass
class LogprobSignal:
    """Aggregate token log-probabilities into one field score (a :class:`~ek.base.Signal`).

    Args:
        aggregator: A logprob aggregator (name registered under ``aggregators`` or a
            callable). Defaults to :func:`geo_mean`.
        alpha: Length penalty passed to :func:`length_normalized` only.
        cost_tier: ``2`` -- free intrinsic signal (already computed at inference).
    """

    aggregator: Any = geo_mean
    alpha: float = DEFAULT_LENGTH_ALPHA
    cost_tier: int = 2

    def __call__(self, token_logps: Sequence[float]) -> float:
        agg = self.aggregator
        if isinstance(agg, str):
            from ..registry import get

            agg = get("aggregators", agg)
        if agg is length_normalized:
            return agg(token_logps, alpha=self.alpha)
        return agg(token_logps)


# ---------------------------------------------------------------------------
# Per-unit posterior aggregation: Sequence[prob] -> field probability score
# ---------------------------------------------------------------------------


def _confidences_of(obj: Any) -> list:
    """Extract per-unit confidences from an OcrResult-shaped object or a sequence.

    Prefers per-block confidences (``.blocks[i].confidence``); falls back to a
    single ``.mean_confidence``; otherwise treats ``obj`` as a sequence of numbers.
    Null-safe: ``None`` and non-finite confidences are dropped (the VLM/markdown
    case); the rest are clamped into the documented ``[0, 1]`` range.
    """
    blocks = getattr(obj, "blocks", None)
    if blocks:
        confs = _clean_confs(getattr(b, "confidence", None) for b in blocks)
        if confs:
            return confs
    mean_conf = getattr(obj, "mean_confidence", _SENTINEL)
    if mean_conf is not _SENTINEL:
        return _clean_confs([mean_conf])
    return _clean_confs(obj)


def _clean_confs(values: Any) -> list:
    """Coerce to floats in ``[0, 1]``, dropping ``None`` and non-finite entries."""
    out = []
    for c in values:
        if c is None:
            continue
        c = float(c)
        if math.isfinite(c):
            out.append(0.0 if c < 0.0 else 1.0 if c > 1.0 else c)
    return out


#: Allowed pooling strategies for :class:`IntrinsicConfidenceSignal` (an unknown
#: name is rejected rather than silently treated as "mean").
_CONF_POOLS = ("min", "mean", "geo_mean")


@dataclass
class IntrinsicConfidenceSignal:
    """Aggregate an extractor's per-unit confidences into a field score.

    Args:
        pool: ``"min"`` (weakest unit), ``"mean"``, or ``"geo_mean"`` over the
            per-unit confidences (already probabilities in ``[0, 1]``).
        cost_tier: ``2`` -- free (the posteriors are emitted at inference).

    Example:
        >>> IntrinsicConfidenceSignal(pool="min")([0.99, 0.4, 0.95])
        0.4
    """

    pool: str = "min"
    cost_tier: int = 2

    def __call__(self, source: Any) -> float:
        if self.pool not in _CONF_POOLS:
            raise ValueError(
                f"pool must be one of {sorted(_CONF_POOLS)}, got {self.pool!r}"
            )
        confs = _confidences_of(source)
        if not confs:
            return 1.0
        if self.pool == "min":
            return min(confs)
        if self.pool == "geo_mean":
            return math.exp(
                sum(math.log(max(c, DEFAULT_PROB_FLOOR)) for c in confs) / len(confs)
            )
        return sum(confs) / len(confs)
