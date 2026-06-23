"""Selective prediction: turn a calibrated probability into accept / flag / block.

Calibration makes a score *mean* something; a **decision policy** turns it into an
action with a guarantee (``misc/docs/ek_03`` §3-4). This is the formal home of
"accept / flag / block" -- a selective predictor that abstains (routes to a human)
rather than emit a low-confidence answer. Every policy here reads **only calibrated
probabilities** (Hard Rule 1): a raw score must already have passed through a
:class:`~ek.base.Calibrator`.

Three policies, by what you want to guarantee:

- :class:`CostSensitiveGate` -- **the default**. The accept threshold falls out of the
  cost ratio ``rho = c_FN / c_FP`` on the calibrated probability; no labelled set
  needed once you know your costs.
- :class:`ConformalGate` -- split (inductive) conformal: a finite-sample,
  distribution-free **marginal** guarantee that at most ``alpha`` of truly-correct
  items are flagged. Per-field-type validity needs the Mondrian
  (:class:`GroupConformalGate`) variant -- conditional coverage is otherwise
  impossible (Barber et al. 2019).
- :class:`RiskControlGate` -- conformal risk control: bound a monotone loss (the rate
  of accepted-yet-wrong items) at a target level.

:func:`risk_coverage_curve` exposes the trade-off so stakeholders pick the operating
point explicitly. The pure-Python gates need no extra dependencies; the
``ek[calibration]`` extra (MAPIE/crepes) offers library-backed conformal for an
sklearn-centric stack. Exchangeability is the load-bearing assumption -- track
realised coverage and re-fit on drift.

Example:
    >>> from ek.base import Decision
    >>> gate = CostSensitiveGate(rho=9.0)          # a false-negative costs 9x a review
    >>> gate(0.95) is Decision.ACCEPT              # tau = 1 - 1/9 ~ 0.889
    True
    >>> gate(0.5) is Decision.FLAG
    True
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..base import Decision
from ..registry import register, requires_extra

#: Default target miscoverage for conformal gates (flag at most ~alpha of correct items).
DEFAULT_ALPHA = 0.1

#: Default risk bound for :class:`RiskControlGate` (accepted-error rate ceiling).
DEFAULT_RISK_TARGET = 0.05


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _finite(p: float, *, what: str = "confidence") -> float:
    """Coerce to float and fail fast on a non-finite (NaN/inf) gate input."""
    p = float(p)
    if not math.isfinite(p):
        raise ValueError(f"{what} must be finite, got {p!r}")
    return p


# ---------------------------------------------------------------------------
# Cost-sensitive gate (the default; threshold from the cost ratio)
# ---------------------------------------------------------------------------


@dataclass
class CostSensitiveGate:
    """Accept/flag/block from the cost ratio ``rho = c_FN / c_FP`` on calibrated ``p``.

    Accept when the expected cost of accepting (an undetected error) is below the
    cost of routing to a human: ``(1 - p) * c_FN <= c_FP``, i.e. ``p >= 1 - 1/rho``.
    A larger ``rho`` (misses much costlier than reviews) raises the bar to
    auto-accept. ``block_threshold`` adds an optional hard-fail band at the bottom.

    Args:
        rho: Cost ratio ``c_FN / c_FP`` (>= 0). The single lever; no magic numbers.
        block_threshold: Calibrated probability at or below which to ``block`` rather
            than ``flag`` (default ``0`` -> never block on probability alone; a
            failed verifier is what forces a block, upstream).
        accept_threshold: Override the derived accept threshold explicitly.
    """

    rho: float = 1.0
    block_threshold: float = 0.0
    accept_threshold: Optional[float] = None

    @property
    def tau(self) -> float:
        """The accept threshold on calibrated probability (derived from ``rho``)."""
        if self.accept_threshold is not None:
            return _clip01(self.accept_threshold)
        return _clip01(1.0 - 1.0 / self.rho) if self.rho > 0 else 0.0

    def __call__(self, confidence: float) -> Decision:
        p = _finite(confidence)
        # block_threshold == 0 (the default) disables the block band entirely; a
        # block is otherwise reserved for an explicit hard-fail floor.
        if self.block_threshold > 0.0 and p <= self.block_threshold:
            return Decision.BLOCK
        return Decision.ACCEPT if p >= self.tau else Decision.FLAG


# ---------------------------------------------------------------------------
# Split conformal gate (marginal, distribution-free)
# ---------------------------------------------------------------------------


def split_conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The split-conformal threshold: the ``ceil((n+1)(1-alpha)) / n`` empirical quantile.

    ``scores`` are nonconformity scores on an exchangeable calibration set. Returns
    ``+inf`` when ``n`` is too small to guarantee ``1 - alpha`` (so nothing is
    flagged -- the honest behaviour at that sample size), and ``-inf`` at the
    ``alpha -> 1`` limit (flag everything).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    s = sorted(scores)
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    if k <= 0:  # alpha -> 1: the quantile collapses below every score
        return -math.inf
    return s[k - 1]


@dataclass
class ConformalGate:
    """Split-conformal accept/flag with a finite-sample **marginal** guarantee.

    Fit on ``(calibrated_prob, correct)``: the nonconformity scores of the *correct*
    calibration items set a threshold ``q`` such that, among truly-correct test items
    exchangeable with them, at most ``alpha`` are flagged. The guarantee is
    *marginal*, not per-field-type -- use :class:`GroupConformalGate` for that.

    Args:
        alpha: Target false-flag rate on correct items (e.g. ``0.1``).
        nonconformity: Maps calibrated confidence -> nonconformity (default
            ``1 - confidence``: low confidence is nonconforming).
    """

    alpha: float = DEFAULT_ALPHA
    nonconformity: Callable[[float], float] = lambda p: 1.0 - p
    q: float = math.inf
    _cal: List[float] = field(default_factory=list)

    def fit(self, probs: Sequence[float], correct: Sequence[bool]) -> "ConformalGate":
        """Fit ``q`` from the nonconformity scores of the *correct* calibration items.

        Non-finite confidences are dropped (a single ``NaN`` would otherwise sort to
        the end and corrupt the quantile, silently breaking the coverage guarantee).
        """
        self._cal = sorted(
            self.nonconformity(float(p))
            for p, c in zip(probs, correct)
            if c and math.isfinite(float(p))
        )
        self.q = split_conformal_quantile(self._cal, self.alpha)
        return self

    def p_value(self, confidence: float) -> float:
        """Per-instance conformal p-value (small = atypical of correct items)."""
        if not self._cal:
            return 1.0
        s = self.nonconformity(float(confidence))
        return (1 + sum(1 for si in self._cal if si >= s)) / (len(self._cal) + 1)

    def __call__(self, confidence: float) -> Decision:
        return (
            Decision.ACCEPT
            if self.nonconformity(_finite(confidence)) <= self.q
            else Decision.FLAG
        )


@dataclass
class GroupConformalGate:
    """Mondrian (class-conditional) conformal: one :class:`ConformalGate` per group.

    Restores approximate per-field-type validity by calibrating separately per group
    key (e.g. ``NodeType``/``FieldSpec``), with a pooled fallback for unseen groups.
    """

    alpha: float = DEFAULT_ALPHA
    nonconformity: Callable[[float], float] = lambda p: 1.0 - p
    by_group: dict = field(default_factory=dict)
    pooled: Optional[ConformalGate] = None

    def fit(
        self, probs: Sequence[float], correct: Sequence[bool], *, groups: Sequence[Any]
    ) -> "GroupConformalGate":
        buckets: dict = {}
        for p, c, g in zip(probs, correct, groups):
            buckets.setdefault(g, ([], []))
            buckets[g][0].append(p)
            buckets[g][1].append(c)
        self.by_group = {
            g: ConformalGate(alpha=self.alpha, nonconformity=self.nonconformity).fit(ps, cs)
            for g, (ps, cs) in buckets.items()
        }
        self.pooled = ConformalGate(alpha=self.alpha, nonconformity=self.nonconformity).fit(
            list(probs), list(correct)
        )
        return self

    def __call__(self, confidence: float, *, group: Any = None) -> Decision:
        gate = self.by_group.get(group, self.pooled)
        return gate(confidence) if gate is not None else Decision.FLAG


# ---------------------------------------------------------------------------
# Conformal risk control (bound a monotone loss)
# ---------------------------------------------------------------------------


@dataclass
class RiskControlGate:
    """Conformal Risk Control: accept ``p >= lambda`` bounding the accepted-error rate.

    Picks the *smallest* threshold ``lambda`` (the most coverage) whose finite-sample
    risk bound is at or below ``target``. The controlled quantity is
    ``E[ loss(item) * 1(accepted) ]`` -- with the default 0/1 loss, the population
    rate of accepted-yet-wrong items.

    Args:
        target: Upper bound on the accepted-error rate (e.g. ``0.05``).
        loss: ``correct -> loss`` in ``[0, loss_bound]``; default ``0`` if correct
            else ``1``. Any monotone loss bounded by ``loss_bound`` works.
        loss_bound: The loss's upper bound ``B`` -- the CRC finite-sample slack is
            ``+B/(n+1)``. Must match ``loss``'s range or the guarantee is invalid;
            keep the default ``1.0`` for the 0/1 loss.
    """

    target: float = DEFAULT_RISK_TARGET
    loss: Callable[[bool], float] = lambda correct: 0.0 if correct else 1.0
    loss_bound: float = 1.0
    lam: float = 1.0

    def fit(self, probs: Sequence[float], correct: Sequence[bool]) -> "RiskControlGate":
        """Find the smallest accept threshold whose risk bound is <= ``target``."""
        pairs = list(zip((float(p) for p in probs), correct))
        n = len(pairs)
        if n == 0:
            self.lam = 1.0
            return self
        candidates = [0.0] + sorted({p for p, _ in pairs})
        chosen = 1.0 + 1e-9  # accept nothing if no threshold satisfies the bound
        for lam in candidates:
            risk = sum(self.loss(c) for p, c in pairs if p >= lam) / n
            bound = (n * risk + self.loss_bound) / (n + 1)  # CRC slack +B/(n+1)
            if bound <= self.target:
                chosen = lam
                break
        self.lam = chosen
        return self

    def __call__(self, confidence: float) -> Decision:
        return Decision.ACCEPT if _finite(confidence) >= self.lam else Decision.FLAG


# ---------------------------------------------------------------------------
# Risk-coverage curve (publish so stakeholders pick the operating point)
# ---------------------------------------------------------------------------


def risk_coverage_curve(
    probs: Sequence[float],
    correct: Sequence[bool],
    *,
    thresholds: Optional[Sequence[float]] = None,
) -> List[dict]:
    """The risk-coverage trade-off: ``{threshold, coverage, selective_risk}`` per point.

    Coverage is the fraction auto-accepted at each threshold; selective risk is the
    error rate *among the accepted*. Lower-left is better; pick the operating point
    against a target risk or coverage.
    """
    pairs = list(zip((float(p) for p in probs), correct))
    n = len(pairs)
    if n == 0:
        return []
    taus = sorted(set(thresholds)) if thresholds is not None else sorted({p for p, _ in pairs})
    out = []
    for tau in taus:
        accepted = [c for p, c in pairs if p >= tau]
        cov = len(accepted) / n
        risk = (sum(1 for c in accepted if not c) / len(accepted)) if accepted else 0.0
        out.append({"threshold": tau, "coverage": cov, "selective_risk": risk})
    return out


# ---------------------------------------------------------------------------
# Optional library-backed conformal (ek[calibration]) -- opt-in, Mondrian via crepes
# ---------------------------------------------------------------------------


@requires_extra("calibration", packages=["crepes"])
def crepes_mondrian_gate(*args: Any, **kwargs: Any):
    """A Mondrian (class-conditional) conformal gate backed by ``crepes`` (``ek[calibration]``).

    The pure-Python :class:`GroupConformalGate` is the dependency-free default; use
    this for parity with a crepes/sklearn conformal stack (full CDFs, normalised
    variants). Raises an actionable install hint if ``crepes`` is absent.
    """
    import crepes  # noqa: F401  (presence checked by @requires_extra)

    raise NotImplementedError(
        "crepes-backed Mondrian conformal is a thin wrapper to add when a crepes "
        "stack is in use; the pure-Python GroupConformalGate is the default."
    )


# Register the built-in policies so they resolve by name (open-closed).
register("policies", "cost_sensitive", CostSensitiveGate)
register("policies", "conformal", ConformalGate)
register("policies", "risk_control", RiskControlGate)
register("policies", "group_conformal", GroupConformalGate)
