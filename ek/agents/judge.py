"""LLM-as-judge: a reference-free :class:`~ek.base.Signal`, a reference-based
:class:`~ek.base.Metric`, and the validation that must precede either.

When a task has no checkable outcome, an LLM judge is the fallback. It is also a **liability
unless validated and calibrated**, because its known failure modes have double-digit effect
sizes: *position* bias (which answer came first), *verbosity* bias (longer looks better),
*self-preference* (a model prefers its own text). ``misc/docs/ek_09`` has the numbers.

Three seams, deliberately kept apart (they are not interchangeable):

- :class:`JudgeSignal` -- **reference-free, criteria-only**. This is the only judge that is a
  ``Signal``: it scores an output against *criteria*, with no gold, so it plugs into
  :func:`ek.estimate_quality`'s ``signal -> calibrate -> validate -> decide`` cascade. Its
  ``cost_tier`` is the highest in the framework, so an escalation policy runs the cheap signals
  first and pays for the judge only on residual uncertainty.
- :class:`JudgeMetric` -- **reference-based**. A judge that needs the gold answer is not a
  Signal at all; it is a Metric, and it belongs on the :func:`ek.score` side.
- :func:`pairwise_judge` -- pairwise comparison needs *two* outputs and cannot fit
  ``Signal.__call__(one_output)``. It is a helper, and it **swaps the order and averages** --
  the standard, necessary correction for position bias.

**Hard Rule 1 applies to judges.** A raw judge score is not a probability. Calibrate it before
any :class:`~ek.base.DecisionPolicy` reads it -- :func:`ek.estimate_quality` already refuses to
gate an uncalibrated signal, and a judge is no exception (-> ``misc/docs/ek_03``).

**Known facade limitation, stated plainly:** :func:`ek.estimate_quality` currently averages *all*
raw signals into one confidence and calibrates that single mean. Mixing a judge score with, say,
a logprob -- different scales, different reliabilities -- and fitting one calibrator over the
mean is not statistically sound. Until per-signal-family calibration lands, either run the judge
as the *only* signal, or calibrate it separately and pass the calibrated value.

``ek`` never imports an LLM SDK: ``judge`` is an **injected callable**.

Example:
    >>> stub = lambda output, *, criteria="", reference=None: 0.9 if "meow" in output else 0.1
    >>> sig = JudgeSignal(stub, criteria="Is the answer helpful?")
    >>> sig("the cat says meow")
    0.9
    >>> sig.cost_tier                       # the most expensive signal family
    5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..base import Finding, GraphGrammar, Score, Severity
from ..harness import cohen_kappa, krippendorff_alpha, percent_agreement

#: Judges are the most expensive signal family -- run cheap signals first and escalate.
JUDGE_COST_TIER = 5

#: Krippendorff conventions: >= 0.8 is good agreement, 0.67-0.8 only tentative.
ALPHA_GOOD = 0.8
ALPHA_TENTATIVE = 0.67


_ABSENT = object()


def _output_of(target: Any) -> Any:
    """The text a judge should read, off an Episode, a FieldEstimate, or a raw value.

    Presence of the attribute decides, not truthiness: an ``Episode(output=None)`` must hand the
    judge ``None`` (an empty answer -- which a judge should score badly), never the Episode object.
    """
    for attr in ("output", "value", "text"):
        got = getattr(target, attr, _ABSENT)
        if got is not _ABSENT:
            return got
    return target


@dataclass
class JudgeSignal:
    """A reference-free, criteria-only LLM judge as a :class:`~ek.base.Signal`.

    Args:
        judge: An **injected** callable ``(output, *, criteria) -> float``. ``ek`` core never
            imports an LLM client; you pass your own (or an ``ek[judge]`` adapter).
        criteria: The rubric the judge scores against, bound at construction (exactly as
            :class:`~ek.qe.signals.LogprobSignal` binds its aggregator), so the object still
            satisfies ``Signal.__call__(output) -> float``.
        cost_tier: :data:`JUDGE_COST_TIER` -- the most expensive tier.
    """

    judge: Callable
    criteria: str = ""
    cost_tier: int = JUDGE_COST_TIER
    name: str = "judge"

    def __call__(self, target: Any) -> float:
        return float(self.judge(_output_of(target), criteria=self.criteria))

    def finding(self, target: Any, rationale: str) -> Finding:
        """Surface a judge's *rationale* as a :class:`~ek.base.Finding` (not just a scalar).

        A judge's reasoning is auditable evidence; a bare number throws it away.
        """
        return Finding(
            field="", layer="judge", severity=Severity.FLAG, message=rationale
        )


class JudgeMetric:
    """A **reference-based** LLM judge as a :class:`~ek.base.Metric` (the ``score()`` side).

    A judge that reads the gold answer is not a reference-free signal, so it does not belong in
    :func:`ek.estimate_quality`. It lives here.

    Args:
        judge: Injected ``(output, *, criteria, reference) -> float`` in ``[0, 1]``.
        criteria: The rubric.
    """

    name = "judge"

    def __init__(self, judge: Callable, criteria: str = ""):
        self.judge = judge
        self.criteria = criteria

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        value = float(
            self.judge(_output_of(pred), criteria=self.criteria, reference=gold)
        )
        return Score(
            value=value,
            metric="judge",
            detail={"criteria": self.criteria, "higher_is_better": True},
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Mean judge score (a judge score *is* per-item; there is no global accumulation)."""
        return (sum(float(s) for s in scores) / len(scores)) if scores else 0.0


def pairwise_judge(
    judge: Callable,
    a: Any,
    b: Any,
    *,
    criteria: str = "",
    swap: bool = True,
) -> float:
    """Pairwise preference for ``a`` over ``b``, **corrected for position bias**.

    Position bias is not a rounding error -- judges materially prefer whichever answer they see
    first. The standard correction is to ask twice with the order swapped and average, which is
    why this cannot be a plain ``Signal`` (it needs two outputs).

    Args:
        judge: Injected ``(a, b, *, criteria) -> float`` in ``[0, 1]``, the preference for ``a``.
        swap: Ask again with the arguments swapped and average (default; disable only if your
            judge is known to be order-invariant).

    Returns:
        Preference for ``a`` in ``[0, 1]`` (``0.5`` = a tie).

    Example:
        >>> first_wins = lambda x, y, *, criteria="": 1.0     # a maximally position-biased judge
        >>> pairwise_judge(first_wins, "A", "B")              # swapping exposes it as a tie
        0.5
    """
    forward = float(judge(_output_of(a), _output_of(b), criteria=criteria))
    if not swap:
        return forward
    backward = float(judge(_output_of(b), _output_of(a), criteria=criteria))
    # backward is the preference for b; the preference for a is its complement.
    return (forward + (1.0 - backward)) / 2.0


def judge_validation(
    judge_labels: Sequence,
    human_labels: Sequence,
    *,
    level: str = "nominal",
) -> dict:
    """Certify a judge against human labels **before** trusting it. No new code -- reuse the IAA.

    A judge is just another annotator, so ``ek``'s existing inter-annotator-agreement machinery
    (which already tells you the *ceiling* of a gold standard) is exactly the right instrument:
    Krippendorff's alpha is the default (it alone handles any number of raters, any measurement
    level, and missing data); Cohen's kappa applies for exactly two raters on complete nominal
    data. This reproduces the MT-Bench judge-vs-human protocol.

    Returns:
        ``alpha``, ``kappa``, ``percent_agreement``, ``n``, and a ``verdict`` against the
        conventional thresholds (alpha >= 0.8 good, 0.67-0.8 tentative, below that unreliable).

    Example:
        >>> r = judge_validation([1, 1, 0, 0], [1, 1, 0, 0])
        >>> r["alpha"], r["verdict"]
        (1.0, 'good')
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge and human must label the same items")
    alpha = krippendorff_alpha([list(judge_labels), list(human_labels)], level=level)
    result = {
        "n": len(judge_labels),
        "alpha": alpha,
        "percent_agreement": percent_agreement(list(judge_labels), list(human_labels)),
        "verdict": _alpha_verdict(alpha),
    }
    if level == "nominal":
        result["kappa"] = cohen_kappa(list(judge_labels), list(human_labels))
    return result


def _alpha_verdict(alpha: float) -> str:
    if alpha >= ALPHA_GOOD:
        return "good"
    if alpha >= ALPHA_TENTATIVE:
        return "tentative"
    return "unreliable"
