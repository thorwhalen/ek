"""Task-success and cost-per-successful-task: the two metrics that drive the whole thing.

Two reference-based metrics over an :class:`~ek.agents.base.Episode`, both of which fit
:func:`ek.evaluate` exactly because their corpus aggregate is a **ratio of two sums**, never a
mean of per-item scores -- the same discipline that makes ``ek``'s WER accumulate globally and
its F1 micro-average (``misc/docs/ek_08``, ``misc/docs/ek_11``):

- :class:`TaskSuccessMetric` -- grade the **final state**, not the surface text. The oracle is
  *injected*: ``ek`` never owns an LLM or a sandbox, so a checker is a plugged-in strategy
  (a database-state comparison, a hidden-test run, a programmatic predicate, a judge).
  Aggregate = the success **rate**.
- :class:`CostPerSuccessMetric` -- **Cost-of-Pass**: ``E[cost] / P(success)``. Aggregate =
  ``sum(dollars) / sum(successes)``, diverging to ``inf`` when nothing succeeded. A benchmark
  that scores accuracy *without* cost lets an agent chase tiny gains with unbounded API calls,
  so this metric is the point, not a nicety.

**A safety property, not a nicety:** the checker must run somewhere the agent's trajectory
cannot reach. An audited agent-benchmark red-team produced a ~10-line ``conftest.py`` that
pytest auto-loads and that rewrote *every* test outcome to "passed" -- scoring near-perfect
without solving a single task. Verifier isolation is a security boundary.

Example:
    >>> from ek.agents.base import Cost, Episode
    >>> from ek.agents.cost import per_million
    >>> ep = Episode(task_id="t1", output="42", cost=Cost(input_tokens=1_000_000))
    >>> TaskSuccessMetric()(ep, "42").value
    1.0
    >>> m = CostPerSuccessMetric(price=per_million(2.0, 6.0))
    >>> round(m(ep, "42").value, 4)          # this episode's dollars
    2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Protocol, runtime_checkable

from ..base import GraphGrammar, Score
from ..canonicalize import default_canonicalizer
from ..registry import register, resolve
from .base import Episode
from .cost import ModelPrice, cost_of_pass, episode_dollars


@runtime_checkable
class Checker(Protocol):
    """The success oracle: did this episode leave the world in the required state?

    Injected, never owned by ``ek`` -- a real suite plugs in a database-state comparison, a
    hidden-test runner, or a programmatic predicate. It must run **isolated** from the agent.
    """

    def __call__(self, episode: Any, gold: Any) -> bool: ...


# ---------------------------------------------------------------------------
# Built-in checkers (the trivial ones; real suites inject their own)
# ---------------------------------------------------------------------------


@register("checkers", "output")
def output_match(episode: Any, gold: Any) -> bool:
    """Canonicalized exact match on the final answer (the GAIA-style oracle).

    The weakest useful oracle, and the default only because it needs nothing. Prefer a
    state-based or executable checker whenever the task has one.
    """
    canon = default_canonicalizer()
    pred = getattr(episode, "output", episode)
    expected = gold.get("output", gold) if isinstance(gold, Mapping) else gold
    if isinstance(pred, str) and isinstance(expected, str):
        return canon(pred) == canon(expected)
    return pred == expected


@register("checkers", "final_state")
def final_state_match(episode: Any, gold: Any) -> bool:
    """Compare the episode's **final state** with the goal state (the tau-bench oracle)."""
    state = getattr(episode, "final_state", None)
    expected = gold.get("final_state", gold) if isinstance(gold, Mapping) else gold
    return state == expected


@register("checkers", "recorded")
def recorded_success(episode: Any, gold: Any) -> bool:
    """Trust a ``success`` already recorded on the episode (e.g. by an external harness)."""
    success = getattr(episode, "success", None)
    if success is None:
        raise ValueError(
            "checker 'recorded': episode carries no success verdict; grade it with a real "
            "checker (see ek.agents.Checker) or use check='output'/'final_state'."
        )
    return bool(success)


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------


class TaskSuccessMetric:
    """Did the episode complete the task? (``1.0``/``0.0``; aggregate = success rate).

    Args:
        check: A :class:`Checker` -- a registered name (``"output"``, ``"final_state"``,
            ``"recorded"``) or any ``(episode, gold) -> bool`` callable. Defaults to
            ``"output"``; **inject a state-based or executable oracle for a real suite.**
    """

    name = "task_success"

    def __init__(self, check: Any = None):
        self.check = resolve("checkers", check, default="output")

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        ok = bool(self.check(pred, gold))
        detail: dict = {"success": ok, "higher_is_better": True}
        cost = getattr(pred, "cost", None)
        if cost is not None:
            detail["total_tokens"] = cost.total_tokens
            detail["latency_s"] = cost.latency_s
        return Score(value=1.0 if ok else 0.0, metric="task_success", detail=detail)

    def aggregate(self, scores: Sequence[Score]) -> float:
        """The success **rate** over the corpus."""
        if not scores:
            return 0.0
        return sum(1 for s in scores if s.detail.get("success")) / len(scores)


class CostPerSuccessMetric:
    """Cost-of-Pass: dollars per **successfully completed** task (**lower is better**).

    The per-episode ``Score.value`` is that episode's dollar cost -- the real number is the
    corpus ``aggregate``, ``sum(dollars) / sum(successes)``, which is ``inf`` when nothing
    succeeded (infeasibility is the honest answer; a per-token metric reporting "$0.003/call"
    for an agent that never succeeds is actively misleading).

    Args:
        check: The success oracle (as in :class:`TaskSuccessMetric`).
        price: A :class:`~ek.agents.cost.ModelPrice` applied to every episode, or ``None`` to
            resolve each episode's model against ``prices``.
        prices: A ``{model: ModelPrice}`` catalog (see :func:`ek.agents.load_prices`).
    """

    name = "cost_per_success"

    def __init__(
        self,
        check: Any = None,
        *,
        price: Optional[ModelPrice] = None,
        prices: Optional[Mapping[str, ModelPrice]] = None,
    ):
        self.check = resolve("checkers", check, default="output")
        self.price = price
        self.prices = prices

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        if not isinstance(pred, Episode):
            raise TypeError(
                "CostPerSuccessMetric scores an Episode (it needs the episode's Cost); got "
                f"{type(pred).__name__}"
            )
        ok = bool(self.check(pred, gold))
        money = episode_dollars(pred, prices=self.prices, price=self.price)
        return Score(
            value=money,
            metric="cost_per_success",
            detail={
                "dollars": money,
                "success": ok,
                # Without this stamp the regression gate would read cost_per_success as
                # higher-is-better and wave a cost regression straight through.
                "higher_is_better": False,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """``sum(dollars) / sum(successes)`` -- a ratio of two sums, never a mean of ratios."""
        total = sum(s.detail.get("dollars", 0.0) for s in scores)
        n_success = sum(1 for s in scores if s.detail.get("success"))
        return cost_of_pass(total, n_success)


# Register default-configured instances so `score(..., metric="tool_call")` Just Works;
# pass a constructed instance when you need non-default configuration.
def _register_defaults() -> None:
    from .toolcalls import ToolCallMetric
    from .trajectory import TrajectoryMetric

    register("metrics", "task_success", TaskSuccessMetric())
    register("metrics", "cost_per_success", CostPerSuccessMetric())
    register("metrics", "tool_call", ToolCallMetric())
    register("metrics", "trajectory", TrajectoryMetric())


_register_defaults()
