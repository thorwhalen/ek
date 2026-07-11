"""Trajectory (process) evaluation: a cost-weighted **sequence** edit distance over steps.

Outcome grading tells you *whether* the agent arrived; trajectory evaluation tells you *how*.
This scores the ordered sequence of tool calls against a reference trajectory, with per-step
and per-argument costs read from the Layer-A grammar -- so taking a needless detour through a
*destructive* tool costs more than a harmless one.

**Why not reuse the flagship** :class:`~ek.metrics.graphs.TypedGraphMetric`? It is tempting
(a trajectory *is* a typed graph), but the graph-edit-distance engine is the wrong tool for a
**linear** object, on four counts:

1. **It ignores order.** ``networkx.graph_edit_distance`` is an isomorphism search; node
   substitution matches on type + fields and ignores node identity. Chain edges add
   substitutable mass; they do not impose an ordering constraint. Two identical calls at
   different positions become interchangeable -- which is exactly the error a trajectory metric
   exists to catch.
2. **It is capped.** ``DEFAULT_MAX_NODES = 60`` raises on a 100-step episode.
3. **It is nondeterministic.** The NP-hard search runs under a wall-clock ``timeout`` and
   returns a best-so-far bound -- a wall-clock-dependent score for what is a *linear* alignment.
4. **Its denominator is polluted** by the synthetic ordering edges.

So we reuse the **cost model** (the grammar's importance weights) and not the **engine**: an
O(n*m) Needleman-Wunsch alignment, which is order-honoring, exact, deterministic and unbounded
in length. (A genuine DAG mode -- for trajectories carrying explicit step *dependencies* --
would justify the graph engine; ``Step`` does not model dependency edges yet, so it is
deliberately not offered rather than shipped as a mode that quietly ignores order.)

The four schemes **disagree with each other** -- pick one deliberately, exactly as with the
partial-match schemes on the IE side (``misc/docs/ek_02``, ``misc/docs/ek_10``).

Example:
    >>> from ek.agents.base import Step
    >>> gold = [Step("search", {"q": "cat"}), Step("answer", {"a": "meow"})]
    >>> TrajectoryMetric()(gold, gold).value                      # identical -> distance 0
    0.0
    >>> detour = [Step("search", {"q": "cat"}), Step("search", {"q": "cat"}),
    ...           Step("answer", {"a": "meow"})]
    >>> round(TrajectoryMetric()(detour, gold).value, 3) > 0      # a needless extra step costs
    True
    >>> TrajectoryMetric(scheme="superset")(detour, gold).value   # but it IS a superset of gold
    0.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from ..base import GraphGrammar, Score
from ..canonicalize import default_canonicalizer, resolve_canonicalizer
from .base import as_call
from .toolcalls import (
    _MISSING,
    _arg_costs,
    _calls_of,
    _call_matches,
    _norm_value,
    _tool_weight,
    match_calls,
)

#: The trajectory-match schemes. They disagree; choose deliberately.
SCHEMES = ("in_order", "exact", "any_order", "superset")


def _mass(call: tuple, grammar: Optional[GraphGrammar]) -> float:
    """The total cost of a step: its call weight plus all of its argument weights."""
    tool, args = call
    return _tool_weight(grammar, tool) + sum(_arg_costs(grammar, tool, args).values())


def _sub_cost(p: tuple, g: tuple, grammar: Optional[GraphGrammar], canon) -> float:
    """Cost of substituting predicted step ``p`` for gold step ``g``.

    A *different tool* is not a substitution at all -- it is a delete plus an insert (the same
    rule ``TypedGraphMetric`` applies to an incompatible node type). Same tool, different
    arguments costs only the weight of the arguments that disagree.
    """
    p_tool, p_args = p
    g_tool, g_args = g
    if p_tool != g_tool:
        return _mass(p, grammar) + _mass(g, grammar)
    costs = _arg_costs(grammar, g_tool, set(p_args) | set(g_args))
    # Use the _MISSING sentinel, not .get(name) -> None: an *omitted* argument must not compare
    # equal to a gold argument whose value legitimately IS None (which would make skipping it,
    # or hallucinating an extra `{"x": None}`, entirely free).
    return sum(
        w
        for name, w in costs.items()
        if _norm_value(p_args.get(name, _MISSING), canon)
        != _norm_value(g_args.get(name, _MISSING), canon)
    )


def _needleman_wunsch(pred: list, gold: list, grammar, canon) -> float:
    """Order-honoring cost-weighted sequence edit distance (O(n*m), exact, deterministic)."""
    n, m = len(pred), len(gold)
    # Row-wise DP; only the previous row is needed.
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + _mass(gold[j - 1], grammar)
    for i in range(1, n + 1):
        cur = [prev[0] + _mass(pred[i - 1], grammar)] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j - 1] + _sub_cost(pred[i - 1], gold[j - 1], grammar, canon),
                prev[j] + _mass(pred[i - 1], grammar),  # delete a predicted step
                cur[j - 1] + _mass(gold[j - 1], grammar),  # insert a missing gold step
            )
        prev = cur
    return prev[m]


class TrajectoryMetric:
    """Cost-weighted trajectory distance as a :class:`~ek.base.Metric` (**lower is better**).

    Args:
        scheme: One of :data:`SCHEMES`.
            ``in_order`` (default) -- Needleman-Wunsch edit distance; order matters, gaps allowed.
            ``exact`` -- 0 iff the call sequences are identical, else the full mass.
            ``any_order`` -- multiset distance; order ignored (the calls, not the path).
            ``superset`` -- 0 iff every gold call occurs in the prediction; extra steps are free
            (use when the reference lists *required* calls rather than the whole path).
        grammar: Layer-A grammar supplying step/argument cost weights (may be passed per call).
        canonicalizer: ``str -> str`` applied to string arguments before comparison.
    """

    name = "trajectory"

    def __init__(
        self,
        scheme: str = "in_order",
        *,
        grammar: Optional[GraphGrammar] = None,
        canonicalizer=None,
    ):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        self.scheme = scheme
        self.grammar = grammar
        # As in ToolCallMetric: argument comparison folds case/whitespace by default.
        self.canonicalizer = (
            resolve_canonicalizer(canonicalizer)
            if canonicalizer is not None
            else default_canonicalizer()
        )

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        g = grammar if grammar is not None else self.grammar
        canon = self.canonicalizer
        pred_calls = [as_call(x) for x in _calls_of(pred)]
        gold_calls = [as_call(x) for x in _calls_of(gold)]

        pred_mass = sum(_mass(c, g) for c in pred_calls)
        gold_mass = sum(_mass(c, g) for c in gold_calls)

        if self.scheme == "superset":
            denom = gold_mass
            raw = self._superset_cost(pred_calls, gold_calls, g, canon)
        else:
            denom = pred_mass + gold_mass
            if self.scheme == "exact":
                raw = 0.0 if _same(pred_calls, gold_calls, canon) else denom
            elif self.scheme == "any_order":
                raw = self._any_order_cost(pred_calls, gold_calls, g, canon)
            else:
                raw = _needleman_wunsch(pred_calls, gold_calls, g, canon)

        if denom == 0:  # nothing expected and nothing done -> identical
            normalized = 0.0
        else:
            normalized = min(1.0, raw / denom)

        return Score(
            value=normalized,
            metric="trajectory",
            detail={
                "raw_distance": raw,
                "denom": denom,
                "similarity": 1.0 - normalized,
                "scheme": self.scheme,
                "n_pred_steps": len(pred_calls),
                "n_gold_steps": len(gold_calls),
                "higher_is_better": False,
                "exact": True,  # unlike GED, this alignment is exact, not budget-truncated
            },
        )

    def _any_order_cost(self, pred_calls, gold_calls, g, canon) -> float:
        """Multiset distance: match calls ignoring order, charge the residue."""
        pairs, extra, missed = match_calls(pred_calls, gold_calls, canon=canon)
        cost = sum(_sub_cost(p, gl, g, canon) for p, gl in pairs)
        cost += sum(_mass(c, g) for c in extra)
        cost += sum(_mass(c, g) for c in missed)
        return cost

    def _superset_cost(self, pred_calls, gold_calls, g, canon) -> float:
        """Cost of the gold calls the prediction *failed to make* (extras are free)."""
        pairs, _, missed = match_calls(pred_calls, gold_calls, canon=canon)
        # A matched-but-wrong-arguments call still fails to cover its gold call.
        wrong = [gl for p, gl in pairs if not _call_matches(p[1], gl[1], canon=canon)]
        return sum(_mass(c, g) for c in missed) + sum(_mass(c, g) for c in wrong)

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Corpus distance = total raw distance / total maximum distance (never a mean)."""
        total_raw = sum(s.detail.get("raw_distance", 0.0) for s in scores)
        total_denom = sum(s.detail.get("denom", 0.0) for s in scores)
        return (total_raw / total_denom) if total_denom else 0.0


def _same(pred_calls: list, gold_calls: list, canon) -> bool:
    """Whether two call sequences are identical after canonicalization (order included)."""
    if len(pred_calls) != len(gold_calls):
        return False
    return all(
        p[0] == g[0] and _call_matches(p[1], g[1], canon=canon)
        for p, g in zip(pred_calls, gold_calls)
    )
