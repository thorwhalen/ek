"""Tool-call correctness: BFCL-style AST matching over a *multiset* of calls.

The Berkeley Function Calling Leaderboard defines the reference algorithm: parse the emitted
call and -- **without executing it** -- check that the function name matches, that every
expected argument is present with the right value (case-insensitive, whitespace-folded string
comparison), and that no argument outside the schema was hallucinated. Arguments are matched
**by name**, so argument *order* is irrelevant; element order *within a list value* is not
(``misc/docs/ek_08``, ``misc/docs/ek_12``).

**Why this cannot just reuse** :class:`~ek.metrics.fields.FieldMetric`. That metric compares two
flat records *keyed by field name* -- the key **is** the alignment. A trajectory's tool calls are
a **keyless multiset**: two ``search(q=...)`` calls collide onto one key, and reordering or
repeating a call silently miscounts. So this module adds the missing layer -- an **assignment**
between predicted and gold calls -- and then reuses ``FieldMetric``'s exact TP/FP/FN accounting
and micro-averaged aggregation on top of it.

Cost-sensitivity is the reason to build this rather than borrow one: the per-argument weights
come from the Layer-A grammar (``FieldSpec.importance``), so a wrong argument to a *destructive*
tool is not one unit of error. No off-the-shelf tool-call metric models that.

Example:
    >>> from ek.agents.base import Step
    >>> pred = [Step("search", {"q": "cats"}), Step("answer", {"a": "meow"})]
    >>> gold = [("search", {"q": "cats"}), ("answer", {"a": "meow"})]
    >>> ToolCallMetric()(pred, gold).f1
    1.0
    >>> hallucinated = [Step("search", {"q": "cats"}), Step("delete_all", {})]
    >>> round(ToolCallMetric()(hallucinated, gold).f1, 3)     # 1 TP, 1 FP, 1 FN
    0.5
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..base import GraphGrammar, Score
from ..canonicalize import default_canonicalizer, resolve_canonicalizer
from .base import as_call

_MISSING = object()


def _f1(precision: float, recall: float) -> float:
    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )


def _norm_value(value: Any, canon) -> Any:
    """Canonicalize a value for comparison (strings folded; lists elementwise, order kept)."""
    if isinstance(value, str):
        return canon(value) if canon is not None else value
    if isinstance(value, (list, tuple)):
        return tuple(_norm_value(v, canon) for v in value)
    return value


def _arg_costs(grammar: Optional[GraphGrammar], tool: str, names) -> dict:
    """Per-argument cost weights for a tool, read off the Layer-A grammar (default 1.0)."""
    if grammar is None:
        return {n: 1.0 for n in names}
    return {n: grammar.field_cost(tool, n) for n in names}


def _tool_weight(grammar: Optional[GraphGrammar], tool: str) -> float:
    return grammar.node_cost(tool) if grammar is not None else 1.0


def _arg_diff(
    pred_args: Mapping, gold_args: Mapping, *, tool: str, grammar, canon
) -> tuple:
    """Weighted TP/FP/FN over one call's arguments (the FieldMetric accounting, weighted).

    A present-but-wrong argument counts as **both** spurious and missed -- the conventional
    slot-error accounting ``FieldMetric`` uses.
    """
    names = set(gold_args) | set(pred_args)
    costs = _arg_costs(grammar, tool, names)
    tp = fp = fn = 0.0
    for name in names:
        g = gold_args.get(name, _MISSING)
        p = pred_args.get(name, _MISSING)
        w = costs.get(name, 1.0)
        if g is _MISSING:  # an argument the schema/gold never had -> hallucinated
            fp += w
        elif p is _MISSING:  # a required argument the agent omitted
            fn += w
        elif _norm_value(p, canon) == _norm_value(g, canon):
            tp += w
        else:
            fp += w
            fn += w
    return tp, fp, fn


def _call_matches(pred_args: Mapping, gold_args: Mapping, *, canon) -> bool:
    """Exact AST match on one call's arguments (all present, all equal, none extra)."""
    if set(pred_args) != set(gold_args):
        return False
    return all(
        _norm_value(pred_args[k], canon) == _norm_value(gold_args[k], canon)
        for k in gold_args
    )


def match_calls(pred: Sequence, gold: Sequence, *, canon=None) -> tuple:
    """Assign predicted calls to gold calls -- the layer ``FieldMetric`` does not provide.

    Calls are grouped by tool name (a call to the *wrong tool* is never a match). Within a tool
    name, candidate pairs are scored and taken **globally best-first** (exact-argument matches
    before partial ones, more agreeing arguments before fewer), so identical repeated calls are
    interchangeable and a reordered pair is still matched -- while a strong pairing can never be
    stolen by a weaker one that merely happened to come first in the prediction.

    This is a deterministic greedy approximation to the optimal assignment. It is exact for the
    common cases (distinct calls, repeats, reorderings); a full min-cost bipartite matching
    (Hungarian) could differ only on pathological many-to-many partial-overlap sets, and is not
    worth a dependency here.

    Returns ``(pairs, unmatched_pred, unmatched_gold)``, ``pairs`` being ``(pred_call, gold_call)``.

    Example:
        >>> pairs, up, ug = match_calls([("s", {"q": "b"}), ("s", {"q": "a"})],
        ...                             [("s", {"q": "a"}), ("s", {"q": "b"})])
        >>> len(pairs), up, ug          # order-insensitive within a tool name
        (2, [], [])
    """
    pred_calls = [as_call(x) for x in pred]
    gold_calls = [as_call(x) for x in gold]

    by_tool_gold: dict = {}
    for i, (tool, _args) in enumerate(gold_calls):
        by_tool_gold.setdefault(tool, []).append(i)

    used_pred: set = set()
    used_gold: set = set()
    pairs: list = []

    # Fast path: only when NEITHER side repeats a tool name is the assignment actually forced.
    # (Requiring it of gold alone is a bug: with gold=[s(q=a)] and pred=[s(q=z), s(q=a)] the
    # first-come rule would hand the gold call to the WRONG pred and discard the exact match.
    # Then there is a genuine choice to make, and only the scored path can make it.)
    by_tool_pred: dict = {}
    for i, (tool, _args) in enumerate(pred_calls):
        by_tool_pred.setdefault(tool, []).append(i)
    forced = all(len(v) == 1 for v in by_tool_gold.values()) and all(
        len(v) == 1 for v in by_tool_pred.values()
    )

    if forced:
        for p_i, (tool, _p_args) in enumerate(pred_calls):
            candidates = by_tool_gold.get(tool, ())
            if not candidates:
                continue
            g_i = candidates[0]
            used_pred.add(p_i)
            used_gold.add(g_i)
            pairs.append((p_i, g_i))
    else:
        # Score every same-tool candidate pair, then take them globally best-first.
        candidates_scored: list = []
        for p_i, (tool, p_args) in enumerate(pred_calls):
            for g_i in by_tool_gold.get(tool, ()):
                g_args = gold_calls[g_i][1]
                exact = _call_matches(p_args, g_args, canon=canon)
                overlap = sum(
                    1
                    for key in set(g_args) & set(p_args)
                    if _norm_value(p_args[key], canon) == _norm_value(g_args[key], canon)
                )
                # Prefer exact, then most agreeing args; ties broken by position (determinism).
                candidates_scored.append((0 if exact else 1, -overlap, p_i, g_i))
        candidates_scored.sort()

        for _rank, _neg_overlap, p_i, g_i in candidates_scored:
            if p_i in used_pred or g_i in used_gold:
                continue
            used_pred.add(p_i)
            used_gold.add(g_i)
            pairs.append((p_i, g_i))

    unmatched_pred = [c for i, c in enumerate(pred_calls) if i not in used_pred]
    unmatched_gold = [c for i, c in enumerate(gold_calls) if i not in used_gold]
    resolved = [(pred_calls[p], gold_calls[g]) for p, g in sorted(pairs)]
    return resolved, unmatched_pred, unmatched_gold


class ToolCallMetric:
    """Tool-call correctness (BFCL AST match) as a cost-weighted :class:`~ek.base.Metric`.

    Args:
        level: ``"call"`` (default) scores whole calls -- a call is a TP only on an exact AST
            match; ``"arg"`` gives partial credit at argument granularity.
        canonicalizer: ``str -> str`` applied to string arguments before comparison
            (BFCL's case/whitespace folding). Defaults to ``ek``'s default canonicalizer.
        grammar: Layer-A grammar supplying per-tool and per-argument cost weights (may also be
            passed per call).

    The ``Score`` carries ``tp``/``fp``/``fn`` in ``detail`` so :func:`ek.evaluate` aggregates a
    correct **micro**-averaged F1 over the corpus rather than averaging per-episode F1s.
    """

    name = "tool_call"

    def __init__(
        self,
        level: str = "call",
        *,
        canonicalizer=None,
        grammar: Optional[GraphGrammar] = None,
    ):
        if level not in ("call", "arg"):
            raise ValueError(f"level must be 'call' or 'arg', got {level!r}")
        self.level = level
        # BFCL compares argument values case-insensitively with whitespace folding, so unlike
        # the IE metrics (where None means "compare raw") the default here is the standard
        # canonicalizer. Pass ``canonicalizer=str`` for a raw, byte-exact comparison.
        self.canonicalizer = (
            resolve_canonicalizer(canonicalizer)
            if canonicalizer is not None
            else default_canonicalizer()
        )
        self.grammar = grammar

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        g = grammar if grammar is not None else self.grammar
        canon = self.canonicalizer
        pred_calls = _calls_of(pred)
        gold_calls = _calls_of(gold)

        pairs, extra, missed = match_calls(pred_calls, gold_calls, canon=canon)

        tp = fp = fn = 0.0
        for (p_tool, p_args), (g_tool, g_args) in pairs:
            if self.level == "call":
                w = _tool_weight(g, g_tool)
                if _call_matches(p_args, g_args, canon=canon):
                    tp += w
                else:  # right tool, wrong arguments -> both spurious and missed
                    fp += w
                    fn += w
            else:
                a_tp, a_fp, a_fn = _arg_diff(
                    p_args, g_args, tool=g_tool, grammar=g, canon=canon
                )
                tp += a_tp
                fp += a_fp
                fn += a_fn

        # A call the agent invented (including a should-not-call) is pure FP; one it skipped, FN.
        for tool, args in extra:
            fp += _tool_weight(g, tool) + (
                0.0 if self.level == "call" else sum(_arg_costs(g, tool, args).values())
            )
        for tool, args in missed:
            fn += _tool_weight(g, tool) + (
                0.0 if self.level == "call" else sum(_arg_costs(g, tool, args).values())
            )

        precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
        f1 = _f1(precision, recall)
        return Score(
            value=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            metric="tool_call",
            detail={
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "level": self.level,
                "n_matched": len(pairs),
                "n_spurious": len(extra),
                "n_missed": len(missed),
                "higher_is_better": True,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide) -- never a mean."""
        tp = sum(s.detail.get("tp", 0.0) for s in scores)
        fp = sum(s.detail.get("fp", 0.0) for s in scores)
        fn = sum(s.detail.get("fn", 0.0) for s in scores)
        if tp + fp + fn == 0:
            return 1.0  # nothing to call and nothing called -> perfect
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return _f1(precision, recall)


def _calls_of(x: Any) -> Sequence:
    """Read a call sequence off an Episode, a Trajectory, or a bare list of calls."""
    traj = getattr(x, "trajectory", None)
    if traj is not None:
        return list(traj.steps)
    steps = getattr(x, "steps", None)
    if steps is not None:
        return list(steps)
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return list(x)
    raise TypeError(
        f"ToolCallMetric needs an Episode, a Trajectory, or a sequence of calls; "
        f"got {type(x).__name__}"
    )
