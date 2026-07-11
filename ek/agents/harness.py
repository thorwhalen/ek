"""The agent task-suite harness: k trials, per-slice cuts, and a gate that survives noise.

:func:`run_suite` is :func:`ek.evaluate_store` specialized for agents: it runs the system under
test over a suite **k times per task** (reliability needs repeats), grades each episode with an
*injected* checker, and reports ``pass^k`` + cost + latency per slice.

**Why a separate gate.** ``ek``'s offline :func:`ek.regression_gate` is a scalar point
comparison with ``tolerance=0.0`` -- correct for a deterministic OCR benchmark, **unsound** for
an agent. Agent metrics are random variables: re-running the same agent moves the number. A
point gate therefore (a) flags sampling noise as a regression and (b) misses a real regression
that hides inside the tolerance band. :func:`agent_regression_gate` compares **intervals**
(Wilson on the success rate, bootstrap on ``pass^k`` and on the skewed cost-per-success ratio)
and only fails when the *whole* current interval sits on the wrong side of the baseline.

**Hidden eval variables.** In a tau-bench-style suite the *user simulator is itself an LLM* --
change its model or prompt and the scores move, silently. So a baseline records its
:class:`~ek.agents.base.RunProvenance` and the gate **refuses** to compare across a changed
simulator, suite version, or scaffold, exactly as the offline gate already refuses to compare
two different metrics. Comparing across them is not a regression check, it is a category error.

**Verifier isolation is a security property.** The checker must run where the agent's trajectory
cannot reach it (a red-team produced a ~10-line ``conftest.py`` that rewrote every test outcome
to "passed"). ``ek`` cannot enforce your sandbox -- it only refuses to pretend the problem does
not exist.

Example:
    >>> from ek.agents.base import Cost, Episode, TaskSpec
    >>> tasks = [TaskSpec("t1", input="2+2", gold="4"), TaskSpec("t2", input="3+3", gold="6")]
    >>> def agent(task):
    ...     answers = {"2+2": "4", "3+3": "7"}      # gets t2 wrong, every time
    ...     return Episode(output=answers[task.input], cost=Cost(input_tokens=10))
    >>> report = run_suite(agent, tasks, k=1)
    >>> report.success_rate, report.pass_hat_k
    (0.5, 0.5)
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from ..registry import resolve
from ..stores import json_store
from .base import Episode, RunProvenance, TaskSpec, suite_grammar
from .cost import ModelPrice, cost_of_pass, cost_report, episode_dollars
from .reliability import (
    ReliabilityReport,
    bootstrap_ci,
    difference_upper_bound,
    newcombe_difference,
    reliability,
)


def _as_tasks(tasks: Any) -> list:
    """Coerce a task suite to a list of :class:`~ek.agents.base.TaskSpec`."""
    if isinstance(tasks, Mapping):
        out = []
        for task_id, spec in tasks.items():
            if isinstance(spec, TaskSpec):
                out.append(spec)
            elif isinstance(spec, Mapping):
                out.append(TaskSpec(task_id=task_id, **spec))
            else:
                out.append(TaskSpec(task_id=task_id, input=spec))
        return out
    return list(tasks)


def _as_episode(result: Any, task: TaskSpec) -> Episode:
    """Accept an Episode from the agent, or wrap a bare answer into one."""
    if isinstance(result, Episode):
        return replace(result, task_id=result.task_id or task.task_id)
    if hasattr(result, "trajectory") and hasattr(result, "task_id"):
        return result  # episode-shaped duck
    return Episode(task_id=task.task_id, output=result)


def run_suite(
    agent: Callable[[TaskSpec], Any],
    tasks: Any,
    *,
    k: int = 1,
    check: Any = None,
    metrics: Optional[Mapping] = None,
    price: Optional[ModelPrice] = None,
    prices: Optional[Mapping[str, ModelPrice]] = None,
    run: Optional[RunProvenance] = None,
    seed: Optional[int] = None,
    persist: bool = False,
    run_id: Optional[str] = None,
    rootdir: Optional[str] = None,
) -> ReliabilityReport:
    """Run ``agent`` over a task suite, ``k`` trials per task, and report reliability + cost.

    Args:
        agent: The system under test: ``TaskSpec -> Episode`` (a bare answer is wrapped). It
            receives the whole spec -- ``.input``, ``.tools``, ``.gold`` -- not just the input.
            If it accepts a keyword ``seed``, the per-trial seed is passed to it.
        tasks: :class:`~ek.agents.base.TaskSpec` s (or a ``{task_id: spec}`` mapping).
        k: Trials per task. ``k > 1`` is what makes ``pass^k`` meaningful -- and it only means
            anything if the agent is genuinely stochastic (the report warns if it is not).
        check: The success oracle -- a registered checker name or an ``(episode, gold) -> bool``
            callable. Defaults to ``"output"``; **inject a state-based/executable oracle** for a
            real suite, and run it isolated from the agent.
        metrics: Optional ``{name: Metric}`` scored on every episode against ``task.gold``, with
            the **suite's Layer-A grammar injected** -- this is how the tool/argument cost
            weights reach a ``tool_call``/``trajectory`` metric. Results land in
            ``report.detail["metrics"]``.
        price/prices: Rates for costing the episodes (see :mod:`ek.agents.cost`).
        run: :class:`~ek.agents.base.RunProvenance` for this run (model, simulator, suite
            version, scaffold). Recorded on every episode and on any baseline saved from it.
        seed: Base RNG seed. Trial *i* of each task uses ``seed + i``: it is recorded on the
            episode's provenance **and passed to the agent** when the agent accepts a ``seed``
            keyword.
        persist/run_id/rootdir: Persist the run to the ``runs``/``results`` stores.

    Returns:
        A :class:`~ek.agents.reliability.ReliabilityReport` carrying ``pass@k``, ``pass^k`` (with
        a bootstrap CI), the success rate (with a Wilson CI), the per-slice cuts, and the cost
        report -- because reliability without cost is only half the answer.
    """
    task_list = _as_tasks(tasks)
    checker = resolve("checkers", check, default="output")
    base_run = run or RunProvenance()
    # The suite's Layer-A grammar (tools + their arg cost weights). This is what makes the
    # extra metrics cost-sensitive: a wrong argument to a *destructive* tool is not one unit
    # of error. Built once and injected -- it is the same frozen cost SSOT the IE side uses.
    grammar = suite_grammar(task_list)
    accepts_seed = _accepts_seed(agent)

    episodes: list = []
    by_task: dict = {}
    for task in task_list:
        for trial in range(k):
            trial_seed = None if seed is None else seed + trial
            trial_run = replace(base_run, seed=trial_seed)
            # Actually *hand the agent* its seed when it can take one, rather than only
            # stamping it on provenance and calling the run "reproducible".
            result = (
                agent(task, seed=trial_seed)
                if (accepts_seed and trial_seed is not None)
                else agent(task)
            )
            episode = _as_episode(result, task)
            episode = replace(episode, run=episode.run or trial_run)
            episode = episode.graded(bool(checker(episode, task.gold)))
            episodes.append(episode)
            by_task.setdefault(task.task_id, []).append(episode)

    slices = {t.task_id: t.slice for t in task_list if t.slice is not None}
    costs = cost_report(episodes, prices=prices, price=price)
    costs["cost_ci"] = _bootstrap_cost_ci(episodes, prices=prices, price=price)

    report = reliability(episodes, k=k, slices=slices or None, cost=costs)
    report.detail["episodes"] = len(episodes)
    report.detail["provenance"] = _provenance_dict(base_run)
    report.detail["value_weighted_success"] = _value_weighted_success(task_list, by_task)
    if metrics:
        report.detail["metrics"] = _score_metrics(metrics, task_list, by_task, grammar)

    if persist:
        _persist(report, run_id=run_id, rootdir=rootdir)
    return report


def _accepts_seed(fn: Any) -> bool:
    """Whether the agent takes a keyword ``seed`` (same capability probe the facade uses)."""
    try:
        return "seed" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _value_weighted_success(task_list: Sequence[TaskSpec], by_task: Mapping) -> float:
    """Success rate weighted by each task's ``value`` -- not all tasks are worth the same.

    ``TaskSpec.value`` is the task-level analogue of ``FieldSpec.importance``: finishing a
    high-value task counts for more than finishing a trivial one.
    """
    total = hit = 0.0
    for task in task_list:
        eps = by_task.get(task.task_id, ())
        if not eps:
            continue
        total += task.value * len(eps)
        hit += task.value * sum(1 for e in eps if e.success)
    return (hit / total) if total else 0.0


def _score_metrics(
    metrics: Mapping, task_list: Sequence[TaskSpec], by_task: Mapping, grammar
) -> dict:
    """Score every episode with each extra metric, injecting the suite's Layer-A grammar.

    This is how the cost weights on ``ToolSpec``/``FieldSpec.importance`` actually reach a
    tool-call or trajectory metric on the harness path (otherwise they are inert).
    """
    gold_less = [t.task_id for t in task_list if t.gold is None]
    if gold_less:
        # Dropping them in silence would report an empty/partial metrics block as if it covered
        # the suite. Say what was excluded and why.
        warnings.warn(
            f"run_suite(metrics=...): {len(gold_less)} task(s) have no `gold` and were excluded "
            f"from the extra metrics (e.g. {gold_less[:3]}). Set TaskSpec.gold to score them.",
            stacklevel=3,
        )

    out: dict = {}
    for name, metric in metrics.items():
        scores = []
        for task in task_list:
            if task.gold is None:
                continue
            for episode in by_task.get(task.task_id, ()):
                scores.append(metric(episode, task.gold, grammar=grammar))
        if not scores:
            continue
        aggregate = (
            metric.aggregate(scores)
            if hasattr(metric, "aggregate")
            else sum(float(s) for s in scores) / len(scores)
        )
        out[name] = {"aggregate": aggregate, "n": len(scores), "n_excluded": len(gold_less)}
    return out


def _bootstrap_cost_ci(episodes: Sequence[Episode], *, prices, price) -> tuple:
    """Bootstrap CI on the cost-per-success **ratio** (skewed: a normal interval is meaningless).

    ``(None, None)`` when no rates were supplied -- ek does not invent prices.
    """
    if price is None and prices is None:
        return (None, None)
    pairs = [
        (episode_dollars(e, prices=prices, price=price), 1 if e.success else 0)
        for e in episodes
    ]
    if not pairs:
        return (float("nan"), float("nan"))

    def ratio(sample) -> float:
        return cost_of_pass(sum(d for d, _ in sample), sum(s for _, s in sample))

    return bootstrap_ci(pairs, ratio)


def _provenance_dict(run: RunProvenance) -> dict:
    return {
        "model": run.model,
        "simulator": run.simulator,
        "suite_version": run.suite_version,
        "scaffold": run.scaffold,
        "temperature": run.temperature,
    }


def _persist(report: ReliabilityReport, *, run_id: Optional[str], rootdir) -> None:
    if run_id is None:
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("agent-run-%Y%m%dT%H%M%S")
    summary = _summary(report)
    json_store("runs", rootdir=rootdir)[run_id] = summary
    json_store("results", rootdir=rootdir)[run_id] = {
        **summary,
        "per_task": report.per_task,
    }


def _json_num(x: Any) -> Any:
    """JSON has no ``Infinity``/``NaN``. Persist them as ``null`` rather than emit invalid JSON.

    ``cost_per_success`` is legitimately ``inf`` when nothing succeeded; Python's ``json`` would
    happily write a bare ``Infinity`` token that every strict parser (jq, a JS dashboard) rejects.
    ``null`` here means "not measurable", which is exactly what an infinite cost-of-pass is.
    """
    if isinstance(x, (int, float)) and not math.isfinite(x):
        return None
    return x


def _summary(report: ReliabilityReport) -> dict:
    return {
        "kind": "agent",
        "k": report.k,
        "n_tasks": report.n_tasks,
        "n_trials": report.n_trials,
        # Persist the raw COUNTS, not just the rate: a proper two-sample test on the difference
        # needs (successes, n) for both runs, and a rate alone cannot reconstruct them.
        "n_success": report.n_success,
        "success_rate": report.success_rate,
        "success_ci": list(report.success_ci),
        "pass_at_k": _json_num(report.pass_at_k),
        "pass_hat_k": _json_num(report.pass_hat_k),
        "pass_hat_k_ci": list(report.pass_hat_k_ci),
        "stochastic": report.stochastic,
        "per_slice": report.per_slice,
        "cost_per_success": _json_num(report.cost.get("cost_per_success")),
        "cost_ci": [_json_num(v) for v in (report.cost.get("cost_ci") or (None, None))],
        "total_dollars": _json_num(report.cost.get("total_dollars")),
        "provenance": report.detail.get("provenance", {}),
    }


# ---------------------------------------------------------------------------
# Baselines + the variance-aware gate
# ---------------------------------------------------------------------------


def save_agent_baseline(
    report: ReliabilityReport, name: str, *, rootdir: Optional[str] = None
) -> dict:
    """Freeze a run as a named baseline (including its provenance, so the gate can guard it).

    Raises:
        ValueError: if the run evaluated nothing. An empty baseline is worse than no baseline --
            it is a *permanent free pass*, because every bound in it is the maximally-uncertain
            default and no later run can be shown to fall below it.
    """
    if report.n_trials == 0:
        raise ValueError(
            f"save_agent_baseline({name!r}): refusing to freeze a run that evaluated ZERO "
            "trials. Such a baseline can never be regressed against -- it would pass every "
            "future run, including a totally broken one."
        )
    record = _summary(report)
    json_store("baselines", rootdir=rootdir)[name] = record
    return record


def load_agent_baseline(name: str, *, rootdir: Optional[str] = None) -> Optional[dict]:
    """Load a named agent baseline (or ``None``)."""
    store = json_store("baselines", rootdir=rootdir)
    return store[name] if name in store else None


@dataclass
class AgentGateResult:
    """Outcome of a variance-aware agent regression check."""

    passed: bool = True
    reasons: list = field(default_factory=list)
    pass_hat_k: Optional[float] = None
    pass_hat_k_baseline: Optional[float] = None
    success_rate: Optional[float] = None
    success_rate_baseline: Optional[float] = None
    cost_per_success: Optional[float] = None
    cost_per_success_baseline: Optional[float] = None
    underpowered: bool = False

    def __bool__(self) -> bool:
        return self.passed


def _comparability(record: Mapping) -> tuple:
    p = record.get("provenance", {}) or {}
    return (p.get("simulator", ""), p.get("suite_version", ""), p.get("scaffold", ""))


def agent_regression_gate(
    report: ReliabilityReport,
    baseline: Any,
    *,
    tolerance: float = 0.0,
    cost_tolerance: float = 0.0,
    rootdir: Optional[str] = None,
) -> AgentGateResult:
    """Fail only when the evidence says the agent **really** got worse -- not on noise.

    A regression is declared only if the *entire* current interval lies on the wrong side of the
    baseline point (minus ``tolerance``): the upper bound of the ``pass^k`` / success-rate CI is
    below the baseline, or the lower bound of the cost-per-success CI is above it. Overlapping
    intervals are *not* a regression -- they are an underpowered experiment, and the result says so.

    Raises:
        ValueError: if the baseline was produced with a different **user simulator**, suite
            version, or scaffold -- those runs are not comparable, and silently comparing them
            would be a category error (the same reason the offline gate refuses to compare two
            different metrics).
    """
    if isinstance(baseline, str):
        base = load_agent_baseline(baseline, rootdir=rootdir)
    elif isinstance(baseline, ReliabilityReport):
        base = _summary(baseline)  # comparing two live reports is the natural call
    else:
        base = baseline

    # An empty run must never be green. With no tasks every CI is the maximally-uncertain
    # [0, 1], so *no* regression test can fire and a broken loader / bad glob / over-filtered
    # suite would sail through as a pass. Absence of evidence is not evidence of no regression.
    if report.n_trials == 0:
        return AgentGateResult(
            passed=False,
            reasons=[
                "the run evaluated ZERO trials -- an empty suite cannot pass a regression "
                "gate (check the task loader/filter)."
            ],
        )

    if base is None:  # no baseline yet -> first run always passes
        return AgentGateResult(
            passed=True,
            pass_hat_k=report.pass_hat_k,
            success_rate=report.success_rate,
            cost_per_success=report.cost.get("cost_per_success"),
        )

    # A baseline recorded from an EMPTY run is a permanent free pass: every one of its bounds is
    # the maximally-uncertain default, so nothing can ever be shown to fall below it. Guarding
    # only the *current* run would just move the hole one hop upstream.
    if not base.get("n_trials"):
        raise ValueError(
            "agent_regression_gate: the baseline recorded ZERO trials, so it cannot be "
            "regressed against -- every bound in it is the maximally-uncertain default, and "
            "any run at all would 'pass'. Re-baseline from a real run."
        )

    current_key = _comparability(_summary(report))
    base_key = _comparability(base)
    if current_key != base_key:
        raise ValueError(
            "agent_regression_gate: refusing to compare runs with different evaluation "
            f"conditions -- baseline (simulator, suite_version, scaffold)={base_key!r} vs "
            f"current={current_key!r}. A changed user-simulator or suite version moves the "
            "score on its own; re-baseline instead of comparing across them."
        )

    reasons: list = []
    base_hat = base.get("pass_hat_k")
    base_rate = base.get("success_rate")
    base_cost = base.get("cost_per_success")
    cur_cost = report.cost.get("cost_per_success")
    cost_ci = report.cost.get("cost_ci") or (None, None)

    # A suite that shrank is not the same experiment. (`is not None`, not truthiness: a
    # zero-task baseline must not slip through the check by being falsy.)
    base_tasks = base.get("n_tasks")
    if base_tasks is not None and report.n_tasks < base_tasks:
        reasons.append(
            f"the suite shrank: {report.n_tasks} tasks vs {base_tasks} in the baseline -- "
            "not the same experiment (re-baseline, or restore the missing tasks)."
        )

    # THE core rule: test the DIFFERENCE, with both runs' uncertainty folded in.
    #
    # Two wrong ways to do this, both of which we have now been bitten by:
    #   * compare our interval to the baseline's POINT -- the baseline is itself a noisy
    #     estimate, so one flake against a lucky baseline reads as a confident regression;
    #   * ask whether the two 95% intervals OVERLAP -- that is not a 5% test but roughly a
    #     0.5% one, and it buys its calm by going blind (a 15-point drop on a 60-task suite
    #     becomes invisible). A gate that cannot see a real regression is as broken as one
    #     that cries wolf.
    # So: a proper two-sample interval on (current - baseline). Regression iff its UPPER bound
    # is below -tolerance, i.e. we are confident the true change is negative.
    base_succ, base_trials = base.get("n_success"), base.get("n_trials")
    if base_succ is not None and base_trials:
        _lo, hi = newcombe_difference(
            report.n_success, report.n_trials, base_succ, base_trials
        )
        if hi < -tolerance:
            reasons.append(
                f"success rate regressed: {report.success_rate:.3f} vs baseline "
                f"{base_rate:.3f}; the 95% interval for the change tops out at {hi:+.3f} "
                f"(tolerance {tolerance})"
            )
    elif base_rate is not None and report.success_ci[1] < base_rate - tolerance:
        # Legacy baseline with no counts: fall back to the (cruder) interval-vs-point test.
        reasons.append(
            f"success rate regressed: CI {_fmt_ci(report.success_ci)} lies below baseline "
            f"{base_rate:.3f} (tolerance {tolerance})"
        )

    # pass^k is a mean of per-task estimators, not a raw proportion, so Newcombe does not apply
    # -- but the same principle does: bound the DIFFERENCE using both runs' uncertainty.
    if base_hat is not None and report.pass_hat_k is not None:
        hi = difference_upper_bound(
            report.pass_hat_k,
            report.pass_hat_k_ci,
            base_hat,
            base.get("pass_hat_k_ci") or (base_hat, base_hat),
        )
        if hi is not None and hi < -tolerance:
            reasons.append(
                f"pass^{report.k} regressed: {report.pass_hat_k:.3f} vs baseline "
                f"{base_hat:.3f}; the interval for the change tops out at {hi:+.3f} "
                f"(tolerance {tolerance})"
            )
    # Cost: fail only if we are CONFIDENT it rose (our low above the baseline's high).
    cost_ceiling = _upper(base, "cost_ci", base_cost)
    if (
        cost_ceiling is not None
        and cost_ci[0] is not None
        and _finite(cost_ceiling)
        and _finite(cost_ci[0])
        and cost_ci[0] > cost_ceiling * (1.0 + cost_tolerance)
    ):
        reasons.append(
            f"cost per success regressed: CI low {cost_ci[0]:.4g} exceeds the baseline CI "
            f"high {cost_ceiling:.4g} (+{cost_tolerance:.0%})"
        )

    # An overlapping interval is not a clean pass -- it is an underpowered experiment. Say so.
    underpowered = (
        not reasons
        and base_hat is not None
        and report.pass_hat_k is not None
        and report.pass_hat_k < base_hat
    )
    if underpowered:
        warnings.warn(
            f"agent_regression_gate: pass^{report.k} fell from {base_hat:.3f} to "
            f"{report.pass_hat_k:.3f}, but the intervals still overlap -- the run is "
            "underpowered, not clean. Add trials or tasks before trusting this pass.",
            stacklevel=2,
        )

    return AgentGateResult(
        passed=not reasons,
        reasons=reasons,
        pass_hat_k=report.pass_hat_k,
        pass_hat_k_baseline=base_hat,
        success_rate=report.success_rate,
        success_rate_baseline=base_rate,
        cost_per_success=cur_cost,
        cost_per_success_baseline=base_cost,
        underpowered=underpowered,
    )


def _lower(base: Mapping, ci_key: str, point: Any) -> Any:
    """The baseline's lower confidence bound (falling back to its point estimate)."""
    ci = base.get(ci_key)
    if ci and ci[0] is not None:
        return ci[0]
    return point


def _upper(base: Mapping, ci_key: str, point: Any) -> Any:
    """The baseline's upper confidence bound (falling back to its point estimate)."""
    ci = base.get(ci_key)
    if ci and len(ci) > 1 and ci[1] is not None:
        return ci[1]
    return point


def _fmt_ci(ci: Sequence) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
