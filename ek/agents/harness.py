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

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from ..registry import resolve
from ..stores import json_store
from .base import Episode, RunProvenance, TaskSpec
from .cost import ModelPrice, cost_of_pass, cost_report, episode_dollars
from .reliability import ReliabilityReport, bootstrap_ci, reliability


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
        tasks: :class:`~ek.agents.base.TaskSpec` s (or a ``{task_id: spec}`` mapping).
        k: Trials per task. ``k > 1`` is what makes ``pass^k`` meaningful -- and it only means
            anything if the agent is genuinely stochastic (the report warns if it is not).
        check: The success oracle -- a registered checker name or an ``(episode, gold) -> bool``
            callable. Defaults to ``"output"``; **inject a state-based/executable oracle** for a
            real suite, and run it isolated from the agent.
        price/prices: Rates for costing the episodes (see :mod:`ek.agents.cost`).
        run: :class:`~ek.agents.base.RunProvenance` for this run (model, simulator, suite
            version, scaffold). Recorded on every episode and on any baseline saved from it.
        seed: Base RNG seed; trial *i* of each task records ``seed + i`` so trials are
            distinguishable and a run is reproducible.
        persist/run_id/rootdir: Persist the run to the ``runs``/``results`` stores.

    Returns:
        A :class:`~ek.agents.reliability.ReliabilityReport` carrying ``pass@k``, ``pass^k`` (with
        a bootstrap CI), the success rate (with a Wilson CI), the per-slice cuts, and the cost
        report -- because reliability without cost is only half the answer.
    """
    task_list = _as_tasks(tasks)
    checker = resolve("checkers", check, default="output")
    base_run = run or RunProvenance()

    episodes: list = []
    for task in task_list:
        for trial in range(k):
            trial_run = replace(
                base_run, seed=None if seed is None else seed + trial
            )
            result = agent(task)
            episode = _as_episode(result, task)
            episode = replace(episode, run=episode.run or trial_run)
            episode = episode.graded(bool(checker(episode, task.gold)))
            episodes.append(episode)

    slices = {t.task_id: t.slice for t in task_list if t.slice is not None}
    costs = cost_report(episodes, prices=prices, price=price)
    costs["cost_ci"] = _bootstrap_cost_ci(episodes, prices=prices, price=price)

    report = reliability(episodes, k=k, slices=slices or None, cost=costs)
    report.detail["episodes"] = len(episodes)
    report.detail["provenance"] = _provenance_dict(base_run)

    if persist:
        _persist(report, run_id=run_id, rootdir=rootdir)
    return report


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


def _summary(report: ReliabilityReport) -> dict:
    return {
        "kind": "agent",
        "k": report.k,
        "n_tasks": report.n_tasks,
        "n_trials": report.n_trials,
        "success_rate": report.success_rate,
        "success_ci": list(report.success_ci),
        "pass_at_k": report.pass_at_k,
        "pass_hat_k": report.pass_hat_k,
        "pass_hat_k_ci": list(report.pass_hat_k_ci),
        "stochastic": report.stochastic,
        "per_slice": report.per_slice,
        "cost_per_success": report.cost.get("cost_per_success"),
        "total_dollars": report.cost.get("total_dollars"),
        "provenance": report.detail.get("provenance", {}),
    }


# ---------------------------------------------------------------------------
# Baselines + the variance-aware gate
# ---------------------------------------------------------------------------


def save_agent_baseline(
    report: ReliabilityReport, name: str, *, rootdir: Optional[str] = None
) -> dict:
    """Freeze a run as a named baseline (including its provenance, so the gate can guard it)."""
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
    base = (
        load_agent_baseline(baseline, rootdir=rootdir)
        if isinstance(baseline, str)
        else baseline
    )
    if base is None:  # no baseline yet -> first run always passes
        return AgentGateResult(
            passed=True,
            pass_hat_k=report.pass_hat_k,
            success_rate=report.success_rate,
            cost_per_success=report.cost.get("cost_per_success"),
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

    # Reliability: fail only if we are CONFIDENT it dropped (whole CI below the baseline).
    if base_hat is not None and report.pass_hat_k_ci[1] < base_hat - tolerance:
        reasons.append(
            f"pass^{report.k} regressed: CI {_fmt_ci(report.pass_hat_k_ci)} lies below "
            f"baseline {base_hat:.3f} (tolerance {tolerance})"
        )
    if base_rate is not None and report.success_ci[1] < base_rate - tolerance:
        reasons.append(
            f"success rate regressed: CI {_fmt_ci(report.success_ci)} lies below baseline "
            f"{base_rate:.3f} (tolerance {tolerance})"
        )
    # Cost: fail only if we are CONFIDENT it rose.
    if (
        base_cost is not None
        and cost_ci[0] is not None
        and _finite(base_cost)
        and _finite(cost_ci[0])
        and cost_ci[0] > base_cost * (1.0 + cost_tolerance)
    ):
        reasons.append(
            f"cost per success regressed: CI low {cost_ci[0]:.4g} exceeds baseline "
            f"{base_cost:.4g} (+{cost_tolerance:.0%})"
        )

    # An overlapping interval is not a pass -- it is an underpowered experiment. Say so.
    underpowered = not reasons and base_hat is not None and report.pass_hat_k < base_hat
    if underpowered:
        warnings.warn(
            f"agent_regression_gate: pass^{report.k} fell from {base_hat:.3f} to "
            f"{report.pass_hat_k:.3f}, but the CI {_fmt_ci(report.pass_hat_k_ci)} still covers "
            "the baseline -- the run is underpowered, not clean. Add trials or tasks before "
            "trusting this pass.",
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


def _fmt_ci(ci: Sequence) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _finite(x: Any) -> bool:
    import math

    return isinstance(x, (int, float)) and math.isfinite(x)
