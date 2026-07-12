"""Reliability: ``pass@k`` (capability) vs ``pass^k`` (consistency), and their error bars.

The single most important distinction in agent evaluation. ``pass@k`` asks *can it ever do
this?*; ``pass^k`` asks *does it do this every single time?* -- and only the second is a
production metric. A customer-facing agent that succeeds "usually" is not shippable, and
``pass^k`` is the number that exposes it: on tau-bench, gpt-4o clears ~61% of retail tasks at
``pass^1`` but only ~25% at ``pass^8`` (``misc/docs/ek_07``, ``misc/docs/ek_08``).

**These are not** :class:`~ek.base.Metric` **s.** A metric scores one ``(pred, gold)`` pair and
:func:`ek.evaluate` aggregates independent cases; ``pass@k``/``pass^k`` are *cross-task*
quantities over **k trials of the same task**, and the trial grouping is the harness's job.
They ship here as pure functions plus a :class:`ReliabilityReport`.

Two caveats the literature forces:

- ``pass^k`` presumes **genuine run-to-run stochasticity**. On a deterministic agent every
  trial is identical, ``pass^k`` collapses to ``pass^1``, and it tells you nothing new --
  :func:`reliability` detects this and warns rather than reporting a falsely reassuring number.
- Suites are small (a few hundred tasks), so a point estimate is not a result. Report an
  interval: :func:`wilson_interval` for a success rate, :func:`bootstrap_ci` for the skewed
  cost-per-success ratio. The regression gate compares **intervals, not points**.

Example:
    >>> pass_at_k(n=5, c=2, k=1)          # capability: any 1 of 5 trials succeeded
    0.4
    >>> round(pass_at_k(n=5, c=2, k=3), 4)   # some 3-subset contains a success
    0.9
    >>> pass_hat_k(n=5, c=2, k=3)         # reliability: ALL of a 3-subset succeed
    0.0
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .base import Episode

#: z for a 95% normal interval (the Wilson default). Keyword-tunable, not a magic number.
DEFAULT_Z = 1.96

#: Resamples for the bootstrap CI on the (skewed) cost-per-success ratio.
DEFAULT_BOOTSTRAP_RESAMPLES = 2000


# ---------------------------------------------------------------------------
# The two unbiased estimators
# ---------------------------------------------------------------------------


def _check(n: int, c: int, k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"need n >= k trials to estimate at k={k}, got n={n}")
    if not 0 <= c <= n:
        raise ValueError(f"successes c must be in [0, n]; got c={c}, n={n}")


def pass_at_k(*, n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k``: probability that **at least one** of k samples succeeds.

    The HumanEval estimator ``1 - C(n-c, k) / C(n, k)`` over ``n`` trials with ``c``
    successes. This is the **capability** metric -- right when a single success is enough
    (offline candidate generation behind a verifier).

    Example:
        >>> round(pass_at_k(n=10, c=1, k=1), 3)
        0.1
        >>> pass_at_k(n=10, c=10, k=5)
        1.0
    """
    _check(n, c, k)
    if n - c < k:  # too few failures to fill a k-subset -> some success is certain
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(*, n: int, c: int, k: int) -> float:
    """Unbiased ``pass^k``: probability that **all k** independent trials succeed.

    The tau-bench estimator ``C(c, k) / C(n, k)``; for a per-trial success probability ``p``
    it decays to ``p**k``. This is the **reliability** metric -- the one that matters when
    consistency *is* the product.

    Example:
        >>> pass_hat_k(n=8, c=8, k=8)
        1.0
        >>> pass_hat_k(n=8, c=4, k=2)
        0.21428571428571427
    """
    _check(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Error bars (a point estimate on a 200-task suite is not a result)
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, *, z: float = DEFAULT_Z) -> tuple:
    """Wilson score interval for a proportion -- correct in the tails, unlike Wald/CLT.

    The naive normal ("Wald") interval is badly wrong at small n and near 0/1 -- exactly
    where agent success rates live. Wilson is the cheap, dependency-free fix.

    Example:
        >>> lo, hi = wilson_interval(0, 10)
        >>> lo == 0.0, round(hi, 3)
        (True, 0.278)
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def newcombe_difference(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    *,
    z: float = DEFAULT_Z,
) -> tuple:
    """Newcombe score interval for the **difference** of two proportions ``p_a - p_b``.

    This is the right instrument for "did the agent get worse than its baseline?", and it is
    strictly better than the tempting shortcut of *checking whether two 95% CIs overlap*.
    Non-overlap is not a 5% test -- it is roughly a 0.5% one, so it buys its low false-alarm
    rate by going nearly blind: on a 60-task suite it cannot see a 15-point success-rate drop.
    A difference interval keeps the same false-alarm rate and recovers most of the power,
    because the variance of a difference is not the sum of two interval widths.

    Newcombe (1998) method 10: build each proportion's Wilson interval, then

        lower = (p_a - p_b) - sqrt((p_a - l_a)^2 + (u_b - p_b)^2)
        upper = (p_a - p_b) + sqrt((u_a - p_a)^2 + (p_b - l_b)^2)

    Returns ``(lower, upper)``. The difference is a *confident regression* when ``upper < 0``.

    Example:
        >>> lo, hi = newcombe_difference(50, 100, 90, 100)   # 50% now vs 90% before
        >>> hi < 0                                            # confidently worse
        True
        >>> lo, hi = newcombe_difference(89, 100, 90, 100)   # 89% vs 90% -- noise
        >>> hi > 0
        True
    """
    if n_a <= 0 or n_b <= 0:
        return (-1.0, 1.0)  # no information -> the maximally uninformative interval
    p_a, p_b = successes_a / n_a, successes_b / n_b
    l_a, u_a = wilson_interval(successes_a, n_a, z=z)
    l_b, u_b = wilson_interval(successes_b, n_b, z=z)
    delta = p_a - p_b
    lower = delta - math.sqrt((p_a - l_a) ** 2 + (u_b - p_b) ** 2)
    upper = delta + math.sqrt((u_a - p_a) ** 2 + (p_b - l_b) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


def _se_from_ci(ci: Sequence, *, z: float = DEFAULT_Z) -> Optional[float]:
    """Back out a standard error from a symmetric-ish CI (for statistics we only store as CIs)."""
    if not ci or ci[0] is None or ci[1] is None:
        return None
    width = float(ci[1]) - float(ci[0])
    if not math.isfinite(width) or width < 0:
        return None
    return width / (2 * z)


def difference_upper_bound(
    value_a: float,
    ci_a: Sequence,
    value_b: float,
    ci_b: Sequence,
    *,
    z: float = DEFAULT_Z,
) -> Optional[float]:
    """Upper bound of ``a - b`` for two statistics known only through their CIs.

    Used for ``pass^k`` (a mean of per-task estimators, not a raw proportion), where a Newcombe
    interval does not apply but the same principle does: test the **difference**, combining both
    runs' uncertainty, rather than asking whether two intervals happen to overlap.
    """
    se_a, se_b = _se_from_ci(ci_a, z=z), _se_from_ci(ci_b, z=z)
    if se_a is None or se_b is None:
        return None
    return (value_a - value_b) + z * math.sqrt(se_a**2 + se_b**2)


def bootstrap_ci(
    values: Sequence[float],
    statistic,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
) -> tuple:
    """Percentile bootstrap CI for any ``statistic`` over ``values`` (deterministic by seed).

    Used for the **cost-per-success ratio**, whose distribution is skewed and heavy-tailed
    (a normal interval on it is meaningless).

    Example:
        >>> lo, hi = bootstrap_ci([1.0] * 20, lambda v: sum(v) / len(v), resamples=50)
        >>> lo == hi == 1.0
        True
    """
    values = list(values)
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo_i = int((alpha / 2) * len(stats))
    hi_i = min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))
    return (stats[lo_i], stats[hi_i])


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityReport:
    """Corpus-level reliability over a k-trial run, with per-task and per-slice cuts.

    ``success_rate`` is micro (over all trials); ``pass_at_k``/``pass_hat_k`` are averaged
    over *tasks* (each task's own unbiased estimator), never over trials.
    """

    k: int = 1
    n_tasks: int = 0
    n_trials: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    success_ci: tuple = (0.0, 1.0)
    # ``None`` (not 0.0) when no task had enough trials to estimate at k: "we could not
    # measure this" and "the agent failed everything" are opposite facts, and reporting the
    # second when the first is true is how a perfect agent gets shown as a total failure.
    pass_at_k: Optional[float] = None
    pass_hat_k: Optional[float] = None
    pass_hat_k_ci: tuple = (0.0, 1.0)
    stochastic: bool = True
    per_task: dict = field(default_factory=dict)
    per_slice: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def __float__(self) -> float:
        """The headline number is the *reliability* one (``nan`` if it could not be measured)."""
        return float("nan") if self.pass_hat_k is None else float(self.pass_hat_k)


def _group_trials(source: Any) -> dict:
    """Group ``{task_id: [bool, ...]}`` from episodes or an already-grouped mapping."""
    if isinstance(source, Mapping):
        return {tid: [bool(x) for x in outcomes] for tid, outcomes in source.items()}
    grouped: dict = {}
    for ep in source:
        if not isinstance(ep, Episode) and not hasattr(ep, "task_id"):
            raise TypeError(
                "reliability() takes Episodes or a {task_id: [bool, ...]} mapping; "
                f"got {type(ep).__name__}"
            )
        if ep.success is None:
            raise ValueError(
                f"episode for task {ep.task_id!r} is ungraded (success is None); "
                "run it through a checker (see ek.agents.TaskSuccessMetric) first."
            )
        grouped.setdefault(ep.task_id, []).append(bool(ep.success))
    return grouped


def reliability(
    source: Any,
    *,
    k: int = 1,
    slices: Optional[Mapping[str, str]] = None,
    cost: Optional[Mapping] = None,
    warn_deterministic: bool = True,
) -> ReliabilityReport:
    """Compute ``pass@k`` / ``pass^k`` / success-rate over a k-trial run.

    Args:
        source: Graded :class:`~ek.agents.base.Episode` s (grouped by ``task_id``), or an
            already-grouped ``{task_id: [bool, ...]}`` mapping.
        k: The k of pass@k / pass^k. Tasks with fewer than k trials are skipped (and
            counted in ``detail['skipped']``) rather than silently mis-estimated.
        slices: Optional ``{task_id: slice_label}`` for per-slice cuts.
        cost: An optional cost report (from :func:`ek.agents.cost_report`) to carry along --
            reliability without cost is only half the picture.
        warn_deterministic: Warn when **no** task shows mixed outcomes across trials, in
            which case ``pass^k`` degenerates to ``pass^1`` and is not informative.

    Example:
        >>> r = reliability({"t1": [True, True, False], "t2": [True, True, True]}, k=2)
        >>> r.n_tasks, r.k
        (2, 2)
        >>> round(r.pass_hat_k, 3)      # t1: C(2,2)/C(3,2)=1/3 ; t2: 1.0
        0.667
    """
    grouped = _group_trials(source)
    per_task: dict = {}
    skipped: list = []
    any_mixed = False

    for task_id, outcomes in grouped.items():
        n, c = len(outcomes), sum(outcomes)
        if n < k:
            skipped.append(task_id)
            continue
        if 0 < c < n:
            any_mixed = True
        per_task[task_id] = {
            "n": n,
            "c": c,
            "pass_at_k": pass_at_k(n=n, c=c, k=k),
            "pass_hat_k": pass_hat_k(n=n, c=c, k=k),
        }

    n_tasks = len(per_task)
    n_trials = sum(v["n"] for v in per_task.values())
    total_c = sum(v["c"] for v in per_task.values())
    # None, not 0.0, when nothing could be estimated at this k (see ReliabilityReport).
    mean_at = (
        sum(v["pass_at_k"] for v in per_task.values()) / n_tasks if n_tasks else None
    )
    mean_hat = (
        sum(v["pass_hat_k"] for v in per_task.values()) / n_tasks if n_tasks else None
    )

    # Dropping tasks from the headline in silence is a mis-estimate, not a nicety: a suite
    # where half the tasks have fewer than k trials would otherwise report pass^k over the
    # other half as if it were the whole suite.
    if skipped:
        warnings.warn(
            f"reliability: {len(skipped)} of {len(grouped)} task(s) have fewer than k={k} "
            f"trials and were EXCLUDED from pass@k/pass^k (e.g. {skipped[:3]}). Run at least "
            "k trials per task, or lower k -- the reported figures cover only the remaining "
            f"{n_tasks} task(s).",
            stacklevel=2,
        )

    per_slice: dict = {}
    if slices:
        buckets: dict = {}
        for task_id, v in per_task.items():
            label = slices.get(task_id)
            if label is not None:
                buckets.setdefault(label, []).append(v)
        for label, vs in buckets.items():
            per_slice[label] = {
                "n_tasks": len(vs),
                "pass_at_k": sum(v["pass_at_k"] for v in vs) / len(vs),
                "pass_hat_k": sum(v["pass_hat_k"] for v in vs) / len(vs),
                "success_rate": sum(v["c"] for v in vs) / sum(v["n"] for v in vs),
            }

    stochastic = any_mixed or k == 1
    if warn_deterministic and k > 1 and not any_mixed and n_tasks:
        warnings.warn(
            "reliability: no task showed mixed outcomes across trials -- the agent looks "
            f"deterministic, so pass^{k} degenerates to pass^1 and measures nothing new. "
            "Ensure real run-to-run stochasticity (temperature, seeds, a simulated user) "
            "or report pass^1 instead.",
            stacklevel=2,
        )

    # Bootstrap the pass^k CI over TASKS (the sampling unit), so the regression gate can
    # compare intervals rather than points on a few-hundred-task suite.
    hat_values = [v["pass_hat_k"] for v in per_task.values()]
    hat_ci = (
        bootstrap_ci(hat_values, lambda v: sum(v) / len(v))
        if hat_values
        else (0.0, 1.0)
    )

    return ReliabilityReport(
        k=k,
        n_tasks=n_tasks,
        n_trials=n_trials,
        n_success=total_c,
        success_rate=(total_c / n_trials) if n_trials else 0.0,
        success_ci=wilson_interval(total_c, n_trials),
        pass_at_k=mean_at,
        pass_hat_k=mean_hat,
        pass_hat_k_ci=hat_ci,
        stochastic=stochastic,
        per_task=per_task,
        per_slice=per_slice,
        cost=dict(cost) if cost else {},
        detail={"skipped": skipped} if skipped else {},
    )
