"""``ek.agents`` -- evaluating AI agents and assistants, in cost per successful task.

The agent instance of ``ek``, mirroring :mod:`ek.ocr`. Agent evaluation is the **same 2x2** the
framework already implements -- (reference availability) x (granularity) -- but the evaluated
object is an **episode** (tool calls + observations ending in a final state) and the objective is
**cost per successfully completed task**, not cost per token.

| corner | agent meaning | facade |
|---|---|---|
| reference-based, one item | grade an episode against a gold outcome/trajectory | :func:`ek.score` |
| reference-based, corpus | task-suite benchmark: ``pass^k`` + cost, per slice | :func:`run_suite` |
| reference-free, one item | judge / verifier / the cascade's confidence gate | :func:`ek.estimate_quality` |

What it reuses (rather than rebuilds): the Layer-A ``GraphGrammar`` becomes the **task/tool
grammar** (``FieldSpec.importance`` = the cost of a wrong argument, so a wrong argument to a
*destructive* tool is not one unit of error); ``FieldMetric``'s TP/FP/FN accounting powers
tool-call scoring; the ROVER agreement seam is self-consistency; the IAA helpers validate a
judge; and the whole ``signal -> calibrate -> validate -> decide`` cascade is the confidence gate.

Quickstart:
    >>> from ek.agents import TaskSpec, run_suite, as_agent
    >>> tasks = [TaskSpec("t1", input="hi", gold="HI"), TaskSpec("t2", input="yo", gold="YO")]
    >>> report = run_suite(as_agent(str.upper), tasks)
    >>> report.success_rate
    1.0

Research: ``misc/docs/ek_07`` (map) - ``ek_08`` (task success) - ``ek_09`` (judge/QE) -
``ek_10`` (trajectory) - ``ek_11`` (cost economics) - ``ek_12`` (libraries + integration).
"""

from __future__ import annotations

from .base import (
    DEFAULT_TASK_VALUE,
    DESTRUCTIVE_WEIGHT,
    Cost,
    Episode,
    RunProvenance,
    Step,
    TaskSpec,
    ToolSpec,
    Trajectory,
    as_call,
    is_episode,
    suite_grammar,
    tool_grammar,
)
from .bridge import (
    as_agent,
    cost_from_usage,
    episode_from_messages,
    from_deepeval_test_case,
    from_inspect_sample,
    trajectory_from_messages,
)
from .cost import (
    ModelPrice,
    UnknownModelPrice,
    cost_of_pass,
    cost_report,
    dollars,
    episode_dollars,
    load_prices,
    per_million,
    price_of,
)
from .harness import (
    AgentGateResult,
    agent_regression_gate,
    load_agent_baseline,
    run_suite,
    save_agent_baseline,
)
from .judge import (
    JudgeMetric,
    JudgeSignal,
    judge_validation,
    pairwise_judge,
)

# Importing .metrics registers the agent metrics + checkers by name in the registry.
from .metrics import (
    Checker,
    CostPerSuccessMetric,
    TaskSuccessMetric,
    final_state_match,
    output_match,
    recorded_success,
)
from .reliability import (
    ReliabilityReport,
    bootstrap_ci,
    difference_upper_bound,
    newcombe_difference,
    pass_at_k,
    pass_hat_k,
    reliability,
    wilson_interval,
)
from .toolcalls import ToolCallMetric, match_calls
from .trajectory import SCHEMES, TrajectoryMetric

__all__ = [
    # Layer A -- the task/tool grammar (cost SSOT)
    "TaskSpec",
    "ToolSpec",
    "tool_grammar",
    "suite_grammar",
    "DEFAULT_TASK_VALUE",
    "DESTRUCTIVE_WEIGHT",
    # Layer B -- the episode
    "Episode",
    "Trajectory",
    "Step",
    "Cost",
    "RunProvenance",
    "as_call",
    "is_episode",
    # reference-based metrics
    "TaskSuccessMetric",
    "CostPerSuccessMetric",
    "ToolCallMetric",
    "TrajectoryMetric",
    "JudgeMetric",
    "SCHEMES",
    "match_calls",
    # checkers (the injected success oracle)
    "Checker",
    "output_match",
    "final_state_match",
    "recorded_success",
    # reliability (harness-owned; NOT Metrics -- they are cross-task)
    "pass_at_k",
    "pass_hat_k",
    "reliability",
    "ReliabilityReport",
    "wilson_interval",
    "bootstrap_ci",
    "newcombe_difference",
    "difference_upper_bound",
    # cost (the other half of the objective)
    "ModelPrice",
    "per_million",
    "dollars",
    "episode_dollars",
    "cost_of_pass",
    "cost_report",
    "load_prices",
    "price_of",
    "UnknownModelPrice",
    # reference-free QE
    "JudgeSignal",
    "judge_validation",
    "pairwise_judge",
    # harness
    "run_suite",
    "save_agent_baseline",
    "load_agent_baseline",
    "agent_regression_gate",
    "AgentGateResult",
    # bridge (duck-typed; no external harness import required)
    "as_agent",
    "trajectory_from_messages",
    "episode_from_messages",
    "cost_from_usage",
    "from_inspect_sample",
    "from_deepeval_test_case",
]
