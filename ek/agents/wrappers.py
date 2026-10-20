"""Optional WRAP adapters behind ``ek[agents]``: uqlm, Ragas and DeepEval as ek strategies.

``ek.agents`` scores episodes in pure python; this module is the batteries-included layer for
teams who already run these libraries. Every adapter imports its backend **lazily** (at call
time, guarded by :func:`~ek.registry.requires_extra`), so ``import ek`` stays light, and each
takes an injectable backend object so it can be tested -- and used -- without network access.

- :class:`UqlmConsistencySignal` -- black-box sample-and-vote self-consistency from ``uqlm``
  (a ``[0, 1]`` confidence), a :class:`~ek.base.Signal` (expensive: ``cost_tier=3``).
- :class:`RagasSignal` -- reference-free Ragas metrics (faithfulness, answer relevancy,
  context precision) as a :class:`~ek.base.Signal` over a RAG sample.
- :class:`DeepEvalMetric` -- DeepEval's ``ToolCorrectnessMetric`` / ``TaskCompletionMetric`` /
  ``GEval`` as ek ``metrics``. Confident-AI telemetry is **disabled** before the import.
  Note ``ToolCorrectnessMetric`` divides by ``len(expected_tools)`` -- it is
  *recall-flavored*, not F1 (compare ``ek``'s own cost-aware ``tool_call``).

Example (a fake backend, no extra needed):
    >>> class FakeUQ:
    ...     def score(self, prompt):
    ...         return 0.75
    >>> UqlmConsistencySignal(scorer=FakeUQ())("What is 2+2?")
    0.75
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping
from typing import Any, Callable, Optional

from ..base import GraphGrammar, Score
from ..registry import register, requires_extra

#: Env var that opts out of DeepEval's Confident-AI telemetry (set before importing it).
DEEPEVAL_TELEMETRY_ENV = "DEEPEVAL_TELEMETRY_OPT_OUT"

#: Sample count for uqlm's black-box scorers (a keyword default, not a magic number).
DEFAULT_NUM_RESPONSES = 5

#: uqlm black-box scorer used when none is named.
DEFAULT_UQLM_SCORER = "noncontradiction"

#: Ragas metric used when none is named.
DEFAULT_RAGAS_METRIC = "faithfulness"


def _run(value: Any) -> Any:
    """Resolve ``value`` if it is awaitable (many of these backends are async-only)."""
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(value))
    raise RuntimeError(
        "Called from inside a running event loop; await the backend directly or use a "
        "thread. (These adapters are synchronous Signals.)"
    )


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


# ---------------------------------------------------------------------------
# uqlm: black-box self-consistency
# ---------------------------------------------------------------------------


class UqlmConsistencySignal:
    """Self-consistency confidence in ``[0, 1]`` from ``uqlm``'s black-box scorers.

    Samples ``num_responses`` generations for the prompt and measures how much they agree
    (non-contradiction, exact match, semantic negentropy, ...). Raw and **uncalibrated**:
    calibrate before gating (Hard Rule 1).

    Args:
        scorer: A backend with ``score(prompt) -> float`` (tests / custom sampling), or
            ``None`` to build uqlm's ``BlackBoxUQ`` from ``llm``.
        llm: A LangChain chat model, required when ``scorer`` is ``None``.
        scorer_name: The uqlm scorer column to read (default ``"noncontradiction"``).
        num_responses: Samples per prompt.
        cost_tier: ``3`` -- it pays for ``num_responses`` extra generations.

    Example:
        >>> class Fake:
        ...     def score(self, prompt): return 2.0   # out-of-range is clamped
        >>> UqlmConsistencySignal(scorer=Fake())("q")
        1.0
    """

    def __init__(
        self,
        scorer: Any = None,
        *,
        llm: Any = None,
        scorer_name: str = DEFAULT_UQLM_SCORER,
        num_responses: int = DEFAULT_NUM_RESPONSES,
        cost_tier: int = 3,
    ):
        self.scorer = scorer
        self.llm = llm
        self.scorer_name = scorer_name
        self.num_responses = num_responses
        self.cost_tier = cost_tier

    @requires_extra("agents", packages=["uqlm"])
    def _build_backend(self) -> Any:
        from uqlm import BlackBoxUQ

        if self.llm is None:
            raise ValueError("UqlmConsistencySignal needs `llm=` (or an injected `scorer=`).")
        return BlackBoxUQ(llm=self.llm, scorers=[self.scorer_name])

    def _score_with_uqlm(self, prompt: str) -> float:
        uq = self._build_backend()
        result = _run(uq.generate_and_score(prompts=[prompt], num_responses=self.num_responses))
        frame = result.to_df()
        return float(frame[self.scorer_name].iloc[0])

    def __call__(self, extractor_output: Any) -> float:
        prompt = (
            extractor_output
            if isinstance(extractor_output, str)
            else str(getattr(extractor_output, "prompt", extractor_output))
        )
        if self.scorer is not None:
            return _clamp01(_run(self.scorer.score(prompt)))
        return _clamp01(self._score_with_uqlm(prompt))


# ---------------------------------------------------------------------------
# Ragas: reference-free RAG signals
# ---------------------------------------------------------------------------

#: ek's neutral key -> Ragas ``SingleTurnSample`` field, for the sample a RagasSignal reads.
_RAGAS_SAMPLE_FIELDS = {
    "user_input": ("user_input", "question", "query", "prompt"),
    "response": ("response", "answer", "output"),
    "retrieved_contexts": ("retrieved_contexts", "contexts", "context"),
}

#: ek's metric name -> Ragas class name.
RAGAS_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "ResponseRelevancy",
    "context_precision": "LLMContextPrecisionWithoutReference",
}


def _pick(sample: Any, aliases: tuple) -> Any:
    for key in aliases:
        value = sample.get(key) if isinstance(sample, Mapping) else getattr(sample, key, None)
        if value is not None:
            return value
    return None


class RagasSignal:
    """A reference-free Ragas metric as a :class:`~ek.base.Signal` over a RAG sample.

    The sample is a mapping/object with ``user_input`` (or ``question``), ``response`` (or
    ``answer``) and ``retrieved_contexts`` (or ``contexts``). The score is in ``[0, 1]``.
    Ragas metrics call an LLM, so pass a configured ``metric`` (or ``llm=`` to build one).

    Args:
        metric: A Ragas metric object exposing ``single_turn_ascore(sample)`` (injected), or
            ``None`` to build ``name`` from ``llm``.
        name: ``faithfulness`` | ``answer_relevancy`` | ``context_precision``.
        llm: A ``ragas`` LLM wrapper, used when ``metric`` is ``None``.
        cost_tier: ``3`` -- an LLM call per sample.

    Example:
        >>> class Fake:
        ...     async def single_turn_ascore(self, sample): return 0.5
        >>> RagasSignal(Fake())({"question": "q", "answer": "a", "contexts": ["c"]})
        0.5
    """

    def __init__(
        self,
        metric: Any = None,
        *,
        name: str = DEFAULT_RAGAS_METRIC,
        llm: Any = None,
        cost_tier: int = 3,
    ):
        if metric is None and name not in RAGAS_METRICS:
            raise ValueError(f"name must be one of {sorted(RAGAS_METRICS)}, got {name!r}")
        self.metric = metric
        self.name = name
        self.llm = llm
        self.cost_tier = cost_tier

    @requires_extra("agents", packages=["ragas"])
    def _build_metric(self) -> Any:
        import ragas.metrics as rm

        return getattr(rm, RAGAS_METRICS[self.name])(llm=self.llm)

    @requires_extra("agents", packages=["ragas"])
    def _make_sample(self, fields: dict) -> Any:
        from ragas import SingleTurnSample

        return SingleTurnSample(**fields)

    def __call__(self, extractor_output: Any) -> float:
        fields = {
            key: _pick(extractor_output, aliases)
            for key, aliases in _RAGAS_SAMPLE_FIELDS.items()
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if self.metric is None:
            self.metric = self._build_metric()
        sample = fields if _is_fake(self.metric) else self._make_sample(fields)
        return _clamp01(_run(self.metric.single_turn_ascore(sample)))


def _is_fake(metric: Any) -> bool:
    """True for an injected, non-Ragas metric (its module is not under ``ragas``)."""
    return not type(metric).__module__.startswith("ragas")


# ---------------------------------------------------------------------------
# DeepEval metrics as ek metrics
# ---------------------------------------------------------------------------

#: ek name -> (DeepEval class name, needs an LLM judge).
DEEPEVAL_KINDS = {
    "tool_correctness": "ToolCorrectnessMetric",
    "task_completion": "TaskCompletionMetric",
    "geval": "GEval",
}


def disable_deepeval_telemetry() -> None:
    """Opt out of Confident-AI telemetry (idempotent; never overrides an explicit choice)."""
    os.environ.setdefault(DEEPEVAL_TELEMETRY_ENV, "YES")


class DeepEvalMetric:
    """A DeepEval metric as an ek ``Metric`` over an :class:`~ek.agents.base.Episode`.

    ``score(episode, gold)`` builds a DeepEval ``LLMTestCase`` from the episode's output and
    tool calls (``gold`` is the expected output string, or a mapping with ``expected_output``
    / ``expected_tools``) and returns DeepEval's score as an ek ``Score``.

    ``tool_correctness`` is **recall-flavored** (divides by ``len(expected_tools)``), not F1:
    the label says so, and ``detail['flavor']`` records it.

    Args:
        kind: ``tool_correctness`` | ``task_completion`` | ``geval``.
        metric: An injected DeepEval-shaped metric (``.measure(case)``, ``.score``); ``None``
            builds the real one lazily (telemetry disabled first).
        **metric_kwargs: Passed to the DeepEval constructor (e.g. ``criteria=`` for GEval).
    """

    def __init__(self, kind: str = "tool_correctness", *, metric: Any = None, **metric_kwargs):
        if kind not in DEEPEVAL_KINDS:
            raise ValueError(f"kind must be one of {sorted(DEEPEVAL_KINDS)}, got {kind!r}")
        self.kind = kind
        self.name = f"deepeval_{kind}"
        self.metric = metric
        self.metric_kwargs = metric_kwargs

    @requires_extra("agents", packages=["deepeval"])
    def _build(self) -> Any:
        disable_deepeval_telemetry()
        import deepeval.metrics as dm

        return getattr(dm, DEEPEVAL_KINDS[self.kind])(**self.metric_kwargs)

    @requires_extra("agents", packages=["deepeval"])
    def _real_case(self, **fields: Any) -> Any:
        disable_deepeval_telemetry()
        from deepeval.test_case import LLMTestCase, ToolCall

        for key in ("tools_called", "expected_tools"):
            if key in fields:
                fields[key] = [ToolCall(name=n) for n in fields[key]]
        return LLMTestCase(**fields)

    def _case_fields(self, episode: Any, gold: Any) -> dict:
        calls = [getattr(s, "tool", s) for s in getattr(getattr(episode, "trajectory", None), "steps", ())]
        fields: dict = {
            "input": str(getattr(episode, "task_id", "")),
            "actual_output": str(getattr(episode, "output", "") or ""),
            "tools_called": calls,
        }
        if isinstance(gold, Mapping):
            if "expected_output" in gold:
                fields["expected_output"] = gold["expected_output"]
            if "expected_tools" in gold:
                fields["expected_tools"] = [getattr(t, "tool", t) for t in gold["expected_tools"]]
        elif gold is not None:
            fields["expected_output"] = str(gold)
        return fields

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        if self.metric is None:
            self.metric = self._build()
        fields = self._case_fields(pred, gold)
        case = fields if _is_fake(self.metric) else self._real_case(**fields)
        value = _clamp01(self._measure(case))
        detail: dict = {"higher_is_better": True, "backend": "deepeval"}
        if self.kind == "tool_correctness":
            detail["flavor"] = "recall (divides by len(expected_tools)), not F1"
        reason = getattr(self.metric, "reason", None)
        if reason:
            detail["reason"] = reason
        return Score(value=value, metric=self.name, detail=detail)

    def _measure(self, case: Any) -> float:
        returned = _run(self.metric.measure(case))
        score = getattr(self.metric, "score", None)
        return float(score if score is not None else returned)

    def aggregate(self, scores: Any) -> float:
        """Mean of the per-episode scores (DeepEval scores are per-case, not count-based)."""
        scores = list(scores)
        return sum(s.value for s in scores) / len(scores) if scores else 0.0


def _register_defaults() -> None:
    register("signals", "uqlm_consistency", UqlmConsistencySignal)
    register("signals", "ragas_faithfulness", RagasSignal)
    register("metrics", "deepeval_tool_correctness", DeepEvalMetric("tool_correctness"))
    register("metrics", "deepeval_task_completion", DeepEvalMetric("task_completion"))


_register_defaults()
