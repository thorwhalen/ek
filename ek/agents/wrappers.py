"""Optional WRAP adapters: Inspect AI, uqlm, Ragas and DeepEval as ek strategies.

``ek.agents`` scores episodes in pure python; this module is the batteries-included layer for
teams who already run these libraries. Every adapter imports its backend **lazily** (at call
time, guarded by :func:`~ek.registry.requires_extra`), so ``import ek`` stays light, and each
takes an injectable backend object so it can be tested -- and used -- without network access.
The dependency direction stays ``ek -> backend``, never the reverse. Inspect AI, Ragas and
DeepEval come with ``ek[agents]``; uqlm (heavy: torch/transformers) with ``ek[agreement]``.

- :func:`run_inspect` -- run an Inspect AI eval and get graded :class:`~ek.agents.Episode` s
  back (Inspect's ``epochs`` are ek's trials, so the result feeds ``reliability`` directly).
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
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Optional

from ..base import GraphGrammar, Score
from ..registry import register, requires_extra
from .bridge import DEFAULT_SUCCESS_THRESHOLD, episodes_from_inspect_log

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
# Inspect AI: run an eval, get Episodes
# ---------------------------------------------------------------------------


@requires_extra("agents", packages=["inspect_ai"])
def run_inspect(
    tasks: Any,
    *,
    model: Any = None,
    epochs: Optional[int] = None,
    scorer: Optional[str] = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    log_dir: Optional[str] = None,
    **eval_kwargs: Any,
) -> list:
    """Run an Inspect AI eval and return its samples as graded :class:`~ek.agents.Episode` s.

    A thin convenience over ``inspect_ai.eval`` + :func:`~ek.agents.episodes_from_inspect_log`:
    one episode per sample per epoch, graded by the sample's score (``scorer``, else the first)
    reaching ``success_threshold``. Run with ``epochs=k`` and the result is a k-trial run --
    hand it to :func:`~ek.agents.reliability` for ``pass^k``, or to
    :func:`~ek.agents.cost_report` for cost per successful task.

    Args:
        tasks: Anything ``inspect_ai.eval`` accepts (a ``Task``, a task function, a registry
            name, a list of them).
        model: The model under test (e.g. ``"openai/gpt-4o"``); ``None`` lets Inspect use
            ``INSPECT_EVAL_MODEL``. ``"mockllm/model"`` runs offline at no cost.
        epochs: Trials per sample (Inspect's ``epochs``); ``None`` keeps the task's own setting.
        scorer: Which scorer's value grades success (default: each sample's first score).
        success_threshold: A sample succeeds when its score value is ``>=`` this.
        log_dir: Where Inspect writes its ``.eval`` logs. ``None`` uses a temporary directory
            that is removed once the episodes are read -- pass a path to keep the logs.
        **eval_kwargs: Passed to ``inspect_ai.eval`` (``limit=``, ``solver=``, ...). ``display``
            defaults to ``"none"``.

    Raises:
        RuntimeError: if any eval did not finish with status ``success`` (a partial run is not
            a k-trial run, and grading it as one would understate failures).

    Note: ``inspect_ai.eval`` is synchronous and refuses to run inside an event loop; from async
    code, call ``inspect_ai.eval_async`` and pass each log to ``episodes_from_inspect_log``.
    """
    import tempfile

    from inspect_ai import eval as inspect_eval

    eval_kwargs.setdefault("display", "none")
    if epochs is not None:
        eval_kwargs["epochs"] = epochs

    def collect(directory: str) -> list:
        logs = inspect_eval(tasks, model=model, log_dir=directory, **eval_kwargs)
        failed = [log for log in logs if getattr(log, "status", None) != "success"]
        if failed:
            errors = [str(getattr(log, "error", None) or log.status) for log in failed]
            raise RuntimeError(
                f"run_inspect: {len(failed)} of {len(logs)} Inspect eval(s) did not succeed: "
                f"{errors[:3]}"
            )
        # Several logs (a list of tasks) pool their samples; prefix the task name so sample ids
        # that repeat across tasks are not counted as extra trials of one task.
        names = [str(getattr(getattr(log, "eval", None), "task", "") or "") for log in logs]
        if len(logs) > 1 and len(set(names)) < len(names):
            names = [f"{name}#{i}" for i, name in enumerate(names)]
        # Read the samples while the log files still exist (Inspect may load them lazily).
        episodes = [
            episode
            for log, name in zip(logs, names)
            for episode in episodes_from_inspect_log(
                log,
                scorer=scorer,
                success_threshold=success_threshold,
                task_prefix=name if len(logs) > 1 else "",
            )
        ]
        ungraded = sum(e.success is None for e in episodes)
        if not episodes or ungraded:
            warnings.warn(
                f"run_inspect: {len(episodes)} episode(s), {ungraded} ungraded (no scalar "
                "score). Check log_samples / fail_on_error / the scorer before trusting a "
                "success rate.",
                stacklevel=3,
            )
        return episodes

    if log_dir is not None:
        return collect(log_dir)
    with tempfile.TemporaryDirectory(prefix="ek-inspect-") as directory:
        return collect(directory)


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
        llm: A LangChain chat model, required when ``scorer`` is ``None``. uqlm reads its
            ``temperature`` attribute, so it must have one.
        scorer_name: The uqlm scorer column to read (default ``"noncontradiction"``, which
            downloads an NLI model on first use; ``"exact_match"`` needs none).
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

    @requires_extra("agreement", packages=["uqlm"])
    def _build_backend(self) -> Any:
        from uqlm import BlackBoxUQ

        if self.llm is None:
            raise ValueError(
                "UqlmConsistencySignal needs `llm=` (or an injected `scorer=`)."
            )
        return BlackBoxUQ(llm=self.llm, scorers=[self.scorer_name])

    def _score_with_uqlm(self, prompt: str) -> float:
        uq = self._build_backend()
        kwargs: dict = {"prompts": [prompt], "num_responses": self.num_responses}
        if "show_progress_bars" in inspect.signature(uq.generate_and_score).parameters:
            kwargs["show_progress_bars"] = False  # a Signal is called per item; keep it quiet
        result = _run(uq.generate_and_score(**kwargs))
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

#: ek's metric name -> Ragas class name in the legacy ``ragas.metrics`` API (deprecated since
#: Ragas 0.4) -- metrics take a ``BaseRagasLLM`` and expose ``single_turn_ascore(sample)``.
RAGAS_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "ResponseRelevancy",
    "context_precision": "LLMContextPrecisionWithoutReference",
}

#: ek's metric name -> Ragas class name in ``ragas.metrics.collections`` (Ragas >= 0.4, the API
#: that survives 1.0) -- metrics take an ``InstructorBaseRagasLLM`` (``llm_factory``) and expose
#: ``ascore(**fields) -> MetricResult``. ``answer_relevancy`` also needs ``embeddings=``.
RAGAS_COLLECTION_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_precision": "ContextPrecisionWithoutReference",
}


def _pick(sample: Any, aliases: tuple) -> Any:
    for key in aliases:
        value = (
            sample.get(key)
            if isinstance(sample, Mapping)
            else getattr(sample, key, None)
        )
        if value is not None:
            return value
    return None


class RagasSignal:
    """A reference-free Ragas metric as a :class:`~ek.base.Signal` over a RAG sample.

    The sample is a mapping/object with ``user_input`` (or ``question``), ``response`` (or
    ``answer``) and ``retrieved_contexts`` (or ``contexts``). The score is in ``[0, 1]``.
    Ragas metrics call an LLM, so pass a configured ``metric`` (or ``llm=`` to build one).

    Both Ragas metric APIs are accepted: the collections API (``ascore(**fields)`` returning a
    ``MetricResult``; ``ragas.metrics.collections``, the one that survives Ragas 1.0) and the
    legacy one (``single_turn_ascore(SingleTurnSample)``; deprecated since Ragas 0.4).

    Args:
        metric: A Ragas metric object (either API), or ``None`` to build ``name`` from ``llm``.
        name: ``faithfulness`` | ``answer_relevancy`` | ``context_precision``.
        llm: The judge, used when ``metric`` is ``None``. An ``InstructorBaseRagasLLM`` (from
            ``ragas.llms.llm_factory``) builds the collections metric; any other Ragas LLM
            wrapper (e.g. ``LangchainLLMWrapper``) builds the legacy one.
        cost_tier: ``3`` -- an LLM call per sample.
        **metric_kwargs: Passed to the Ragas constructor (e.g. ``embeddings=`` for
            ``answer_relevancy``).

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
        **metric_kwargs: Any,
    ):
        if metric is None and name not in RAGAS_METRICS:
            raise ValueError(
                f"name must be one of {sorted(RAGAS_METRICS)}, got {name!r}"
            )
        self.metric = metric
        self.name = name
        self.llm = llm
        self.cost_tier = cost_tier
        self.metric_kwargs = metric_kwargs

    @requires_extra("agents", packages=["ragas"])
    def _build_metric(self) -> Any:
        try:  # Ragas >= 0.4: the collections API, built from an Instructor-style LLM
            import ragas.metrics.collections as collections
            from ragas.llms import InstructorBaseRagasLLM
        except ImportError:
            collections = InstructorBaseRagasLLM = None
        if collections is not None and isinstance(self.llm, InstructorBaseRagasLLM):
            cls = getattr(collections, RAGAS_COLLECTION_METRICS[self.name])
            return cls(llm=self.llm, **self.metric_kwargs)
        import ragas.metrics as rm

        cls = getattr(rm, RAGAS_METRICS[self.name])
        return cls(llm=self.llm, **self.metric_kwargs)

    def _score(self, fields: dict) -> Any:
        """Score with whichever API the metric speaks (legacy sample, or collections fields)."""
        metric = self.metric
        if hasattr(metric, "single_turn_ascore"):
            real = _is_backend(metric, "ragas")
            sample = self._make_sample(fields) if real else fields
            return _run(metric.single_turn_ascore(sample))
        params = inspect.signature(metric.ascore).parameters
        takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
        kwargs = {k: v for k, v in fields.items() if takes_any or k in params}
        result = _run(metric.ascore(**kwargs))
        return getattr(result, "value", result)  # a MetricResult carries .value

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
        return _clamp01(self._score(fields))


def _is_backend(metric: Any, package: str) -> bool:
    """True when ``metric`` is a real ``package`` object (its class lives under ``package``).

    An injected fake (tests, a custom backend) gets the plain-dict sample/case instead of the
    library's own type. Per-package on purpose: a single "is it Ragas?" check once routed real
    DeepEval metrics down the fake path, handing them a ``dict`` instead of an ``LLMTestCase``.
    """
    module = type(metric).__module__
    return module == package or module.startswith(package + ".")


# ---------------------------------------------------------------------------
# DeepEval metrics as ek metrics
# ---------------------------------------------------------------------------

#: ek name -> (DeepEval class name, needs an LLM judge).
DEEPEVAL_KINDS = {
    "tool_correctness": "ToolCorrectnessMetric",
    "task_completion": "TaskCompletionMetric",
    "geval": "GEval",
}


#: Test-case fields a GEval judge reads when ``evaluation_params`` is not given. DeepEval >= 4
#: refuses a plain ``LLMTestCase`` without them (it would otherwise evaluate a *trace*).
DEFAULT_GEVAL_PARAMS = ("input", "actual_output")


def _geval_kwargs(kwargs: dict, test_case_module: Any) -> dict:
    """Fill GEval's ``name`` / ``evaluation_params`` defaults; accept params as plain strings.

    The params enum was ``LLMTestCaseParams`` before DeepEval 4 and is ``SingleTurnParams`` since;
    either is resolved here so callers can write ``evaluation_params=["input", "actual_output"]``.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("name", "ek_geval")
    enum = getattr(test_case_module, "SingleTurnParams", None) or getattr(
        test_case_module, "LLMTestCaseParams"
    )
    params = kwargs.get("evaluation_params") or DEFAULT_GEVAL_PARAMS
    kwargs["evaluation_params"] = [enum(p) if isinstance(p, str) else p for p in params]
    return kwargs


def _no_judge_model() -> Any:
    """A ``DeepEvalBaseLLM`` that raises if called (for metrics that need no LLM judge)."""
    from deepeval.models.base_model import DeepEvalBaseLLM

    class NoJudge(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__(model="ek-no-judge")

        def load_model(self):
            return self

        def generate(self, *args, **kwargs):
            raise RuntimeError(
                "this DeepEval metric asked for an LLM judge; pass `model=` (a model name or a "
                "DeepEvalBaseLLM) to DeepEvalMetric."
            )

        async def a_generate(self, *args, **kwargs):
            return self.generate(*args, **kwargs)

        def get_model_name(self):
            return "ek-no-judge"

    return NoJudge()


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
        **metric_kwargs: Passed to the DeepEval constructor (e.g. ``criteria=`` for GEval, and
            ``model=`` -- a model name or a ``DeepEvalBaseLLM`` -- for any of them: DeepEval >= 4
            builds its default OpenAI judge at construction, so without ``model=`` it needs
            ``OPENAI_API_KEY`` even for ``tool_correctness``). For GEval, ``name`` defaults to
            ``"ek_geval"`` and ``evaluation_params`` to ``DEFAULT_GEVAL_PARAMS`` (strings are
            accepted).
    """

    def __init__(
        self, kind: str = "tool_correctness", *, metric: Any = None, **metric_kwargs
    ):
        if kind not in DEEPEVAL_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(DEEPEVAL_KINDS)}, got {kind!r}"
            )
        self.kind = kind
        self.name = f"deepeval_{kind}"
        self.metric = metric
        self.metric_kwargs = metric_kwargs

    @requires_extra("agents", packages=["deepeval"])
    def _build(self) -> Any:
        disable_deepeval_telemetry()
        import deepeval.metrics as dm
        import deepeval.test_case as tc

        kwargs = self.metric_kwargs
        if self.kind == "geval":
            kwargs = _geval_kwargs(kwargs, tc)
        elif (
            self.kind == "tool_correctness"
            and "model" not in kwargs
            and not kwargs.get("available_tools")
        ):
            # Without available_tools this metric is deterministic -- it never asks a judge --
            # yet DeepEval >= 4 builds an OpenAI client at construction and demands a key.
            # A judge that refuses to be called makes it work offline, and fails loudly if a
            # future DeepEval starts calling it after all.
            kwargs = {**kwargs, "model": _no_judge_model()}
        return getattr(dm, DEEPEVAL_KINDS[self.kind])(**kwargs)

    @requires_extra("agents", packages=["deepeval"])
    def _real_case(self, fields: dict, args: dict) -> Any:
        """Build a real ``LLMTestCase``; tool calls carry their arguments as ``input_parameters``."""
        disable_deepeval_telemetry()
        from deepeval.test_case import LLMTestCase, ToolCall

        fields = dict(fields)
        for key in ("tools_called", "expected_tools"):
            if key in fields:
                fields[key] = [
                    ToolCall(name=name, input_parameters=params or None)
                    for name, params in zip(fields[key], args.get(key, ()))
                ]
        return LLMTestCase(**fields)

    @staticmethod
    def _calls(items: Any) -> list:
        """``[(tool_name, args), ...]`` from Steps, ``(name, args)`` pairs or bare names."""
        out = []
        for item in items or ():
            if isinstance(item, str):
                out.append((item, {}))
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], (Mapping, type(None)))
            ):
                out.append((str(item[0]), dict(item[1] or {})))
            else:
                args = getattr(item, "args", None)
                name = getattr(item, "tool", item)
                out.append((str(name), dict(args) if isinstance(args, Mapping) else {}))
        return out

    def _case_args(self, episode: Any, gold: Any) -> dict:
        """The tool-call arguments that go with :meth:`_case_fields`' tool names (same order)."""
        steps = getattr(getattr(episode, "trajectory", None), "steps", ())
        args = {"tools_called": [a for _, a in self._calls(steps)]}
        if isinstance(gold, Mapping) and "expected_tools" in gold:
            args["expected_tools"] = [a for _, a in self._calls(gold["expected_tools"])]
        return args

    def _case_fields(self, episode: Any, gold: Any) -> dict:
        steps = getattr(getattr(episode, "trajectory", None), "steps", ())
        calls = [name for name, _ in self._calls(steps)]
        fields: dict = {
            "input": str(getattr(episode, "task_id", "")),
            "actual_output": str(getattr(episode, "output", "") or ""),
            "tools_called": calls,
        }
        if isinstance(gold, Mapping):
            if "expected_output" in gold:
                fields["expected_output"] = gold["expected_output"]
            if "expected_tools" in gold:
                fields["expected_tools"] = [
                    name for name, _ in self._calls(gold["expected_tools"])
                ]
        elif gold is not None:
            fields["expected_output"] = str(gold)
        return fields

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        if self.metric is None:
            self.metric = self._build()
        fields = self._case_fields(pred, gold)
        case = (
            self._real_case(fields, self._case_args(pred, gold))
            if _is_backend(self.metric, "deepeval")
            else fields
        )
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
