"""Offline smoke tests of the WRAP adapters against the REAL libraries (skipped when absent).

CI does not install ``ek[agents]`` / ``ek[agreement]``, so these skip there; run them locally
after installing the library to re-validate the adapters against its current API. Every judge
or model here is a local fake (Inspect's ``mockllm/model``, a scripted LangChain chat model, a
canned Ragas/DeepEval judge): **no network, no API key, no cost**.
"""

import importlib.util
import os
import typing
import warnings

import pytest

from ek.agents import Episode, Step, Trajectory


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(not _has("inspect_ai"), reason="inspect_ai not installed")
def test_run_inspect_with_mockllm_and_a_tool_call():
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.scorer import includes
    from inspect_ai.solver import generate, use_tools
    from inspect_ai.tool import tool

    from ek.agents import reliability, run_inspect

    @tool
    def add():
        async def execute(x: int, y: int):
            """Add two numbers.

            Args:
                x: First number.
                y: Second number.
            """
            return x + y

        return execute

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(model="mockllm/model", tool_name="add",
                                      tool_arguments={"x": 1, "y": 2}),
            ModelOutput.from_content(model="mockllm/model", content="The answer is 3"),
        ],
    )
    task = Task(dataset=[Sample(input="1+2?", target="3", id="a")],
                solver=[use_tools(add()), generate()], scorer=includes())
    (ep,) = run_inspect(task, model=model)
    assert ep.success is True and ep.task_id == "a"
    assert [(s.tool, s.args, s.observation) for s in ep.trajectory.steps] == [
        ("add", {"x": 1, "y": 2}, "3")]

    plain = Task(dataset=[Sample(input="hi", target="Default", id="s1"),
                          Sample(input="bye", target="nope", id="s2")],
                 solver=[generate()], scorer=includes())
    eps = run_inspect(plain, model="mockllm/model", epochs=3)
    assert len(eps) == 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert reliability(eps, k=3).pass_hat_k == 0.5


@pytest.mark.skipif(not _has("deepeval"), reason="deepeval not installed")
def test_deepeval_metrics_offline():
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    from deepeval.models.base_model import DeepEvalBaseLLM

    from ek.agents import DeepEvalMetric

    ep = Episode(task_id="t", output="done",
                 trajectory=Trajectory([Step("search", {"q": "x"}), Step("book", {})]))
    # tool_correctness needs no judge: works with no model and no OPENAI_API_KEY.
    s = DeepEvalMetric("tool_correctness")(ep, {"expected_tools": ["search", "pay"]})
    assert s.value == 0.5 and "recall" in s.detail["flavor"]

    class CannedJudge(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__(model="canned")

        def load_model(self):
            return self

        def generate(self, prompt, schema=None, **kw):
            return _fill(schema)

        async def a_generate(self, prompt, schema=None, **kw):
            return self.generate(prompt, schema=schema)

        def get_model_name(self):
            return "canned"

    for kind, kw in (("task_completion", {}), ("geval", {"criteria": "is it right"})):
        s = DeepEvalMetric(kind, model=CannedJudge(), async_mode=False, **kw)(ep, "done")
        assert 0.0 <= s.value <= 1.0


@pytest.mark.skipif(not _has("ragas"), reason="ragas not installed")
def test_ragas_collections_metrics_offline():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms import InstructorBaseRagasLLM

    from ek.agents import RagasSignal

    class CannedJudge(InstructorBaseRagasLLM):
        def generate(self, prompt, response_model):
            return _fill(response_model)

        async def agenerate(self, prompt, response_model):
            return self.generate(prompt, response_model)

    class UnitEmbedding(BaseRagasEmbedding):
        def embed_text(self, text, **kw):
            return [1.0, 0.0, 0.0]

        async def aembed_text(self, text, **kw):
            return self.embed_text(text)

    sample = {"question": "sky colour?", "answer": "blue", "contexts": ["the sky is blue"]}
    for name, kw in (("faithfulness", {}), ("answer_relevancy", {"embeddings": UnitEmbedding()}),
                     ("context_precision", {})):
        sig = RagasSignal(name=name, llm=CannedJudge(), **kw)
        assert 0.0 <= sig(sample) <= 1.0
        assert type(sig.metric).__module__.startswith("ragas.metrics.collections")


@pytest.mark.skipif(not (_has("uqlm") and _has("langchain_core")), reason="uqlm not installed")
def test_uqlm_consistency_offline():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from ek.agents import UqlmConsistencySignal

    class ScriptedChat(FakeListChatModel):
        temperature: float = 1.0  # uqlm reads llm.temperature

    agree = ScriptedChat(responses=["4"] * 12)
    split = ScriptedChat(responses=["4", "5", "four", "4", "6", "7"] * 2)
    assert UqlmConsistencySignal(llm=agree, scorer_name="exact_match")("2+2?") == 1.0
    assert UqlmConsistencySignal(llm=split, scorer_name="exact_match")("2+2?") < 1.0


def _fill(model):
    """A schema instance with every field filled (a canned, offline LLM-judge answer)."""
    from pydantic import BaseModel

    if model is None:
        return '{"score": 1, "reason": "ok"}'

    def value(t):
        origin = typing.get_origin(t)
        if isinstance(t, type) and issubclass(t, BaseModel):
            return t(**{n: value(f.annotation) for n, f in t.model_fields.items()})
        if origin is list:
            (arg,) = typing.get_args(t) or (str,)
            return [value(arg)]
        if origin is dict:
            return {}
        if origin is typing.Union:
            return value(typing.get_args(t)[0])
        if origin is typing.Literal:
            return typing.get_args(t)[0]
        return {str: "the sky is blue", bool: True, int: 1, float: 1.0}.get(t)

    return value(model)
