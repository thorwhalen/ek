"""Tests for the Inspect AI runner and the WRAP-adapter API-drift fixes (issue #37).

The real libraries are not installed in CI, so these use log-shaped stand-ins and fake modules.
They pin the behaviours that were validated once against the real libraries (inspect-ai 0.3.282
with ``mockllm/model``, deepeval 4.3.1, ragas 0.4.3, uqlm 0.6.7 -- all offline):

- an Inspect log becomes one graded Episode per sample per epoch, ready for ``reliability``;
- a real DeepEval metric gets an ``LLMTestCase``, not the fake-path ``dict`` (the old
  ``_is_fake`` check only recognised Ragas, so every real DeepEval metric got a dict);
- Ragas' collections API (``ascore(**fields) -> MetricResult``) is scored as well as the legacy
  ``single_turn_ascore``;
- GEval gets the ``evaluation_params`` DeepEval >= 4 requires, with strings accepted.
"""

import enum
import importlib.util
import sys
import types
import warnings
from types import SimpleNamespace as NS

import pytest

from ek.agents import (
    DeepEvalMetric,
    RagasSignal,
    episodes_from_inspect_log,
    inspect_score_value,
    reliability,
    run_inspect,
)
from ek.agents.base import Episode, Step, Trajectory
from ek.agents.wrappers import _geval_kwargs, _is_backend
from ek.registry import MissingExtraError


def _sample(sid, epoch, value, *, tool=None):
    messages = [{"role": "user", "content": "q"}]
    if tool:
        call = NS(function=tool, arguments={"x": 1})  # Inspect's flat ToolCall shape
        messages.append(NS(role="assistant", content="", tool_calls=[call]))
        messages.append({"role": "tool", "content": "2"})
    messages.append({"role": "assistant", "content": "done"})
    return NS(
        id=sid,
        epoch=epoch,
        messages=messages,
        output=NS(completion="done", usage={"input_tokens": 3, "output_tokens": 4}),
        scores={"match": NS(value=value), "other": NS(value=0.0)},
    )


def _log(samples, *, status="success"):
    return NS(samples=samples, eval=NS(model="mockllm/model", task_version=0), status=status)


# ---------------------------------------------------------------------------
# Inspect: log -> Episodes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("C", 1.0), ("I", 0.0), ("P", 0.5), ("N", 0.0), ("yes", 1.0), ("0.75", 0.75),
     (True, 1.0), (0, 0.0), (0.3, 0.3), ({"a": 1}, None), ("maybe", None), (None, None)],
)
def test_inspect_score_value(value, expected):
    assert inspect_score_value(value) == expected


def test_episodes_from_inspect_log_grades_each_epoch_and_feeds_reliability():
    log = _log([_sample("a", 1, "C"), _sample("a", 2, "C"), _sample("b", 1, "C"),
                _sample("b", 2, "I")])
    eps = list(episodes_from_inspect_log(log))
    assert [(e.task_id, e.meta["epoch"], e.success) for e in eps] == [
        ("a", 1, True), ("a", 2, True), ("b", 1, True), ("b", 2, False)]
    assert eps[0].run.model == "mockllm/model" and eps[0].run.suite_version == "0"
    assert eps[0].cost.input_tokens == 3 and eps[0].output == "done"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = reliability(eps, k=2)
    assert r.n_tasks == 2 and r.pass_hat_k == 0.5  # a: 1.0, b: 0.0


def test_episodes_from_inspect_log_scorer_threshold_and_ungraded():
    log = _log([_sample("a", 1, "P")])
    assert next(episodes_from_inspect_log(log)).success is False  # P = 0.5 < 1.0
    assert next(episodes_from_inspect_log(log, success_threshold=0.5)).success is True
    # Named scorer: "other" is 0.0 for every sample.
    assert next(episodes_from_inspect_log(log, scorer="other")).success is False
    # A non-scalar score stays UNGRADED (None), never a silent failure.
    assert next(episodes_from_inspect_log(_log([_sample("a", 1, {"acc": 1})]))).success is None
    assert next(episodes_from_inspect_log(_log([_sample("a", 1, "nan")]))).success is None
    # A scorer NO sample has is a typo, not "ungraded": fail loudly.
    with pytest.raises(ValueError, match="no sample has a score named 'nope'"):
        list(episodes_from_inspect_log(_log([_sample("a", 1, "C")]), scorer="nope"))


def test_episodes_from_inspect_log_task_prefix():
    ep = next(episodes_from_inspect_log(_log([_sample("1", 1, "C")]), task_prefix="math"))
    assert ep.task_id == "math/1"


def test_episodes_from_inspect_log_reads_inspect_tool_calls():
    ep = next(episodes_from_inspect_log(_log([_sample("a", 1, "C", tool="add")])))
    assert [(s.tool, s.args, s.observation) for s in ep.trajectory.steps] == [("add", {"x": 1}, "2")]


@pytest.mark.skipif(importlib.util.find_spec("inspect_ai") is not None, reason="inspect_ai installed")
def test_run_inspect_missing_extra_is_actionable():
    with pytest.raises(MissingExtraError, match=r"ek\[agents\]"):
        run_inspect(object(), model="mockllm/model")


@pytest.fixture
def fake_inspect_ai(monkeypatch):
    """A stand-in ``inspect_ai`` module whose ``eval`` records its call and returns logs."""
    calls = {}
    module = types.ModuleType("inspect_ai")
    module.__spec__ = importlib.util.spec_from_loader("inspect_ai", loader=None)

    def fake_eval(tasks, *, model=None, log_dir=None, **kwargs):
        calls.update(tasks=tasks, model=model, log_dir=log_dir, **kwargs)
        return calls.get("logs") or [_log([_sample("a", 1, "C"), _sample("a", 2, "I")])]

    module.eval = fake_eval
    monkeypatch.setitem(sys.modules, "inspect_ai", module)
    return calls


def test_run_inspect_passes_through_and_uses_a_temp_log_dir(fake_inspect_ai, tmp_path):
    eps = run_inspect("my_task", model="mockllm/model", epochs=2, limit=5)
    assert [e.success for e in eps] == [True, False]
    assert fake_inspect_ai["model"] == "mockllm/model" and fake_inspect_ai["epochs"] == 2
    assert fake_inspect_ai["limit"] == 5 and fake_inspect_ai["display"] == "none"
    assert "ek-inspect-" in fake_inspect_ai["log_dir"]

    run_inspect("my_task", log_dir=str(tmp_path))
    assert fake_inspect_ai["log_dir"] == str(tmp_path)


def test_run_inspect_keeps_tasks_apart_when_pooling_logs(fake_inspect_ai):
    """Two tasks reusing sample ids must not merge into extra trials of one task."""
    def task_log(name):
        log = _log([_sample("1", 1, "C")])
        log.eval.task = name
        return log

    fake_inspect_ai["logs"] = [task_log("math"), task_log("code")]
    assert [e.task_id for e in run_inspect(["m", "c"])] == ["math/1", "code/1"]
    fake_inspect_ai["logs"] = [task_log("t"), task_log("t")]
    assert [e.task_id for e in run_inspect(["t", "t"])] == ["t#0/1", "t#1/1"]


def test_run_inspect_warns_on_ungraded_or_empty(fake_inspect_ai):
    fake_inspect_ai["logs"] = [_log([])]
    with pytest.warns(UserWarning, match="0 episode"):
        assert run_inspect("t") == []
    fake_inspect_ai["logs"] = [_log([_sample("a", 1, {"acc": 1})])]
    with pytest.warns(UserWarning, match="1 ungraded"):
        run_inspect("t")


def test_run_inspect_refuses_a_failed_eval(fake_inspect_ai):
    fake_inspect_ai["logs"] = [_log([], status="error")]
    with pytest.raises(RuntimeError, match="did not succeed"):
        run_inspect("my_task", model="mockllm/model")


# ---------------------------------------------------------------------------
# API drift in the WRAP adapters
# ---------------------------------------------------------------------------


def _class_in(module_name, **attrs):
    cls = type("Backend", (), attrs)
    cls.__module__ = module_name
    return cls


def test_is_backend_is_per_package():
    assert _is_backend(_class_in("deepeval.metrics.tool_correctness")(), "deepeval")
    assert _is_backend(_class_in("ragas.metrics._faithfulness")(), "ragas")
    assert not _is_backend(_class_in("deepeval.metrics.x")(), "ragas")
    assert not _is_backend(_class_in("ragas_fork.metrics")(), "ragas")
    assert not _is_backend(_class_in("tests.fakes")(), "deepeval")


def test_real_deepeval_metric_gets_a_test_case_not_a_dict(monkeypatch):
    """The bug: the old check only knew Ragas, so a real DeepEval metric was handed a dict."""
    seen = {}

    def measure(self, case):
        seen["case"] = case
        self.score = 1.0

    metric = _class_in("deepeval.metrics.tool_correctness", measure=measure, score=None)()
    m = DeepEvalMetric("tool_correctness", metric=metric)
    built = {}

    def fake_real_case(fields, args):
        built.update(fields=fields, args=args)
        return "LLMTestCase"

    monkeypatch.setattr(m, "_real_case", fake_real_case)
    ep = Episode(task_id="t", output="x",
                 trajectory=Trajectory([Step("search", {"q": "a"}), Step("pay", {})]))
    assert m(ep, {"expected_tools": [("search", {"q": "a"}), "pay"]}).value == 1.0
    assert seen["case"] == "LLMTestCase"
    assert built["fields"]["tools_called"] == ["search", "pay"]
    assert built["args"] == {"tools_called": [{"q": "a"}, {}],
                             "expected_tools": [{"q": "a"}, {}]}


def test_ragas_collections_api_is_scored():
    """Ragas >= 0.4: ``ascore(**fields) -> MetricResult``; only the fields it accepts are sent."""
    seen = {}

    class AnswerRelevancy:
        async def ascore(self, user_input, response):
            seen.update(user_input=user_input, response=response)
            return NS(value=0.7, reason="r")

    sig = RagasSignal(AnswerRelevancy())
    assert sig({"question": "q", "answer": "a", "contexts": ["c"]}) == 0.7
    assert seen == {"user_input": "q", "response": "a"}


def test_ragas_kwargs_only_backend_gets_every_field():
    class Wrapped:
        async def ascore(self, **fields):
            return NS(value=len(fields) / 10)

    assert RagasSignal(Wrapped())({"question": "q", "answer": "a", "contexts": ["c"]}) == 0.3


def test_deepeval_calls_tolerate_odd_shapes():
    calls = DeepEvalMetric._calls(["a", ("b", {"x": 1}), ("search", "query"), Step("c", {})])
    assert calls == [("a", {}), ("b", {"x": 1}), ("('search', 'query')", {}), ("c", {})]


def test_ragas_extra_kwargs_reach_the_constructor():
    sig = RagasSignal(name="answer_relevancy", llm="judge", embeddings="emb")
    assert sig.metric_kwargs == {"embeddings": "emb"}


def test_geval_kwargs_fill_deepeval4_params():
    class SingleTurnParams(enum.Enum):
        INPUT = "input"
        ACTUAL_OUTPUT = "actual_output"
        EXPECTED_OUTPUT = "expected_output"

    new = types.SimpleNamespace(SingleTurnParams=SingleTurnParams)
    kw = _geval_kwargs({"criteria": "c"}, new)
    assert kw["name"] == "ek_geval" and kw["criteria"] == "c"
    assert kw["evaluation_params"] == [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    kw = _geval_kwargs({"name": "mine", "evaluation_params": ["expected_output"]}, new)
    assert kw["name"] == "mine" and kw["evaluation_params"] == [SingleTurnParams.EXPECTED_OUTPUT]

    # DeepEval < 4 called the enum LLMTestCaseParams.
    old = types.SimpleNamespace(LLMTestCaseParams=SingleTurnParams)
    assert _geval_kwargs({}, old)["evaluation_params"][0] is SingleTurnParams.INPUT
