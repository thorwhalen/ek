"""Tests for the optional WRAP adapters (fake backends; real-library tests skip if absent)."""

import importlib.util

import pytest

import ek
from ek.agents import (
    DeepEvalMetric,
    RagasSignal,
    UqlmConsistencySignal,
    bundled_prices_path,
    load_prices,
)
from ek.agents.base import Episode, Step, Trajectory
from ek.registry import MissingExtraError


def _has(mod):
    return importlib.util.find_spec(mod) is not None


def test_bundled_prices_load():
    table = load_prices()
    assert bundled_prices_path().exists()
    assert len(table) > 100
    assert all(p.input >= 0 and p.output >= 0 for p in table.values())


def test_uqlm_signal_fake_and_clamp():
    class Fake:
        def score(self, prompt):
            return 0.4 if prompt == "q" else 9

    sig = UqlmConsistencySignal(scorer=Fake())
    assert sig("q") == 0.4
    assert sig("other") == 1.0
    assert sig.cost_tier == 3


@pytest.mark.skipif(_has("uqlm"), reason="uqlm installed")
def test_uqlm_missing_extra_is_actionable():
    with pytest.raises(MissingExtraError, match=r"ek\[agents\]"):
        UqlmConsistencySignal(llm=object())("q")


def test_ragas_signal_fake():
    seen = {}

    class Fake:
        async def single_turn_ascore(self, sample):
            seen.update(sample)
            return 0.8

    sig = RagasSignal(Fake())
    assert sig({"question": "q", "answer": "a", "contexts": ["c"]}) == 0.8
    assert seen == {"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}


def test_ragas_rejects_unknown_name():
    with pytest.raises(ValueError):
        RagasSignal(name="nope")


@pytest.mark.skipif(_has("ragas"), reason="ragas installed")
def test_ragas_missing_extra_is_actionable():
    with pytest.raises(MissingExtraError, match=r"ek\[agents\]"):
        RagasSignal()({"question": "q", "answer": "a", "contexts": []})


class _FakeDE:
    reason = "because"

    def __init__(self):
        self.score = None
        self.case = None

    def measure(self, case):
        self.case = case
        self.score = 0.5


def test_deepeval_metric_fake_labels_recall():
    ep = Episode(task_id="t", output="x", trajectory=Trajectory([Step("search", {})]))
    fake = _FakeDE()
    m = DeepEvalMetric("tool_correctness", metric=fake)
    s = m(ep, {"expected_tools": ["search"], "expected_output": "x"})
    assert s.value == 0.5
    assert "recall" in s.detail["flavor"]
    assert fake.case["tools_called"] == ["search"]
    assert fake.case["expected_tools"] == ["search"]
    assert m.aggregate([s, s]) == 0.5


def test_deepeval_registered_and_telemetry_off(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    from ek.agents.wrappers import disable_deepeval_telemetry
    import os

    disable_deepeval_telemetry()
    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
    assert "deepeval_tool_correctness" in set(ek.registry.names("metrics"))


@pytest.mark.skipif(_has("deepeval"), reason="deepeval installed")
def test_deepeval_missing_extra_is_actionable():
    ep = Episode(task_id="t", output="x")
    with pytest.raises(MissingExtraError, match=r"ek\[agents\]"):
        DeepEvalMetric("tool_correctness")(ep, "x")
