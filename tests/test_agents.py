"""Tests for ``ek.agents`` -- agent & assistant evaluation in cost per successful task.

These deliberately pin the *design decisions* that an adversarial review flagged, not just the
happy paths:

- trajectory scoring must be **order-sensitive** (the graph-edit-distance engine is not);
- tool-call scoring must handle a **multiset** of calls (``FieldMetric``'s key-based alignment
  cannot: repeated/reordered calls collide);
- ``cost_per_success`` must stamp ``higher_is_better=False`` or the regression gate inverts;
- the agent gate must be **variance-aware** (not flag noise, not miss a real regression) and must
  **refuse** to compare runs whose user-simulator/suite version changed;
- Cost-of-Pass must diverge to ``inf`` when nothing succeeds;
- ``pass^k`` must warn when the agent is deterministic (it degenerates to ``pass^1``).
"""

import json
import math
import random
import warnings

import pytest

import ek
from ek.agents import (
    Cost,
    CostPerSuccessMetric,
    Episode,
    JudgeSignal,
    ModelPrice,
    RunProvenance,
    Step,
    TaskSpec,
    ToolCallMetric,
    ToolSpec,
    TrajectoryMetric,
    TaskSuccessMetric,
    Trajectory,
    agent_regression_gate,
    as_agent,
    as_call,
    cost_of_pass,
    cost_report,
    dollars,
    episode_from_messages,
    from_deepeval_test_case,
    from_inspect_sample,
    judge_validation,
    load_prices,
    match_calls,
    pairwise_judge,
    pass_at_k,
    pass_hat_k,
    per_million,
    reliability,
    run_suite,
    save_agent_baseline,
    tool_grammar,
    trajectory_from_messages,
    wilson_interval,
)
from ek.agents.cost import UnknownModelPrice
from ek.base import FieldSpec

# ---------------------------------------------------------------------------
# Layer A: the task/tool grammar carries the cost weights
# ---------------------------------------------------------------------------


def test_tool_grammar_carries_cost_weights():
    search = ToolSpec("search", params={"q": FieldSpec("q", "string")})
    refund = ToolSpec(
        "refund",
        params={"amount": FieldSpec("amount", "number", importance=20.0)},
        destructive=True,
    )
    g = tool_grammar(search, refund)
    assert g.node_cost("search") == 1.0
    assert g.node_cost("refund") == 10.0  # destructive default
    assert g.field_cost("refund", "amount") == 20.0
    assert g.field_cost("search", "q") == 1.0


def test_explicit_importance_beats_destructive_default():
    t = ToolSpec("wipe", importance=3.0, destructive=True)
    assert t.weight == 3.0


def test_as_call_accepts_step_tuple_and_mapping():
    assert as_call(Step("s", {"q": "x"})) == ("s", {"q": "x"})
    assert as_call(("s", {"q": "x"})) == ("s", {"q": "x"})
    assert as_call({"tool": "s", "args": {"q": "x"}}) == ("s", {"q": "x"})
    assert as_call({"name": "s", "arguments": {"q": "x"}}) == ("s", {"q": "x"})
    assert as_call("s") == ("s", {})
    with pytest.raises(TypeError):
        as_call(3.7)


def test_cost_adds():
    total = Cost(input_tokens=10, output_tokens=2) + Cost(
        input_tokens=5, output_tokens=1, retries=2
    )
    assert (total.input_tokens, total.output_tokens, total.retries) == (15, 3, 2)
    assert sum([Cost(input_tokens=1), Cost(input_tokens=2)], Cost()).input_tokens == 3


# ---------------------------------------------------------------------------
# Reliability: pass@k vs pass^k
# ---------------------------------------------------------------------------


def test_pass_at_k_matches_humaneval_estimator():
    # all fail -> 0; all succeed -> 1
    assert pass_at_k(n=5, c=0, k=1) == 0.0
    assert pass_at_k(n=5, c=5, k=3) == 1.0
    # 1 - C(n-c, k)/C(n, k) = 1 - C(3,2)/C(5,2) = 1 - 3/10
    assert pass_at_k(n=5, c=2, k=2) == pytest.approx(0.7)


def test_pass_hat_k_is_the_reliability_estimator():
    # C(c,k)/C(n,k) = C(2,2)/C(5,2) = 1/10
    assert pass_hat_k(n=5, c=2, k=2) == pytest.approx(0.1)
    assert pass_hat_k(n=5, c=1, k=2) == 0.0  # fewer successes than k -> impossible
    assert pass_hat_k(n=4, c=4, k=4) == 1.0


def test_pass_hat_k_decays_geometrically():
    """A 'usually works' agent collapses under pass^k -- the whole point of the metric."""
    n, c = 100, 90  # 90% per-trial success
    assert pass_at_k(n=n, c=c, k=8) > 0.99  # capability looks perfect
    assert pass_hat_k(n=n, c=c, k=8) < 0.45  # reliability does not (~0.9**8 = 0.43)


def test_estimators_validate_their_inputs():
    with pytest.raises(ValueError):
        pass_at_k(n=2, c=1, k=3)  # k > n
    with pytest.raises(ValueError):
        pass_hat_k(n=5, c=9, k=1)  # c > n
    with pytest.raises(ValueError):
        pass_at_k(n=5, c=1, k=0)


def test_reliability_report_and_slices():
    r = reliability(
        {"t1": [True, True, False], "t2": [True, True, True]},
        k=2,
        slices={"t1": "hard", "t2": "easy"},
    )
    assert r.n_tasks == 2 and r.n_trials == 6
    assert r.success_rate == pytest.approx(5 / 6)
    # t1: C(2,2)/C(3,2)=1/3 ; t2: 1.0  -> mean 2/3
    assert r.pass_hat_k == pytest.approx(2 / 3)
    assert set(r.per_slice) == {"hard", "easy"}
    assert r.per_slice["easy"]["pass_hat_k"] == 1.0
    lo, hi = r.pass_hat_k_ci
    assert 0.0 <= lo <= r.pass_hat_k <= hi <= 1.0


def test_reliability_warns_when_agent_is_deterministic():
    """pass^k on a deterministic agent degenerates to pass^1 -- say so, don't reassure."""
    with pytest.warns(UserWarning, match="deterministic"):
        reliability({"t1": [True, True], "t2": [False, False]}, k=2)


def test_reliability_rejects_ungraded_episodes():
    with pytest.raises(ValueError, match="ungraded"):
        reliability([Episode(task_id="t1")], k=1)


def test_wilson_interval_is_sane_in_the_tails():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.5  # not the degenerate [0, 0] a Wald interval gives
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi


# ---------------------------------------------------------------------------
# Cost: Cost-of-Pass
# ---------------------------------------------------------------------------


def test_dollars_prices_token_kinds_asymmetrically():
    price = per_million(1.0, 10.0)  # output is 10x input
    assert dollars(Cost(input_tokens=1_000_000), price) == pytest.approx(1.0)
    assert dollars(Cost(output_tokens=1_000_000), price) == pytest.approx(10.0)
    # cached input defaults to 10% of the input rate
    assert dollars(Cost(cached_input_tokens=1_000_000), price) == pytest.approx(0.1)


def test_batch_discount():
    price = per_million(1.0, 1.0)
    full = dollars(Cost(input_tokens=1_000_000), price)
    assert dollars(Cost(input_tokens=1_000_000), price, batch=True) == pytest.approx(
        full * 0.5
    )


def test_cost_of_pass_diverges_when_nothing_succeeds():
    """A model that never succeeds is not cheap -- it is unusable. inf, not 0."""
    assert cost_of_pass(10.0, 0) == math.inf
    assert cost_of_pass(10.0, 4) == 2.5


def test_unknown_model_price_is_actionable():
    with pytest.raises(UnknownModelPrice, match=r"load_prices|prices="):
        dollars(Cost(input_tokens=1), "gpt-nonexistent", prices={})


def test_load_prices_parses_litellm_shape():
    table = load_prices(
        {
            "m1": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "cache_read_input_token_cost": 1e-7,
            },
            "embed": {"input_cost_per_token": None, "output_cost_per_token": None},
        }
    )
    assert "embed" not in table  # unpriced/non-completion entries are skipped
    assert table["m1"].output == 2e-6
    assert table["m1"].cached_rate == 1e-7


def test_cost_report_refuses_to_invent_prices():
    eps = [
        Episode(task_id="a", cost=Cost(input_tokens=100), success=True),
        Episode(task_id="b", cost=Cost(input_tokens=100), success=False),
    ]
    unpriced = cost_report(eps)
    assert unpriced["total_dollars"] is None  # not a fabricated 0.0
    assert unpriced["cost_per_success"] is None
    assert unpriced["total_tokens"] == 200  # tokens still reported
    priced = cost_report(eps, price=per_million(1.0, 1.0))
    assert priced["cost_per_success"] == pytest.approx(2e-4)


# ---------------------------------------------------------------------------
# Tool calls: the multiset matcher FieldMetric cannot provide
# ---------------------------------------------------------------------------


def test_tool_call_exact_match():
    gold = [("search", {"q": "cats"}), ("answer", {"a": "meow"})]
    pred = [Step("search", {"q": "cats"}), Step("answer", {"a": "meow"})]
    assert ToolCallMetric()(pred, gold).f1 == 1.0


def test_tool_call_scores_hallucinated_and_missed_calls():
    gold = [("search", {"q": "cats"}), ("answer", {"a": "meow"})]
    pred = [Step("search", {"q": "cats"}), Step("delete_all", {})]
    s = ToolCallMetric()(pred, gold)
    assert s.detail["n_spurious"] == 1 and s.detail["n_missed"] == 1
    assert s.f1 == pytest.approx(0.5)  # 1 TP, 1 FP, 1 FN


def test_tool_call_handles_repeated_and_reordered_calls():
    """The multiset case: two identical calls cannot collide onto one dict key.

    Note the alignment choice: a *duplicate* same-tool call is charged as a wrong-argument
    error against the still-unmatched gold call (tp=1, fp=1, fn=1), which scores identically
    to calling it spurious+missed but additionally tells you *which argument* was wrong.
    """
    gold = [("s", {"q": "a"}), ("s", {"q": "b"})]
    reordered = [Step("s", {"q": "b"}), Step("s", {"q": "a"})]
    assert ToolCallMetric()(reordered, gold).f1 == 1.0  # order-insensitive by call
    # A duplicated call is not a free pass: it is still one right call and one wrong one.
    duped = [Step("s", {"q": "a"}), Step("s", {"q": "a"})]
    s = ToolCallMetric()(duped, gold)
    assert s.detail["tp"] == 1 and s.detail["fp"] >= 1 and s.detail["fn"] >= 1
    assert s.f1 == pytest.approx(0.5)


def test_tool_call_spurious_tool_is_not_absorbed_by_a_same_tool_match():
    """A call to a tool gold never uses is spurious, not a wrong-argument error."""
    s = ToolCallMetric()(
        [Step("s", {"q": "a"}), Step("rm", {"path": "/"})], [("s", {"q": "a"})]
    )
    assert s.detail["n_spurious"] == 1 and s.detail["n_missed"] == 0


def test_tool_call_wrong_argument_counts_as_both_spurious_and_missed():
    s = ToolCallMetric()([Step("s", {"q": "wrong"})], [("s", {"q": "right"})])
    assert s.detail["tp"] == 0 and s.detail["fp"] > 0 and s.detail["fn"] > 0


def test_tool_call_is_cost_weighted_by_the_grammar():
    """A wrong argument to a destructive tool is not one unit of error."""
    g = tool_grammar(
        ToolSpec("note", params={"text": FieldSpec("text")}),
        ToolSpec("refund", params={"amt": FieldSpec("amt", importance=50.0)}, destructive=True),
    )
    m = ToolCallMetric(level="arg", grammar=g)
    cheap = m([Step("note", {"text": "x"})], [("note", {"text": "y"})])
    pricey = m([Step("refund", {"amt": 1})], [("refund", {"amt": 999})])
    assert pricey.detail["fn"] > cheap.detail["fn"] * 10


def test_tool_call_folds_case_and_whitespace_by_default():
    assert ToolCallMetric()([Step("s", {"q": "Cats "})], [("s", {"q": "cats"})]).f1 == 1.0


def test_tool_call_aggregates_micro_not_mean():
    """Micro-averaged over pooled TP/FP/FN -- a corpus of unequal episodes must not be meaned."""
    m = ToolCallMetric()
    scores = [
        m([Step("a", {})] * 3, [("a", {})] * 3),  # 3 TP
        m([Step("b", {})], [("c", {})]),  # 1 FP + 1 FN
    ]
    naive_mean = sum(s.f1 for s in scores) / len(scores)  # (1.0 + 0.0) / 2 = 0.5
    micro = m.aggregate(scores)  # P=R=3/4 -> F1=0.75
    assert micro == pytest.approx(0.75)
    assert micro != pytest.approx(naive_mean)


# ---------------------------------------------------------------------------
# Trajectory: order sensitivity -- the property the GED engine loses
# ---------------------------------------------------------------------------


def test_trajectory_identical_is_zero_distance():
    traj = [Step("a", {}), Step("b", {})]
    assert TrajectoryMetric()(traj, traj).value == 0.0


def test_trajectory_in_order_is_order_sensitive():
    """The load-bearing property: a graph-edit-distance engine would score these equal."""
    gold = [Step("a", {}), Step("b", {})]
    swapped = [Step("b", {}), Step("a", {})]
    in_order = TrajectoryMetric(scheme="in_order")(swapped, gold).value
    any_order = TrajectoryMetric(scheme="any_order")(swapped, gold).value
    assert in_order > 0.0, "a reordered trajectory must cost something"
    assert any_order == 0.0, "any_order deliberately ignores order"


def test_trajectory_penalizes_a_needless_extra_step():
    gold = [Step("search", {"q": "cat"}), Step("answer", {"a": "meow"})]
    detour = [Step("search", {"q": "cat"}), Step("search", {"q": "cat"}),
              Step("answer", {"a": "meow"})]
    assert TrajectoryMetric()(detour, gold).value > 0.0
    # ...but the same detour IS a superset of the required calls.
    assert TrajectoryMetric(scheme="superset")(detour, gold).value == 0.0


def test_trajectory_superset_flags_a_missing_required_call():
    gold = [Step("verify", {}), Step("send", {})]
    skipped_verify = [Step("send", {})]
    assert TrajectoryMetric(scheme="superset")(skipped_verify, gold).value > 0.0


def test_trajectory_exact_scheme_is_all_or_nothing():
    gold = [Step("a", {"x": 1}), Step("b", {})]
    assert TrajectoryMetric(scheme="exact")(gold, gold).value == 0.0
    assert TrajectoryMetric(scheme="exact")([Step("a", {"x": 2}), Step("b", {})], gold).value == 1.0


def test_trajectory_is_cost_weighted_and_unbounded_in_length():
    g = tool_grammar(ToolSpec("ping"), ToolSpec("wipe", destructive=True))
    m = TrajectoryMetric(grammar=g)
    gold = [Step("ping", {})]
    cheap_detour = [Step("ping", {}), Step("ping", {})]
    costly_detour = [Step("ping", {}), Step("wipe", {})]
    assert m(costly_detour, gold).detail["raw_distance"] > m(cheap_detour, gold).detail[
        "raw_distance"
    ]
    # No max_nodes cap: a long trajectory scores rather than raising (GED would refuse).
    long_traj = [Step("ping", {}) for _ in range(200)]
    assert 0.0 <= m(long_traj, gold).value <= 1.0


def test_trajectory_scheme_is_validated():
    with pytest.raises(ValueError, match="scheme"):
        TrajectoryMetric(scheme="nonsense")


def test_trajectory_direction_is_lower_is_better():
    s = TrajectoryMetric()([Step("a", {})], [Step("b", {})])
    assert s.detail["higher_is_better"] is False
    assert s.detail["exact"] is True  # not a timeout-truncated approximation


# ---------------------------------------------------------------------------
# Task success + cost-per-success
# ---------------------------------------------------------------------------


def test_task_success_uses_an_injected_checker():
    ep = Episode(task_id="t", output="42", final_state={"db": 1})
    assert TaskSuccessMetric()(ep, "42").value == 1.0  # default: output match
    assert TaskSuccessMetric(check="final_state")(ep, {"db": 1}).value == 1.0
    assert TaskSuccessMetric(check="final_state")(ep, {"db": 2}).value == 0.0
    # any callable works -- ek never owns the sandbox/LLM
    assert TaskSuccessMetric(check=lambda e, g: True)(ep, "nope").value == 1.0


def test_task_success_aggregate_is_the_success_rate():
    m = TaskSuccessMetric()
    scores = [m(Episode(output="a"), "a"), m(Episode(output="b"), "c")]
    assert m.aggregate(scores) == 0.5


def test_cost_per_success_aggregate_is_cost_of_pass():
    price = per_million(1.0, 1.0)
    m = CostPerSuccessMetric(price=price)
    eps = [
        Episode(task_id="a", output="x", cost=Cost(input_tokens=1_000_000)),  # success
        Episode(task_id="b", output="y", cost=Cost(input_tokens=1_000_000)),  # failure
    ]
    scores = [m(eps[0], "x"), m(eps[1], "x")]
    # $2 total spent, 1 success -> $2 per successful task (NOT the $1 mean per episode)
    assert m.aggregate(scores) == pytest.approx(2.0)


def test_cost_per_success_is_infinite_when_nothing_succeeds():
    m = CostPerSuccessMetric(price=per_million(1.0, 1.0))
    ep = Episode(task_id="a", output="wrong", cost=Cost(input_tokens=1_000_000))
    assert m.aggregate([m(ep, "right")]) == math.inf


def test_cost_per_success_stamps_lower_is_better():
    """Without this stamp the regression gate would wave a cost regression straight through."""
    m = CostPerSuccessMetric(price=per_million(1.0, 1.0))
    s = m(Episode(task_id="a", output="x", cost=Cost(input_tokens=10)), "x")
    assert s.detail["higher_is_better"] is False


def test_cost_per_success_gate_direction_is_read_by_the_core_harness(tmp_path):
    """End-to-end: the core regression_gate must treat rising cost as a regression."""
    price = per_million(1.0, 1.0)
    m = CostPerSuccessMetric(price=price)
    cheap = [(Episode(task_id="a", output="x", cost=Cost(input_tokens=1_000_000)), "x")]
    dear = [(Episode(task_id="a", output="x", cost=Cost(input_tokens=9_000_000)), "x")]
    base_report = ek.evaluate(cheap, metric=m)
    ek.save_baseline(base_report, "costbase", rootdir=str(tmp_path))
    worse = ek.evaluate(dear, metric=m)
    gate = ek.regression_gate(worse, "costbase", rootdir=str(tmp_path))
    assert not gate, "a 9x cost increase must fail the gate"
    assert gate.higher_is_better is False


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def test_judge_signal_is_a_reference_free_signal():
    sig = JudgeSignal(lambda out, *, criteria="": 0.9 if "meow" in out else 0.1,
                      criteria="helpful?")
    assert sig("the cat says meow") == 0.9
    assert sig(Episode(output="the cat says meow")) == 0.9  # reads .output off an Episode
    assert sig.cost_tier == 5  # the most expensive family -> escalate to it last


def test_pairwise_judge_corrects_position_bias():
    """A judge that always prefers whichever answer it sees first must come out a tie."""
    always_first = lambda a, b, *, criteria="": 1.0  # noqa: E731
    assert pairwise_judge(always_first, "A", "B") == 0.5
    assert pairwise_judge(always_first, "A", "B", swap=False) == 1.0  # bias visible unswapped


def test_pairwise_judge_still_reports_a_real_preference():
    prefer_longer = lambda a, b, *, criteria="": 1.0 if len(a) > len(b) else 0.0  # noqa: E731
    assert pairwise_judge(prefer_longer, "long answer", "s") == 1.0


def test_judge_validation_reuses_the_iaa_harness():
    perfect = judge_validation([1, 1, 0, 0], [1, 1, 0, 0])
    assert perfect["alpha"] == 1.0 and perfect["verdict"] == "good"
    assert perfect["kappa"] == 1.0
    bad = judge_validation([1, 0, 1, 0], [0, 1, 0, 1])
    assert bad["verdict"] == "unreliable"


def test_judge_validation_requires_aligned_labels():
    with pytest.raises(ValueError):
        judge_validation([1, 0], [1])


def test_estimate_quality_refuses_to_gate_an_uncalibrated_judge():
    """Hard Rule 1 applies to judges too: a raw judge score is not a probability."""
    sig = JudgeSignal(lambda out, *, criteria="": 0.9)
    with pytest.raises(ValueError, match="calibrat"):
        ek.estimate_quality("some answer", signals=[sig], policy=lambda c: ek.Decision.ACCEPT)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def test_trajectory_from_openai_messages():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "content": "18C"},
        {"role": "assistant", "content": "It is 18C."},
    ]
    traj = trajectory_from_messages(messages)
    assert traj.tools == ("get_weather",)
    assert traj.steps[0].args == {"city": "Paris"}
    assert traj.steps[0].observation == "18C"


def test_trajectory_from_anthropic_messages():
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "search", "input": {"q": "cats"}}]},
        {"role": "user", "content": "results..."},
    ]
    assert trajectory_from_messages(messages).tools == ("search",)


def test_malformed_tool_arguments_are_evaluable_not_fatal():
    messages = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "f", "arguments": "{not json"}}]}]
    step = trajectory_from_messages(messages).steps[0]
    assert "_malformed" in step.args  # a bad payload is a finding, not a crash


def test_episode_from_messages_extracts_output_and_cost():
    messages = [
        {"role": "assistant", "content": "final answer"},
    ]
    ep = episode_from_messages(
        messages, task_id="t1", usage={"prompt_tokens": 10, "completion_tokens": 3}
    )
    assert ep.output == "final answer"
    assert ep.cost.input_tokens == 10 and ep.cost.output_tokens == 3


def test_as_agent_wraps_a_plain_function():
    ep = as_agent(str.upper)(TaskSpec("t", input="hi"))
    assert isinstance(ep, Episode) and ep.output == "HI"


class _InspectToolCall:
    """Inspect AI's real shape: `.function` IS the name (a str), `.arguments` is a dict."""

    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments


class _InspectMessage:
    def __init__(self, role, content="", tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _InspectSample:
    def __init__(self, messages, output=None, id="s1"):
        self.messages = messages
        self.output = output
        self.id = id


class _InspectOutput:
    def __init__(self, completion, usage=None):
        self.completion = completion
        self.usage = usage


def test_from_inspect_sample_reads_object_shaped_tool_calls():
    """Inspect's ToolCall is an OBJECT, not a mapping.

    Getting this wrong drops every call and scores the episode as 'made no calls' -- a silent
    wrong answer, not an error. Regression test for exactly that.
    """
    sample = _InspectSample(
        messages=[
            _InspectMessage("user", "weather?"),
            _InspectMessage(
                "assistant",
                tool_calls=[_InspectToolCall("get_weather", {"city": "Paris"})],
            ),
            _InspectMessage("tool", "18C"),
            _InspectMessage("assistant", "It is 18C."),
        ],
        output=_InspectOutput("It is 18C.", usage={"input_tokens": 40, "output_tokens": 9}),
    )
    ep = from_inspect_sample(sample)
    assert ep.trajectory.tools == ("get_weather",), "Inspect tool calls were silently dropped"
    assert ep.trajectory.steps[0].args == {"city": "Paris"}
    assert ep.trajectory.steps[0].observation == "18C"
    assert ep.output == "It is 18C."
    assert ep.cost.input_tokens == 40
    assert ep.task_id == "s1"


def test_openai_sdk_object_shaped_tool_calls():
    """The OpenAI SDK nests the name: call.function.name / call.function.arguments (a JSON str)."""

    class _Fn:
        name, arguments = "search", '{"q": "cats"}'

    class _Call:
        function = _Fn()

    class _Msg:
        role, content, tool_calls = "assistant", "", [_Call()]

    traj = trajectory_from_messages([_message_dict_for_test(_Msg())])
    assert traj.tools == ("search",)
    assert traj.steps[0].args == {"q": "cats"}


def _message_dict_for_test(m):
    return {"role": m.role, "content": m.content, "tool_calls": m.tool_calls}


def test_from_deepeval_test_case():
    class _Tool:
        name = "search"
        input_parameters = {"q": "cats"}

    class _Case:
        actual_output = "meow"
        tools_called = [_Tool()]

    ep = from_deepeval_test_case(_Case(), task_id="d1")
    assert ep.output == "meow" and ep.trajectory.tools == ("search",)
    assert ep.trajectory.steps[0].args == {"q": "cats"}


# ---------------------------------------------------------------------------
# Harness: k trials, provenance, and the variance-aware gate
# ---------------------------------------------------------------------------


def _flaky_agent(fail_task="t2", p_fail=1.0):
    """An agent that always gets `fail_task` wrong."""

    def agent(task):
        wrong = task.task_id == fail_task
        return Episode(
            output=("WRONG" if wrong else str(task.gold)),
            cost=Cost(input_tokens=1000),
        )

    return agent


def test_run_suite_reports_reliability_and_cost():
    tasks = [TaskSpec("t1", input="a", gold="A"), TaskSpec("t2", input="b", gold="B")]
    report = run_suite(_flaky_agent(), tasks, k=1, price=per_million(1.0, 1.0))
    assert report.n_tasks == 2 and report.n_trials == 2
    assert report.success_rate == 0.5
    assert report.pass_hat_k == 0.5
    assert report.cost["total_tokens"] == 2000
    assert report.cost["cost_per_success"] == pytest.approx(2e-3)


def test_run_suite_runs_k_trials_per_task():
    tasks = [TaskSpec("t1", input="a", gold="A")]
    # A deterministic agent makes pass^k meaningless -- the harness must say so, not reassure.
    with pytest.warns(UserWarning, match="deterministic"):
        report = run_suite(as_agent(str.upper), tasks, k=4)
    assert report.n_trials == 4 and report.n_tasks == 1
    assert report.pass_at_k == 1.0
    assert report.stochastic is False


def test_run_suite_records_provenance():
    tasks = [TaskSpec("t1", input="a", gold="A")]
    run = RunProvenance(model="m1", simulator="sim@2", suite_version="v3", scaffold="react")
    report = run_suite(as_agent(str.upper), tasks, k=1, run=run)
    assert report.detail["provenance"]["simulator"] == "sim@2"
    assert report.detail["provenance"]["suite_version"] == "v3"


def test_run_suite_accepts_a_task_mapping():
    report = run_suite(as_agent(str.upper), {"t1": TaskSpec("t1", input="a", gold="A")}, k=1)
    assert report.success_rate == 1.0


def test_run_suite_slices():
    tasks = [
        TaskSpec("t1", input="a", gold="A", slice="easy"),
        TaskSpec("t2", input="b", gold="B", slice="hard"),
    ]
    report = run_suite(_flaky_agent(), tasks, k=1)
    assert report.per_slice["easy"]["success_rate"] == 1.0
    assert report.per_slice["hard"]["success_rate"] == 0.0


def test_agent_gate_passes_on_a_first_run(tmp_path):
    tasks = [TaskSpec("t1", input="a", gold="A")]
    report = run_suite(as_agent(str.upper), tasks, k=1)
    assert agent_regression_gate(report, "nonexistent", rootdir=str(tmp_path))


def test_agent_gate_flags_a_real_regression(tmp_path):
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(30)]
    good = run_suite(as_agent(str.upper), tasks, k=1)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    broken = run_suite(as_agent(lambda s: "WRONG"), tasks, k=1)
    gate = agent_regression_gate(broken, "base", rootdir=str(tmp_path))
    assert not gate
    assert any("regressed" in r for r in gate.reasons)


def test_agent_gate_does_not_flag_pure_noise(tmp_path):
    """An identical re-run must not be called a regression -- that is the whole point."""
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(20)]
    first = run_suite(as_agent(str.upper), tasks, k=1)
    save_agent_baseline(first, "base", rootdir=str(tmp_path))
    second = run_suite(as_agent(str.upper), tasks, k=1)
    assert agent_regression_gate(second, "base", rootdir=str(tmp_path))


def _stochastic_agent(p_fail: float, seed: int):
    """A genuinely FLAKY agent -- the only kind that can exercise a variance-aware gate."""
    rng = random.Random(seed)

    def agent(task):
        wrong = rng.random() < p_fail
        return Episode(output=("WRONG" if wrong else str(task.gold)),
                       cost=Cost(input_tokens=1000))

    return agent


def test_agent_gate_tolerates_real_sampling_noise(tmp_path):
    """THE load-bearing gate property, and the one a deterministic agent cannot test.

    Two runs of the SAME flaky agent (same quality, different luck) must not be called a
    regression. A gate that compares our interval to the baseline's *point* fails this: a
    Wilson upper bound only reaches 1.0 when every trial passed, so a single unlucky flake
    against a lucky baseline would read as a 'confident regression'.
    """
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(60)]
    lucky = run_suite(_stochastic_agent(p_fail=0.05, seed=1), tasks, k=1)
    save_agent_baseline(lucky, "base", rootdir=str(tmp_path))
    for seed in (2, 3, 4, 5):  # same agent quality, different luck
        rerun = run_suite(_stochastic_agent(p_fail=0.05, seed=seed), tasks, k=1)
        gate = agent_regression_gate(rerun, "base", rootdir=str(tmp_path))
        assert gate.passed, f"noise flagged as a regression (seed={seed}): {gate.reasons}"


def test_agent_gate_still_catches_a_regression_that_hides_under_noise(tmp_path):
    """...but the tolerance must not be so wide that a genuine quality drop slips through."""
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(60)]
    good = run_suite(_stochastic_agent(p_fail=0.05, seed=1), tasks, k=1)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    much_worse = run_suite(_stochastic_agent(p_fail=0.60, seed=2), tasks, k=1)
    gate = agent_regression_gate(much_worse, "base", rootdir=str(tmp_path))
    assert not gate.passed and gate.reasons


def test_agent_gate_has_real_statistical_POWER(tmp_path):
    """The gate must not buy its calm by going blind.

    Testing whether two 95% CIs *overlap* is not a 5% test but roughly a 0.5% one -- it makes a
    modest-but-real regression invisible. This pins actual power: a 90% -> 70% drop on a
    200-task suite (a smaller effect than the 'obvious' test above, on a realistic suite size)
    MUST be caught. A non-overlap rule detects this ~0% of the time.
    """
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(200)]
    good = run_suite(_stochastic_agent(p_fail=0.10, seed=1), tasks, k=1)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    worse = run_suite(_stochastic_agent(p_fail=0.30, seed=2), tasks, k=1)
    gate = agent_regression_gate(worse, "base", rootdir=str(tmp_path))
    assert not gate.passed, (
        f"a 20-point regression went undetected -- the gate is underpowered "
        f"({good.success_rate:.2f} -> {worse.success_rate:.2f})"
    )


def test_gate_skips_the_pass_k_arm_at_k_equals_1(tmp_path):
    """At k=1, pass^1 IS the success rate -- re-testing it with a worse method only adds noise.

    pass^1 = C(c,1)/C(n,1) = c/n, identically the success rate. Running the (less well
    calibrated) SE-from-CI arm on it would contribute zero power and only false alarms.
    """
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(40)]
    perfect = run_suite(as_agent(str.upper), tasks, k=1)
    assert perfect.pass_hat_k == perfect.success_rate  # the identity that makes the arm redundant
    save_agent_baseline(perfect, "base", rootdir=str(tmp_path))

    # A perfect baseline has a ZERO-WIDTH pass^k CI. The old arm would treat that as certainty
    # and collapse back to interval-vs-point -- flagging a single flake as a regression.
    def one_flake(task):
        return Episode(output=("WRONG" if task.task_id == "t3" else "A"))

    gate = agent_regression_gate(run_suite(one_flake, tasks, k=1), "base", rootdir=str(tmp_path))
    assert gate.passed, f"one flake vs a perfect baseline must not fail: {gate.reasons}"
    assert not any("pass^" in r for r in gate.reasons)


def test_gate_rejects_malformed_baselines(tmp_path):
    """A malformed record must fail loudly, never silently skip the check it cannot make."""
    store = ek.json_store("baselines", rootdir=str(tmp_path))
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(10)]
    report = run_suite(as_agent(str.upper), tasks, k=1)

    store["no_trials_key"] = {"kind": "agent", "success_rate": 1.0, "provenance": {}}
    with pytest.raises(ValueError, match="no 'n_trials'"):
        agent_regression_gate(report, "no_trials_key", rootdir=str(tmp_path))

    # n_success > n_trials used to blow up with a raw `math domain error` from wilson_interval.
    store["impossible"] = {
        "kind": "agent", "n_trials": 10, "n_success": 99, "success_rate": 1.0, "provenance": {},
    }
    with pytest.raises(ValueError, match="n_success"):
        agent_regression_gate(report, "impossible", rootdir=str(tmp_path))

    # counts but no rate at all -> the success check would be SILENTLY skipped.
    store["no_rate"] = {"kind": "agent", "n_trials": 10, "provenance": {}}
    with pytest.raises(ValueError, match="silently skipped"):
        agent_regression_gate(report, "no_rate", rootdir=str(tmp_path))


def test_agent_gate_refuses_an_empty_baseline(tmp_path):
    """An EMPTY baseline is worse than none: it is a permanent free pass.

    Every bound in it is the maximally-uncertain default, so no run can be shown to fall below
    it -- a totally broken agent would gate green. Guarding only the current run just moves the
    hole one hop upstream.
    """
    empty = run_suite(as_agent(str.upper), [], k=1)
    with pytest.raises(ValueError, match="ZERO trials"):
        save_agent_baseline(empty, "empty", rootdir=str(tmp_path))

    # ...and even if such a record exists (hand-written / from an older version), reject it.
    ek.json_store("baselines", rootdir=str(tmp_path))["empty"] = {
        "kind": "agent", "n_tasks": 0, "n_trials": 0, "success_rate": 0.0,
        "success_ci": [0.0, 1.0], "pass_hat_k": None, "pass_hat_k_ci": [0.0, 1.0],
        "provenance": {},
    }
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(50)]
    totally_broken = run_suite(as_agent(lambda s: "WRONG"), tasks, k=1)
    assert totally_broken.success_rate == 0.0
    with pytest.raises(ValueError, match="ZERO trials"):
        agent_regression_gate(totally_broken, "empty", rootdir=str(tmp_path))


def test_agent_gate_fails_an_empty_run(tmp_path):
    """An empty suite must NEVER be green: with no tasks every CI is [0,1], so no test can fire.

    A broken loader or an over-filtered suite would otherwise sail through as a pass --
    absence of evidence presented as evidence of no regression.
    """
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(10)]
    good = run_suite(as_agent(str.upper), tasks, k=1)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    empty = run_suite(as_agent(str.upper), [], k=1)
    gate = agent_regression_gate(empty, "base", rootdir=str(tmp_path))
    assert not gate.passed
    assert any("ZERO trials" in r for r in gate.reasons)


def test_agent_gate_flags_a_shrunken_suite(tmp_path):
    """Fewer tasks than the baseline is not the same experiment."""
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(10)]
    good = run_suite(as_agent(str.upper), tasks, k=1)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    fewer = run_suite(as_agent(str.upper), tasks[:4], k=1)
    gate = agent_regression_gate(fewer, "base", rootdir=str(tmp_path))
    assert not gate.passed and any("shrank" in r for r in gate.reasons)


def test_agent_gate_accepts_a_report_as_the_baseline(tmp_path):
    """Comparing two live reports is the most natural call -- it must not AttributeError."""
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(10)]
    first = run_suite(as_agent(str.upper), tasks, k=1)
    second = run_suite(as_agent(str.upper), tasks, k=1)
    assert agent_regression_gate(second, first)


def test_agent_gate_refuses_to_compare_across_a_changed_simulator(tmp_path):
    """A tau-bench-style user simulator is itself an LLM -- a hidden eval variable."""
    tasks = [TaskSpec("t1", input="a", gold="A")]
    base = run_suite(
        as_agent(str.upper), tasks, k=1, run=RunProvenance(simulator="sim@1")
    )
    save_agent_baseline(base, "base", rootdir=str(tmp_path))
    later = run_suite(
        as_agent(str.upper), tasks, k=1, run=RunProvenance(simulator="sim@2")
    )
    with pytest.raises(ValueError, match="simulator|conditions"):
        agent_regression_gate(later, "base", rootdir=str(tmp_path))


def test_agent_gate_warns_when_underpowered(tmp_path):
    """A drop the CI cannot resolve is not a pass -- it is an underpowered experiment."""
    tasks = [TaskSpec(f"t{i}", input="a", gold="A") for i in range(4)]
    good = run_suite(as_agent(str.upper), tasks, k=1)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    # one of four tasks now fails: a drop, but four tasks cannot prove it
    agent = _flaky_agent(fail_task="t0")
    noisy = run_suite(agent, tasks, k=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gate = agent_regression_gate(noisy, "base", rootdir=str(tmp_path), tolerance=0.5)
    assert gate.underpowered or not gate.passed
    if gate.passed:
        assert any("underpowered" in str(w.message) for w in caught)


def test_run_suite_persists(tmp_path):
    tasks = [TaskSpec("t1", input="a", gold="A")]
    run_suite(
        as_agent(str.upper), tasks, k=1, persist=True, run_id="r1", rootdir=str(tmp_path)
    )
    assert ek.json_store("runs", rootdir=str(tmp_path))["r1"]["kind"] == "agent"


def test_persisted_run_is_valid_json_even_with_infinite_cost(tmp_path):
    """cost-per-success is legitimately `inf` when nothing succeeds -- but JSON has no Infinity."""
    tasks = [TaskSpec("t1", input="a", gold="NEVER")]
    run_suite(
        as_agent(str.upper), tasks, k=1, price=per_million(1.0, 1.0),
        persist=True, run_id="r2", rootdir=str(tmp_path),
    )
    raw = (tmp_path / "runs" / "r2.json").read_text()
    assert "Infinity" not in raw, "invalid JSON: strict parsers (jq, JS) reject Infinity"
    assert json.loads(raw)["cost_per_success"] is None  # null = "not measurable"


def test_run_suite_passes_the_seed_to_an_agent_that_takes_one():
    """The seed must actually reach the agent, not just be stamped on provenance."""
    seen = []

    def agent(task, *, seed=None):
        seen.append(seed)
        return Episode(output=str(task.gold))

    run_suite(agent, [TaskSpec("t1", input="a", gold="A")], k=3, seed=100)
    assert seen == [100, 101, 102]


def test_run_suite_injects_the_suite_grammar_into_extra_metrics():
    """The Layer-A cost weights must actually reach the metrics on the harness path."""
    refund = ToolSpec("refund", params={"amt": FieldSpec("amt", importance=50.0)},
                      destructive=True)
    task = TaskSpec("t1", input="x", gold=[("refund", {"amt": 10})], tools=[refund])

    def agent(spec):
        return Episode(trajectory=Trajectory([Step("refund", {"amt": 999})]), output="x")

    report = run_suite(agent, [task], k=1, check=lambda e, g: True,
                       metrics={"tool_call": ToolCallMetric(level="arg")})
    detail = report.detail["metrics"]["tool_call"]
    assert detail["n"] == 1
    # With the grammar injected, the wrong `amt` is weighted 50x -- F1 must reflect that.
    assert detail["aggregate"] < 0.5


def test_newcombe_difference_is_a_real_two_sample_test():
    from ek.agents.reliability import newcombe_difference

    # A confident regression: 50/100 vs 90/100.
    assert newcombe_difference(50, 100, 90, 100)[1] < 0
    # Noise: 89/100 vs 90/100 -- the interval must straddle zero.
    lo, hi = newcombe_difference(89, 100, 90, 100)
    assert lo < 0 < hi
    # And it must have more POWER than a non-overlap rule: 160/200 vs 180/200 (80% vs 90%)
    # is a real drop that overlapping-CIs would miss.
    assert newcombe_difference(160, 200, 180, 200)[1] < 0


def test_run_suite_warns_when_metrics_cannot_score_a_goldless_task():
    task = TaskSpec("t1", input="x", gold=None)
    with pytest.warns(UserWarning, match="no `gold`"):
        run_suite(as_agent(str.upper), [task], k=1, check=lambda e, g: True,
                  metrics={"tool_call": ToolCallMetric()})


def test_run_suite_value_weights_the_success_rate():
    """TaskSpec.value: not all completed tasks are worth the same."""
    tasks = [
        TaskSpec("cheap", input="a", gold="A", value=1.0),
        TaskSpec("precious", input="b", gold="B", value=99.0),
    ]
    # fails only the high-value task
    agent = lambda t: Episode(output="A" if t.task_id == "cheap" else "WRONG")  # noqa: E731
    report = run_suite(agent, tasks, k=1)
    assert report.success_rate == 0.5
    assert report.detail["value_weighted_success"] < 0.02  # the valuable one failed


# ---------------------------------------------------------------------------
# Facade integration: agent metrics resolve through the registry
# ---------------------------------------------------------------------------


def test_agent_metrics_resolve_by_name_through_the_score_facade():
    ep = Episode(task_id="t", trajectory=Trajectory([Step("s", {"q": "cats"})]))
    gold = [("s", {"q": "cats"})]
    assert ek.score(ep, gold, metric="tool_call").f1 == 1.0
    assert ek.score(ep, gold, metric="trajectory").value == 0.0
    assert ek.score(Episode(output="42"), "42", metric="task_success").value == 1.0


def test_evaluate_aggregates_agent_metrics_correctly():
    cases = [
        (Episode(output="a"), "a"),
        (Episode(output="b"), "c"),
        (Episode(output="d"), "d"),
    ]
    report = ek.evaluate(cases, metric="task_success")
    assert report.aggregate == pytest.approx(2 / 3)  # the success rate, not a mean of means
    assert report.n == 3


def test_cost_per_success_metric_requires_an_episode():
    m = CostPerSuccessMetric(price=per_million(1.0, 1.0))
    with pytest.raises(TypeError, match="Episode"):
        m("just a string", "gold")


def test_cost_per_success_by_name_fails_with_an_actionable_message():
    """Resolved by name it has no rates -- say the real thing, not 'no price for model ""'."""
    ep = Episode(task_id="a", output="x", cost=Cost(input_tokens=10))
    with pytest.raises(UnknownModelPrice, match="cannot be used by name alone"):
        ek.score(ep, "x", metric="cost_per_success")


# ---------------------------------------------------------------------------
# Edge cases the first round of tests missed entirely
# ---------------------------------------------------------------------------


def test_reliability_warns_and_excludes_when_k_exceeds_the_trial_count():
    """A PERFECT agent must not be reported as a total failure just because k > n."""
    with pytest.warns(UserWarning, match="fewer than k"):
        r = reliability({"t1": [True] * 4, "t2": [True] * 4}, k=8)
    assert r.n_tasks == 0
    assert r.pass_hat_k is None, "None ('not measurable') -- never 0.0 ('it failed everything')"
    assert math.isnan(float(r))


def test_reliability_partial_skip_is_not_silent():
    with pytest.warns(UserWarning, match="fewer than k"):
        r = reliability({"t1": [True] * 8, "t2": [True] * 4}, k=8)
    assert r.n_tasks == 1 and r.detail["skipped"] == ["t2"]


def test_cost_report_refuses_ungraded_episodes():
    """An ungraded episode is not a failed one -- do not silently report an infinite cost."""
    with pytest.raises(ValueError, match="ungraded"):
        cost_report([Episode(task_id="a", cost=Cost(input_tokens=1))],
                    price=per_million(1.0, 1.0))


def test_cost_sums_with_builtin_sum():
    """sum() starts from int 0, so a naive __radd__ = __add__ would raise."""
    total = sum([Cost(input_tokens=1), Cost(input_tokens=2), Cost(input_tokens=3)])
    assert total.input_tokens == 6


def test_trajectory_is_immutable_even_if_the_caller_mutates_their_list():
    steps = [Step("a", {})]
    traj = Trajectory(steps)
    steps.append(Step("b", {}))
    assert len(traj) == 1, "a frozen Trajectory must not grow when the caller's list does"


def test_graded_does_not_share_meta_with_the_original():
    ep = Episode(task_id="t", meta={"k": 1})
    graded = ep.graded(True)
    graded.meta["k"] = 2
    assert ep.meta["k"] == 1, "replace() is a shallow copy -- meta must be copied"


def test_missing_argument_is_not_free_against_a_gold_none():
    """A skipped argument must not compare equal to a gold argument whose value IS None."""
    m = TrajectoryMetric()
    assert m([Step("s", {})], [("s", {"x": None})]).value > 0.0
    assert m([Step("s", {"x": None})], [("s", {})]).value > 0.0  # nor a hallucinated one


def test_match_calls_fast_path_does_not_change_results():
    """The no-repeats fast path must be a PURE optimisation.

    Regression test: an earlier version took the fast path whenever *gold* had no repeated tool
    name -- but with gold=[s(q=a)] and pred=[s(q=z), s(q=a)] there is still a real choice, and
    first-come handed the gold call to the WRONG pred, discarding the exact match (F1 0.0).
    The assignment is only forced when NEITHER side repeats a tool name.
    """
    gold = [("s", {"q": "a"})]
    pred = [("s", {"q": "z"}), ("s", {"q": "a"})]  # the EXACT match comes second
    pairs, extra, _missed = match_calls(pred, gold)
    assert pairs == [(("s", {"q": "a"}), ("s", {"q": "a"}))], "exact match was discarded"
    assert extra == [("s", {"q": "z"})]
    # 1 TP, 1 spurious FP, 0 FN -> P=0.5, R=1.0
    assert ToolCallMetric()(pred, gold).f1 == pytest.approx(2 / 3)


def test_match_calls_conservation_law():
    """Every predicted call is matched exactly once or unmatched -- never both, never twice."""
    rng = random.Random(11)
    tools, vals = ["a", "b", "c"], ["x", "y", "z"]
    for _ in range(200):
        gen = lambda: [  # noqa: E731
            (rng.choice(tools), {"q": rng.choice(vals)}) for _ in range(rng.randint(0, 5))
        ]
        pred, gold = gen(), gen()
        pairs, extra, missed = match_calls(pred, gold)
        assert len(pairs) + len(extra) == len(pred)
        assert len(pairs) + len(missed) == len(gold)


def test_match_calls_does_not_mis_assign_on_partial_overlap():
    """Global best-first: a strong pairing must not be stolen by a weaker earlier one."""
    gold = [("s", {"a": 1, "b": 1}), ("s", {"a": 1})]
    pred = [("s", {"a": 1, "z": 1}), ("s", {"a": 1, "b": 1, "z": 1})]
    pairs, extra, missed = match_calls(pred, gold)
    assert not extra and not missed
    # A pair is (pred_call, gold_call), each being (tool, args).
    # pred[1] {a,b,z} agrees with gold[0] {a,b} on BOTH a and b -> it must claim gold[0];
    # a naive in-order greedy would hand gold[0] to pred[0] (which agrees only on `a`).
    for (_p_tool, p_args), (_g_tool, g_args) in pairs:
        if g_args == {"a": 1, "b": 1}:
            assert p_args == {"a": 1, "b": 1, "z": 1}, "strong pairing was stolen by a weaker one"
            break
    else:
        pytest.fail("gold call {a,b} was never matched")


def test_judge_reads_a_none_output_rather_than_the_episode_object():
    seen = []
    sig = JudgeSignal(lambda out, *, criteria="": seen.append(out) or 0.0)
    sig(Episode(output=None))
    assert seen == [None], "an empty answer must reach the judge as None, not as an Episode"


def test_empty_trajectories_and_suites_do_not_explode():
    assert TrajectoryMetric()([], []).value == 0.0
    assert ToolCallMetric()([], []).f1 == 1.0  # nothing to call, nothing called
    empty = run_suite(as_agent(str.upper), [], k=1)
    assert empty.n_trials == 0 and empty.pass_hat_k is None
