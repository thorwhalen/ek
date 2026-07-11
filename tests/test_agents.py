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

import math
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
    judge_validation,
    load_prices,
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
