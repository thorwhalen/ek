"""Tests for the ``pass^k`` standard error used by the agent regression gate (issue #32).

The gate's ``pass^k`` arm (k > 1) used to back-derive each run's SE from its bootstrap CI. That
under-estimates near p = 1 (a truncated percentile interval) and reads a *perfect* run's zero-width
CI as certainty. It now uses the per-task sample SD persisted in the summary,
``SE = sd / sqrt(n_tasks)``, floored Agresti--Coull-style near the boundary. These tests pin:

- the SD is computed over the per-task estimators and persisted (JSON-safe) with ``n_tasks``;
- in the interior the SE is the plain ``sd / sqrt(n)``;
- at the boundary (p -> 1, a perfect baseline) the SE is positive and equals Agresti--Coull's;
- the gate still has power, and old summaries without the SD still work (CI fallback).
"""

import json
import math
import random
import warnings

import pytest

import ek
from ek.agents import (
    Cost,
    Episode,
    TaskSpec,
    agent_regression_gate,
    as_agent,
    mean_standard_error,
    reliability,
    run_suite,
    save_agent_baseline,
)

Z = 1.96


def _quiet(fn, *args, **kwargs):
    """Run ``fn`` with the deterministic/underpowered warnings silenced (they are not under test)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _flaky_agent(p_fail: float, seed: int):
    rng = random.Random(seed)

    def agent(task):
        wrong = rng.random() < p_fail
        return Episode(output=("WRONG" if wrong else str(task.gold)), cost=Cost(input_tokens=1))

    return agent


def _tasks(n: int) -> list:
    return [TaskSpec(f"t{i}", input="a", gold="A") for i in range(n)]


def _one_flaky_task(task):
    return Episode(output=("WRONG" if task.task_id == "t3" else "A"))


def test_pass_hat_k_sd_is_persisted_and_gives_the_plain_se_in_the_interior(tmp_path):
    rng = random.Random(0)
    grouped = {f"t{i}": [rng.random() < 0.5 for _ in range(4)] for i in range(200)}
    r = _quiet(reliability, grouped, k=2)
    hats = [v["pass_hat_k"] for v in r.per_task.values()]
    m = sum(hats) / len(hats)
    sd = math.sqrt(sum((h - m) ** 2 for h in hats) / (len(hats) - 1))
    assert r.pass_hat_k_sd == pytest.approx(sd)

    summary = save_agent_baseline(r, "base", rootdir=str(tmp_path))
    assert summary["pass_hat_k_sd"] == pytest.approx(sd) and summary["n_tasks"] == 200
    assert json.loads(json.dumps(summary))["pass_hat_k_sd"] == pytest.approx(sd)

    # Away from the boundary the Agresti-Coull floor barely moves the plain SE.
    se = r.pass_hat_k_se()
    assert se == pytest.approx(sd / math.sqrt(200), rel=0.05)
    assert se == mean_standard_error(r.pass_hat_k, sd, 200)


def test_single_task_has_no_sample_sd_but_still_an_se():
    r = _quiet(reliability, {"t1": [True, False, True]}, k=2)
    assert r.pass_hat_k_sd is None
    assert r.pass_hat_k_se() > 0


def test_mean_standard_error_is_agresti_coull_at_the_boundary():
    """p -> 1: a zero sample SD must not mean zero uncertainty; for n/n it IS Agresti-Coull."""
    for n in (10, 60, 200):
        se = mean_standard_error(1.0, 0.0, n)
        p_ac = (n + Z * Z / 2) / (n + Z * Z)
        ac = math.sqrt(p_ac * (1 - p_ac) / (n + Z * Z))
        assert se > 0
        assert se == pytest.approx(ac, rel=0.05)
    assert mean_standard_error(0.0, 0.0, 60) == pytest.approx(mean_standard_error(1.0, 0.0, 60))
    assert mean_standard_error(1.0, 0.0, 200) < mean_standard_error(1.0, 0.0, 60)


def test_mean_standard_error_floor_bites_only_near_the_boundary():
    # 0/1 data with one failure in 60: the plain SE is too small; the floor raises it.
    vals = [1.0] * 59 + [0.0]
    m = sum(vals) / 60
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / 59)
    assert mean_standard_error(m, sd, 60) > sd / math.sqrt(60)
    # Interior: within a few percent of the plain SE.
    assert mean_standard_error(0.5, 0.3, 200) == pytest.approx(0.3 / math.sqrt(200), rel=0.03)


def test_mean_standard_error_with_nothing_to_estimate():
    assert mean_standard_error(None, 0.1, 10) is None
    assert mean_standard_error(0.5, 0.1, 0) is None
    assert mean_standard_error(0.5, None, 10) > 0  # no SD (one unit): the floor alone


def test_gate_perfect_baseline_at_k_gt_1_does_not_fake_certainty(tmp_path):
    """A perfect baseline has sd == 0 and a zero-width CI; its SE must still be positive.

    Otherwise one flaky task against it reads as a regression (interval-vs-point, the very bug
    the gate exists to avoid).
    """
    tasks = _tasks(40)
    perfect = _quiet(run_suite, as_agent(str.upper), tasks, k=2)
    assert perfect.pass_hat_k == 1.0 and perfect.pass_hat_k_sd == 0.0
    assert perfect.pass_hat_k_se() > 0
    save_agent_baseline(perfect, "base", rootdir=str(tmp_path))

    cur = _quiet(run_suite, _one_flaky_task, tasks, k=2)
    gate = _quiet(agent_regression_gate, cur, "base", rootdir=str(tmp_path))
    assert gate.passed, f"one flaky task vs a perfect baseline must not fail: {gate.reasons}"


def test_gate_still_catches_a_pass_hat_k_regression_at_k_gt_1(tmp_path):
    """The floor must not blind the gate: a large consistency drop is still caught."""
    tasks = _tasks(100)
    good = _quiet(run_suite, _flaky_agent(p_fail=0.02, seed=1), tasks, k=4)
    save_agent_baseline(good, "base", rootdir=str(tmp_path))
    worse = _quiet(run_suite, _flaky_agent(p_fail=0.25, seed=2), tasks, k=4)
    gate = _quiet(agent_regression_gate, worse, "base", rootdir=str(tmp_path))
    assert not gate.passed
    assert any("pass^4 regressed" in r for r in gate.reasons)


def test_gate_accepts_legacy_baselines_without_pass_hat_k_sd(tmp_path):
    """Summaries written before #32 carry no per-task SD: fall back to the CI-derived SE."""
    tasks = _tasks(100)
    good = _quiet(run_suite, _flaky_agent(p_fail=0.02, seed=1), tasks, k=4)
    record = save_agent_baseline(good, "tmp", rootdir=str(tmp_path))
    legacy = {k: v for k, v in record.items() if k != "pass_hat_k_sd"}
    ek.json_store("baselines", rootdir=str(tmp_path))["legacy"] = legacy

    same = _quiet(run_suite, _flaky_agent(p_fail=0.02, seed=3), tasks, k=4)
    assert _quiet(agent_regression_gate, same, "legacy", rootdir=str(tmp_path)).passed

    worse = _quiet(run_suite, _flaky_agent(p_fail=0.25, seed=2), tasks, k=4)
    gate = _quiet(agent_regression_gate, worse, "legacy", rootdir=str(tmp_path))
    assert any("pass^4 regressed" in r for r in gate.reasons)


def test_gate_floors_a_legacy_perfect_baseline(tmp_path):
    """A legacy PERFECT baseline (zero-width CI, no SD) is floored, not treated as certain."""
    tasks = _tasks(40)
    perfect = _quiet(run_suite, as_agent(str.upper), tasks, k=2)
    record = save_agent_baseline(perfect, "p", rootdir=str(tmp_path))
    legacy = {k: v for k, v in record.items() if k != "pass_hat_k_sd"}
    assert legacy["pass_hat_k_ci"][0] == legacy["pass_hat_k_ci"][1] == 1.0
    cur = _quiet(run_suite, _one_flaky_task, tasks, k=2)
    gate = _quiet(agent_regression_gate, cur, legacy)
    assert gate.passed, gate.reasons
