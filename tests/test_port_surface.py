"""Characterization tests pinning ROVER's tie behaviour: the port's safety net.

These pin what ``ek.qe.rover`` does *today* where several answers are equally good,
because that is exactly where a faithful-looking re-implementation (rapidfuzz, a Rust
kernel, a "cleaner" DP) silently changes the consensus:

- the alignment backtrace (``_align``) walks from the bottom-right corner and, among
  equal-cost moves, prefers the **diagonal** (match/substitution), then the
  **deletion** (spine slot -> NULL), then the **insertion** (new slot);
- the slot vote (``_vote``) maximises ``(score, real-over-NULL, votes, -first_seen)``.

Every alignment test runs on each ``_align`` backend; the native one is skipped when
the optional ``ek-rs`` extension is not installed. See ``misc/docs/port-log.md``.
"""

import functools
import importlib
import importlib.util

import pytest

# ``ek.qe.rover`` resolves to the function (ek.qe re-exports it), so import the module
rover_mod = importlib.import_module("ek.qe.rover")
from ek.qe.rover import _vote, rover  # noqa: E402

HAS_NATIVE = importlib.util.find_spec("ek_rs") is not None
BACKENDS = [
    "python",
    pytest.param(
        "rust", marks=pytest.mark.skipif(not HAS_NATIVE, reason="ek-rs not installed")
    ),
]


def _aligner(backend):
    return functools.partial(rover_mod._align, backend=backend)


# (spine, hyp, expected ops) -- outputs of ek 0.1.15, checked by hand.
TIE_CASES = [
    ([], [], []),
    (["a"], [], [("del", 0, -1)]),
    ([], ["a"], [("ins", -1, 0)]),
    (["a"], ["a"], [("match", 0, 0)]),
    # a substitution (cost 1) beats delete + insert (cost 2)
    (["a"], ["b"], [("match", 0, 0)]),
    # one extra token: the backtrace consumes the diagonal last-to-last first, so the
    # *leading* token becomes the insertion (rapidfuzz picks the trailing one)
    (["a"], ["a", "a"], [("ins", -1, 0), ("match", 0, 1)]),
    (["a", "a"], ["a"], [("del", 0, -1), ("match", 1, 0)]),
    (["x", "y"], ["y", "x", "y"], [("ins", -1, 0), ("match", 0, 1), ("match", 1, 2)]),
    # diagonal (two substitutions) preferred over del+ins, all three cost 2
    (["a", "b"], ["b", "a"], [("match", 0, 0), ("match", 1, 1)]),
    (["a", "b"], ["b", "c"], [("match", 0, 0), ("match", 1, 1)]),
    (
        ["a", "a", "b"],
        ["a", "b", "a"],
        [("match", 0, 0), ("match", 1, 1), ("match", 2, 2)],
    ),
    # at the corner diagonal is not optimal but deletion and insertion tie: deletion wins
    (
        ["a", "b", "a"],
        ["b", "a", "b"],
        [("ins", -1, 0), ("match", 0, 1), ("match", 1, 2), ("del", 2, -1)],
    ),
    (
        ["a", "b", "c"],
        ["c", "a", "b"],
        [("ins", -1, 0), ("match", 0, 1), ("match", 1, 2), ("del", 2, -1)],
    ),
    # a NULL spine representative is just a token that equals nothing in the hypothesis
    ([None], ["a"], [("match", 0, 0)]),
    ([None, "a"], ["a"], [("del", 0, -1), ("match", 1, 0)]),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("spine", "hyp", "expected"), TIE_CASES)
def test_align_tie_preference_is_pinned(backend, spine, hyp, expected):
    assert _aligner(backend)(spine, hyp) == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_align_returns_list_of_str_int_int_tuples(backend):
    ops = _aligner(backend)(["a", "b"], ["b"])
    assert type(ops) is list
    for op in ops:
        assert type(op) is tuple and len(op) == 3
        assert type(op[0]) is str and type(op[1]) is int and type(op[2]) is int


@pytest.mark.parametrize("backend", BACKENDS)
def test_align_accepts_any_sequence_of_equal_comparable_tokens(backend):
    align = _aligner(backend)
    # non-str tokens use ==, so 1 == 1.0 == True match one another
    assert align([1, 2], [1.0, True]) == [("match", 0, 0), ("match", 1, 1)]
    assert align((1,), (True, 1)) == [("ins", -1, 0), ("match", 0, 1)]
    assert align("ab", "b") == [("del", 0, -1), ("match", 1, 0)]
    # hash(-1) == hash(-2) in CPython: any hash-based equality would confuse them
    assert align([-1], [-2]) == [("match", 0, 0)]
    assert align([-1, -2], [-2]) == [("del", 0, -1), ("match", 1, 0)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_align_rejects_what_it_rejected(backend):
    align = _aligner(backend)
    with pytest.raises(TypeError):
        align(iter(["a"]), ["a"])  # len() of an iterator
    with pytest.raises(TypeError):
        align(["a"], None)


def test_vote_prefers_a_real_token_over_null_on_equal_score():
    # 'a' and NULL both score 0.5 * 1/2 + 0.5 * 0.7 = 0.6; NULL comes first.
    entries = [(None, 0.0), ("a", 0.7)]
    winner, score, share = _vote(entries, n_engines=2, conf_weight=0.5, null_conf=0.7)
    assert (winner, share) == ("a", 0.5)
    assert score == pytest.approx(0.6)


def test_vote_prefers_more_votes_on_equal_score():
    # confidence-only scoring: 'b' (1 vote) and 'a' (2 votes) both score 0.5
    entries = [("b", 0.5), ("a", 0.5), ("a", 0.5)]
    winner, score, share = _vote(entries, n_engines=3, conf_weight=1.0, null_conf=0.7)
    assert (winner, score, share) == ("a", 0.5, 2 / 3)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [([("a", 1.0), ("b", 1.0)], "a"), ([("b", 1.0), ("a", 1.0)], "b")],
)
def test_vote_prefers_first_seen_on_full_tie(entries, expected):
    winner, _, _ = _vote(entries, n_engines=2, conf_weight=0.5, null_conf=0.7)
    assert winner == expected


@pytest.mark.parametrize(
    ("hyps", "slot_entries", "tokens"),
    [
        (
            ["a", "a a"],
            [[(None, 0.0), ("a", 1.0)], [("a", 1.0), ("a", 1.0)]],
            ["a", "a"],
        ),
        (
            ["a a", "a"],
            [[("a", 1.0), (None, 0.0)], [("a", 1.0), ("a", 1.0)]],
            ["a", "a"],
        ),
        (
            ["a b", "b a"],
            [[("a", 1.0), ("b", 1.0)], [("b", 1.0), ("a", 1.0)]],
            ["a", "b"],
        ),
        (
            ["x y", "y x y", "x y"],
            [
                [(None, 0.0), ("y", 1.0), (None, 0.0)],
                [("x", 1.0), ("x", 1.0), ("x", 1.0)],
                [("y", 1.0), ("y", 1.0), ("y", 1.0)],
            ],
            ["x", "y"],
        ),
    ],
)
def test_rover_network_under_ties_is_pinned(hyps, slot_entries, tokens):
    c = rover(hyps)
    assert [s.entries for s in c.slots] == slot_entries
    assert c.tokens == tokens


def test_align_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown _align backend 'fortran'"):
        rover_mod._align(["a"], ["a"], backend="fortran")


def test_align_rust_backend_without_ek_rs_names_the_package(monkeypatch):
    from ek.registry import MissingExtraError

    monkeypatch.setitem(__import__("sys").modules, "ek_rs", None)  # import fails
    with pytest.raises(MissingExtraError, match=r"ek-rs") as info:
        rover_mod._align(["a"], ["a"], backend="rust")
    # the install hint is the extra, which pulls in the ek-rs wheel
    assert "ek[rust]" in str(info.value)
    assert isinstance(info.value.__cause__, ImportError)  # the real cause is kept


def test_rover_default_backend_is_python():
    assert (
        rover_mod._ALIGN_BACKEND == "python"
        or "EK_ROVER_ALIGN_BACKEND" in __import__("os").environ
    )
    assert rover_mod._align.__kwdefaults__ == {"backend": "python"}


def test_align_rust_backend_off_domain_needs_no_extension(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "ek_rs", None)  # import fails
    assert rover_mod._align([1], [1.0], backend="rust") == [("match", 0, 0)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_align_none_tokens_on_both_sides(backend):
    assert _aligner(backend)([None], [None, "a"]) == [("match", 0, 0), ("ins", -1, 1)]


def test_a_bad_backend_environment_variable_fails_at_import():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", "import ek"],
        env={**__import__("os").environ, "EK_ROVER_ALIGN_BACKEND": "Rust"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "EK_ROVER_ALIGN_BACKEND='Rust'" in proc.stderr
