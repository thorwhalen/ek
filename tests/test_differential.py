"""Differential tests: every ``_align`` backend must behave exactly like the frozen reference.

Built from the ``pyo3-port`` differential-test template. The port under test is
``ek.qe.rover._align(..., backend=BACKEND)`` -- the facade, not the extension -- and
``rover()`` end to end with its alignment backend switched; the oracle is the frozen
``ek._reference._align`` (and ``rover()`` on the Python backend).

``BACKEND`` is ``"rust"`` when the optional ``ek-rs`` extension is importable, else
``"python"`` (then the run checks the facade's routing against the reference, which
is all there is to check). Set ``EK_DIFF_BACKEND=rust`` to *require* the native
backend (CI does), so a missing or stale build fails instead of skipping.

Each call compares the *outcome*: the return value or the exception class, plus the
arguments as they look afterwards. Values compare by type as well as by value.

Profiles: ``fast`` (default; deterministic, for every PR) and ``thorough`` (before
merging a step and after any divergence)::

    HYPOTHESIS_PROFILE=thorough pytest tests/test_differential.py
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import importlib
import importlib.util
import itertools
import math
import os
import string

import pytest
from hypothesis import example, given, note, settings
from hypothesis import strategies as st

# isort: split
from ek import _reference as ref  # the frozen pre-port implementation

rover_mod = importlib.import_module("ek.qe.rover")  # the module, not the function

settings.register_profile("fast", max_examples=300, derandomize=True, deadline=None)
settings.register_profile("thorough", max_examples=5000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))

HAS_NATIVE = importlib.util.find_spec("ek_rs") is not None
BACKEND = os.environ.get("EK_DIFF_BACKEND", "rust" if HAS_NATIVE else "python")
port_align = functools.partial(rover_mod._align, backend=BACKEND)
port_align.__name__ = "_align"


# --- Strategies seeded with the Python -> Rust traps --------------------------------

TRAP_CHARS = (
    "\x1c\x1d\x1e\x1f\x85\xa0 "  # whitespace that Python and Rust classify differently
    "_‿⁀"  # connector punctuation: _ ‿ ⁀
    "́ͅि"  # combining marks: Mn, Mn with Other_Alphabetic, Mc
    "²½①Ⅻ"  # non-decimal numerics: ² ½ ① Ⅻ
    "İΣßẞǅ"  # case traps: İ Σ ß ẞ ǅ
    "一\U0001f600"  # CJK, emoji (4-byte UTF-8)
)


def py_text(*, max_size: int = 40, surrogates: bool = True) -> st.SearchStrategy[str]:
    """Python ``str`` values, including what a Rust ``&str`` cannot hold (lone surrogates).

    Half are free text; half are short tokens joined by separators, so that a single
    trap character often stands alone as a whole word or field.
    """
    any_char = st.characters(exclude_categories=() if surrogates else ("Cs",))
    common = st.sampled_from(string.ascii_letters + string.digits + " " + TRAP_CHARS)
    free = st.text(st.one_of(common, common, any_char), max_size=max_size)
    token = st.one_of(
        st.sampled_from(TRAP_CHARS),
        st.text(st.one_of(common, any_char), min_size=1, max_size=4),
    )
    joined = st.lists(token, max_size=max(1, max_size // 4)).flatmap(
        lambda toks: st.sampled_from([" ", ",", "\t", ""]).map(
            lambda sep: sep.join(toks)
        )
    )
    return st.one_of(free, joined)


def py_ints(*, small: tuple[int, int] = (-2, 8)) -> st.SearchStrategy[int]:
    """Mostly small ints (where sizes and thresholds interact with short inputs), plus the
    64-bit edges and unbounded Python ints."""
    edges = [0, -1, 2**31 - 1, 2**31, 2**63 - 1, 2**63, 2**64 - 1, 2**64, -(2**63) - 1]
    return st.one_of(
        st.integers(*small), st.integers(-3, 100), st.sampled_from(edges), st.integers()
    )


def py_floats() -> st.SearchStrategy[float]:
    """Floats with NaN, infinities, signed zero and subnormals."""
    return st.one_of(
        st.floats(), st.sampled_from([0.0, -0.0, math.nan, math.inf, -math.inf, 5e-324])
    )


# --- Outcome comparison ---------------------------------------------------------------


def same(a, b, *, rel: float = 0.0, abs_: float = 0.0) -> bool:
    """Structural equality that also compares types; floats may get a tolerance."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if a == b:
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            same(x, y, rel=rel, abs_=abs_) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict):  # insertion order is part of a dict's observable behaviour
        return list(a) == list(b) and all(
            same(a[k], b[k], rel=rel, abs_=abs_) for k in a
        )
    return a == b


def close(*, rel: float = 1e-12, abs_: float = 0.0):
    """A ``compare`` for float results that may differ in the last bits (exp/log, sums)."""
    return lambda a, b: same(a, b, rel=rel, abs_=abs_)


def outcome(fn, args: tuple, kwargs: dict, *, messages: bool = False):
    """Run ``fn`` on copies of the inputs: ('ok', value, args after) or ('err', class, args after)."""
    args, kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
    try:
        result = ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - exception classes are part of the API
        result = ("err", type(exc), str(exc)) if messages else ("err", type(exc))
    return (*result, args, kwargs)


_NO_ARGS, _NO_KWARGS = st.just(()), st.just({})


def differential(
    ref_fn,
    port_fn,
    /,
    *,
    args=_NO_ARGS,
    kwargs=_NO_KWARGS,
    compare=same,
    messages=False,
    examples=(),
):
    """Build a Hypothesis test asserting ``port_fn`` behaves like ``ref_fn`` on generated inputs.

    ``examples`` is the seed corpus: ``(args, kwargs)`` pairs always run first. Add every
    divergence found during the port, and any trap too narrow for random generation to hit.
    """

    def test(args, kwargs):
        expected = outcome(ref_fn, args, kwargs, messages=messages)
        actual = outcome(port_fn, args, kwargs, messages=messages)
        note(f"reference: {expected[:-2]!r}")
        note(f"port:      {actual[:-2]!r}")
        if expected[0] == actual[0] == "ok":
            assert compare(actual[1], expected[1]), "results differ"
        else:
            assert actual[:-2] == expected[:-2], (
                "outcomes differ (value vs exception, or exception class)"
            )
        assert same(actual[-2:], expected[-2:]), "arguments were mutated differently"

    for ex_args, ex_kwargs in examples:
        test = example(args=ex_args, kwargs=ex_kwargs)(test)
    test = given(args=args, kwargs=kwargs)(test)
    test.__qualname__ = test.__name__ = (
        f"test_{getattr(ref_fn, '__name__', 'fn')}_matches_reference"
    )
    return test


def trap_grid(ref_fn, port_fn, /, *, args, kwargs=({},), compare=same, messages=False):
    """Build a parametrised test over every ``args`` x ``kwargs`` combination.

    Random generation rarely hits a conjunction (one trap character *and* one exact
    parameter value), so pair the Hypothesis test with this deterministic grid: every
    trap character alone and inside a word, times the small parameter values.
    """
    cases = [(a, k) for a in args for k in kwargs]

    @pytest.mark.parametrize(("case_args", "case_kwargs"), cases)
    def test(case_args, case_kwargs):
        expected = outcome(ref_fn, case_args, case_kwargs, messages=messages)
        actual = outcome(port_fn, case_args, case_kwargs, messages=messages)
        if expected[0] == actual[0] == "ok":
            assert compare(actual[1], expected[1]), (actual[:-2], expected[:-2])
        else:
            assert actual[:-2] == expected[:-2], (actual[:-2], expected[:-2])
        assert same(actual[-2:], expected[-2:]), "arguments were mutated differently"

    test.__qualname__ = test.__name__ = (
        f"test_{getattr(ref_fn, '__name__', 'fn')}_trap_grid"
    )
    return test


def trap_strings() -> list[str]:
    """Each trap character alone, inside a word, and doubled: inputs for ``trap_grid``."""
    return [
        "",
        *TRAP_CHARS,
        *(f"ab{c}cd" for c in TRAP_CHARS),
        *(c + c for c in TRAP_CHARS),
    ]


# --- The native module must be the one under test -------------------------------------


@pytest.mark.skipif(BACKEND != "rust", reason="native backend not under test")
def test_native_extension_is_loaded():
    """Fails if a stale build, a fallback or the source tree is what got imported."""
    import ek_rs

    # maturin wraps a pure-Rust module in a package: ek_rs/ek_rs.<abi3>.so
    native = importlib.import_module("ek_rs.ek_rs")
    assert native.__file__.endswith((".so", ".pyd")), native.__file__
    assert ek_rs.align is native.align
    assert ek_rs.DEBUG_BUILD in (True, False)


# --- Inputs ---------------------------------------------------------------------------

# A tiny alphabet makes ties (the whole point) common; py_text tokens bring the text
# traps (surrogates, combining marks, case traps), which the facade interns in Python.
small_tok = st.sampled_from(["a", "b", "c"])
any_tok = st.one_of(small_tok, small_tok, py_text(max_size=6))
spine_tok = st.one_of(any_tok, any_tok, any_tok, st.none())


def seqs(tok, *, max_size=24):
    lists = st.lists(tok, max_size=max_size)
    return st.one_of(lists, lists, lists.map(tuple))


# Outside the native domain (exact ``str`` or ``None`` tokens in a list or tuple): the
# facade must route these to the reference, so the outcome is still identical.
class Loose(str):
    """A str subclass whose == ignores case: interning by hash would disagree."""

    def __eq__(self, other):
        return isinstance(other, str) and self.lower() == other.lower()

    __hash__ = str.__hash__


class Stream:
    """Iterable without ``len()`` (like a generator), comparable by contents."""

    def __init__(self, items):
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)

    def __eq__(self, other):
        return type(other) is Stream and other.items == self.items

    __hash__ = None


odd_tok = st.one_of(
    st.integers(-3, 3),  # hash(-1) == hash(-2)
    st.sampled_from([math.nan, 1.0, True, 0.0, -0.0, (), b"a", Loose("A"), Loose("b")]),
    small_tok,
)
odd_seq = st.one_of(
    st.lists(odd_tok, max_size=8),
    st.lists(small_tok, max_size=8).map(lambda xs: "".join(xs)),  # a str is a Sequence
    st.lists(small_tok, max_size=4).map(Stream),  # no len(): TypeError on every backend
    st.just(None),
)


# --- One line per ported function -----------------------------------------------------

test_align = differential(
    ref._align,
    port_align,
    args=st.tuples(seqs(spine_tok), seqs(st.one_of(any_tok, any_tok, st.none()))),
    examples=[
        ((["a"], ["a", "a"]), {}),  # rapidfuzz takes the other insertion
        ((["a", "b", "a"], ["b", "a", "b"]), {}),  # deletion beats insertion
        ((["a", "b"], ["b", "a"]), {}),  # diagonal beats both
        (([None, "a"], ["a"]), {}),
        (([None], [None, "a"]), {}),  # None == None: interned to one id
        ((["a", None], [None, None]), {}),
        (([], []), {}),
        ((["\ud800"], ["\ud800", "x"]), {}),  # a lone surrogate token
    ],
)
test_align_off_domain = differential(
    ref._align,
    port_align,
    args=st.tuples(st.one_of(odd_seq, seqs(spine_tok, max_size=6)), odd_seq),
    examples=[
        (([math.nan], [math.nan]), {}),  # nan != nan, though a dict would intern it
        (([-1], [-2]), {}),
        (([Loose("A")], ["a"]), {}),
        ((["a"], [Loose("A")]), {}),
        (([1], [1.0]), {}),
    ],
)
test_align_trap_grid = trap_grid(
    ref._align,
    port_align,
    args=[([a], [b, a]) for a in trap_strings() for b in ("", "ab")]
    + [([a, None], [a]) for a in trap_strings()],
)


def test_align_exhaustive_small_grid():
    """Every spine over {a, b, NULL} x hypothesis over {a, b}, lengths 0..5 (22 932 pairs)."""

    def words(alphabet, n):
        return [
            list(w) for k in range(n + 1) for w in itertools.product(alphabet, repeat=k)
        ]

    hyps = words("ab", 5)
    for spine in words(["a", "b", None], 5):
        for hyp in hyps:
            assert port_align(spine, hyp) == ref._align(spine, hyp), (spine, hyp)


def _rover_with(backend):
    def run(hyps, **kwargs):
        saved = rover_mod._ALIGN_BACKEND
        rover_mod._ALIGN_BACKEND = backend
        try:
            return dataclasses.asdict(rover_mod.rover(hyps, **kwargs))
        finally:
            rover_mod._ALIGN_BACKEND = saved

    run.__name__ = f"rover_{backend}"
    return run


words_vocab = st.sampled_from(["the", "cat", "sat", "bat", "sit", "on", "mat", "a"])
hyp_text = st.lists(words_vocab, max_size=12).map(" ".join)
hyp_pairs = st.lists(
    st.tuples(words_vocab, st.floats(0, 1, allow_nan=False)), max_size=12
)

test_rover_end_to_end = differential(
    _rover_with("python"),
    _rover_with(BACKEND),
    args=st.tuples(st.lists(st.one_of(hyp_text, hyp_pairs), max_size=5)),
    kwargs=st.fixed_dictionaries(
        {},
        optional={
            "use_confidence": st.booleans(),
            "conf_weight": st.sampled_from([0.0, 0.3, 0.5, 1.0]),
            "null_conf": st.sampled_from([0.0, 0.7, 1.0]),
        },
    ),
    examples=[
        ((["the cat sat", "the cat sit", "the bat sat"],), {}),
        ((["a", "a a"],), {}),
        ((["x y", "y x y", "x y"],), {}),
    ],
)
