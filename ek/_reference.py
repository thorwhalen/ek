"""Frozen pre-port implementation of ``ek.qe.rover._align`` -- do not edit.

This is the oracle of the Rust port of ROVER's alignment kernel: the differential
tests (``tests/test_differential.py``) compare every backend of
:func:`ek.qe.rover._align` against it, the benchmark (``benchmarks/bench_port.py``)
times it as the "before" column, and the facade's ``backend="python"`` path and the
native path's fallback (inputs outside the native domain) both call it.

It is a verbatim copy of ``_align`` from ``ek/qe/rover.py`` at commit ``70f1019``
(ek 0.1.15). Never "fix" it silently: a deliberate behaviour change is a new test
asserting the new behaviour, applied to every backend, plus a changelog entry.
See ``misc/docs/port-log.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Tuple


def _align(
    spine: Sequence[Optional[str]], hyp: Sequence[str]
) -> List[Tuple[str, int, int]]:
    """Needleman-Wunsch alignment of a network ``spine`` to a hypothesis token list.

    Returns operations in left-to-right order, each ``(op, s_idx, h_idx)`` where
    ``op`` is ``"match"`` (tokens equal or substituted onto the same slot),
    ``"del"`` (a spine slot the hypothesis skips -> NULL), or ``"ins"`` (a
    hypothesis token with no spine slot -> a new slot). Pure-Python so the engine
    runs in a bare environment and over arbitrary (non-string) tokens.
    """
    s, t = len(spine), len(hyp)
    # dp[i][j] = min edit cost aligning spine[:i] with hyp[:j].
    dp = [[0] * (t + 1) for _ in range(s + 1)]
    for i in range(1, s + 1):
        dp[i][0] = i
    for j in range(1, t + 1):
        dp[0][j] = j
    for i in range(1, s + 1):
        for j in range(1, t + 1):
            sub = dp[i - 1][j - 1] + (0 if spine[i - 1] == hyp[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    ops: List[Tuple[str, int, int]] = []
    i, j = s, t
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if spine[i - 1] == hyp[j - 1] else 1)
        ):
            ops.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))  # spine slot, hypothesis has no token
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))  # hypothesis token, no spine slot
            j -= 1
    ops.reverse()
    return ops
