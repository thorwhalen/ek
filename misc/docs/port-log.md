# Port log: `ek.qe.rover._align` to Rust

The state of the port, kept where the next session will find it. Method: the
`pyo3-port` skill (oracle first, one function per step, each gated on a
differential test), with `rust-port-assess` for the assessment and the publish
rule. Status: **landing owed**. A cloud session did the port and opened the PR. The
package's own session must land it (see [Landing](#landing-owed)). (It lives in
`misc/docs/` rather than `docs/` because `docs/` is gitignored build output here.)

## Contents

- [1. Assessment](#1-assessment)
- [Gate: rapidfuzz](#gate-rapidfuzz)
- [2. Oracle](#2-oracle)
- [Decisions](#decisions)
- [Steps](#steps)
- [Benchmarks](#benchmarks)
- [Publish rule](#publish-rule)
- [Landing owed](#landing-owed)

## 1. Assessment

The assessment (2026-11-02, read-only) gave a score of 65 and this verdict: **do not
port for speed**, because no pipeline calls `_align` in a loop. It meets both parts of
the publish rule: it transfers well (interned ints in, integer ops out) and there is
probably no ROVER crate. The session's brief asked for the port anyway, to get the
crate, with the native path opt-in.

Checked cheaply on 2026-11-02 (`rust-port-assess` `hotspots.py ek --top 8`):

- `_align` (now `ek/_reference.py:21`) ranks #2 in the package (score 108: two nested
  loops, 24 subscripts). It is O(L x L') time and memory, a pure DP over a list of
  lists.
- Callers: `rover()`, i.e. `AgreementSignal` (`ek/qe/rover.py`),
  `estimate_quality(sources=...)` (`ek/facade.py:315`) and the `ek rover` CLI
  (`ek/tools.py:34`). None of them is called in a loop by any pipeline in the repo.
  The verdict stands.
- Gate 1 (an algorithmic fix in Python): none. The DP is already the textbook
  algorithm. A row-local rewrite would save a constant factor (about 2x, estimated)
  and nothing asymptotic. The Python version also uses about 200 MB at the
  5 000-token guard (a list of 5 001 lists of 5 001 ints). The Rust version keeps one
  byte per cell (25 MB).
- Gate 2 (depend on a library): rapidfuzz is already a dependency. It does not
  reproduce the ops (next section).
- Related DPs, not duplicates: `ek/metrics/tables.py:_align_pairs` (float scores,
  free indels) and `ek/agents/trajectory.py:_needleman_wunsch` (per-cell Python
  cost callbacks). They have a different shape, take callbacks, and are not ported.
- Determinism contract: the ops are not content-addressed or stored, but they decide
  the consensus. Contract: **bit-exact**, identical ops on every input.

## Gate: rapidfuzz

`rapidfuzz.distance.Levenshtein.opcodes` on token lists, expanded to `_align`'s
`("match"|"del"|"ins", s_idx, h_idx)` triples, was compared with `_align` on every
pair of strings over `{a, b, c}` up to length 5 (132 496 pairs), in four
orientations:

| orientation | diverging pairs |
|---|---|
| `opcodes(spine, hyp)` | 56 496 (43%) |
| `opcodes(hyp, spine)`, swapped back | 57 942 |
| both reversed, ops reversed back | 53 238 |
| reversed and swapped | 51 708 |

The first divergence is `spine=["a"], hyp=["a", "a"]`. `_align` gives
`[ins 0, match 0-1]`: its backtrace starts at the corner, prefers the diagonal, and
so turns the *leading* token into the insertion. rapidfuzz gives
`[match 0-0, ins 1]`. rapidfuzz's tie order is an internal detail of its bit-parallel
traceback, so no orientation reproduces it. It also compares non-`str` elements by
`hash()`, and in CPython `hash(-1) == hash(-2)`. **Gate failed: port.** The pinned
cases are in `tests/test_port_surface.py`.

## 2. Oracle

- **Baseline suite** (CPython 3.13.18, `uv pip install -e ".[dev]"` plus `[metrics]`,
  hypothesis and coverage): 449 passed, 0 failed, 5 skipped.
- **Port-surface coverage** (`port_coverage.py --target ek.qe.rover:_align`): 100%
  lines, 100% branches (14/14) **before** any new test. The existing
  `tests/test_rover.py` already reaches every branch. Nothing to explain.
- **Tie tests** (`tests/test_port_surface.py`). They pin the backtrace preference
  (diagonal, then deletion, then insertion; `rover.py:152-164` at 0.1.15), the vote's
  tie-breaking (`(score, real-over-NULL, votes, -first_seen)`, `rover.py:241`), the
  slot network `rover()` builds under ties, the return types, `==` semantics on
  non-`str` tokens (`1 == 1.0 == True`, and `-1` vs `-2`), and the `TypeError`s. They
  are parametrised over the `_align` backend.
- **Frozen reference**: `ek/_reference.py` is a verbatim copy of `_align` at
  `70f1019`. A test of `inspect.getsource` equality was run once, at the copy. The
  facade's Python backend now calls it, so the DP has one source.
- **Differential tests** (`tests/test_differential.py`, from the `pyo3-port`
  template). `_align` on generated spines and hypotheses: a small alphabet so ties are
  common, `py_text` tokens with surrogates and text traps, `None` in the spine, lists
  and tuples. An off-domain test: ints with `-1`/`-2`, `nan`, `1`/`1.0`/`True`,
  `-0.0`, bytes, a case-insensitive `str` subclass, a `str` as the sequence, an
  iterable with no `len()`, `None`. A trap grid. An exhaustive grid: every spine over
  `{a, b, NULL}` against every hypothesis over `{a, b}`, lengths 0 to 5
  (22 932 pairs). `rover()` end to end with the backend switched. Reference against
  reference: fast 249 passed, thorough 223 passed (5 000 examples per test).

## Decisions

1. **Tokens are interned in Python, the kernel sees ints.** The facade maps each
   distinct token to a small int with one dict, and passes two `list[int]` to the
   native `align`. Equality on exact `str` and `None` is the dict's equality, which is
   `==`. So the text traps (surrogates, Unicode versions, `\w`, case) cannot arise:
   no string crosses the boundary.
2. **Native domain: a `list` or `tuple` of exact `str`/`None` tokens, on both sides.**
   Everything else goes to the reference: other sequence types, iterables with no
   `len()`, `str` subclasses (they can override `==`), and non-`str` tokens (`nan`,
   `1` vs `1.0`, hash collisions). Interning by dict would disagree with `==` on
   those.
3. **Seam: a keyword-only `backend=` on `_align`, default `"python"`.** `rover()`'s
   signature does not change. It reads the private `_ALIGN_BACKEND`, set from
   `EK_ROVER_ALIGN_BACKEND` at import and `"python"` by default, so every `rover()`
   test can run on either backend.
4. **Packaging: a separate, optional distribution `ek-rs`, not a mixed package.**
   `layout-and-packaging.md` offers two shapes. The separate kernels package is the
   one for "the package must stay a pure-Python wheel, the native path is opt-in",
   and both hold here. The speed verdict is "do not port". The default stays Python.
   Turning `ek` itself into a maturin package would make every user install a
   compiled wheel (and need Rust for the sdist) for a function no pipeline calls hot.
   It would also force wads publishing off (i2mint/wads#166). So `ek`'s build,
   version and release are untouched. `rust/py` builds the extension module `ek_rs`
   with its own `pyproject.toml` (maturin). `ek` declares `ek[rust] = ["ek-rs"]` (not
   in `all`). The landing session may still choose the mixed package.
5. **The core keeps one byte per cell, not the cost matrix.** It stores each cell's
   backtrace direction, chosen with the reference's own preference (diagonal if it
   is optimal, else deletion if optimal, else insertion), plus two rows of costs. The
   backtrace then reads the directions. This is exactly what the reference's
   backtrace computes from the full matrix, at a quarter of the memory of a `u32`
   matrix. The differential tests decide.

## Steps

| step | status | notes |
|---|---|---|
| 1 assessment | done | above |
| 2 oracle | done | coverage, tie tests, frozen reference, differential tests, before benchmark |
| 3 layout | done | `rust/` workspace, `maturin develop -m rust/py/Cargo.toml` builds, `import ek_rs` loads |
| 4 skeleton | done | `ek_rs.align(spine, hyp, /)` raising `NotImplementedError`; facade routes `backend="rust"` to it. **Failing run, as required**: 239 failed / 29 passed in `test_differential.py` + `test_port_surface.py`. Every native-path case failed; the passes were the Python-backend params, the off-domain inputs (routed to the reference by design), and `test_native_extension_is_loaded` |

## Benchmarks

Conditions: Intel(R) Xeon(R) Processor @ 2.10GHz (model 207), 4 vCPU, Linux
x86_64 cloud VM, CPython 3.13.18 (GIL build). The output of `lscpu | head -15` is in
the PR body.

### Before (step 2; both columns time Python, which checks the harness)

| case | size | reference (median, min-max) | facade, `backend="python"` | ratio (spread) |
|---|---|---|---|---|
| `_align` | 300 tokens | 13.1 ms (12.5 ms-15.2 ms) | 12.9 ms (12.6 ms-14 ms) | 1.01x (0.89-1.20) |
| `_align` | 1000 tokens | 204 ms (203 ms-249 ms) | 210 ms (196 ms-269 ms) | 0.97x (0.75-1.27) |
| `_align` | 5000 tokens | 8.25 s (7.89 s-10.2 s) | 8.18 s (6.05 s-10.2 s) | 1.01x (0.77-1.69) |
| `rover (3 hypotheses)` | 1000 tokens | 474 ms (446 ms-553 ms) | 546 ms (503 ms-652 ms) | 0.87x (0.68-1.10) |

Run: `python benchmarks/bench_port.py --repeat 5` (median of 5 auto-ranged runs). The
300-token case is about 145 ns per DP cell in Python. At 5 000 tokens it is 8.25 s per
alignment, close to the brief's 10 s estimate. The ratio column is noise around 1.0
because both columns are Python.

## Publish rule

To do (step 7).

## Landing owed

To do (wrap-up).
