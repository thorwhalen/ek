# Port log: `ek.qe.rover._align` to Rust

The state of the port, kept where the next session will find it. Method: the
`pyo3-port` skill (oracle first, one function per step, each gated on a
differential test), with `rust-port-assess` for the assessment and the publish
rule. Status: **landed** (see [Landing record](#landing-record)). A cloud session did the port and opened the PR. The
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
   with its own `pyproject.toml` (maturin). `ek` declares **no** `rust` extra for now (landing session: `uv lock` / `uv sync`
   resolve every extra and fail while `ek-rs` is not on PyPI; add `rust = ["ek-rs"]`
   after the first `ek-rs` release). The landing session may still choose the mixed package.
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
| 5 `align` leaf | done | `rover_align::align` (core) + `ek_rs.align` (binding). Core: 5 unit tests, 660 golden cases from the reference, an exhaustive property test (complete, ordered, minimal scripts on all pairs over `{a,b,c}` up to length 5), 3 doctests (README included). Python on the release build: fast 271 passed, **thorough 224 passed** (5 000 examples per test), whole suite with `EK_ROVER_ALIGN_BACKEND=rust` 720 passed. Hand mutants, each caught by the suite: insertion wins deletion ties (4 failures), the diagonal loses ties (9), an off-by-one in column 0 (5) |
| review | done | `rust-reviewer` (fresh context, Opus-class): **0 blockers, 0 majors, 4 minors, all fixed**: the stub was untracked; no `None` in `hyp` strategies; `@requires_extra` ran before the domain check (off-domain input with no `ek-rs` raised instead of falling back); `EK_ROVER_ALIGN_BACKEND` was not validated at import. After the fixes: 724 passed on each backend; thorough rerun below |
| 6 errors | done | `AlignError::TableTooLarge` (from `checked_mul` / `try_reserve_exact`, never an abort or a panic) maps to `MemoryError`, which is what the Python table allocation raises. Not tested from Python: no safe input reaches it. The Rust unit test covers it with zero-sized tokens |
| 7 interpreter | done | the DP runs in `py.detach`. `gil_used`: PyO3's free-threading default is kept, because the module has no globals and no interior mutability. **Not run on 3.14t**, and abi3 wheels do not load there anyway, so free-threading is not claimed |
| 8 stubs | done | `rust/py/ek_rs.pyi` becomes `ek_rs/__init__.pyi` plus `py.typed` in the wheel. `mypy.stubtest ek_rs` is clean; the allowlist has one entry, maturin's inner `ek_rs.ek_rs` module |
| 9 packaging and CI | done; **green on GitHub at `433ac52`**: `rust-ci` (lint, test 3.10 + 3.14, deny x2, core-publishable), `rust-wheels` (manylinux x86_64 + aarch64, macOS arm64, Windows x64 built and smoke-tested; macOS x86_64 built only, it is cross-compiled; sdist), wads `ci` (Linux 3.10/3.12, Windows), `license-gate`. The first run (`c42f452`) failed twice, and both were this PR's fault: the wads CI has no hypothesis (now `importorskip`), and nightly cargo warned about an unused `workspace.package.version` (removed) | the release wheel installed in a clean venv with ek installed from source, tested from outside the repo with `--import-mode=importlib`: 286 passed. The sdist builds and imports in a clean 3.10 venv (the core is inside it). `rover-align` passes `cargo +1.85 test` (MSRV). New workflows: `rust-ci.yml` (fmt, clippy with warnings denied, core tests, the full suite twice on the native backend on 3.10 and 3.14, stubtest, cargo-deny, core publishability: dry-run, MSRV, docs.rs build) and `rust-wheels.yml` (abi3 wheel matrix plus sdist on PRs touching `rust/`; publish on an `ek-rs-v*` tag through PyPI trusted publishing, no secret). Not run here: actionlint (this session was not allowed to download it) and cargo-deny (not installed; it runs in CI) |

## Benchmarks

Conditions: Intel(R) Xeon(R) Processor @ 2.10GHz (model 207), 4 vCPU, Linux
x86_64 cloud VM, CPython 3.13.18 (GIL build). The output of `lscpu | head -15` is in
the PR body.

### After (step 5, release build; same machine and harness as the before run)

| case | size | before: Python (median, min-max) | after: Rust via the facade (median, min-max) | ratio (spread) |
|---|---|---|---|---|
| `_align` | 300 tokens | 17.3 ms (15.1 ms-19 ms) | 357 µs (355 µs-381 µs) | **48x** (40-54x) |
| `_align` | 1000 tokens | 220 ms (209 ms-239 ms) | 4.48 ms (3.66 ms-4.94 ms) | **49x** (42-65x) |
| `_align` | 5000 tokens | 8.51 s (7.91 s-8.89 s) | 99.4 ms (96.1 ms-119 ms) | **86x** (67-93x) |
| `rover (3 hypotheses)` | 1000 tokens | 457 ms (436 ms-562 ms) | 12 ms (11.1 ms-12.9 ms) | **38x** (34-50x) |

Ratios come from one process timing both columns, so they are valid on a noisy VM;
the absolute times are not comparable with another machine. The `rust-reviewer` subagent
was running `cargo` on the same 4 vCPUs during part of this run, so spreads are wider
than they would be on an idle machine. The kernel costs about 4 ns per cell, boundary
and interning included (1 000 tokens: 4.5 ms for 10^6 cells). The 5 000-token ratio is
higher (86x) because the Python table (200 MB of lists) also stresses the allocator.
Every function that moved gains, so none goes back to the reference.

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

Applied to `rover-align` (`rust/core`) with `rust-port-assess`'s `publish-rule.md`
(2026-11-02):

```text
Publish: yes: flagship (a = 3, b = 2): the user publishes; this session never does
  (a) 3/3  a pure transformation: two &[T: PartialEq] in, Vec<AlignOp> out, no config,
           no callbacks, deterministic (rust/core/src/lib.rs `align`)
  (b) 2/3  incumbents: levenshtein-diff 0.2.4 (MIT, generic edit script, last release
           2022-11-15, so unmaintained by the rule's two-year test); edit-distance 2.2.2
           (Apache-2.0, distance only, strings); strsim 0.11 (distances only); similar 3.2
           (Apache-2.0, LCS/Myers diff: no substitutions, other semantics); bio 4.2 (MIT,
           pairwise alignment over byte slices, heavy); sesdiff 0.3 (GPL-3.0+). None
           documents a tie order. ROVER: nothing (`cargo search rover`, crates.io
           q=rover voting / word transition network: unrelated crates only)
  name: rover-align, free (crates.io API 404 for rover-align and rover_align)
  layout: (i) publish rust/core; ek-rs (rust/py) stays publish = false
```

Vetoes, checked: none applies. It is meaningful without Python: the types are plain
Rust, and the tie order is a documented contract, not a Python quirk. It has no
private data. Its only dependency is `thiserror` (on crates.io). The name is free. The
API is one function plus two types, small enough to commit to at 0.1. The licence is
MIT, and so is its only dependency.

Done for publishing: version 0.1.0 and `publish = true` on the core only; description,
keywords, categories, readme, repository and docs.rs metadata; `README.md` (its example
is a doctest), `CHANGELOG.md` and `LICENSE` in `rust/core/`; `cargo package --list`
holds only the crate's files; `cargo publish --dry-run -p rover-align --locked` passes
(see the PR body for the run); MSRV 1.85 tested; rustdoc with `-D warnings` is clean;
every public item is documented with `# Errors` / `# Examples`; `AlignError` is
`#[non_exhaustive]`. `AlignOp` is deliberately exhaustive: the three edit operations
are a closed set, and `#[non_exhaustive]` would force a useless wildcard arm on every
user.

**Owed for flagship** (before or after the first publish; the user decides):
criterion benchmarks against `levenshtein-diff` and `edit-distance` with numbers in the
README; the Rust API Guidelines checklist walked item by item; `examples/`; a second
owner; `cargo-semver-checks` in CI once a baseline exists; an announcement. The first
publish needs `CARGO_REGISTRY_TOKEN` from the user's machine, because trusted
publishing cannot create a crate. After that, crates.io trusted publishing for
`rust-wheels.yml` (or a `crates` job) is an option.

## Landing owed

This session (cloud, unattended) pushed the branch and opened the PR. It did not merge
it, and it did not publish anything. The package's own session must:

1. Re-run here, on the user's machine: `uv pip install -e ".[dev,metrics]"`,
   `maturin develop --release -m rust/py/Cargo.toml`,
   `EK_DIFF_BACKEND=rust HYPOTHESIS_PROFILE=thorough pytest tests/test_differential.py`,
   `EK_ROVER_ALIGN_BACKEND=rust pytest`, `python benchmarks/bench_port.py`
   (macOS arm64 numbers are not comparable with the table above; ratios are).
2. Check that `rust-ci` and `rust-wheels` ran green on the PR. The macOS and Windows
   wheels have only ever been built there; this VM is Linux x86_64 only.
3. Decide the packaging (decision 4): keep `ek-rs` as a separate optional wheel (this
   PR), or fold it into ek as a mixed maturin package. The second makes every ek
   install compiled and conflicts with wads publishing (i2mint/wads#166).
4. Before the first `ek-rs` release: register `rust-wheels.yml` (environment `pypi`)
   as a trusted publisher of `ek-rs` on PyPI, then tag `ek-rs-v0.0.1`. Until `ek-rs`
   is on PyPI, `pip install ek-rs` fails; build it from `rust/`. After the first release, add
   `rust = ["ek-rs"]` to ek's extras and point the missing-extension error at `pip install ek[rust]`.
5. Decide on publishing `rover-align` (verdict above). If yes, run
   `cargo publish -p rover-align --locked` from a clean checkout with the user's token.
6. Squash-merge. ek's version is untouched; the wads bump happens on merge as usual.

## Landing record

Second session (cloud, unattended), 2026-11-04, Linux x86_64 4 vCPU, CPython 3.13.18,
rustc 1.99.1 (the pinned toolchain). Everything below was re-run, not copied.

| check | result | PR body said |
|---|---|---|
| `maturin develop --release --uv` (rust/py) | built, `ek_rs` imports | n/a |
| full suite, default backend | 724 passed / 0 failed / 5 skipped | 724 / 0 / 5 |
| full suite, `EK_ROVER_ALIGN_BACKEND=rust` | 724 passed / 0 failed / 5 skipped | same |
| differential, `HYPOTHESIS_PROFILE=thorough EK_DIFF_BACKEND=rust` | 224 passed | 224 |
| `cargo test --locked` | core: 5 unit, 1 golden, 1 property, 3 doctests, all passed | same |
| `cargo clippy --all-targets` with `CARGO_BUILD_WARNINGS=deny`, `cargo fmt --check` | clean | n/a |
| benchmark (ratios) | 42x (300), 50x (1000), 87x (5000), 44x (rover, 3 hyp, 1000) | 48x / 49x / 86x / 38x |

Independent review (fresh Opus-class subagent over `git diff origin/main...HEAD`, B1-B6,
P1-P5, O1): **0 blockers, 0 majors, 6 minors**. It also fed the built extension hostile
inputs (`-1`, `2**64`, `['a']`, `True`: `OverflowError` / `TypeError`, no panic) and ran two
mutants of `ek_rs.align` against the differential suite (a rapidfuzz-style tie order failed
154 tests, a repeated-token bug failed 8). Minors, not fixed here: (1) a direct
`_align(..., backend="rust")` call has no token cap, so a huge input raises `MemoryError`
with a message, or hits the OOM killer under overcommit (`rover()` caps at 5 000 tokens);
(2) `#[pymodule]` keeps PyO3's default `gil_used` without a comment at the site (the reason
is in step 7); (3) `_EXTRA_PROBES` has no entry for the missing-extension case; (4) a bad
`EK_ROVER_ALIGN_BACKEND` makes `import ek` itself fail (deliberate fail-fast); (5) `_align`'s
docstring still says "Pure-Python"; (6) `rust-wheels.yml` `publish` also runs on a manual
dispatch with `publish` ticked from any ref (intended: a manual dispatch is an explicit act).

**Fixed in the landing session (a blocker under the landing brief):** the `ek[rust]` extra named
`ek-rs`, which is not on PyPI (404). `uv lock` failed with "ek[rust]'s requirements are
unsatisfiable" (ek has no committed `uv.lock`, and the wads CI does not lock, but any
`uv lock` / `uv sync` / `uv run` in a checkout would fail). The extra is dropped; the
missing-extension error now names `ek-rs` and how to build it, and keeps the original
`ImportError` as `__cause__`. The test was changed first and failed against the old code.
`uv lock` then resolves (430 packages).

Release safety: the push-to-main path of `rust-ci.yml` only runs `cargo publish --dry-run`;
`rust-wheels.yml` publishes only on an `ek-rs-v*` tag or a manual dispatch with `publish`
ticked; nothing publishes `rover-align` (no `cargo publish` without `--dry-run` anywhere).
`uv build` gives `ek-0.1.15` as a pure-Python wheel with no `rust/` or `ek_rs` files, and an
sdist with no `rust/` tree. In a fresh venv holding only that wheel, `import ek`, `rover()`
and `_align` on the default backend work, and `backend="rust"` / `EK_ROVER_ALIGN_BACKEND=rust`
raise `MissingExtraError` with the message above. A plain `pip install` of the wheel works.

**Deferred to the user:** re-run the benchmark on the Mac (ratios here stand in); register
`rust-wheels.yml` (environment `pypi`) as a PyPI trusted publisher of `ek-rs` and tag
`ek-rs-v0.0.1`; then add `rust = ["ek-rs"]` to ek's extras; set `CARGO_REGISTRY_TOKEN` and run
`cargo publish -p rover-align --locked` if rover-align should go to crates.io; the flagship items
for `rover-align` (criterion benchmarks, API-guidelines checklist, examples, a second owner).

## Mac benchmark and workflow tightening (2026-11-11)

Re-run on the user's Mac (Apple M1 Max, arm64, CPython 3.12.1 GIL build, release build of `ek-rs` 0.0.1 from `maturin develop --release --uv`), `python benchmarks/bench_port.py --repeat 5`, reference and native in the same process. Ratios only; absolute times are not comparable with the cloud VM above.

| case | size | reference | native | ratio (spread) |
|---|---|---|---|---|
| `_align` | 300 tokens | 19.05 ms | 486.93 µs | 39.1x (38.8-39.3x) |
| `_align` | 1000 tokens | 274.46 ms | 5.00 ms | 54.9x (54.4-55.9x) |
| `_align` | 5000 tokens | 7.49 s | 120.89 ms | 62.0x (61.4-62.3x) |
| `rover (3 hypotheses)` | 1000 tokens | 578.63 ms | 14.36 ms | 40.3x (39.9-40.7x) |

The Mac ratios (39-62x) sit within the cloud VM's range (38-86x); the 5000-token case is lower on the Mac because the Python reference is relatively faster there. Minor 6 above is fixed: `rust-wheels.yml` now publishes only from an `ek-rs-v*` tag (the `publish` input on manual dispatch is gone, so a manual run only builds and smoke-tests), and the `version` job already fails a tag that differs from the versions in `rust/py/`. The redundant `claude/land-pr-*` branch was already deleted from the remote.

