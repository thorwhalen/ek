#!/usr/bin/env python3
"""Benchmark ROVER's alignment backends against the frozen reference, end to end.

Built from the ``pyo3-port`` benchmark template. Times ``ek._reference._align`` (the
"before") and the facade ``ek.qe.rover._align(..., backend=BACKEND)`` (the "after",
boundary and token interning included) on the same deterministic inputs, at the
sizes from the port assessment (300, 1 000 and 5 000 tokens), plus ``rover()`` end
to end on three hypotheses. ``BACKEND`` is ``"rust"`` when ``ek-rs`` is importable,
else ``"python"`` (both columns then time Python). Stdlib only::

    python benchmarks/bench_port.py              # all cases
    python benchmarks/bench_port.py --quick      # smallest size of each case, fewer repeats
    python benchmarks/bench_port.py --json out.json

It refuses to time a debug build of ``ek_rs`` (``ek_rs.DEBUG_BUILD``): numbers from
``maturin develop`` without ``--release`` are meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import timeit
from statistics import median

import importlib
import importlib.util

from ek import _reference as ref

rover_mod = importlib.import_module("ek.qe.rover")
BACKEND = os.environ.get(
    "EK_BENCH_BACKEND", "rust" if importlib.util.find_spec("ek_rs") else "python"
)

SEED = 0  # inputs are deterministic, so two runs time the same work


VOCAB = [f"w{i}" for i in range(500)]


def _tokens(size: int, rng: random.Random) -> list[str]:
    """``size`` tokens with a Zipf-like word distribution (OCR/ASR-shaped)."""
    weights = [1 / (r + 1) for r in range(len(VOCAB))]
    return rng.choices(VOCAB, weights=weights, k=size)


def _noisy(tokens: list[str], rng: random.Random, *, rate: float = 0.1) -> list[str]:
    """A second engine's view: ~``rate`` of tokens substituted, dropped or doubled."""
    out: list[str] = []
    for tok in tokens:
        r = rng.random()
        if r < rate / 3:
            out.append(rng.choice(VOCAB))  # substitution
        elif r < 2 * rate / 3:
            continue  # deletion
        elif r < rate:
            out += [tok, rng.choice(VOCAB)]  # insertion
        else:
            out.append(tok)
    return out


def _align_case(size: int, rng: random.Random):
    base = _tokens(size, rng)
    return (base, _noisy(base, rng)), {}


def _rover_case(size: int, rng: random.Random):
    base = _tokens(size, rng)
    return ([" ".join(_noisy(base, rng)) for _ in range(3)],), {}


def _ref_rover(hyps):
    return _rover_on("python", hyps)


def _port_rover(hyps):
    return _rover_on(BACKEND, hyps)


def _rover_on(backend: str, hyps):
    saved = rover_mod._ALIGN_BACKEND
    rover_mod._ALIGN_BACKEND = backend
    try:
        return rover_mod.rover(hyps)
    finally:
        rover_mod._ALIGN_BACKEND = saved


def _port_align(spine, hyp):
    return rover_mod._align(spine, hyp, backend=BACKEND)


# (label, reference fn, port fn, factory(size, rng) -> (args, kwargs), sizes)
CASES = [
    ("_align", ref._align, _port_align, _align_case, [300, 1_000, 5_000]),
    ("rover (3 hypotheses)", _ref_rover, _port_rover, _rover_case, [1_000]),
]


def conditions() -> dict:
    """What a speedup needs next to it to mean anything."""
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "build": "GIL" if gil else "free-threaded",
        "machine": f"{platform.system()} {platform.machine()}",
        "cpus": os.cpu_count(),
    }


def time_call(
    fn, args: tuple, kwargs: dict, *, repeat: int
) -> tuple[float, float, float]:
    """(median, min, max) seconds per call over ``repeat`` runs, each auto-ranged to >= 0.2 s."""
    timer = timeit.Timer(lambda: fn(*args, **kwargs))
    number, _ = timer.autorange()
    runs = [t / number for t in timer.repeat(repeat=repeat, number=number)]
    return median(runs), min(runs), max(runs)


def _fmt(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if seconds >= scale:
            return f"{seconds / scale:7.2f} {unit}"
    return f"{seconds / 1e-9:7.2f} ns"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="smallest size only, 3 repeats"
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    opts = parser.parse_args(argv)

    core = importlib.import_module("ek_rs") if BACKEND == "rust" else None
    if core is not None and getattr(core, "DEBUG_BUILD", False):
        print(
            "refusing to benchmark a debug build: run `maturin develop --release` first"
        )
        return 2
    if not CASES:
        print("no CASES defined: add one entry per ported function")
        return 2

    repeat = 3 if opts.quick else opts.repeat
    cond = conditions()
    cond["backend"] = BACKEND
    print("conditions: " + ", ".join(f"{k}={v}" for k, v in cond.items()))
    rows = []
    for label, ref_fn, port_fn, factory, sizes in CASES:
        for size in sizes[:1] if opts.quick else sizes:
            args, kwargs = factory(size, random.Random(SEED))
            assert port_fn(*args, **kwargs) == ref_fn(*args, **kwargs), (
                "backends disagree"
            )
            t_ref, ref_lo, ref_hi = time_call(ref_fn, args, kwargs, repeat=repeat)
            t_port, port_lo, port_hi = time_call(port_fn, args, kwargs, repeat=repeat)
            # the ratio's spread: worst and best pairing of the two ranges
            lo, hi = ref_lo / port_hi, ref_hi / port_lo
            rows.append(
                {
                    "case": label,
                    "size": size,
                    "reference_s": t_ref,
                    "reference_range_s": [ref_lo, ref_hi],
                    "native_s": t_port,
                    "native_range_s": [port_lo, port_hi],
                    "speedup": t_ref / t_port,
                    "speedup_range": [lo, hi],
                }
            )
            print(
                f"{label:<22} n={size:<6} reference {_fmt(t_ref)}   native {_fmt(t_port)}"
                f"   x{t_ref / t_port:,.1f} (x{lo:,.1f}-x{hi:,.1f})"
            )
    if opts.json:
        with open(opts.json, "w", encoding="utf-8") as f:
            json.dump({"conditions": cond, "results": rows}, f, indent=2)
        print(f"wrote {opts.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
