"""Write golden alignment cases from the frozen Python reference (``ek._reference``).

The Rust core's tests read ``align.json``, so the core is checked against Python
without building the extension. Run from the repository root, with ek importable:

    python rust/core/tests/golden/make_golden.py
"""

import itertools
import json
import pathlib
import random

from ek._reference import _align

OUT = pathlib.Path(__file__).with_name("align.json")


def cases():
    words = ["a", "b", None]
    for n, m in itertools.product(range(4), repeat=2):  # every pair up to length 3
        for spine in itertools.product(words, repeat=n):
            for hyp in itertools.product(words[:2], repeat=m):
                yield list(spine), list(hyp)
    rng = random.Random(0)
    vocab = ["the", "cat", "sat", "bat", "İ", "数据", "\U0001f600", ""]
    for _ in range(60):  # longer realistic ones
        spine = [rng.choice(vocab + [None]) for _ in range(rng.randrange(40))]
        hyp = [rng.choice(vocab) for _ in range(rng.randrange(40))]
        yield spine, hyp


rows = [[s, h, [list(op) for op in _align(s, h)]] for s, h in cases()]
OUT.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} cases to {OUT}")
