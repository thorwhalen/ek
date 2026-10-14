# ek.tools

Small, CLI-friendly functions over the `ek` core (the dispatch surface).

These are thin, string-in/value-out wrappers designed for triple dispatch (CLI
today, HTTP/UI later). The single `_dispatch_funcs` list is the SSOT of what
`ek.__main__` exposes, so there is no duplicated command registration, and
each function’s *signature* is its command-line interface.

Run them from the shell:

```default
python -m ek cer "hello wrld" "hello world"
python -m ek where
python -m ek check tesseract
```

### Functions

| [`cer`](#ek.tools.cer)(prediction, gold, \*[, normalize])         | Character Error Rate between a prediction and a gold string (lower is better).     |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`check`](#ek.tools.check)(engine)                                  | Report what an OCR engine needs to run (delegates to ocracy when installed).       |
| [`cost_per_success`](#ek.tools.cost_per_success)(total_dollars, successes, \*) | Cost-of-Pass: dollars per *successfully completed* task (`inf` if none succeeded). |
| [`engines`](#ek.tools.engines)()                                      | List installed OCR backends (requires the `ek[ocr]` extra).                        |
| [`pass_k`](#ek.tools.pass_k)(n, c[, k])                              | pass@k (capability) and pass^k (reliability) from `c` successes in `n` trials.     |
| [`rover`](#ek.tools.rover)(\*hypotheses[, confidence])              | ROVER-fuse 2+ transcriptions into a consensus + mean agreement (reference-free).   |
| [`version`](#ek.tools.version)()                                      | Print the installed `ek` version.                                                  |
| [`wer`](#ek.tools.wer)(prediction, gold, \*[, normalize])         | Word Error Rate between a prediction and a gold string (lower is better).          |
| [`where`](#ek.tools.where)()                                        | Print the local data folder where `ek` persists gold/results/runs.                 |

### ek.tools.cer(prediction, gold, , normalize=None)

Character Error Rate between a prediction and a gold string (lower is better).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.tools.check(engine)

Report what an OCR engine needs to run (delegates to ocracy when installed).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.tools.cost_per_success(total_dollars, successes, , attempts=None)

Cost-of-Pass: dollars per *successfully completed* task (`inf` if none succeeded).

The unit that matters for an agent – tokens spent on a failed episode are pure waste.

Example:

```default
python -m ek cost-per-success 12.50 5 --attempts 20
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.tools.engines()

List installed OCR backends (requires the `ek[ocr]` extra).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### ek.tools.pass_k(n, c, k=1)

pass@k (capability) and pass^k (reliability) from `c` successes in `n` trials.

`pass@k` asks “can it *ever* do this?”; `pass^k` asks “does it do this *every* time?”
– and only the second is a production number.

Example:

```default
python -m ek pass-k 10 9 --k 8      # a 90%-reliable agent, judged over 8 trials
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.tools.rover(\*hypotheses, confidence=True)

ROVER-fuse 2+ transcriptions into a consensus + mean agreement (reference-free).

Example:

```default
python -m ek rover "the cat sat" "the cat sit" "the bat sat"
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.tools.version()

Print the installed `ek` version.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.tools.wer(prediction, gold, , normalize=None)

Word Error Rate between a prediction and a gold string (lower is better).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.tools.where()

Print the local data folder where `ek` persists gold/results/runs.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
