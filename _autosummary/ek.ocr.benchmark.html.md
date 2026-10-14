# ek.ocr.benchmark

End-to-end OCR offline benchmark: run a fleet over gold, score, persist.

This is `ek`’s first out-of-the-box experience – “how accurate is my OCR?”. It
pairs each gold item with the engine’s output, scores text with a globally
accumulated CER/WER (the correct corpus aggregation), optionally slices the report,
and persists gold + per-item results + the run aggregate to the `dol` stores
under `~/.local/share/ek/`.

The runner is deliberately source-agnostic: `engine` is any
`image -> OcrResult` callable (or an `ocracy` backend id, bridged for you), so
the benchmark runs – and is fully testable – without any real OCR engine or cloud
credentials.

### Example

```pycon
>>> import tempfile; root = tempfile.mkdtemp()
>>> gold = {
...     "d1": {"image": "ignored", "reference_text": "cat", "slice": "easy"},
...     "d2": {"image": "ignored", "reference_text": "dog", "slice": "easy"},
... }
>>> # a stub engine that always reads "cat" (perfect for d1, wrong for d2):
>>> engine = lambda image: type("R", (), {"text": "cat"})()
>>> report = evaluate_ocr(engine, gold, metric="cer", rootdir=root)
>>> report.n
2
>>> sorted(report.per_slice)
['easy']
```

### Functions

| [`add_gold_item`](#ek.ocr.benchmark.add_gold_item)(key, image, reference_text, \*)   | Add one gold example to the `gold` store.      |
|--------------------------------------------------------------------------------------------------|------------------------------------------------|
| [`evaluate_ocr`](#ek.ocr.benchmark.evaluate_ocr)(engine, gold, \*[, metric, ...])   | Benchmark an OCR engine against a gold corpus. |

### ek.ocr.benchmark.add_gold_item(key, image, reference_text, , slice=None, rootdir=None, \*\*extra)

Add one gold example to the `gold` store.

* **Parameters:**
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Unique id for the document.
  * **image** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An image reference the engine can read (path, URL, …). Stored as a
    reference, not bytes – gold stores hold references + the expected text.
  * **reference_text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The ground-truth transcription.
  * **slice** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional stratification label (doc type, language, scan quality, …).
  * **rootdir** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the data root (tests/ephemeral use).
  * **\*\*extra** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Any additional metadata to store on the record.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ek.ocr.benchmark.evaluate_ocr(engine, gold, , metric='cer', normalize=None, grammar=None, image_key='image', reference_key='reference_text', slice_key='slice', persist=False, run_id=None, rootdir=None)

Benchmark an OCR engine against a gold corpus.

* **Parameters:**
  * **engine** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – An `image -> OcrResult` callable, or an `ocracy` backend id.
  * **gold** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – A mapping `key -> {image, reference_text, [slice], ...}` – a plain
    dict or a `ek` gold store.
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"cer"` (default) or `"wer"` (or any registered string metric).
  * **normalize** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional canonicalizer applied to both sides before scoring.
  * **grammar** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional Layer-A grammar (forwarded to the metric).
  * **image_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Field names within each gold record.
  * **reference_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Field names within each gold record.
  * **slice_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Field names within each gold record.
  * **persist** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If true, write per-item results and the run aggregate to the
    `results` / `runs` stores.
  * **run_id** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Identifier for this run (auto-generated from the clock if omitted).
  * **rootdir** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the data root (tests/ephemeral use).
* **Returns:**
  A [`Report`](ek.base.html.md#ek.base.Report) whose `detail['per_item']` maps each gold key to
  its prediction, reference, slice, score, and engine confidence/backend.
