# ek.ocr.bridge

The `ocracy` bridge: turn the OCR fleet into an `image -> OcrResult` callable.

`ocracy` is a deliberately dependency-free facade over ~16 OCR engines whose
single entry point `ocracy.ocr(image, *, backend=None, **kwargs)` returns a
normalized `OcrResult` (`.text` plus `.blocks` carrying `[0, 1]`-normalized
confidence and geometry). `ek` consumes that shape rather than reinventing engine
integration. The bridge is intentionally tiny – it only adapts `ocracy`’s call
into the [`OcrBackend`](ek.base.html.md#ek.base.OcrBackend) contract and guards the optional dependency
with an actionable install hint.

Null-safety reminder (see `misc/docs/ek_01`): VLM/markdown engines (Claude,
GPT-4o, Mistral OCR) return text only – `blocks == []` and `confidence is
None`. Any code reading confidence/geometry off the result must tolerate `None`.

### Functions

| [`ocracy_backend`](#ek.ocr.bridge.ocracy_backend)([backend])       | Return an `image -> OcrResult` callable bound to one `ocracy` backend.   |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`read_text`](#ek.ocr.bridge.read_text)(image, \*[, backend]) | Convenience: OCR an image and return just the recognized text string.    |

### ek.ocr.bridge.ocracy_backend(backend=None, \*\*default_kwargs)

Return an `image -> OcrResult` callable bound to one `ocracy` backend.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – An `ocracy` backend id (e.g. `"tesseract"`, `"google_vision"`);
    `None` uses `ocracy`’s first installed backend.
  * **\*\*default_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Options forwarded to `ocracy.ocr` on every call
    (e.g. `languages=["en"]`); per-call kwargs override these.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  A callable suitable for [`ek.ocr.evaluate_ocr()`](ek.ocr.html.md#ek.ocr.evaluate_ocr).

### ek.ocr.bridge.read_text(image, , backend=None, \*\*kwargs)

Convenience: OCR an image and return just the recognized text string.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
