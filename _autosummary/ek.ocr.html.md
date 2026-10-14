# ek.ocr

OCR evaluation – `ek`’s first concrete instance, built on `ocracy`.

OCR is the noisiest special case of information extraction. This subpackage adds
the OCR-specific pieces on top of `ek`’s source-agnostic core:

- [`ocracy_backend()`](#ek.ocr.ocracy_backend) / [`read_text()`](#ek.ocr.read_text) – a thin bridge that turns
  `ocracy`’s 16-engine fleet into an `image -> OcrResult` callable. The
  dependency arrow is strict: `ek -> ocracy` (via the `ek[ocr]` extra), never
  the reverse. `ek` core depends only on the `OcrResult` *shape* (`.text` /
  `.blocks` / `.mean_confidence`), so the benchmark below evaluates *any*
  `image -> OcrResult` callable – `ocracy` or your own.
- [`profile()`](#ek.ocr.profile) – per-engine capability profiles (who emits real vs
  model-guessed geometry, who is calibrated, who yields table structure), distilled
  from `misc/docs/ek_01`. `ek` treats these as claims to verify empirically.
- [`evaluate_ocr()`](#ek.ocr.evaluate_ocr) / [`add_gold_item()`](#ek.ocr.add_gold_item) – run a fleet over a gold corpus,
  score with CER/WER (globally accumulated), slice the report, and persist gold +
  results to the `dol` stores under `~/.local/share/ek/`.
- [`table_from_ocr()`](#ek.ocr.table_from_ocr) / [`has_table_structure()`](#ek.ocr.has_table_structure) – recover the existing
  [`ek.metrics.tables.Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult` (whose engine-specific table
  > data lives only in `.raw`/`.markdown`) so the table metrics (TEDS / GriTS) can
  > score OCR table output uniformly. The per-engine extractor is injected/registered
  > under the `"table_parsers"` namespace (open-closed); see [`ek.ocr.tables`](ek.ocr.tables.html.md#module-ek.ocr.tables).

Heavy engine SDKs and credentials are never required to import this module; they
are checked at call time via [`ek.registry.check_requirements()`](ek.registry.html.md#ek.registry.check_requirements).

### Functions

| [`ocracy_backend`](#ek.ocr.ocracy_backend)([backend])                     | Return an `image -> OcrResult` callable bound to one `ocracy` backend.                                                   |
|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`read_text`](#ek.ocr.read_text)(image, \*[, backend])               | Convenience: OCR an image and return just the recognized text string.                                                    |
| [`evaluate_ocr`](#ek.ocr.evaluate_ocr)(engine, gold, \*[, metric, ...]) | Benchmark an OCR engine against a gold corpus.                                                                           |
| [`add_gold_item`](#ek.ocr.add_gold_item)(key, image, reference_text, \*) | Add one gold example to the `gold` store.                                                                                |
| [`profile`](#ek.ocr.profile)(engine)                               | Capability profile for an engine id (a safe default if unknown).                                                         |
| [`table_from_ocr`](#ek.ocr.table_from_ocr)(ocr_result, \*[, parser])      | Extract a normalized [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult`. |
| [`has_table_structure`](#ek.ocr.has_table_structure)(ocr_result, \*[, parser]) | Whether [`table_from_ocr()`](#ek.ocr.table_from_ocr) recovers a non-empty table from `ocr_result`.  |
| [`resolve_table_parser`](#ek.ocr.resolve_table_parser)(ref)                     | Resolve a table parser reference (a registered name or a callable) to a callable.                                        |
| [`engine_yields_tables`](#ek.ocr.engine_yields_tables)(engine)                  | Whether `engine`'s capability profile claims scoreable table structure.                                                  |

### ek.ocr.add_gold_item(key, image, reference_text, , slice=None, rootdir=None, \*\*extra)

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

### ek.ocr.engine_yields_tables(engine)

Whether `engine`’s capability profile claims scoreable table structure.

A convenience over [`ek.ocr.profile()`](#ek.ocr.profile): returns the engine profile’s `tables`
flag. ek treats the profile as a *prior* (Azure DI, AWS Textract, PaddleOCR and the
markdown engines are expected to yield scoreable tables; Google Vision is not), to
be verified empirically by actually running [`table_from_ocr()`](#ek.ocr.table_from_ocr). Use it to skip
table scoring for engines that cannot emit cells in the first place.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.ocr.evaluate_ocr(engine, gold, , metric='cer', normalize=None, grammar=None, image_key='image', reference_key='reference_text', slice_key='slice', persist=False, run_id=None, rootdir=None)

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

### ek.ocr.has_table_structure(ocr_result, , parser=None)

Whether [`table_from_ocr()`](#ek.ocr.table_from_ocr) recovers a non-empty table from `ocr_result`.

A thin predicate over [`table_from_ocr()`](#ek.ocr.table_from_ocr) (same `parser` semantics): true iff
a table with at least one cell can be recovered. Use it to route an OCR result to
the table metrics vs the text-only CER/WER fallback.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.ocr.ocracy_backend(backend=None, \*\*default_kwargs)

Return an `image -> OcrResult` callable bound to one `ocracy` backend.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – An `ocracy` backend id (e.g. `"tesseract"`, `"google_vision"`);
    `None` uses `ocracy`’s first installed backend.
  * **\*\*default_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Options forwarded to `ocracy.ocr` on every call
    (e.g. `languages=["en"]`); per-call kwargs override these.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  A callable suitable for [`ek.ocr.evaluate_ocr()`](#ek.ocr.evaluate_ocr).

### ek.ocr.profile(engine)

Capability profile for an engine id (a safe default if unknown).

Accepts `ocracy`’s canonical hyphenated ids and their underscore variants.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.ocr.read_text(image, , backend=None, \*\*kwargs)

Convenience: OCR an image and return just the recognized text string.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.ocr.resolve_table_parser(ref)

Resolve a table parser reference (a registered name or a callable) to a callable.

Mirrors how other ek strategies resolve from the registry (see
[`ek.registry.resolve()`](ek.registry.html.md#ek.registry.resolve)). A `str` is looked up under the `"table_parsers"`
namespace; an already-callable parser is returned unchanged.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ek.ocr.table_from_ocr(ocr_result, , parser=None)

Extract a normalized [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult`.

The single seam ek’s table metrics (TEDS / GriTS) use to score OCR table output
uniformly. Returns the recovered Table, or `None` when the result carries no
recoverable table structure.

* **Parameters:**
  * **ocr_result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Any `OcrResult`-shaped object. ek reads only its `.raw` (the
    engine-specific payload), `.markdown` (VLM/markdown engines), and
    `.meta` – never ocracy types, so any `image -> OcrResult` callable’s
    output works.
  * **parser** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – 

    How to turn the raw payload into a table (dependency injection /
    open-closed). One of:
    - `None` (default): try the built-in **safe heuristics** – `.raw` is
      already a [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table), a 2-D grid, or a
      `<table>...</table>` HTML string; or `.raw`/`.meta` is a mapping
      carrying one of `{tables, table, cells, grid, html}`; or `.markdown`
      contains an HTML `<table>`. Returns `None` if none match – it never
      guesses wildly.
    - a `str`: the name of a parser registered under `"table_parsers"`
      (per-engine extractor), resolved from [`ek.registry`](ek.registry.html.md#module-ek.registry).
    - a callable `(raw) -> Table | grid | html | None`: an inline per-engine
      extractor. It receives the result’s `.raw` (falling back to the result
      itself when there is no `.raw`).
  * **through** (*Any non-None value an injected parser returns is fed*)
* **Return type:**
  [*Table*](ek.metrics.tables.html.md#ek.metrics.tables.Table) | *None*

:param `Table.coerce()`:
:param so a parser may yield a Table:
:param a 2-D grid:
:param or HTML.:

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)]
* **Returns:**
  A non-empty [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table), or `None` when no table
  structure is recoverable. An empty Table (no rows/cells) is normalized to
  `None` so the predicate [`has_table_structure()`](#ek.ocr.has_table_structure) and downstream metrics
  see “no table”, not a degenerate one.

### Modules

| [`benchmark`](ek.ocr.benchmark.html.md#module-ek.ocr.benchmark)   | End-to-end OCR offline benchmark: run a fleet over gold, score, persist.           |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`bridge`](ek.ocr.bridge.html.md#module-ek.ocr.bridge)         | The `ocracy` bridge: turn the OCR fleet into an `image -> OcrResult` callable.     |
| [`profiles`](ek.ocr.profiles.html.md#module-ek.ocr.profiles)     | Per-engine OCR capability profiles (what QE signal each engine actually emits).    |
| [`tables`](ek.ocr.tables.html.md#module-ek.ocr.tables)         | Recover a normalized table from an `OcrResult` so ek's table metrics can score it. |
