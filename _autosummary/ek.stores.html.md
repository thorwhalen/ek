# ek.stores

Local-file persistence for `ek`, as plain `MutableMapping` stores.

Everything `ek` persists – gold corpora, evaluation results, fitted
calibrators, human corrections, run logs – lives under the per-user data
directory `~/.local/share/ek/` (resolved cross-platform by
`config2py.AppData`), one subfolder per *kind*. Each store is a JSON
key-value store from `dol`: `store[key] = value` writes
`<kind>/<key>.json`, `store[key]` reads it back, iteration lists keys – so
callers get a dict-like facade rather than a bespoke god-class.

Progressive disclosure:

```pycon
>>> import tempfile; root = tempfile.mkdtemp()
>>> s = json_store("gold", rootdir=root)
>>> s["doc-001"] = {"reference_text": "hello world"}
>>> s["doc-001"]["reference_text"]
'hello world'
>>> list(s)
['doc-001']
```

A [`mall()`](#ek.stores.mall) groups every kind into one stores-of-stores mapping, with optional
`mall["gold", "doc-001"]` tuple-key access.

Tests and ephemeral use can redirect the root via the `rootdir` argument or the
`EK_DATA_HOME` environment variable, so nothing touches the real user folder.

### Module Attributes

| [`cache_this`](#ek.stores.cache_this)([func, cache, key, pre_cache, ...])   | Unified caching decorator for properties and methods with persistent storage support.   |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`KINDS`](#ek.stores.KINDS)                                            | The kinds of artifact `ek` persists; each becomes a subfolder + a store.                |

### Functions

| [`app_folder`](#ek.stores.app_folder)()                                 | The root `ek` data folder (e.g. `~/.local/share/ek`).                                                       |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [`data_dir`](#ek.stores.data_dir)(kind, \*[, rootdir])                | Filesystem directory backing `kind` (under the data root or `rootdir`).                                     |
| [`json_store`](#ek.stores.json_store)(kind, \*[, rootdir])              | A JSON `MutableMapping` for one `kind` of artifact.                                                         |
| [`mall`](#ek.stores.mall)(\*[, rootdir])                          | The full stores-of-stores mall (one JSON store per [`KINDS`](#ek.stores.KINDS)). |
| [`persistent_cache`](#ek.stores.persistent_cache)([func, kind, key, rootdir]) | Memoize a plain function's result in a `kind` JSON store, across sessions.                                  |

### Classes

| [`Mall`](#ek.stores.Mall)(\*[, kinds, rootdir])   | A read-only mapping of `kind -> json_store(kind)` (a dol "mall").   |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------|

### ek.stores.KINDS *= ('gold', 'results', 'calibrators', 'corrections', 'runs', 'baselines')*

The kinds of artifact `ek` persists; each becomes a subfolder + a store.

### *class* ek.stores.Mall(, kinds=('gold', 'results', 'calibrators', 'corrections', 'runs', 'baselines'), rootdir=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A read-only mapping of `kind -> json_store(kind)` (a dol “mall”).

Supports both `mall["gold"]` (the store) and `mall["gold", key]` (the
value), the latter being sugar for `mall["gold"][key]`.

### ek.stores.app_folder()

The root `ek` data folder (e.g. `~/.local/share/ek`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ek.stores.cache_this(func=None, , cache=None, key=None, pre_cache=False, as_property=None, ignore=None, serialize=None, deserialize=None)

Re-export for the idiomatic class-property persisted cache, e.g.
`@cache_this(cache=lambda self: json_store("runs"), key=...)` on a method.

### ek.stores.data_dir(kind, , rootdir=None)

Filesystem directory backing `kind` (under the data root or `rootdir`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ek.stores.json_store(kind, , rootdir=None)

A JSON `MutableMapping` for one `kind` of artifact.

Keys are plain names (no extension); values are JSON-serializable objects.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

### ek.stores.mall(, rootdir=None)

The full stores-of-stores mall (one JSON store per [`KINDS`](#ek.stores.KINDS)).

* **Return type:**
  [`Mall`](#ek.stores.Mall)

### ek.stores.persistent_cache(func=None, , kind='runs', key=None, rootdir=None)

Memoize a plain function’s result in a `kind` JSON store, across sessions.

Use as `@persistent_cache(kind="runs", key=...)`. Returned values must be
JSON-serializable. `key` may be a constant or a callable receiving the call
arguments; it defaults to the function’s qualified name (so provide a `key`
for functions that take arguments). For caching a *class property/method*,
prefer [`cache_this`](#ek.stores.cache_this) (dol’s descriptor) instead.

### Example

```pycon
>>> import tempfile; root = tempfile.mkdtemp()
>>> calls = []
>>> @persistent_cache(kind="runs", key="answer", rootdir=root)
... def compute():
...     calls.append(1)
...     return {"answer": 42}
>>> compute(), compute(), len(calls)
({'answer': 42}, {'answer': 42}, 1)
```
