# ek.registry

Strategy registry, optional-dependency guarding, and requirement checks.

`ek` is open-closed: every swappable behaviour (a [`Metric`](ek.base.html.md#ek.base.Metric), a
[`Signal`](ek.base.html.md#ek.base.Signal), a calibrator, an OCR backend, …) is registered under a
*namespace* and resolved by name. Built-in strategies register themselves at
import time; third parties add their own either by calling [`register()`](#ek.registry.register) or by
declaring an entry point in the `ek.<namespace>` group – no edit to `ek` needed.

Two more concerns live here so they stay out of the core logic:

- [`requires_extra()`](#ek.registry.requires_extra) – a decorator that turns a missing optional dependency
  into an *actionable* error (`pip install ek[ocr]`) instead of an opaque
  `ImportError` raised deep inside a call.
- [`check_requirements()`](#ek.registry.check_requirements) – a helper that reports, without raising, what an
  engine/feature needs (Python packages, system binaries, credentials), and
  defers to `ocracy`’s own doctor for OCR engines when it is installed.

### Example

```pycon
>>> @register("metrics", "shout")
... def shout(pred, gold, *, grammar=None):
...     return pred == gold
>>> get("metrics", "shout")("a", "a")
True
>>> "shout" in names("metrics")
True
```

### Functions

| [`check_requirements`](#ek.registry.check_requirements)(\*[, engine, extra])   | Report what an OCR engine or feature extra needs, without raising.      |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`get`](#ek.registry.get)(namespace, name)                      | Resolve a registered strategy by name (discovering entry points first). |
| [`names`](#ek.registry.names)(namespace)                          | All registered names in a namespace (after entry-point discovery).      |
| [`register`](#ek.registry.register)(namespace, name[, obj])          | Register `obj` under `namespace`/`name` (usable as a decorator).        |
| [`requires_extra`](#ek.registry.requires_extra)(extra, \*[, packages])     | Decorator: fail with an actionable install hint if an extra is missing. |
| [`resolve`](#ek.registry.resolve)(namespace, ref, \*[, default])    | Coerce a strategy reference to a callable.                              |

### Exceptions

| [`MissingExtraError`](#ek.registry.MissingExtraError)   | Raised when an optional feature is used without its extra installed.   |
|----------------------------------------------------------------------|------------------------------------------------------------------------|

### *exception* ek.registry.MissingExtraError

Bases: [`ImportError`](https://docs.python.org/3/builtins/exceptions.html#ImportError)

Raised when an optional feature is used without its extra installed.

### ek.registry.check_requirements(, engine=None, extra=None)

Report what an OCR engine or feature extra needs, without raising.

For an OCR `engine` and an installed `ocracy`, defers to `ocracy`’s own
doctor (which knows each backend’s pip package, system binaries and
credentials). Otherwise returns a generic hint.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  A dict with at least `{"ok": bool, "hint": str}`.

### ek.registry.get(namespace, name)

Resolve a registered strategy by name (discovering entry points first).

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – with the list of available names, if `name` is unknown.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ek.registry.names(namespace)

All registered names in a namespace (after entry-point discovery).

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ek.registry.register(namespace, name, obj=None)

Register `obj` under `namespace`/`name` (usable as a decorator).

* **Parameters:**
  * **namespace** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A strategy family, e.g. `"metrics"`, `"signals"`,
    `"calibrators"`, `"policies"`, `"normalizers"`, `"ocr"`.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The lookup key within the namespace.
  * **obj** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The object to register; omit to use as a decorator.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  `obj` (so it can be used as a decorator).

### ek.registry.requires_extra(extra, , packages=None)

Decorator: fail with an actionable install hint if an extra is missing.

The wrapped callable runs only once every package in `packages` imports; the
common case (everything installed) adds a single cheap import check.

* **Parameters:**
  * **extra** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The extra name, used in the hint `pip install ek[<extra>]`.
  * **packages** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Import names to probe (defaults to `[extra]`).
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### Example

```pycon
>>> @requires_extra("ocr", packages=["definitely_not_installed_pkg"])
... def run():
...     return "ran"
>>> try:
...     run()
... except MissingExtraError as e:
...     print("ek[ocr]" in str(e))
True
```

### ek.registry.resolve(namespace, ref, , default=None)

Coerce a strategy reference to a callable.

`ref` may be a registered name (`str`), an already-resolved object, or
`None` (in which case `default`, itself a name or object, is used).

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
