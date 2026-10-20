# ek.canonicalize

Versioned, composable canonicalization applied *before* scoring.

Under-specified normalization is the single biggest source of bogus evaluation
scores – it can swing results by tens of points, and a reused English normalizer
catastrophically corrupts Indic/abugida scripts by stripping Unicode Mark-class
characters. So canonicalization is a first-class, *versioned* component here, not
a metric detail: a [`Canonicalizer`](#ek.canonicalize.Canonicalizer) is an ordered pipeline of small named
steps with a stable `version` you can persist alongside a gold set.

Steps are plain `str -> str` functions registered under the `"normalizers"`
namespace, so a schema can name a per-field normalizer (via
`normalizer`) and third parties can add their own.

### Example

```pycon
>>> canon = Canonicalizer(["nfc", "lower", "collapse_whitespace"])
>>> canon("  Héllo   WORLD ")
'héllo world'
>>> canon.version  # stable hash of the step names
'nfc+lower+collapse_whitespace'
```

Always audit before folding a non-Latin script – [`mark_class_count()`](#ek.canonicalize.mark_class_count) reports
how many combining marks a string carries, the signal that aggressive folding
would destroy meaning:

```pycon
>>> mark_class_count("e\u0301")  # 'e' + combining acute
1
```

### Functions

| [`collapse_whitespace`](#ek.canonicalize.collapse_whitespace)(s)           | Collapse runs of whitespace to a single space and strip the ends.           |
|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`default_canonicalizer`](#ek.canonicalize.default_canonicalizer)()          | A conservative default: NFC -> casefold -> collapse whitespace.             |
| [`lower`](#ek.canonicalize.lower)(s)                         | Casefold to lowercase (full Unicode casefold).                              |
| [`mark_class_count`](#ek.canonicalize.mark_class_count)(s)              | Number of Unicode Mark-class (combining) characters in `s`.                 |
| [`nfc`](#ek.canonicalize.nfc)(s)                           | Unicode NFC normalization (compose); the safe default.                      |
| [`nfkc`](#ek.canonicalize.nfkc)(s)                          | Unicode NFKC normalization (compatibility compose); lossy -- use with care. |
| [`resolve_canonicalizer`](#ek.canonicalize.resolve_canonicalizer)(normalize) | Coerce a `normalize` argument to a callable canonicalizer (or `None`).      |
| [`strip`](#ek.canonicalize.strip)(s)                         | Strip leading/trailing whitespace.                                          |
| [`strip_punctuation`](#ek.canonicalize.strip_punctuation)(s)             | Remove Unicode punctuation characters (category starting with `P`).         |

### Classes

| [`Canonicalizer`](#ek.canonicalize.Canonicalizer)([steps, version, audit_marks])   | An ordered, named, versioned pipeline of `str -> str` steps.   |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------|

### *class* ek.canonicalize.Canonicalizer(steps=(), , version='', audit_marks=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An ordered, named, versioned pipeline of `str -> str` steps.

* **Parameters:**
  * **steps** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – Step references – names registered under `"normalizers"` or
    plain callables. Applied left to right.
  * **version** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override the auto-derived version string (the `+`-joined names).
  * **audit_marks** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If true, warn when a step would run on text carrying combining
    marks (the cross-script damage guard).

#### *property* steps *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)*

The resolved step names, in order.

#### *property* version *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Stable identifier of this pipeline (persist it next to a gold set).

### ek.canonicalize.collapse_whitespace(s)

Collapse runs of whitespace to a single space and strip the ends.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.canonicalize.default_canonicalizer()

A conservative default: NFC -> casefold -> collapse whitespace.

* **Return type:**
  [`Canonicalizer`](#ek.canonicalize.Canonicalizer)

### ek.canonicalize.lower(s)

Casefold to lowercase (full Unicode casefold).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.canonicalize.mark_class_count(s)

Number of Unicode Mark-class (combining) characters in `s`.

A non-zero count on a non-Latin script is a warning sign that case/diacritic
folding will corrupt meaning (the Indic-script catastrophe). Use it to gate or
audit a canonicalizer before trusting cross-script scores.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### ek.canonicalize.nfc(s)

Unicode NFC normalization (compose); the safe default.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.canonicalize.nfkc(s)

Unicode NFKC normalization (compatibility compose); lossy – use with care.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.canonicalize.resolve_canonicalizer(normalize)

Coerce a `normalize` argument to a callable canonicalizer (or `None`).

Accepts `None`, a registered normalizer name, a callable, a sequence of
steps, or an existing [`Canonicalizer`](#ek.canonicalize.Canonicalizer).

* **Return type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Canonicalizer`](#ek.canonicalize.Canonicalizer), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]

### ek.canonicalize.strip(s)

Strip leading/trailing whitespace.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ek.canonicalize.strip_punctuation(s)

Remove Unicode punctuation characters (category starting with `P`).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
