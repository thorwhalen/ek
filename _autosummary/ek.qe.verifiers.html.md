# ek.qe.verifiers

Deterministic verifier signals – the cheapest, first-to-run QE layer.

Reference-free quality estimation should try the \*\*free, label-free, deterministic
checks first\*\* (`misc/docs/ek_03` §1d, §5; `ek_04` Layers 0-3): they cost
nothing, need no training data, and catch exactly the errors a confidence signal
misses – a value the extractor was *confident* about but that is structurally
impossible (a checksum that does not validate, a total that does not reconcile, an
enum value out of range). Only escalate to intrinsic confidence, agreement (ROVER),
or LLM signals on the residual that verifiers cannot settle.

The check primitives here (Luhn / IBAN / ISBN checksums, type/range/enum/regex,
cross-field reconciliation) are pure-Python and dependency-free *by design*: they
are reused both as reference-free [`Signal`](ek.base.html.md#ek.base.Signal) evidence (this module,
the [`ek.estimate_quality()`](ek.html.md#ek.estimate_quality) side) and – later – as the L1-L2 layers of the
flag-vs-correct validation pipeline (`ek/validate.py`, issue #7). Implementing the
checksums directly (rather than depending on the LGPL `python-stdnum`) keeps the
permissive-core promise intact.

A verifier emits raw evidence – a pass/fail [`Finding`](ek.base.html.md#ek.base.Finding) and a
fraction-passed score – never a calibrated probability or a decision.

### Example

```pycon
>>> luhn_check("79927398713")          # a classic Luhn-valid number
True
>>> luhn_check("79927398710")
False
>>> isbn_check("0-306-40615-2")         # ISBN-10
True
```

### Module Attributes

| [`DEFAULT_TOTALS_TOL`](#ek.qe.verifiers.DEFAULT_TOTALS_TOL)   | Default absolute tolerance for the cross-field totals reconciliation check.     |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`CHECKSUMS`](#ek.qe.verifiers.CHECKSUMS)            | Built-in checksum checks, by name (register more under the `checks` namespace). |

### Functions

| [`checksum_validator`](#ek.qe.verifiers.checksum_validator)([kind, field_name, layer])     | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value failing a named checksum.            |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`enum_validator`](#ek.qe.verifiers.enum_validator)(members, \*[, field_name, layer])  | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not in an allowed set.               |
| [`iban_check`](#ek.qe.verifiers.iban_check)(iban)                                  | IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1).                                                                 |
| [`isbn10_check`](#ek.qe.verifiers.isbn10_check)(value)                               | ISBN-10 checksum (weighted mod-11; a trailing `X` counts as 10).                                                                       |
| [`isbn13_check`](#ek.qe.verifiers.isbn13_check)(value)                               | ISBN-13 / EAN-13 checksum (alternating 1/3 weights, mod-10).                                                                           |
| [`isbn_check`](#ek.qe.verifiers.isbn_check)(value)                                 | Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored).                                                              |
| [`luhn_check`](#ek.qe.verifiers.luhn_check)(number)                                | Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs.                                                                        |
| [`range_validator`](#ek.qe.verifiers.range_validator)(lo, hi, \*[, field_name, layer])  | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a number outside `[lo, hi]`.                 |
| [`regex_validator`](#ek.qe.verifiers.regex_validator)(pattern, \*[, field_name, layer]) | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not fully matching `pattern`.        |
| [`schema_validator`](#ek.qe.verifiers.schema_validator)(value, \*[, spec])               | A spec-driven [`Validator`](ek.base.html.md#ek.base.Validator): type and `domain` checks from a FieldSpec. |
| [`totals_consistent`](#ek.qe.verifiers.totals_consistent)(record, \*, total_key, ...)     | Flag when `record[total_key]` does not equal the sum of `item_keys` (±tol).                                                            |

### Classes

| [`VerifierSignal`](#ek.qe.verifiers.VerifierSignal)([validators, cost_tier])   | Deterministic verifier evidence as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).   |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|

### ek.qe.verifiers.CHECKSUMS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [bool](https://docs.python.org/3/builtins/functions.html#bool)]]* *= {'iban': <function iban_check>, 'isbn': <function isbn_check>, 'isbn10': <function isbn10_check>, 'isbn13': <function isbn13_check>, 'luhn': <function luhn_check>}*

Built-in checksum checks, by name (register more under the `checks` namespace).

### ek.qe.verifiers.DEFAULT_TOTALS_TOL *= 0.01*

Default absolute tolerance for the cross-field totals reconciliation check.

### *class* ek.qe.verifiers.VerifierSignal(validators=<factory>, cost_tier=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Deterministic verifier evidence as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).

Cost tier 1 – the free, first-to-run layer. Runs a list of
[`Validator`](ek.base.html.md#ek.base.Validator) s on a value and returns the **fraction that passed**
in `[0, 1]` as the raw signal (`1.0` = every check passed). It also exposes
the [`Finding`](ek.base.html.md#ek.base.Finding) s it produced via [`findings()`](#ek.qe.verifiers.VerifierSignal.findings), so the same
object feeds both the score path (-> calibrate) and the audit path (-> review).
Like every signal it is uncalibrated: a verifier failing is strong evidence, not
a probability, so a [`Calibrator`](ek.base.html.md#ek.base.Calibrator) still runs before any gate.

### Example

```pycon
>>> v = VerifierSignal([checksum_validator("luhn")])
>>> v("79927398713")          # valid -> all checks pass
1.0
>>> v("79927398710")          # invalid -> the one check fails
0.0
```

#### findings(value, , spec=None)

All findings the validators produce for `value` (empty == all clear).

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### ek.qe.verifiers.checksum_validator(kind='luhn', , field_name='', layer='checksum')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value failing a named checksum.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.verifiers.enum_validator(members, , field_name='', layer='enum')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not in an allowed set.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.verifiers.iban_check(iban)

IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.verifiers.isbn10_check(value)

ISBN-10 checksum (weighted mod-11; a trailing `X` counts as 10).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.verifiers.isbn13_check(value)

ISBN-13 / EAN-13 checksum (alternating 1/3 weights, mod-10).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.verifiers.isbn_check(value)

Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.verifiers.luhn_check(number)

Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.qe.verifiers.range_validator(lo, hi, , field_name='', layer='range')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a number outside `[lo, hi]`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.verifiers.regex_validator(pattern, , field_name='', layer='format')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not fully matching `pattern`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.qe.verifiers.schema_validator(value, , spec=None)

A spec-driven [`Validator`](ek.base.html.md#ek.base.Validator): type and `domain` checks from a FieldSpec.

Reads the [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec) `type` and `domain` – a `(lo, hi)`
numeric range, an enum tuple, or a regex string – and applies the matching
check. With no spec it is a no-op, so it composes safely in any validator list.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### ek.qe.verifiers.totals_consistent(record, , total_key, item_keys, tol=0.01, layer='cross_field')

Flag when `record[total_key]` does not equal the sum of `item_keys` (±tol).

The canonical cross-field check (invoice total = Σ line items). Skipped (rather
than raising or flagging) when the total or *all* item keys are absent or
non-numeric, so it is safe on partial records.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]
