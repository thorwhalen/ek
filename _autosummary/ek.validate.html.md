# ek.validate

Reference-free post-extraction validation & correction (issue #7).

A layered, **cheapest -> most-expensive** pipeline with an explicit
**FLAG-vs-CORRECT** distinction (`misc/docs/ek_04`). Every method estimates
`argmax_v P(v|o)` without a gold reference: the prior `P(v)` strengthens in
layers – schema/type/range (cheapest, strongest), then lexicon/gazetteer, then
language-model surprisal (softest) – with cross-field consistency and statistical
anomaly checks on top.

The single most important architectural line is **FLAG vs CORRECT**:

- A **Validator** only *flags*: it yields [`Finding`](ek.base.html.md#ek.base.Finding) s with
  `severity=FLAG` and never rewrites the value. The deterministic verifier layer
  (`ek.qe.verifiers`: checksums, regex, range, enum, schema, totals) is re-exported
  here so a caller composes the whole stack from one place.
- A **Corrector** can also *correct*: it yields a `severity=CORRECT` Finding whose
  `suggestion` is the proposed replacement. Only a narrow, safe, deterministic
  class corrects automatically – canonicalization (L0) and closed-set lexicon
  resolution (L2); free invention is reserved for the gated L5 neural corrector.

Both share one callable shape – `(value, *, spec) -> Iterable[Finding]` – so
[`validation_pipeline()`](#ek.validate.validation_pipeline) composes them uniformly. A CORRECT finding’s suggestion
is applied to the value *before the next layer runs* (the noisy-channel chain: a
cheap deterministic fix short-circuits the expensive layers), and an injectable
`stop_when` policy ends the pass early (e.g. once a value is corrected or clean).

The deterministic layers here need no extra dependencies (`rapidfuzz` is core). The
softer/heavier layers ship as **dependency-injected** layers: the layer is dep-free
and you inject the model/LLM call (your in-domain LM scorer, your LLM client), so the
expensive backend is your deployment choice, not a default dependency.

- **L3 LM surprisal** (FLAG): [`lm_surprisal_validator()`](#ek.validate.lm_surprisal_validator) takes an injected
  `(str) -> float` surprisal scorer (n-gram / masked-LM PLL, e.g. `minicons`) and
  flags low-probability spans. Statistical anomaly without an LM:
  [`benford_findings()`](#ek.validate.benford_findings) (leading-digit) and [`zscore_anomaly_findings()`](#ek.validate.zscore_anomaly_findings)
  > (robust median/MAD magnitude outliers), both dep-free.
- **L5 neural/LLM correction** (CORRECT, gated): [`llm_corrector()`](#ek.validate.llm_corrector) takes an
  injected `(str) -> Optional[str]` correction call; it is the only layer that
  invents content, so it fires only on already-flagged values (`only_flagged`),
  reads the pipeline’s findings, and keeps the original for audit.
- **L4 constrained generation** is a *generation*-time concern (it constrains a
  generative extractor, not post-hoc output – ek_04 separates “constrain-on-generate”
  from “post-hoc validation”), so it lives on the extraction side
  (`ek[constrained]`: `outlines`/XGrammar/`instructor`), not in this pipeline.

### Example

```pycon
>>> from ek.validate import validation_pipeline, lexicon_corrector, stop_on_correction
>>> from ek.base import Severity
>>> fix = validation_pipeline(
...     lexicon_corrector(["France", "Germany", "Spain"]),
...     stop_when=stop_on_correction,
... )
>>> r = fix("Frnace")               # one transposition from a closed enum
>>> r.value
'France'
>>> r.corrected
True
>>> r.findings[0].severity is Severity.CORRECT
True
```

### Module Attributes

| [`Layer`](#ek.validate.Layer)                     | a value (and optional spec) -> zero+ Findings.                                                                                                  |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DEFAULT_LEXICON_THRESHOLD`](#ek.validate.DEFAULT_LEXICON_THRESHOLD) | Default minimum similarity (`rapidfuzz` ratio in [0, 1]) for a closed-set lexicon match to be auto-applied as a correction rather than flagged. |
| [`BENFORD_EXPECTED`](#ek.validate.BENFORD_EXPECTED)          | Benford's law expected first-significant-digit frequencies, `P(d)=log10(1+1/d)`.                                                                |
| [`DEFAULT_BENFORD_TOL`](#ek.validate.DEFAULT_BENFORD_TOL)       | Default max absolute per-digit deviation from Benford before a field is flagged.                                                                |
| [`DEFAULT_BENFORD_MIN_N`](#ek.validate.DEFAULT_BENFORD_MIN_N)     | Default minimum sample size before applying Benford (the law is asymptotic).                                                                    |
| [`DEFAULT_ZSCORE_THRESHOLD`](#ek.validate.DEFAULT_ZSCORE_THRESHOLD)  | Modified (median/MAD) z-score above which a value is flagged an outlier (Iglewicz & Hoaglin recommend 3.5).                                     |

### Functions

| [`benford_findings`](#ek.validate.benford_findings)(numbers, \*[, field, tol, ...])   | FLAG a numeric field whose first-digit distribution deviates from Benford's law.   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`canonicalize_corrector`](#ek.validate.canonicalize_corrector)(normalize, \*[, ...])       | L0: fold a value to canonical form, emitting a CORRECT finding when it changes.    |
| [`cross_field_validator`](#ek.validate.cross_field_validator)(predicate, \*, message)      | FLAG a record when `predicate(record)` is False (a general cross-field check).     |
| [`lexicon_corrector`](#ek.validate.lexicon_corrector)(vocabulary, \*[, ...])           | L2: resolve a value against a **closed** vocabulary by fuzzy match (`rapidfuzz`).  |
| [`llm_corrector`](#ek.validate.llm_corrector)(correct_fn, \*[, only_flagged, ...]) | L5: a **gated** neural/LLM corrector -- the only layer that invents content.       |
| [`lm_surprisal_validator`](#ek.validate.lm_surprisal_validator)(scorer, \*, threshold)      | L3: FLAG a value whose language-model surprisal crosses `threshold`.               |
| [`ordering_validator`](#ek.validate.ordering_validator)(keys, \*[, strict, layer])      | FLAG a record whose comparable values at `keys` are not ascending.                 |
| [`stop_on_correction`](#ek.validate.stop_on_correction)(findings)                       | Stop policy: end the pass as soon as any layer proposes a correction.              |
| [`stop_on_flag`](#ek.validate.stop_on_flag)(findings)                             | Stop policy: end the pass at the first FLAG (fail fast).                           |
| [`validate`](#ek.validate.validate)(value, \*[, layers, spec, ...])           | Run one value through a validation/correction pipeline (progressive disclosure).   |
| [`validation_pipeline`](#ek.validate.validation_pipeline)(\*layers[, ...])               | Compose validators/correctors into a cheapest -> most-expensive pipeline.          |
| [`zscore_anomaly_findings`](#ek.validate.zscore_anomaly_findings)(numbers, \*[, field, ...]) | FLAG numeric values that are robust-z-score outliers (median + MAD based).         |

### Classes

| [`Corrector`](#ek.validate.Corrector)(\*args, \*\*kwargs)                 | A [`Validator`](ek.base.html.md#ek.base.Validator) that may also CORRECT.            |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`ValidationResult`](#ek.validate.ValidationResult)(original, value[, findings]) | The outcome of running a value through a [`validation_pipeline()`](#ek.validate.validation_pipeline). |

### ek.validate.BENFORD_EXPECTED *= {1: 0.3010299956639812, 2: 0.17609125905568124, 3: 0.12493873660829992, 4: 0.09691001300805642, 5: 0.07918124604762482, 6: 0.06694678963061322, 7: 0.05799194697768673, 8: 0.05115252244738129, 9: 0.04575749056067514}*

Benford’s law expected first-significant-digit frequencies, `P(d)=log10(1+1/d)`.

### *class* ek.validate.Corrector(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A [`Validator`](ek.base.html.md#ek.base.Validator) that may also CORRECT.

Same callable shape as a Validator, but it can emit a `severity=CORRECT`
[`Finding`](ek.base.html.md#ek.base.Finding) whose `suggestion` is the proposed replacement value.
The `layer` attribute names its place on the cheap->expensive spine.

### ek.validate.DEFAULT_BENFORD_MIN_N *= 30*

Default minimum sample size before applying Benford (the law is asymptotic).

### ek.validate.DEFAULT_BENFORD_TOL *= 0.15*

Default max absolute per-digit deviation from Benford before a field is flagged.

### ek.validate.DEFAULT_LEXICON_THRESHOLD *= 0.8*

Default minimum similarity (`rapidfuzz` ratio in [0, 1]) for a closed-set lexicon
match to be auto-applied as a correction rather than flagged. 0.8 catches typical
single-edit and transposition typos (“Frnace”->”France” ~0.83) while the wide gap
to an unrelated word (~0.3) keeps false corrections rare.

### ek.validate.DEFAULT_ZSCORE_THRESHOLD *= 3.5*

Modified (median/MAD) z-score above which a value is flagged an outlier
(Iglewicz & Hoaglin recommend 3.5).

### ek.validate.Layer

a value (and optional spec) -> zero+ Findings.

* **Type:**
  A Validator/Corrector callable

alias of `Callable`[[…], [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### *class* ek.validate.ValidationResult(original, value, findings=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of running a value through a [`validation_pipeline()`](#ek.validate.validation_pipeline).

`value` is the final value after any applied corrections; `original` is the
input; `findings` is the full audit trail (FLAG and CORRECT), in layer order.

#### *property* clean *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

No findings at all – the value passed every layer untouched.

#### *property* corrected *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether any correction changed the value.

#### *property* flagged *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether any layer raised a FLAG (something a human should look at).

### ek.validate.benford_findings(numbers, , field='', tol=0.15, min_n=30, layer='anomaly')

FLAG a numeric field whose first-digit distribution deviates from Benford’s law.

A reference-free anomaly check for naturally-occurring magnitudes (amounts,
populations, counts): the first significant digit should follow
`P(d)=log10(1+1/d)`. A large deviation flags fabricated or systematically-wrong
data. **Corpus-level** (takes the whole column of values) and **FLAG-only** – it
routes a field to review, never auto-edits, and can false-positive on legitimately
bounded/skewed fields. Skipped (returns `[]`) below `min_n` usable values,
since the law is asymptotic.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### ek.validate.canonicalize_corrector(normalize, , field_name='', layer='canonicalize')

L0: fold a value to canonical form, emitting a CORRECT finding when it changes.

`normalize` is anything [`ek.canonicalize.resolve_canonicalizer()`](ek.canonicalize.html.md#ek.canonicalize.resolve_canonicalizer) accepts
(a registered name, a callable, a step list). The narrowest, safest corrector –
run it first so later layers compare canonical forms.

* **Return type:**
  [`Corrector`](#ek.validate.Corrector)

### ek.validate.cross_field_validator(predicate, , message, fields=(), layer='cross_field')

FLAG a record when `predicate(record)` is False (a general cross-field check).

Skips silently when `predicate` raises on a partial record (a missing field is
not itself a cross-field violation).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.validate.lexicon_corrector(vocabulary, , threshold=0.8, scorer=None, field_name='', layer='lexicon', flag_unmatched=True)

L2: resolve a value against a **closed** vocabulary by fuzzy match (`rapidfuzz`).

The safest corrector in the stack when the candidate set is closed (country codes,
SKUs, enums): an exact member passes untouched; a single close match
(similarity `>= threshold`) is applied as a CORRECT finding; otherwise the value
is FLAGged as out-of-vocabulary (when `flag_unmatched`). Against an *open*
vocabulary, set `flag_unmatched=False` and treat matches as suggestions only.

* **Parameters:**
  * **vocabulary** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The closed set of valid values.
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum `rapidfuzz` similarity in `[0, 1]` to auto-correct.
  * **scorer** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A `rapidfuzz` scorer (default `fuzz.ratio`).
  * **flag_unmatched** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – FLAG a value with no close match (default `True`).
* **Return type:**
  [`Corrector`](#ek.validate.Corrector)

### ek.validate.llm_corrector(correct_fn, , only_flagged=True, field_name='', layer='llm_correct')

L5: a **gated** neural/LLM corrector – the only layer that invents content.

`correct_fn` is an **injected** `(value) -> Optional[str]` (bring your own
LLM/seq2seq call; return a corrected string, or `None` to leave the value
unchanged). It is the most expensive and stochastic layer, so by default
(`only_flagged`) it fires \*\*only on a value that a cheaper layer already
FLAGged\*\* – gate it to the residual, never let it silently rewrite clean fields
(this layer reads the pipeline’s accumulated findings via `wants_findings`).
Emits a CORRECT finding; the pipeline applies the rewrite and keeps the original
for audit. Always verify its output downstream – it can make text worse.

Recipe (Anthropic): `correct_fn=lambda v: client.messages.create(model=...,
messages=[{"role": "user", "content": prompt(v)}]).content[0].text.strip()` with
`client = anthropic.Anthropic()` (MIT). Constrain/verify the result and keep an
audit trail.

* **Return type:**
  [`Corrector`](#ek.validate.Corrector)

### ek.validate.lm_surprisal_validator(scorer, , threshold, higher_is_worse=True, field_name='', layer='lm_prior')

L3: FLAG a value whose language-model surprisal crosses `threshold`.

`scorer` is an **injected** `(str) -> float` returning a surprisal /
perplexity / negative-log-likelihood (dependency injection: bring your own
in-domain n-gram or masked-LM scorer – an *in-domain* prior is what makes this
useful). FLAG-only on its own; pair it with a candidate generator
(`lexicon_corrector` / `llm_corrector`) to actually correct. Set
`higher_is_worse=False` if your scorer returns a probability (higher = better).

Recipe (masked-LM pseudo-log-likelihood): wrap `minicons` –
`from minicons import scorer; m = scorer.MaskedLMScorer("bert-base-uncased", "cpu")`
– and pass `scorer=lambda s: -m.sequence_score([s])[0]` (surprisal = negative
PLL). `minicons` is MIT; install your own LM backend (it pulls torch).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.validate.ordering_validator(keys, , strict=True, layer='cross_field')

FLAG a record whose comparable values at `keys` are not ascending.

The canonical date/sequence check (`start <= end`, issue before due). Only the
keys actually present and mutually comparable are checked, so it is safe on
partial records.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.validate.stop_on_correction(findings)

Stop policy: end the pass as soon as any layer proposes a correction.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.validate.stop_on_flag(findings)

Stop policy: end the pass at the first FLAG (fail fast).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.validate.validate(value, \*, layers=(), spec=None, apply_corrections=True, stop_when=<function \_never_stop>)

Run one value through a validation/correction pipeline (progressive disclosure).

The simple call `validate(v, layers=[...])` Just Works; tune the correction
behaviour with `apply_corrections`/`stop_when`. Compose `layers` from the
factories in this module (`canonicalize_corrector`, `lexicon_corrector`, the
re-exported verifier validators) in cheapest-first order.

* **Return type:**
  [`ValidationResult`](#ek.validate.ValidationResult)

### ek.validate.validation_pipeline(\*layers, apply_corrections=True, stop_when=<function \_never_stop>)

Compose validators/correctors into a cheapest -> most-expensive pipeline.

`layers` run in the given order (the order encodes the cost spine). Each layer
is a `(value, *, spec) -> Iterable[Finding]` callable. When `apply_corrections`
(default), the first `CORRECT` finding a layer emits is applied to the value
before the next layer runs – so a cheap deterministic fix is what the expensive
layers then see (the noisy-channel chain). `stop_when` is an injectable
early-exit policy over the findings accumulated so far (see
[`stop_on_correction()`](#ek.validate.stop_on_correction) / [`stop_on_flag()`](#ek.validate.stop_on_flag)); the default runs every layer.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`ValidationResult`](#ek.validate.ValidationResult)]

### ek.validate.zscore_anomaly_findings(numbers, , field='', threshold=3.5, min_n=30, layer='anomaly')

FLAG numeric values that are robust-z-score outliers (median + MAD based).

A reference-free, **dependency-free** anomaly check that complements
[`benford_findings()`](#ek.validate.benford_findings) (which checks the leading-digit *distribution*; this
checks individual *magnitudes*). For a numeric column it computes each value’s
**modified z-score** – `0.6745 * (x - median) / MAD` – which is robust to the
very outliers it is looking for, and FLAGs those whose absolute score exceeds
`threshold`. One FLAG per outlying value (carrying its index); **FLAG-only** –
it routes a value to review, never edits. Skipped below `min_n` (robust
statistics on a handful of points are noise). When the MAD is 0 (a *near*-constant
column, e.g. many identical values plus one outlier) it falls back to the
mean-absolute-deviation scale (Iglewicz & Hoaglin), so a lone outlier is still
caught; a *truly* constant column (no spread at all) yields no findings. For
*multivariate* outliers, plug an isolation forest via `pyod` (already in
`ek[validation]`) as a custom validator.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
