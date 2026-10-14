# ek.metrics.spans

Span / slot precision-recall-F1 over two backends that legitimately disagree.

Entity/slot F1 is the workhorse for fields-and-spans extraction, but a *single* F1
is meaningless without naming its **match scheme** – and the two standard backends
diverge widely *and disagree with each other* (`misc/docs/ek_02` §2.2):

- **seqeval** (MIT, `chakki-works`) – CoNLL `conlleval`-compatible entity F1
  over BIO/IOBES tag *sequences*. Its `conll` (default) and `strict` (span+type,
  keyed to a tagging scheme) modes are the reference for sequence-labelling NER.
- **nervaluate** (MIT, `MantisAI`) – the SemEval-2013 Task 9.1 scheme over typed
  *spans*, with four schemas: **strict** (span + type), **exact** (span, any type),
  **partial** (boundary overlap, any type), **type** (type with span overlap). It
  counts COR/INC/PAR/MIS/SPU and gives partial spans half credit.

A model that finds “Electric” instead of “General Electric” scores F1=0 under
exact-span matching but ~0.667 under a partial/token scheme – so the scheme is *not*
a default you can hide. This module makes [`MatchScheme`](#ek.metrics.spans.MatchScheme) a **required keyword**
with no default: you must say which contract you are scoring under.

Both backends ship behind the `[metrics]` extra and are imported lazily, so
importing `ek` never pulls them in. Each [`Score`](ek.base.html.md#ek.base.Score) carries the raw
TP/FP/FN (or COR/INC/PAR/MIS/SPU) counts in its `detail` so
[`SpanF1Metric.aggregate()`](#ek.metrics.spans.SpanF1Metric.aggregate) micro-averages over a corpus – never a naive mean of
per-document F1s.

## Input shapes

- `seqeval` schemes ([`MatchScheme.SEQEVAL_CONLL`](#ek.metrics.spans.MatchScheme.SEQEVAL_CONLL),
  [`MatchScheme.SEQEVAL_STRICT`](#ek.metrics.spans.MatchScheme.SEQEVAL_STRICT)): `pred` and `gold` are **tag sequences** –
  > a list of BIO/IOBES tags (e.g. `["B-PER", "I-PER", "O"]`).
- `nervaluate` schemes ([`MatchScheme.STRICT`](#ek.metrics.spans.MatchScheme.STRICT), `EXACT`, `PARTIAL`,
  `TYPE`): `pred` and `gold` are **span lists** – a list of
  `{"label", "start", "end"}` dicts.

### Example

```pycon
>>> from ek.metrics.spans import SpanF1Metric, MatchScheme
>>> m = SpanF1Metric(scheme=MatchScheme.SEQEVAL_CONLL)
>>> s = m(["B-PER", "I-PER", "O"], ["B-PER", "I-PER", "O"])
>>> s.f1
1.0
>>> SpanF1Metric()
Traceback (most recent call last):
TypeError: SpanF1Metric requires an explicit scheme=MatchScheme.<...>
```

### Classes

| [`MatchScheme`](#ek.metrics.spans.MatchScheme)(\*values)                  | The (backend, scheme) contract a span/slot F1 is computed under.                                                    |
|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`SpanF1Metric`](#ek.metrics.spans.SpanF1Metric)([scheme, tagging_scheme]) | Entity/slot precision/recall/F1 under an **explicit** [`MatchScheme`](#ek.metrics.spans.MatchScheme). |

### *class* ek.metrics.spans.MatchScheme(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

The (backend, scheme) contract a span/slot F1 is computed under.

The two backends legitimately disagree (seqeval ignores other-type tags at the
tag level; nervaluate includes them), so the scheme is always explicit. The
`SEQEVAL_*` members consume BIO/IOBES *tag sequences*; the others consume
SemEval-2013 *span lists* via nervaluate.

#### EXACT *= 'exact'*

exact span boundaries, type ignored.

* **Type:**
  nervaluate exact

#### PARTIAL *= 'partial'*

boundary *overlap* (half credit), type ignored.

* **Type:**
  nervaluate partial

#### SEQEVAL_CONLL *= 'seqeval_conll'*

seqeval, CoNLL `conlleval`-compatible entity F1 (the default seqeval mode).

#### SEQEVAL_STRICT *= 'seqeval_strict'*

span boundaries AND type must match, keyed to a scheme.

* **Type:**
  seqeval strict mode

#### STRICT *= 'strict'*

exact span boundaries AND correct type.

* **Type:**
  nervaluate SemEval-2013 strict

#### TYPE *= 'type'*

correct type with at least some span overlap.

* **Type:**
  nervaluate type

### *class* ek.metrics.spans.SpanF1Metric(scheme=None, , tagging_scheme='IOB2')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Entity/slot precision/recall/F1 under an **explicit** [`MatchScheme`](#ek.metrics.spans.MatchScheme).

* **Parameters:**
  * **scheme** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`MatchScheme`](#ek.metrics.spans.MatchScheme)]) – REQUIRED. The (backend, scheme) contract – there is deliberately no
    default, because seqeval and nervaluate disagree and a bare F1 is
    ambiguous (`misc/docs/ek_02` §2.2). Pass a [`MatchScheme`](#ek.metrics.spans.MatchScheme).
  * **tagging_scheme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – For [`MatchScheme.SEQEVAL_STRICT`](#ek.metrics.spans.MatchScheme.SEQEVAL_STRICT) only: the BIO/IOBES
    tagging scheme name (e.g. `"IOB2"`, `"IOBES"`, `"BILOU"`) that
    strict-mode entity decoding is keyed to. Defaults to `"IOB2"`.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `scheme` is omitted (the whole point of this metric).

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum the raw counts, then divide once).

seqeval schemes carry TP/FP/FN; nervaluate schemes carry COR/INC/PAR/MIS/SPU
(PAR worth half). Both micro-average by summing counts globally – never a
mean of per-document F1s. The two backends’ counts are not commensurable, so
a corpus mixing them is rejected rather than silently dropping one.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
