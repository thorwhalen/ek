# ek

`ek` – a framework for building Knowledge Evaluation systems.

`ek` evaluates the outputs of information-extraction systems. OCR is treated as
the noisiest *special case* of a general problem, so the core is source-agnostic
(PDF, DOCX, tables, DB responses, LLM extractors) and the OCR specifics live in
[`ek.ocr`](ek.ocr.html.md#module-ek.ocr).

Two facades cover the two halves of evaluation, both operating on the same typed
schema:

- [`score()`](#ek.score) / [`evaluate()`](#ek.evaluate) – *reference-based* (offline): compare against a
  gold answer, one item or a whole corpus, the metric chosen by output type.
- [`estimate_quality()`](#ek.estimate_quality) – *reference-free* (online): gather signals, calibrate,
  validate, and decide accept/flag/block with no gold answer.

Everything swappable is a `typing.Protocol` strategy resolved from
[`ek.registry`](ek.registry.html.md#module-ek.registry) and injected as a keyword-only argument with a smart default –
so the simple path Just Works and every layer stays replaceable (open-closed).

Quickstart:

```pycon
>>> import ek
>>> round(ek.score("hello wrld", "hello world").value, 3)   # CER by default
0.091
>>> ek.evaluate([("ct", "cat"), ("dg", "dog")], metric="cer").n
2
```

The two-layer data model ([`GraphGrammar`](#ek.GraphGrammar) for the schema + cost weights,
[`AnnotatedExtraction`](#ek.AnnotatedExtraction) for per-extraction verification metadata) is the SSOT
every component plugs into – see [`ek.base`](ek.base.html.md#module-ek.base). Persistence is local-file `dol`
stores under `~/.local/share/ek/` – see [`ek.stores`](ek.stores.html.md#module-ek.stores).

### Functions

| [`score`](#ek.score)(pred, gold, \*[, grammar, metric, ...])      | Score one prediction against one gold reference (reference-based).                                                                     |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`evaluate`](#ek.evaluate)(cases, \*[, metric, grammar, ...])        | Aggregate many comparisons into a [`Report`](ek.base.html.md#ek.base.Report) (corpus level).               |
| [`estimate_quality`](#ek.estimate_quality)(extraction, \*[, sources, ...])   | Estimate the quality of an extraction with no gold reference.                                                                          |
| [`evaluate_store`](#ek.evaluate_store)(predict, gold, \*[, metric, ...])   | Run `predict` over a gold store and score it per slice.                                                                                |
| [`save_baseline`](#ek.save_baseline)(report, name, \*[, rootdir])         | Freeze a report's aggregate + per-slice scores as a named baseline.                                                                    |
| [`load_baseline`](#ek.load_baseline)(name, \*[, rootdir])                 | Load a named baseline (or `None` if it does not exist).                                                                                |
| [`regression_gate`](#ek.regression_gate)(report, baseline, \*[, ...])       | Fail if `report` regresses beyond `tolerance` vs a baseline, per slice.                                                                |
| [`cohen_kappa`](#ek.cohen_kappa)(rater_a, rater_b)                      | Cohen's kappa for two raters on nominal labels (chance-corrected agreement).                                                           |
| [`percent_agreement`](#ek.percent_agreement)(rater_a, rater_b)                | Raw fraction of items two raters labelled identically.                                                                                 |
| [`krippendorff_alpha`](#ek.krippendorff_alpha)(reliability_data, \*[, level])  | Krippendorff's alpha -- the general inter-annotator agreement coefficient.                                                             |
| [`default_cost_weight`](#ek.default_cost_weight)(grammar, ref)                  | The default `CostWeight`: read the importance weight off the schema.                                                                   |
| [`default_canonicalizer`](#ek.default_canonicalizer)()                            | A conservative default: NFC -> casefold -> collapse whitespace.                                                                        |
| [`table_from_ocr`](#ek.table_from_ocr)(ocr_result, \*[, parser])           | Extract a normalized [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult`.               |
| [`has_table_structure`](#ek.has_table_structure)(ocr_result, \*[, parser])      | Whether [`table_from_ocr()`](#ek.table_from_ocr) recovers a non-empty table from `ocr_result`.                |
| [`resolve_table_parser`](#ek.resolve_table_parser)(ref)                          | Resolve a table parser reference (a registered name or a callable) to a callable.                                                      |
| [`engine_yields_tables`](#ek.engine_yields_tables)(engine)                       | Whether `engine`'s capability profile claims scoreable table structure.                                                                |
| [`rover`](#ek.rover)(hypotheses, \*[, use_confidence, ...])       | Align N hypotheses, vote per slot, and emit consensus + per-position agreement.                                                        |
| [`expected_calibration_error`](#ek.expected_calibration_error)(probs, correct, \*)     | Expected Calibration Error: weighted mean gap between confidence and accuracy.                                                         |
| [`reliability_curve`](#ek.reliability_curve)(probs, correct, \*[, n_bins])    | Per-bin `{confidence, accuracy, count}` for a reliability diagram.                                                                     |
| [`save_calibrator`](#ek.save_calibrator)(calibrator, name, \*[, rootdir])   | Persist a fitted calibrator's parameters to the `calibrators` store.                                                                   |
| [`load_calibrator`](#ek.load_calibrator)(name, \*[, rootdir])               | Reconstruct a persisted calibrator by name (dispatched on its `kind`).                                                                 |
| [`mean_probability`](#ek.mean_probability)(probs)                            | Fuse calibrated per-family probabilities by their mean (the default combiner).                                                         |
| [`min_probability`](#ek.min_probability)(probs)                             | Fuse calibrated per-family probabilities by their minimum (conservative).                                                              |
| [`risk_coverage_curve`](#ek.risk_coverage_curve)(probs, correct, \*[, ...])     | The risk-coverage trade-off: `{threshold, coverage, selective_risk}` per point.                                                        |
| [`split_conformal_quantile`](#ek.split_conformal_quantile)(scores, alpha)            | The split-conformal threshold: the `ceil((n+1)(1-alpha)) / n` empirical quantile.                                                      |
| [`checksum_validator`](#ek.checksum_validator)([kind, field_name, layer])      | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value failing a named checksum.            |
| [`regex_validator`](#ek.regex_validator)(pattern, \*[, field_name, layer])  | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not fully matching `pattern`.        |
| [`range_validator`](#ek.range_validator)(lo, hi, \*[, field_name, layer])   | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a number outside `[lo, hi]`.                 |
| [`enum_validator`](#ek.enum_validator)(members, \*[, field_name, layer])   | A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not in an allowed set.               |
| [`schema_validator`](#ek.schema_validator)(value, \*[, spec])                | A spec-driven [`Validator`](ek.base.html.md#ek.base.Validator): type and `domain` checks from a FieldSpec. |
| [`totals_consistent`](#ek.totals_consistent)(record, \*, total_key, ...)      | Flag when `record[total_key]` does not equal the sum of `item_keys` (±tol).                                                            |
| [`luhn_check`](#ek.luhn_check)(number)                                 | Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs.                                                                        |
| [`iban_check`](#ek.iban_check)(iban)                                   | IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1).                                                                 |
| [`isbn_check`](#ek.isbn_check)(value)                                  | Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored).                                                              |
| [`validation_pipeline`](#ek.validation_pipeline)(\*layers[, ...])               | Compose validators/correctors into a cheapest -> most-expensive pipeline.                                                              |
| [`canonicalize_corrector`](#ek.canonicalize_corrector)(normalize, \*[, ...])       | L0: fold a value to canonical form, emitting a CORRECT finding when it changes.                                                        |
| [`lexicon_corrector`](#ek.lexicon_corrector)(vocabulary, \*[, ...])           | L2: resolve a value against a **closed** vocabulary by fuzzy match (`rapidfuzz`).                                                      |
| [`cross_field_validator`](#ek.cross_field_validator)(predicate, \*, message)      | FLAG a record when `predicate(record)` is False (a general cross-field check).                                                         |
| [`ordering_validator`](#ek.ordering_validator)(keys, \*[, strict, layer])      | FLAG a record whose comparable values at `keys` are not ascending.                                                                     |
| [`benford_findings`](#ek.benford_findings)(numbers, \*[, field, tol, ...])   | FLAG a numeric field whose first-digit distribution deviates from Benford's law.                                                       |
| [`zscore_anomaly_findings`](#ek.zscore_anomaly_findings)(numbers, \*[, field, ...]) | FLAG numeric values that are robust-z-score outliers (median + MAD based).                                                             |
| [`lm_surprisal_validator`](#ek.lm_surprisal_validator)(scorer, \*, threshold)      | L3: FLAG a value whose language-model surprisal crosses `threshold`.                                                                   |
| [`llm_corrector`](#ek.llm_corrector)(correct_fn, \*[, only_flagged, ...]) | L5: a **gated** neural/LLM corrector -- the only layer that invents content.                                                           |
| [`stop_on_correction`](#ek.stop_on_correction)(findings)                       | Stop policy: end the pass as soon as any layer proposes a correction.                                                                  |
| [`stop_on_flag`](#ek.stop_on_flag)(findings)                             | Stop policy: end the pass at the first FLAG (fail fast).                                                                               |
| [`register`](#ek.register)(namespace, name[, obj])                   | Register `obj` under `namespace`/`name` (usable as a decorator).                                                                       |
| [`get`](#ek.get)(namespace, name)                               | Resolve a registered strategy by name (discovering entry points first).                                                                |
| [`resolve`](#ek.resolve)(namespace, ref, \*[, default])             | Coerce a strategy reference to a callable.                                                                                             |
| [`names`](#ek.names)(namespace)                                   | All registered names in a namespace (after entry-point discovery).                                                                     |
| [`requires_extra`](#ek.requires_extra)(extra, \*[, packages])              | Decorator: fail with an actionable install hint if an extra is missing.                                                                |
| [`check_requirements`](#ek.check_requirements)(\*[, engine, extra])            | Report what an OCR engine or feature extra needs, without raising.                                                                     |
| [`json_store`](#ek.json_store)(kind, \*[, rootdir])                    | A JSON `MutableMapping` for one `kind` of artifact.                                                                                    |
| [`mall`](#ek.mall)(\*[, rootdir])                                | The full stores-of-stores mall (one JSON store per `KINDS`).                                                                           |
| [`app_folder`](#ek.app_folder)()                                       | The root `ek` data folder (e.g. `~/.local/share/ek`).                                                                                  |
| [`persistent_cache`](#ek.persistent_cache)([func, kind, key, rootdir])       | Memoize a plain function's result in a `kind` JSON store, across sessions.                                                             |
| [`cache_this`](#ek.cache_this)([func, cache, key, pre_cache, ...])     | Unified caching decorator for properties and methods with persistent storage support.                                                  |
| [`run_suite`](#ek.run_suite)(agent, tasks, \*[, k, check, ...])       | Run `agent` over a task suite, `k` trials per task, and report reliability + cost.                                                     |
| [`pass_at_k`](#ek.pass_at_k)(\*, n, c, k)                             | Unbiased `pass@k`: probability that **at least one** of k samples succeeds.                                                            |
| [`pass_hat_k`](#ek.pass_hat_k)(\*, n, c, k)                            | Unbiased `pass^k`: probability that **all k** independent trials succeed.                                                              |

### Classes

| [`GateResult`](#ek.GateResult)(passed, metric, higher_is_better, ...)   | Outcome of a [`regression_gate()`](#ek.regression_gate) check.                                             |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| [`GraphGrammar`](#ek.GraphGrammar)([node_types, edge_types])              | The frozen schema SSOT: typed nodes/edges plus their importance weights.                                                           |
| [`FieldSpec`](#ek.FieldSpec)(name[, type, importance, domain, ...])    | A leaf attribute of a node: its type, validation domain and cost weight.                                                           |
| [`NodeType`](#ek.NodeType)(name[, fields, importance])                | A node kind in the typed graph (e.g. an "invoice" or a "line item").                                                               |
| [`EdgeType`](#ek.EdgeType)(name, src, dst[, importance])              | A directed relation kind between two [`NodeType`](#ek.NodeType) names.                              |
| [`TypeRef`](#ek.TypeRef)(kind, name[, field])                        | A reference into the schema, so an injected cost function can read weights.                                                        |
| [`AnnotatedExtraction`](#ek.AnnotatedExtraction)(grammar[, estimates])           | Layer B: the verification metadata, referencing a frozen Layer-A grammar.                                                          |
| [`FieldEstimate`](#ek.FieldEstimate)(value[, raw_signals, ...])            | One extracted value plus all of its reference-free verification metadata.                                                          |
| [`NodePath`](#ek.NodePath)(node_id, node_type[, field])               | Addresses a node (or one of its fields) in an extracted graph; keys Layer B.                                                       |
| [`Provenance`](#ek.Provenance)([engine, source_span, bbox, ...])        | Where an extracted value came from, for audit and click-to-source review.                                                          |
| [`Finding`](#ek.Finding)(field, layer[, severity, message, ...])     | One validator observation about a value: a flag, optionally a correction.                                                          |
| [`Decision`](#ek.Decision)(\*values)                                  | Terminal action a selective-prediction gate emits for an extracted value.                                                          |
| [`Severity`](#ek.Severity)(\*values)                                  | Whether a [`Finding`](#ek.Finding) merely flags or can also correct a value.                       |
| [`Score`](#ek.Score)(value[, precision, recall, f1, ...])          | The result of comparing one prediction against one reference.                                                                      |
| [`Report`](#ek.Report)([metric, aggregate, n, scores, ...])         | Aggregate of many [`Score`](#ek.Score) s over a corpus, with optional per-slice cuts.            |
| [`QualityReport`](#ek.QualityReport)([calibrated_confidence, ...])         | The result of a reference-free quality estimate for one extraction.                                                                |
| [`Metric`](#ek.Metric)(\*args, \*\*kwargs)                          | Compares one prediction to one gold reference, returning a [`Score`](#ek.Score).                 |
| [`Validator`](#ek.Validator)(\*args, \*\*kwargs)                       | Reference-free check on a value, yielding zero or more [`Finding`](#ek.Finding) s.                 |
| [`Calibrator`](#ek.Calibrator)(\*args, \*\*kwargs)                      | Maps a raw scalar signal to a probability after being `fit` on labels.                                                             |
| [`DecisionPolicy`](#ek.DecisionPolicy)(\*args, \*\*kwargs)                  | Turns a (calibrated) confidence into an accept/flag/block [`Decision`](#ek.Decision).               |
| [`Signal`](#ek.Signal)(\*args, \*\*kwargs)                          | A reference-free quality signal: extractor output -> raw scalar score.                                                             |
| [`Canonicalizer`](#ek.Canonicalizer)([steps, version, audit_marks])        | An ordered, named, versioned pipeline of `str -> str` steps.                                                                       |
| [`StringMetric`](#ek.StringMetric)([mode, canonicalizer])                 | CER (`mode='cer'`) or WER (`mode='wer'`) as a [`Metric`](ek.base.html.md#ek.base.Metric).              |
| [`FieldMetric`](#ek.FieldMetric)(\*[, canonicalizer])                    | Per-field precision/recall/F1 for two dict records.                                                                                |
| [`TypedGraphMetric`](#ek.TypedGraphMetric)([grammar, weights, ...])           | Cost-weighted, type-aware typed-graph edit distance as a [`Metric`](ek.base.html.md#ek.base.Metric).   |
| [`TypedGraph`](#ek.TypedGraph)([nodes, edges])                          | A typed graph: typed nodes plus directed typed edges between them.                                                                 |
| [`TypedNode`](#ek.TypedNode)(node_id, node_type[, fields])             | A node in a typed graph: an id, a node-type tag, and its field values.                                                             |
| [`TypedEdge`](#ek.TypedEdge)(src, dst[, edge_type])                    | A directed, typed relation between two node ids.                                                                                   |
| [`AnlsMetric`](#ek.AnlsMetric)(\*[, threshold])                         | ANLS / ANLS\* (nested-JSON) similarity as a [`Metric`](ek.base.html.md#ek.base.Metric).                |
| [`SpanF1Metric`](#ek.SpanF1Metric)([scheme, tagging_scheme])              | Entity/slot precision/recall/F1 under an **explicit** [`MatchScheme`](#ek.MatchScheme).                |
| [`MatchScheme`](#ek.MatchScheme)(\*values)                               | The (backend, scheme) contract a span/slot F1 is computed under.                                                                   |
| [`TedsMetric`](#ek.TedsMetric)(\*[, structure_only])                    | TEDS / TEDS-Struct as a [`Metric`](ek.base.html.md#ek.base.Metric) (clean-room on `apted`).            |
| [`GritsMetric`](#ek.GritsMetric)(\*[, structure_only])                   | GriTS-Top / GriTS-Con as a [`Metric`](ek.base.html.md#ek.base.Metric) (pure-Python, P/R aware).        |
| [`Table`](#ek.Table)([rows])                                       | A table as an ordered list of rows of [`Cell`](#ek.Cell) s (the logical cells).                 |
| [`Cell`](#ek.Cell)([text, rowspan, colspan])                      | One logical table cell: its text plus its row/column span (>=1 each).                                                              |
| [`RoverConsensus`](#ek.RoverConsensus)([tokens, slots, agreement, ...])     | The result of a ROVER pass over N hypotheses.                                                                                      |
| [`AgreementSignal`](#ek.AgreementSignal)([cost_tier, use_confidence, ...])   | ROVER multi-engine agreement as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).           |
| [`VerifierSignal`](#ek.VerifierSignal)([validators, cost_tier])             | Deterministic verifier evidence as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).        |
| [`LogprobSignal`](#ek.LogprobSignal)([aggregator, alpha, cost_tier])       | Aggregate token log-probabilities into one field score (a [`Signal`](ek.base.html.md#ek.base.Signal)). |
| [`IntrinsicConfidenceSignal`](#ek.IntrinsicConfidenceSignal)([pool, cost_tier])        | Aggregate an extractor's per-unit confidences into a field score.                                                                  |
| [`PlattCalibrator`](#ek.PlattCalibrator)([a, b, max_iter, kind])             | Platt scaling: `sigmoid(a * score + b)`, fit by Newton/IRLS on labels.                                                             |
| [`IsotonicCalibrator`](#ek.IsotonicCalibrator)([x, y, kind])                    | Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.                                                       |
| [`TemperatureCalibrator`](#ek.TemperatureCalibrator)([T, kind, t_min, ...])        | Temperature scaling: `sigmoid(logit / T)` with one `T` fit on a holdout.                                                           |
| [`GroupCalibrator`](#ek.GroupCalibrator)([factory, by_group, pooled, ...])   | Per-group (Mondrian) calibration: one calibrator per `NodeType`/`FieldSpec`.                                                       |
| [`SignalFamilyCalibrator`](#ek.SignalFamilyCalibrator)([calibrators, ...])          | One calibrator per signal family, fused by a documented combination rule.                                                          |
| [`CostSensitiveGate`](#ek.CostSensitiveGate)([rho, block_threshold, ...])      | Accept/flag/block from the cost ratio `rho = c_FN / c_FP` on calibrated `p`.                                                       |
| [`ConformalGate`](#ek.ConformalGate)([alpha, nonconformity, q, \_cal])     | Split-conformal accept/flag with a finite-sample **marginal** guarantee.                                                           |
| [`GroupConformalGate`](#ek.GroupConformalGate)([alpha, nonconformity, ...])     | Mondrian (class-conditional) conformal: one [`ConformalGate`](#ek.ConformalGate) per group.              |
| [`RiskControlGate`](#ek.RiskControlGate)([target, loss, loss_bound, lam])    | Conformal Risk Control: accept `p >= lambda` bounding the accepted-error rate.                                                     |
| [`ValidationResult`](#ek.ValidationResult)(original, value[, findings])       | The outcome of running a value through a [`validation_pipeline()`](#ek.validation_pipeline).                   |
| [`Corrector`](#ek.Corrector)(\*args, \*\*kwargs)                       | A [`Validator`](ek.base.html.md#ek.base.Validator) that may also CORRECT.                              |
| [`Episode`](#ek.Episode)([task_id, trajectory, output, ...])         | Layer B: one agent run over one task -- the object agent evaluation scores.                                                        |
| [`TaskSpec`](#ek.TaskSpec)(task_id[, input, gold, value, ...])        | One task in a suite: its input, its oracle, its value, and its allowed tools.                                                      |
| [`ToolSpec`](#ek.ToolSpec)(name[, params, importance, destructive])   | One tool the agent may call: its argument schema and its error costs.                                                              |

### *class* ek.AgreementSignal(cost_tier=3, use_confidence=True, conf_weight=0.5, null_conf=0.7, tokenize=None, max_tokens=5000)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

ROVER multi-engine agreement as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).

Cost tier 3 (`N` engine runs): try the deterministic verifier layer and any
free intrinsic confidence first. Called on a collection of hypotheses, it runs
[`rover()`](#ek.rover) and returns the mean per-position agreement as the raw signal –
*uncalibrated*, like every signal, so it must pass through a
[`Calibrator`](ek.base.html.md#ek.base.Calibrator) before any gate reads it.

### Example

```pycon
>>> sig = AgreementSignal()
>>> round(sig(["the cat sat", "the cat sit", "the bat sat"]), 3)
0.778
```

### *class* ek.AnlsMetric(, threshold=0.5)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

ANLS / ANLS\* (nested-JSON) similarity as a [`Metric`](ek.base.html.md#ek.base.Metric).

Handles a bare string (classic ANLS) or an arbitrarily nested
dict/list/tuple/`None` structure (ANLS\*); the backend dispatches on type.

* **Parameters:**
  **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – ANLS zeroes a per-leaf similarity below this value before
  averaging (the classic ANLS 0.5 cut tolerates minor OCR/spelling
  noise). `anls_star` exposes no parameter for this – it reads a
  class constant `ANLSTree.THRESHOLD` (default `0.5`) – so the metric
  applies the threshold by setting that constant around the call and
  restoring it. **Caveat:** this mutates a process-global, so it is not
  thread-safe; concurrent `AnlsMetric` calls with different thresholds in
  the same process can interfere. Defaults to `0.5`.

#### aggregate(scores)

Corpus ANLS = mean of per-item ANLS (each is already length-normalized).

Unlike CER/WER (which accumulate global edit counts) or field-F1 (micro-TP/
FP/FN), ANLS is defined as the *average* normalized similarity over the
evaluation set – the “A” in ANLS – so the corpus statistic is the plain
mean of the per-item scores.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.AnnotatedExtraction(grammar, estimates=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Layer B: the verification metadata, referencing a frozen Layer-A grammar.

The grammar is held *by reference* and never mutated; `estimates` maps a
[`NodePath`](#ek.NodePath) to the [`FieldEstimate`](#ek.FieldEstimate) for that node/field.

### *class* ek.Calibrator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Maps a raw scalar signal to a probability after being `fit` on labels.

### *class* ek.Canonicalizer(steps=(), , version='', audit_marks=False)

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

### *class* ek.Cell(text='', rowspan=1, colspan=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One logical table cell: its text plus its row/column span (>=1 each).

### *class* ek.ConformalGate(alpha=0.1, nonconformity=<function ConformalGate.<lambda>>, q=inf, \_cal=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Split-conformal accept/flag with a finite-sample **marginal** guarantee.

Fit on `(calibrated_prob, correct)`: the nonconformity scores of the *correct*
calibration items set a threshold `q` such that, among truly-correct test items
exchangeable with them, at most `alpha` are flagged. The guarantee is
*marginal*, not per-field-type – use [`GroupConformalGate`](#ek.GroupConformalGate) for that.

* **Parameters:**
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Target false-flag rate on correct items (e.g. `0.1`).
  * **nonconformity** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Maps calibrated confidence -> nonconformity (default
    `1 - confidence`: low confidence is nonconforming).

#### fit(probs, correct)

Fit `q` from the nonconformity scores of the *correct* calibration items.

Non-finite confidences are dropped (a single `NaN` would otherwise sort to
the end and corrupt the quantile, silently breaking the coverage guarantee).

* **Return type:**
  [`ConformalGate`](ek.qe.decide.html.md#ek.qe.decide.ConformalGate)

#### p_value(confidence)

Per-instance conformal p-value (small = atypical of correct items).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.Corrector(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A [`Validator`](ek.base.html.md#ek.base.Validator) that may also CORRECT.

Same callable shape as a Validator, but it can emit a `severity=CORRECT`
[`Finding`](ek.base.html.md#ek.base.Finding) whose `suggestion` is the proposed replacement value.
The `layer` attribute names its place on the cheap->expensive spine.

### *class* ek.CostSensitiveGate(rho=1.0, block_threshold=0.0, accept_threshold=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Accept/flag/block from the cost ratio `rho = c_FN / c_FP` on calibrated `p`.

Accept when the expected cost of accepting is below that of routing to a human.
Accepting risks an undetected error: cost `(1 - p) * c_FN`. Flagging a value
that was actually *correct* wastes a review: cost `p * c_FP` (the false-positive
cost is incurred only with probability `p`, not unconditionally). So accept iff
`(1 - p) * c_FN <= p * c_FP`, i.e. the Bayes-optimal threshold
`p >= c_FN / (c_FN + c_FP) = rho / (1 + rho)`. A larger `rho` (misses much
costlier than reviews) raises the bar to auto-accept (`tau -> 1` as
`rho -> inf`); `rho = 1` gives the symmetric `tau = 0.5`.
`block_threshold` adds an optional hard-fail band at the bottom.

* **Parameters:**
  * **rho** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Cost ratio `c_FN / c_FP` (>= 0). The single lever; no magic numbers.
  * **block_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Calibrated probability at or below which to `block` rather
    than `flag` (default `0` -> never block on probability alone; a
    failed verifier is what forces a block, upstream).
  * **accept_threshold** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Override the derived accept threshold explicitly.

#### *property* tau *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The accept threshold on calibrated probability (derived from `rho`).

### *class* ek.Decision(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Terminal action a selective-prediction gate emits for an extracted value.

### *class* ek.DecisionPolicy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Turns a (calibrated) confidence into an accept/flag/block [`Decision`](#ek.Decision).

### *class* ek.EdgeType(name, src, dst, importance=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A directed relation kind between two [`NodeType`](#ek.NodeType) names.

### *class* ek.Episode(task_id='', trajectory=Trajectory(), output=None, final_state=None, cost=None, success=None, run=None, provenance=None, meta=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Layer B: one agent run over one task – the object agent evaluation scores.

The direct analog of `AnnotatedExtraction`: it rides alongside the frozen Layer-A
grammar and carries the run’s metadata. It is scored *offline* against gold
([`ek.score()`](#ek.score) / [`ek.evaluate()`](#ek.evaluate)) and estimated *online* against a judge or a
consensus ([`ek.estimate_quality()`](#ek.estimate_quality)) – the same object, both halves.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Which task this episode answers (the k-trial grouping key).
  * **trajectory** ([`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory)) – The tool calls taken.
  * **output** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The agent’s final user-facing answer.
  * **final_state** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The end state of the world (the DB, the filesystem) – what a
    state-based oracle actually grades. Success is a *final-state* check, not a
    surface-text check.
  * **cost** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)]) – The `Cost` consumed.
  * **success** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Filled in by a checker; `None` until graded.
  * **run** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance)]) – The `RunProvenance` (seed, model, simulator, suite version).
  * **provenance** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Provenance`](ek.base.html.md#ek.base.Provenance)]) – Optional core [`Provenance`](ek.base.html.md#ek.base.Provenance) for click-to-source audit.
  * **meta** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Anything else the bridge captured.

#### *property* calls *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pairs taken, in order.

#### graded(success)

A copy with `success` set (never mutates – grading stays idempotent).

`meta` is copied too: `dataclasses.replace` is a *shallow* copy, so without this
the “copy” would share the original’s mutable `meta` dict.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### *class* ek.FieldEstimate(value, raw_signals=dict(), confidence=None, findings=(), provenance=None, decision=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One extracted value plus all of its reference-free verification metadata.

### *class* ek.FieldMetric(, canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-field precision/recall/F1 for two dict records.

* **Parameters:**
  **canonicalizer** – Optional `str -> str` applied to string values before
  comparison (e.g. casefold/whitespace folding).

#### aggregate(scores)

Micro-averaged F1 over the corpus (sum TP/FP/FN, then divide).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.FieldSpec(name, type='string', importance=1.0, domain=(), normalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A leaf attribute of a node: its type, validation domain and cost weight.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Attribute name.
  * **type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A free-form type tag (e.g. `"string"`, `"number"`, `"date"`);
    `ek` does not impose a type system – the tag drives metric/validator
    dispatch and is yours to define.
  * **importance** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-field cost weight; the lever that makes “two extra digits
    on a donation amount” outweigh “a misspelled city”. Defaults to `1.0`.
  * **domain** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Optional allowed values: an enum tuple, a `(lo, hi)` range, or a
    regex string – consumed by validators, never by metrics.
  * **normalizer** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional key naming a canonicalizer (resolved from the registry)
    to apply to this field before comparison.

### *class* ek.Finding(field, layer, severity=Severity.FLAG, message='', suggestion=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One validator observation about a value: a flag, optionally a correction.

A `suggestion` is present iff the producing layer can *correct* (only the
deterministic canonicalizer and the gated LLM corrector freely do so). The
original value is always retained elsewhere; a Finding never mutates.

### *class* ek.GateResult(passed, metric, higher_is_better, tolerance, aggregate_current=None, aggregate_baseline=None, regressions=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Outcome of a [`regression_gate()`](#ek.regression_gate) check.

### *class* ek.GraphGrammar(node_types=dict(), edge_types=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The frozen schema SSOT: typed nodes/edges plus their importance weights.

Methods return the cost weight for a referenced type, defaulting to `1.0` for
anything not declared, so a partial grammar is always usable.

#### cost(ref)

Importance weight for any [`TypeRef`](#ek.TypeRef) (the default `CostWeight`).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### edge_cost(name)

Importance weight of an edge type (`1.0` if undeclared).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### field_cost(node, field_name)

Importance weight of a field on a node type (`1.0` if undeclared).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### node_cost(name)

Importance weight of a node type (`1.0` if undeclared).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.GritsMetric(, structure_only=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

GriTS-Top / GriTS-Con as a [`Metric`](ek.base.html.md#ek.base.Metric) (pure-Python, P/R aware).

* **Parameters:**
  **structure_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True`, score cell topology only (GriTS-Top); when
  `False` (default), average topology with cell-content similarity
  (GriTS-Con).

#### aggregate(scores)

Corpus GriTS F1: pool overlap and per-table cell counts, then divide once.

GriTS precision = overlap / 

```
|pred cells|
```

 and recall = overlap / 

```
|gold cells|
```

;
the corpus statistic micro-averages by summing the numerators and
denominators globally – never a mean of per-table F1s.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.GroupCalibrator(factory=<class 'ek.qe.calibrate.PlattCalibrator'>, by_group=dict(), pooled=None, kind='group')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-group (Mondrian) calibration: one calibrator per `NodeType`/`FieldSpec`.

Distribution-free *conditional* (per-field-type) coverage is impossible
in general (Barber et al. 2019); calibrating separately per group restores it
approximately. `fit` takes a parallel `groups` sequence; `__call__` routes
by group key, falling back to a pooled calibrator for unseen groups.

* **Parameters:**
  **factory** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Zero-arg callable producing a fresh per-group calibrator (default
  [`PlattCalibrator`](#ek.PlattCalibrator)).

#### factory

alias of [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator)

#### fit(scores, correct, , groups)

Fit one calibrator per distinct group key, plus a pooled fallback.

* **Return type:**
  [`GroupCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.GroupCalibrator)

#### to_dict()

Serialize each per-group calibrator (keyed by stringified group) plus the
pooled fallback, so a fitted Mondrian calibrator round-trips like the others.
Group keys are stringified (they are `NodeType`/`FieldSpec` names).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ek.GroupConformalGate(alpha=0.1, nonconformity=<function GroupConformalGate.<lambda>>, by_group=dict(), pooled=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Mondrian (class-conditional) conformal: one [`ConformalGate`](#ek.ConformalGate) per group.

Restores approximate per-field-type validity by calibrating separately per group
key (e.g. `NodeType`/`FieldSpec`), with a pooled fallback for unseen groups.

### *class* ek.IntrinsicConfidenceSignal(pool='min', cost_tier=2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate an extractor’s per-unit confidences into a field score.

* **Parameters:**
  * **pool** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"min"` (weakest unit), `"mean"`, or `"geo_mean"` over the
    per-unit confidences (already probabilities in `[0, 1]`).
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `2` – free (the posteriors are emitted at inference).

### Example

```pycon
>>> IntrinsicConfidenceSignal(pool="min")([0.99, 0.4, 0.95])
0.4
```

### *class* ek.IsotonicCalibrator(x=list(), y=list(), kind='isotonic')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.

More flexible than Platt; needs more calibration data and can overfit small
sets. Predicts by linear interpolation between fitted points, clipped at the
ends.

#### fit(scores, correct)

Fit the monotonic step function to `(score, correct)` pairs.

* **Return type:**
  [`IsotonicCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.IsotonicCalibrator)

### *class* ek.LogprobSignal(aggregator=<function geo_mean>, alpha=0.6, cost_tier=2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate token log-probabilities into one field score (a [`Signal`](ek.base.html.md#ek.base.Signal)).

* **Parameters:**
  * **aggregator** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A logprob aggregator (name registered under `aggregators` or a
    callable). Defaults to `geo_mean()`.
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Length penalty passed to `length_normalized()` only.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `2` – free intrinsic signal (already computed at inference).

#### aggregator()

Geometric mean of token probabilities, `exp(mean log p)` (perplexity-inverse).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.MatchScheme(\*values)

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

### *class* ek.Metric(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Compares one prediction to one gold reference, returning a [`Score`](#ek.Score).

### *class* ek.NodePath(node_id, node_type, field=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Addresses a node (or one of its fields) in an extracted graph; keys Layer B.

### *class* ek.NodeType(name, fields=dict(), importance=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A node kind in the typed graph (e.g. an “invoice” or a “line item”).

### *class* ek.PlattCalibrator(a=1.0, b=0.0, max_iter=100, kind='platt')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Platt scaling: `sigmoid(a * score + b)`, fit by Newton/IRLS on labels.

The default calibrator: works on any scalar (aggregated confidence/logprob), no
logits required. Uses Platt’s target smoothing so it does not overfit small
calibration sets.

* **Parameters:**
  **max_iter** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Newton iterations (converges in a handful for a 2-parameter model).

#### fit(scores, correct)

Fit `a, b` to maximise the likelihood of `correct` given `scores`.

* **Return type:**
  [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator)

### *class* ek.Provenance(engine='', source_span=None, bbox=None, raw_transcripts=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Where an extracted value came from, for audit and click-to-source review.

`bbox` is intentionally typed `Any` so `ek` core need not import any OCR
package; the OCR instance stores normalized `[0, 1]` top-left-origin geometry.

### *class* ek.QualityReport(calibrated_confidence=None, decision=None, findings=(), raw_signals=dict(), provenance=dict(), per_field=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of a reference-free quality estimate for one extraction.

For a single value the scalar fields carry the verdict. For a whole
[`AnnotatedExtraction`](#ek.AnnotatedExtraction), `per_field` maps each [`NodePath`](#ek.NodePath) to its
updated [`FieldEstimate`](#ek.FieldEstimate) (confidence/findings/decision filled in), and the
scalar fields summarize: `calibrated_confidence` is the weakest field, and
`decision` the most severe (`block` > `flag` > `accept`).

### *class* ek.Report(metric='', aggregate=None, n=0, scores=list(), per_slice=dict(), detail=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Aggregate of many [`Score`](#ek.Score) s over a corpus, with optional per-slice cuts.

`aggregate` is the corpus-level headline (computed by the metric’s own
aggregator – e.g. globally accumulated WER – not a naive mean of per-item
values). `per_slice` maps a slice label to its own aggregate.

### *class* ek.RiskControlGate(target=0.05, loss=<function RiskControlGate.<lambda>>, loss_bound=1.0, lam=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Conformal Risk Control: accept `p >= lambda` bounding the accepted-error rate.

Picks the *smallest* threshold `lambda` (the most coverage) whose finite-sample
risk bound is at or below `target`. The controlled quantity is
`E[ loss(item) * 1(accepted) ]` – with the default 0/1 loss, the population
rate of accepted-yet-wrong items.

* **Parameters:**
  * **target** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Upper bound on the accepted-error rate (e.g. `0.05`).
  * **loss** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – `correct -> loss` in `[0, loss_bound]`; default `0` if correct
    else `1`. Any monotone loss bounded by `loss_bound` works.
  * **loss_bound** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The loss’s upper bound `B` – the CRC finite-sample slack is
    `+B/(n+1)`. Must match `loss`’s range or the guarantee is invalid;
    keep the default `1.0` for the 0/1 loss.

#### fit(probs, correct)

Find the smallest accept threshold whose risk bound is <= `target`.

* **Return type:**
  [`RiskControlGate`](ek.qe.decide.html.md#ek.qe.decide.RiskControlGate)

### *class* ek.RoverConsensus(tokens=list(), slots=list(), agreement=list(), n_engines=0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of a ROVER pass over N hypotheses.

#### *property* mean_agreement *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Mean per-position agreement over consensus tokens, a reference-free
confidence in `[0, 1]`: `1.0` means every engine agreed at every emitted
position; lower means at least one engine dissented.

An **empty** consensus (no token survived voting – NULL won every slot) means
the engines agreed on *nothing*, so this returns `0.0` when 2+ engines were
fused, NOT a misleading `1.0` (which would feed a maximal raw confidence to
the calibrator/gate). A single engine – or none – is vacuously `1.0`.

#### *property* text *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The consensus transcription (space-joined winning tokens).

### *class* ek.Score(value, precision=None, recall=None, f1=None, metric='', detail=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of comparing one prediction against one reference.

`value` is the headline number (higher is better unless a metric documents
otherwise). The optional decomposition fields let structured metrics carry
precision/recall/F1, an alignment, and raw counts (e.g. edit operations) so
that corpus aggregation can be done *correctly* (e.g. global WER accumulation)
rather than by averaging per-item scores. `float(score)` returns `value`.

### *class* ek.Severity(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Whether a [`Finding`](#ek.Finding) merely flags or can also correct a value.

### *class* ek.Signal(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A reference-free quality signal: extractor output -> raw scalar score.

The `cost_tier` attribute (lower is cheaper) lets an escalation policy run
cheap signals first and pay for expensive ones only on residual uncertainty.

### *class* ek.SignalFamilyCalibrator(calibrators=dict(), combine=<function mean_probability>, factory=<class 'ek.qe.calibrate.PlattCalibrator'>, fusion=None, kind='signal_family')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One calibrator per signal family, fused by a documented combination rule.

Averaging heterogeneous raw signals (an LLM-judge score, a token logprob, a ROVER
agreement) and calibrating the mean discards exactly what a calibrator needs:
each family has its own scale, reliability and direction of miscalibration. This
calibrates every signal *by name* with its own calibrator, then fuses the
calibrated probabilities with `combine`.

Pass it (or simply a `Mapping[str, Calibrator]`) as
`estimate_quality(..., calibrator=...)`.

**The fused value must itself be calibrated before it is gated.** Fusing
calibrated probabilities (by mean, min, …) does not yield a calibrated
probability: agreeing families make the mean under-confident, a weak family
drags it toward the base rate. So [`fit()`](#ek.SignalFamilyCalibrator.fit) also learns a second-stage
`fusion` calibrator on `combine`’s output, and `__call__` applies it.
Without a `fusion` calibrator (e.g. a hand-built mapping),
[`ek.estimate_quality()`](#ek.estimate_quality) refuses to gate a value fused from two or more
families unless told `assume_calibrated=True` (Hard Rule 1). A value from a
single family is that family’s calibrated probability and gates as is.

A field’s own upstream `confidence` is just one more family here, named
`"confidence"`: include it in the training rows under that name.

* **Parameters:**
  * **calibrators** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – `signal name -> Calibrator`. A `"*"` entry is the fallback
    for any signal without its own entry. A signal with neither is refused
    (`ValueError`) rather than silently dropped or passed through raw
    (Hard Rule 1).
  * **combine** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – `Mapping[str, float] -> float` fusing the calibrated values;
    [`mean_probability()`](#ek.mean_probability) (default) or [`min_probability()`](#ek.min_probability), or any
    callable of that shape (e.g. a learned stacker).
  * **factory** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Zero-arg callable producing a fresh calibrator, used by [`fit()`](#ek.SignalFamilyCalibrator.fit).
  * **fusion** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Second-stage calibrator applied to `combine`’s output (learned by
    [`fit()`](#ek.SignalFamilyCalibrator.fit)); `None` leaves the fused value uncalibrated.

### Example

```pycon
>>> rows = [{"judge": j, "logprob": l} for j, l in
...         [(0.9, -0.1), (0.8, -0.2), (0.85, -2.0), (0.3, -0.1), (0.2, -2.5), (0.7, -1.8)]]
>>> correct = [True, True, False, True, False, False]
>>> cal = SignalFamilyCalibrator().fit(rows, correct)
>>> sorted(cal.calibrators)
['judge', 'logprob']
>>> 0.0 <= cal({"judge": 0.9, "logprob": -0.1}) <= 1.0
True
```

#### calibrate_signals(raw_signals, , group=None)

`{name: calibrated probability}` for each raw signal (before fusion).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### combine()

Fuse calibrated per-family probabilities by their mean (the default combiner).

Each family’s calibrated value estimates the *same* event (“this field is
correct”), so the mean is a defensible, assumption-light pooling rule.

```pycon
>>> mean_probability({"judge": 0.9, "logprob": 0.7})
0.8
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### factory

alias of [`PlattCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.PlattCalibrator)

#### fit(signals, correct)

Fit one calibrator per signal name, then the `fusion` calibrator.

Each family is fit over the rows where it appears with a finite value. A
family with fewer than two such rows, or only one label class, cannot be
calibrated and is refused (`ValueError`) rather than turned into a
constant that would then dilute the fusion.

* **Return type:**
  [`SignalFamilyCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.SignalFamilyCalibrator)

#### fuse(calibrated)

Fuse per-family calibrated probabilities: `combine`, then `fusion`.

A single family’s value is returned as is: it is already calibrated, and
`fusion` was learned on fused values.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### to_dict()

Serialize every family’s calibrator and the (registered) combiner’s name.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ek.SpanF1Metric(scheme=None, , tagging_scheme='IOB2')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Entity/slot precision/recall/F1 under an **explicit** [`MatchScheme`](#ek.MatchScheme).

* **Parameters:**
  * **scheme** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`MatchScheme`](ek.metrics.spans.html.md#ek.metrics.spans.MatchScheme)]) – REQUIRED. The (backend, scheme) contract – there is deliberately no
    default, because seqeval and nervaluate disagree and a bare F1 is
    ambiguous (`misc/docs/ek_02` §2.2). Pass a [`MatchScheme`](#ek.MatchScheme).
  * **tagging_scheme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – For [`MatchScheme.SEQEVAL_STRICT`](#ek.MatchScheme.SEQEVAL_STRICT) only: the BIO/IOBES
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

### *class* ek.StringMetric(mode='cer', , canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

CER (`mode='cer'`) or WER (`mode='wer'`) as a [`Metric`](ek.base.html.md#ek.base.Metric).

* **Parameters:**
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"cer"` (character) or `"wer"` (word) error rate.
  * **canonicalizer** – Optional `str -> str` applied to both sides before scoring
    (see [`ek.canonicalize`](ek.canonicalize.html.md#module-ek.canonicalize)). Usually supplied by the facade instead.

#### aggregate(scores)

Corpus error rate = total edits / total reference length (the correct way).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.Table(rows=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A table as an ordered list of rows of [`Cell`](#ek.Cell) s (the logical cells).

`rows[i]` is the i-th row’s logical cells in reading order; spans are carried
on each [`Cell`](#ek.Cell). Use [`from_html()`](#ek.Table.from_html) / [`from_grid()`](#ek.Table.from_grid) to build one,
and [`as_grid()`](#ek.Table.as_grid) for the dense 2-D occupancy view GriTS scores.

#### as_grid()

Expand spans into a dense 2-D grid; a spanned position repeats its cell.

Each physical grid position holds the (shared) [`Cell`](#ek.Cell) covering it, so
two tables with the same topology have grids of the same shape – the view
GriTS aligns position-by-position.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Cell`](ek.metrics.tables.html.md#ek.metrics.tables.Cell)]]]

#### *classmethod* coerce(obj)

Coerce a [`Table`](#ek.Table), an HTML string, or a 2-D grid into a Table.

* **Return type:**
  [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)

#### *classmethod* from_grid(grid)

Build a Table from a 2-D grid of cell texts (`None` = spanned-over).

* **Return type:**
  [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)

#### *classmethod* from_html(html)

Parse an HTML `<table>` (with `colspan`/`rowspan`) into a Table.

* **Return type:**
  [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)

### *class* ek.TaskSpec(task_id, input=None, gold=None, value=1.0, tools=(), slice=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One task in a suite: its input, its oracle, its value, and its allowed tools.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Stable identifier (the grouping key for k-trial reliability).
  * **input** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Whatever the agent under test is called with.
  * **gold** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The reference the checker grades against (a goal state, an expected answer,
    a gold trajectory – the checker decides how to read it).
  * **value** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Task-level cost weight: how much a completed task is worth. The task-level
    analog of `FieldSpec.importance`; feeds value-weighted cost accounting.
  * **tools** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`ToolSpec`](ek.agents.base.html.md#ek.agents.base.ToolSpec)]) – The tools this task permits (the Layer-A grammar for the task).
  * **slice** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional slice label (domain, difficulty, language) – per-slice cuts are
    mandatory, not optional (a low pass^k concentrated in one hard slice is a
    different problem from uniform flakiness).

#### grammar()

The Layer-A grammar for this task’s tool set.

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### *class* ek.TedsMetric(, structure_only=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

TEDS / TEDS-Struct as a [`Metric`](ek.base.html.md#ek.base.Metric) (clean-room on `apted`).

* **Parameters:**
  **structure_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True`, ignore cell text and score structure alone
  (TEDS-Struct), isolating table-structure accuracy from OCR noise.
  Defaults to `False` (full TEDS: structure + content).

#### aggregate(scores)

Corpus TEDS = **mean** of per-table TEDS (the PubTabNet/OmniDocBench
convention). Each per-table TEDS is already node-count-normalized, so the
reported corpus statistic is the average, not a pooled edit-rate. The pooled
`1 - sum(dist)/sum(nodes)` is available from the per-item `detail` if a
size-weighted variant is wanted instead.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.TemperatureCalibrator(T=1.0, kind='temperature', t_min=0.05, t_max=10.0, max_iter=60)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Temperature scaling: `sigmoid(logit / T)` with one `T` fit on a holdout.

Use **only when you have logits**: `__call__` expects a *logit*, not a
probability. `T > 1` softens overconfidence; the argmax is unchanged.

#### fit(logits, correct)

Fit `T` by minimising NLL with a bounded 1-D search.

* **Return type:**
  [`TemperatureCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.TemperatureCalibrator)

#### t_min *: [float](https://docs.python.org/3/builtins/functions.html#float)* *= 0.05*

Golden-section search bounds and iteration count (config, not magic numbers).

### *class* ek.ToolSpec(name, params=dict(), importance=None, destructive=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One tool the agent may call: its argument schema and its error costs.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The tool/function name (the AST-match ground truth).
  * **params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec)]) – Argument name -> [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec). Each spec’s `importance`
    is the cost of getting *that argument* wrong, and its `domain` feeds
    validators – the same contract as an extracted field.
  * **importance** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Cost weight of the *call itself* (a missing or spurious call).
    Defaults to `DESTRUCTIVE_WEIGHT` when `destructive` is set, else 1.0.
  * **destructive** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether calling this tool mutates the world irreversibly (a refund,
    a delete, a send). Drives the default weight and the safety validators.

#### to_node_type()

Render this tool as a Layer-A [`NodeType`](ek.base.html.md#ek.base.NodeType) (args as fields).

* **Return type:**
  [`NodeType`](ek.base.html.md#ek.base.NodeType)

#### *property* weight *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The call-level cost weight (explicit, else destructive-aware default).

### *class* ek.TypeRef(kind, name, field=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reference into the schema, so an injected cost function can read weights.

`kind` is one of `"node"`, `"edge"` or `"field"`; for a field, `name`
is the node-type name and `field` the field name.

### *class* ek.TypedEdge(src, dst, edge_type='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A directed, typed relation between two node ids.

### *class* ek.TypedGraph(nodes=(), edges=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A typed graph: typed nodes plus directed typed edges between them.

#### to_networkx()

Build a `networkx.DiGraph` carrying type/fields as node/edge attrs.

### *class* ek.TypedGraphMetric(grammar=None, weights=None, , canonicalizer=None, timeout=10.0, max_nodes=60)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cost-weighted, type-aware typed-graph edit distance as a [`Metric`](ek.base.html.md#ek.base.Metric).

* **Parameters:**
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) supplying cost weights (may
    also be passed per-call to `__call__`).
  * **weights** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar), [`TypeRef`](ek.base.html.md#ek.base.TypeRef)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Optional [`CostWeight`](ek.base.html.md#ek.base.CostWeight) overriding the schema weights.
  * **canonicalizer** – Optional `str -> str` applied to field values before the
    equality check (the facade passes `normalize=` through here); a schema’s
    per-field `FieldSpec.normalizer` takes precedence for the fields it names.
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds budget for the (NP-hard) GED search; on timeout the best
    cost found so far is used (an approximation). Default 10.0.

#### aggregate(scores)

Corpus normalized distance = total raw distance / total max-distance.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* ek.TypedNode(node_id, node_type, fields=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A node in a typed graph: an id, a node-type tag, and its field values.

### *class* ek.ValidationResult(original, value, findings=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of running a value through a [`validation_pipeline()`](#ek.validation_pipeline).

`value` is the final value after any applied corrections; `original` is the
input; `findings` is the full audit trail (FLAG and CORRECT), in layer order.

#### *property* clean *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

No findings at all – the value passed every layer untouched.

#### *property* corrected *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether any correction changed the value.

#### *property* flagged *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether any layer raised a FLAG (something a human should look at).

### *class* ek.Validator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Reference-free check on a value, yielding zero or more [`Finding`](#ek.Finding) s.

### *class* ek.VerifierSignal(validators=list(), cost_tier=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Deterministic verifier evidence as a reference-free [`Signal`](ek.base.html.md#ek.base.Signal).

Cost tier 1 – the free, first-to-run layer. Runs a list of
[`Validator`](ek.base.html.md#ek.base.Validator) s on a value and returns the **fraction that passed**
in `[0, 1]` as the raw signal (`1.0` = every check passed). It also exposes
the [`Finding`](ek.base.html.md#ek.base.Finding) s it produced via [`findings()`](#ek.VerifierSignal.findings), so the same
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

### ek.app_folder()

The root `ek` data folder (e.g. `~/.local/share/ek`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ek.benford_findings(numbers, , field='', tol=0.15, min_n=30, layer='anomaly')

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

### ek.cache_this(func=None, , cache=None, key=None, pre_cache=False, as_property=None, ignore=None, serialize=None, deserialize=None)

Unified caching decorator for properties and methods with persistent storage support.

`cache_this` extends the capabilities of Python’s built-in `functools.cached_property`
and `functools.lru_cache` by providing:

- **Persistent caching**: Store cached values in files, databases, or any MutableMapping
- **Flexible cache backends**: Use instance attributes, external stores, or cache factories
- **Smart key generation**: Automatic argument-based keys for methods with parameter filtering
- **Serialization support**: Custom serialize/deserialize functions for complex data
- **Auto-detection**: Automatically chooses property vs method caching based on signature
- **No LRU eviction**: Unlike lru_cache, values persist until explicitly removed

Unlike functools.cached_property (properties only) and lru_cache (memory-only with eviction),
cache_this provides a unified interface for both use cases with persistent storage options.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], `VT`]) – The function to be decorated (usually left empty).
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[`KT`, `VT`], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – 

    The cache storage. Can be:
    - A MutableMapping instance (shared across instances)
    - A string naming an instance attribute containing a MutableMapping
    - A callable taking (instance) and returning a MutableMapping
      This enables instance-specific caching, e.g.:
      cache=lambda self: Files(f’/cache/{self.user_id}/’)
  * **key** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[`KT`, [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `KT`], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – For properties: the key to store the cache value, can be a callable
    that will be applied to the method name to make a key, or an explicit string.
    For methods: a callable that takes `(self, *args, **kwargs)` and returns a cache key.
  * **pre_cache** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – Default is False. If True, adds an in-memory cache to the method
    to (also) cache the results in memory. If a MutableMapping is given, it will be
    used as the pre-cache.
    This is useful when you want a persistent cache but also want to speed up
    access to the method in the same session.
  * **as_property** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If True, force use of CachedProperty. If False, force use of
    CachedMethod. If None (default), auto-detect based on function signature.
  * **ignore** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Parameter name(s) to exclude from cache key computation.
    Can be a string (single parameter) or list of strings (multiple parameters).
    Commonly used to ignore ‘self’ or parameters like ‘verbose’ that don’t
    affect the result.
  * **serialize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional function to serialize values before caching
    (e.g. `pickle.dumps` for binary file storage).
  * **deserialize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional function to deserialize cached values
    (e.g. `pickle.loads`).
* **Returns:**
  The decorated function.

### Comprehensive Example

Here’s a complete example showcasing all major features of cache_this:

```pycon
>>> import tempfile
>>> import os
>>> from pathlib import Path
>>>
>>> class DataProcessor:
...     def __init__(self, user_id="user123"):
...         self.user_id = user_id
...         self.memory_cache = {}  # In-memory cache
...         self.call_counts = {}   # Track function calls for demo
...
...     # 1. Basic property caching (like functools.cached_property)
...     @cache_this
...     def basic_property(self):
...         '''Cached in instance.__dict__ by default'''
...         self.call_counts['basic_property'] = self.call_counts.get('basic_property', 0) + 1
...         return f"computed_value_{self.call_counts['basic_property']}"
...
...     # 2. Property with custom cache and key
...     @cache_this(cache='memory_cache', key='custom_prop_key')
...     def custom_cached_property(self):
...         '''Cached in instance.memory_cache with custom key'''
...         self.call_counts['custom_cached_property'] = self.call_counts.get('custom_cached_property', 0) + 1
...         return f"custom_value_{self.call_counts['custom_cached_property']}"
...
...     # 3. Method caching with argument-based keys
...     @cache_this(cache='memory_cache')
...     def compute_result(self, x, y, mode='fast'):
...         '''Cached based on arguments (x, y, mode)'''
...         key = ('compute_result', x, y, mode)
...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
...         return x * y * (2 if mode == 'fast' else 3)
...
...     # 4. Method caching with ignored parameters
...     @cache_this(cache='memory_cache', ignore={'verbose', 'debug'})
...     def process_data(self, data, algorithm='default', verbose=False, debug=False):
...         '''Cache ignores verbose and debug parameters'''
...         key = ('process_data', tuple(data), algorithm)
...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
...         if verbose: print(f"Processing {data} with {algorithm}")
...         return sum(data) * (2 if algorithm == 'default' else 3)
...
...     # 5. Instance-specific cache factory
...     @cache_this(cache=lambda self: {f'{self.user_id}_cache': {}}.get(f'{self.user_id}_cache'))
...     def user_specific_computation(self, value):
...         '''Each instance gets its own cache based on user_id'''
...         key = ('user_specific_computation', value)
...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
...         return value ** 2
```

Now let’s test all the features:

```pycon
>>> processor = DataProcessor("alice")
>>>
>>> # Test basic property caching
>>> result1 = processor.basic_property
>>> result2 = processor.basic_property  # Should use cache
>>> assert result1 == result2 == "computed_value_1"
>>> assert 'basic_property' in processor.__dict__  # Cached in instance dict
>>>
>>> # Test custom cache and key
>>> result1 = processor.custom_cached_property
>>> result2 = processor.custom_cached_property  # Should use cache
>>> assert result1 == result2 == "custom_value_1"
>>> assert 'custom_prop_key' in processor.memory_cache
>>>
>>> # Test method caching with arguments
>>> result1 = processor.compute_result(3, 4, 'fast')
>>> result2 = processor.compute_result(3, 4, 'fast')  # Should use cache
>>> result3 = processor.compute_result(3, 4, 'slow')  # Different args, new computation
>>> assert result1 == result2 == 24  # 3 * 4 * 2
>>> assert result3 == 36  # 3 * 4 * 3
>>>
>>> # Test parameter ignoring
>>> result1 = processor.process_data([1, 2, 3], verbose=True)
Processing [1, 2, 3] with default
>>> result2 = processor.process_data([1, 2, 3], verbose=False)  # Should use same cache
>>> result3 = processor.process_data([1, 2, 3], debug=True)     # Should use same cache
>>> assert result1 == result2 == result3 == 12  # sum([1,2,3]) * 2
>>>
>>> # Test instance-specific caching
>>> result1 = processor.user_specific_computation(5)
>>> result2 = processor.user_specific_computation(5)  # Should use cache
>>> assert result1 == result2 == 25  # 5 ** 2
>>>
>>> # Different instance should have separate cache
>>> processor2 = DataProcessor("bob")
>>> result3 = processor2.user_specific_computation(5)  # Fresh computation
>>> assert result3 == 25
```

Used with no arguments, `cache_this` will cache just as the builtin
`cached_property` does – in the instance’s `__dict__` attribute.

```pycon
>>> class SameAsCachedProperty:
...     @cache_this
...     def foo(self):
...         print("In SameAsCachedProperty.foo...")
...         return 42
...
>>> obj = SameAsCachedProperty()
>>> obj.__dict__  # the cache is empty
{}
>>> obj.foo  # when we access foo, it's computed and returned...
In SameAsCachedProperty.foo...
42
>>> obj.__dict__  # ... but also cached
{'foo': 42}
>>> obj.foo  # so that the next time we access foo, it's returned from the cache.
42
```

Not that if you specify `cache=False`, you get a property that is computed
every time it’s accessed:

```pycon
>>> class NoCache:
...     @cache_this(cache=False)
...     def foo(self):
...         print("In NoCache.foo...")
...         return 42
...
>>> obj = NoCache()
>>> obj.foo
In NoCache.foo...
42
>>> obj.foo
In NoCache.foo...
42
```

Specify the cache as a dictionary that lives outside the instance:

```pycon
>>> external_cache = {}
>>>
>>> class CacheWithExternalMapping:
...     @cache_this(cache=external_cache)
...     def foo(self):
...         print("In CacheWithExternalMapping.foo...")
...         return 42
...
>>> obj = CacheWithExternalMapping()
>>> external_cache
{}
>>> obj.foo
In CacheWithExternalMapping.foo...
42
>>> external_cache
{'foo': 42}
>>> obj.foo
42
```

Specify the cache as an attribute of the instance, and an explicit key:

```pycon
>>> class WithCacheInInstanceAttribute:
...
...     def __init__(self):
...         self.my_cache = {}
...
...     @cache_this(cache='my_cache', key='key_for_foo')
...     def foo(self):
...         print("In WithCacheInInstanceAttribute.foo...")
...         return 42
...
>>> obj = WithCacheInInstanceAttribute()
>>> obj.my_cache
{}
>>> obj.foo
In WithCacheInInstanceAttribute.foo...
42
>>> obj.my_cache
{'key_for_foo': 42}
>>> obj.foo
42
```

Now let’s see a more involved example that exhibits how `cache_this` would be used
in real life. Note two things in the example below.

First, that we use `functools.partial` to fix the parameters of our `cache_this`.
This enables us to reuse the same `cache_this` in multiple places without all
the verbosity. We fix that the cache is the attribute `cache` of the instance,
and that the key is a function that will be computed from the name of the method
adding a `'.pkl'` extension to it.

Secondly, we use the `ValueCodecs` from `dol` to provide a pickle codec for storying
values. The backend store used here is a dictionary, so we don’t really need a
codec to store values, but in real life you would use a persistent storage that
would require a codec, such as files or a database.

Thirdly, we’ll use a `pre_cache` to store the values in a different cache “before”
(setting and getting) them in the main cache.
This is useful, for instance, when you want to persist the values (in the main
cache), but keep them in memory for faster access in the same session
(the pre-cache, a dict() instance usually). It can also be used to store and
use things locally (pre-cache) while sharing them with others by storing them in
a remote store (main cache).

Finally, we’ll use a dict that logs any setting and getting of values to show
how the caches are being used.

```pycon
>>> from dol import cache_this
>>>
>>> from functools import partial
>>> from dol import ValueCodecs
>>> from collections import UserDict
>>>
>>>
>>> class LoggedCache(UserDict):
...     name = 'cache'
...
...     def __setitem__(self, key, value):
...         print(f"In {self.name}: setting {key} to {value}")
...         return super().__setitem__(key, value)
...
...     def __getitem__(self, key):
...         print(f"In {self.name}: getting value of {key}")
...         return super().__getitem__(key)
...
>>>
>>> class CacheA(LoggedCache):
...     name = 'CacheA'
...
>>>
>>> class CacheB(LoggedCache):
...     name = 'CacheB'
...
>>>
>>> cache_with_pickle = partial(
...     cache_this,
...     cache='cache',  # the cache can be found on the instance attribute `cache`
...     key=lambda x: f"{x}.pkl",  # the key is the method name with a '.pkl' extension
...     pre_cache=CacheB(),
... )
>>>
>>>
>>> class PickleCached:
...     def __init__(self, backend_store_factory=CacheA):
...         # usually this would be a mapping interface to persistent storage:
...         self._backend_store = backend_store_factory()
...         self.cache = ValueCodecs.default.pickle(self._backend_store)
...
...     @cache_with_pickle
...     def foo(self):
...         print("In PickleCached.foo...")
...         return 42
...
```

```pycon
>>> obj = PickleCached()
>>> list(obj.cache)
[]
```

```pycon
>>> obj.foo
In CacheA: getting value of foo.pkl
In CacheA: getting value of foo.pkl
In PickleCached.foo...
In CacheA: setting foo.pkl to b'\x80\x04K*.'
42
>>> obj.foo
In CacheA: getting value of foo.pkl
In CacheB: setting foo.pkl to 42
42
```

As usual, it’s because the cache now holds something that has to do with `foo`:

```pycon
>>> list(obj.cache)
['foo.pkl']
>>> # == ['foo.pkl']
```

The value of `'foo.pkl'` is indeed `42`:

```pycon
>>> obj.cache['foo.pkl']
In CacheA: getting value of foo.pkl
42
```

But note that the actual way it’s stored in the `_backend_store` is as pickle bytes:

```pycon
>>> obj._backend_store['foo.pkl']
In CacheA: getting value of foo.pkl
b'\x80\x04K*.'
>>> # == b'\x80\x04K*.'
```

### ek.canonicalize_corrector(normalize, , field_name='', layer='canonicalize')

L0: fold a value to canonical form, emitting a CORRECT finding when it changes.

`normalize` is anything [`ek.canonicalize.resolve_canonicalizer()`](ek.canonicalize.html.md#ek.canonicalize.resolve_canonicalizer) accepts
(a registered name, a callable, a step list). The narrowest, safest corrector –
run it first so later layers compare canonical forms.

* **Return type:**
  [`Corrector`](ek.validate.html.md#ek.validate.Corrector)

### ek.check_requirements(, engine=None, extra=None)

Report what an OCR engine or feature extra needs, without raising.

For an OCR `engine` and an installed `ocracy`, defers to `ocracy`’s own
doctor (which knows each backend’s pip package, system binaries and
credentials). Otherwise returns a generic hint.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  A dict with at least `{"ok": bool, "hint": str}`.

### ek.checksum_validator(kind='luhn', , field_name='', layer='checksum')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value failing a named checksum.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.cohen_kappa(rater_a, rater_b)

Cohen’s kappa for two raters on nominal labels (chance-corrected agreement).

Use only for exactly two raters, nominal labels, complete data; prefer
[`krippendorff_alpha()`](#ek.krippendorff_alpha) for the general case (missing data, any measurement
level).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.cross_field_validator(predicate, , message, fields=(), layer='cross_field')

FLAG a record when `predicate(record)` is False (a general cross-field check).

Skips silently when `predicate` raises on a partial record (a missing field is
not itself a cross-field violation).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.default_canonicalizer()

A conservative default: NFC -> casefold -> collapse whitespace.

* **Return type:**
  [`Canonicalizer`](ek.canonicalize.html.md#ek.canonicalize.Canonicalizer)

### ek.default_cost_weight(grammar, ref)

The default `CostWeight`: read the importance weight off the schema.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.engine_yields_tables(engine)

Whether `engine`’s capability profile claims scoreable table structure.

A convenience over [`ek.ocr.profile()`](ek.ocr.html.md#ek.ocr.profile): returns the engine profile’s `tables`
flag. ek treats the profile as a *prior* (Azure DI, AWS Textract, PaddleOCR and the
markdown engines are expected to yield scoreable tables; Google Vision is not), to
be verified empirically by actually running [`table_from_ocr()`](#ek.table_from_ocr). Use it to skip
table scoring for engines that cannot emit cells in the first place.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.enum_validator(members, , field_name='', layer='enum')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not in an allowed set.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.estimate_quality(extraction, , sources=(), signals=(), calibrator=None, validators=(), policy=None, agreement=True, assume_calibrated=False)

Estimate the quality of an extraction with no gold reference.

Composes the strict `signal -> calibrate -> validate -> decide` pipeline
(`misc/docs/ek_03`) over a single value, a [`FieldEstimate`](ek.base.html.md#ek.base.FieldEstimate), or a
whole [`AnnotatedExtraction`](ek.base.html.md#ek.base.AnnotatedExtraction) (scored per field, with per-field
specs feeding the validators and the node type as the Mondrian group key). The
simple call `estimate_quality(value)` Just Works; every stage is injectable.

* **Parameters:**
  * **extraction** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A raw value, a [`FieldEstimate`](ek.base.html.md#ek.base.FieldEstimate), or an
    [`AnnotatedExtraction`](ek.base.html.md#ek.base.AnnotatedExtraction).
  * **sources** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – Additional hypotheses of the *same* content (strings or
    `OcrResult`-shaped objects) to fuse with ROVER – their mean agreement
    becomes a `raw_signals["agreement"]` entry (the flagship online signal).
  * **signals** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – Explicit [`Signal`](ek.base.html.md#ek.base.Signal) callables `target -> float |
    Mapping` producing further raw signals.
  * **calibrator** – A [`Calibrator`](ek.base.html.md#ek.base.Calibrator) mapping raw score -> probability,
    applied to the field’s confidence (else the mean of its raw signals).
    When the signals are *heterogeneous* (e.g. an LLM-judge score next to a
    logprob), pass a `Mapping[str, Calibrator]` keyed by signal name (a
    `"*"` entry is the fallback) or a
    [`SignalFamilyCalibrator`](ek.qe.calibrate.html.md#ek.qe.calibrate.SignalFamilyCalibrator): each signal family is
    then calibrated on its own and the calibrated probabilities are fused by
    its `combine` rule (mean by default) followed by its learned
    `fusion` calibrator. A field’s own `confidence` is the
    `"confidence"` family. The per-family values are reported in
    `provenance["calibrated_signals"]`.
    **Calibration is non-optional before gating** (Hard Rule 1): passing a
    `policy` with no calibrator and `assume_calibrated` false **raises** –
    a raw, uncalibrated score must never reach a DecisionPolicy.
  * **validators** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – [`Validator`](ek.base.html.md#ek.base.Validator) callables yielding findings. A flat
    iterable runs on every field (use spec-driven validators like
    [`schema_validator()`](ek.qe.verifiers.html.md#ek.qe.verifiers.schema_validator)). For per-field scoping pass a
    `Mapping` keyed by field name or node type (`"*"` runs on all).
  * **policy** – A [`DecisionPolicy`](ek.base.html.md#ek.base.DecisionPolicy) producing accept/flag/block from
    the *calibrated* confidence.
  * **agreement** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Auto-run ROVER over `sources` when any are given (default true).
  * **assume_calibrated** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Treat the incoming confidence as already calibrated
    (silences the uncalibrated-gating warning). With a per-family
    calibrator it also waives the check that a value fused from two or more
    families went through a learned `fusion` calibrator.
* **Return type:**
  [`QualityReport`](ek.base.html.md#ek.base.QualityReport)

### ek.evaluate(cases, , metric=None, grammar=None, normalize=None, weights=None)

Aggregate many comparisons into a [`Report`](ek.base.html.md#ek.base.Report) (corpus level).

* **Parameters:**
  * **cases** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – Iterable of `(pred, gold)` or `(pred, gold, slice_label)` tuples.
  * **metric** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Metric`](ek.base.html.md#ek.base.Metric)]) – As in [`score()`](#ek.score) (resolved once from the first case’s types).
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Optional Layer-A grammar passed to every comparison.
  * **normalize** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional canonicalizer applied before every comparison.
* **Return type:**
  [`Report`](ek.base.html.md#ek.base.Report)
* **Returns:**
  A Report whose `aggregate` is computed by the metric’s own aggregator
  (e.g. globally accumulated CER/WER, micro-F1) – never a naive mean.

### ek.evaluate_store(predict, gold, , metric=None, grammar=None, normalize=None, weights=None, input_key='input', reference_key='reference', slice_key='slice', persist=False, run_id=None, rootdir=None)

Run `predict` over a gold store and score it per slice.

* **Parameters:**
  * **predict** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `input -> prediction` callable (the system under test).
  * **gold** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Mapping `key -> {input, reference, [slice], ...}` (a dict or a
    `ek` gold store).
  * **metric** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Metric name/callable (defaults to type-dispatch on the first case).
  * **grammar/normalize/weights** – forwarded to scoring.
  * **input_key/reference_key/slice_key** – field names within each gold record.
  * **persist/run_id/rootdir** – persist the run to the `results`/`runs` stores.
* **Returns:**
  A [`Report`](ek.base.html.md#ek.base.Report); `detail['per_item']` maps each key to its
  prediction, reference, slice, and score.

### ek.expected_calibration_error(probs, correct, , n_bins=10)

Expected Calibration Error: weighted mean gap between confidence and accuracy.

Bins predictions by confidence into `n_bins` equal-width bins and averages
`|mean_confidence - accuracy|` weighted by bin population. `0` is perfect.
Non-finite probs are skipped; out-of-range probs are clamped into `[0, 1]`.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.get(namespace, name)

Resolve a registered strategy by name (discovering entry points first).

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – with the list of available names, if `name` is unknown.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ek.has_table_structure(ocr_result, , parser=None)

Whether [`table_from_ocr()`](#ek.table_from_ocr) recovers a non-empty table from `ocr_result`.

A thin predicate over [`table_from_ocr()`](#ek.table_from_ocr) (same `parser` semantics): true iff
a table with at least one cell can be recovered. Use it to route an OCR result to
the table metrics vs the text-only CER/WER fallback.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.iban_check(iban)

IBAN validity via the ISO 13616 / ISO 7064 mod-97 rule (must equal 1).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.isbn_check(value)

Validate an ISBN-10 or ISBN-13 by length (digits/hyphens/spaces ignored).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.json_store(kind, , rootdir=None)

A JSON `MutableMapping` for one `kind` of artifact.

Keys are plain names (no extension); values are JSON-serializable objects.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

### ek.krippendorff_alpha(reliability_data, , level='nominal')

Krippendorff’s alpha – the general inter-annotator agreement coefficient.

Pure-Python (no dependency, permissive core): any number of raters, any
measurement `level` (`"nominal"`, `"ordinal"`, `"interval"`, `"ratio"`),
and missing data (use `None` or `float('nan')` for an unlabelled cell).
`reliability_data` is one row per rater, one column per item. Alpha is `1.0`
for perfect agreement, `0.0` at chance, and goes negative for systematic
disagreement. Computed via the coincidence-matrix method (Krippendorff 2011):
observed vs expected disagreement under the level-specific difference metric.

### Example

```pycon
>>> data = [[1, 1, 1, 1], [1, 1, 1, 1]]   # two raters, perfect agreement
>>> krippendorff_alpha(data)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.lexicon_corrector(vocabulary, , threshold=0.8, scorer=None, field_name='', layer='lexicon', flag_unmatched=True)

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
  [`Corrector`](ek.validate.html.md#ek.validate.Corrector)

### ek.llm_corrector(correct_fn, , only_flagged=True, field_name='', layer='llm_correct')

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
  [`Corrector`](ek.validate.html.md#ek.validate.Corrector)

### ek.lm_surprisal_validator(scorer, , threshold, higher_is_worse=True, field_name='', layer='lm_prior')

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

### ek.load_baseline(name, , rootdir=None)

Load a named baseline (or `None` if it does not exist).

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.load_calibrator(name, , rootdir=None)

Reconstruct a persisted calibrator by name (dispatched on its `kind`).

Validates the stored record so a malformed or unknown `kind` fails with an
actionable error rather than a raw `KeyError` or a load-then-crash-later.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ek.luhn_check(number)

Luhn (mod-10) checksum: credit cards, IMEIs, some national IDs.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.mall(, rootdir=None)

The full stores-of-stores mall (one JSON store per `KINDS`).

* **Return type:**
  [`Mall`](ek.stores.html.md#ek.stores.Mall)

### ek.mean_probability(probs)

Fuse calibrated per-family probabilities by their mean (the default combiner).

Each family’s calibrated value estimates the *same* event (“this field is
correct”), so the mean is a defensible, assumption-light pooling rule.

```pycon
>>> mean_probability({"judge": 0.9, "logprob": 0.7})
0.8
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.min_probability(probs)

Fuse calibrated per-family probabilities by their minimum (conservative).

Use when any one family flagging a problem should be enough to lower the gate.

```pycon
>>> min_probability({"judge": 0.9, "logprob": 0.7})
0.7
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.names(namespace)

All registered names in a namespace (after entry-point discovery).

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ek.ordering_validator(keys, , strict=True, layer='cross_field')

FLAG a record whose comparable values at `keys` are not ascending.

The canonical date/sequence check (`start <= end`, issue before due). Only the
keys actually present and mutually comparable are checked, so it is safe on
partial records.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.pass_at_k(, n, c, k)

Unbiased `pass@k`: probability that **at least one** of k samples succeeds.

The HumanEval estimator `1 - C(n-c, k) / C(n, k)` over `n` trials with `c`
successes. This is the **capability** metric – right when a single success is enough
(offline candidate generation behind a verifier).

### Example

```pycon
>>> round(pass_at_k(n=10, c=1, k=1), 3)
0.1
>>> pass_at_k(n=10, c=10, k=5)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.pass_hat_k(, n, c, k)

Unbiased `pass^k`: probability that **all k** independent trials succeed.

The tau-bench estimator `C(c, k) / C(n, k)`; for a per-trial success probability `p`
it decays to `p**k`. This is the **reliability** metric – the one that matters when
consistency *is* the product.

### Example

```pycon
>>> pass_hat_k(n=8, c=8, k=8)
1.0
>>> pass_hat_k(n=8, c=4, k=2)
0.21428571428571427
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.percent_agreement(rater_a, rater_b)

Raw fraction of items two raters labelled identically.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.persistent_cache(func=None, , kind='runs', key=None, rootdir=None)

Memoize a plain function’s result in a `kind` JSON store, across sessions.

Use as `@persistent_cache(kind="runs", key=...)`. Returned values must be
JSON-serializable. `key` may be a constant or a callable receiving the call
arguments; it defaults to the function’s qualified name (so provide a `key`
for functions that take arguments). For caching a *class property/method*,
prefer [`cache_this`](#ek.cache_this) (dol’s descriptor) instead.

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

### ek.range_validator(lo, hi, , field_name='', layer='range')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a number outside `[lo, hi]`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.regex_validator(pattern, , field_name='', layer='format')

A [`Validator`](ek.base.html.md#ek.base.Validator) that flags a value not fully matching `pattern`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]]

### ek.register(namespace, name, obj=None)

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

### ek.regression_gate(report, baseline, , tolerance=0.0, higher_is_better=None, rootdir=None)

Fail if `report` regresses beyond `tolerance` vs a baseline, per slice.

* **Parameters:**
  * **report** – The current [`Report`](ek.base.html.md#ek.base.Report).
  * **baseline** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A baseline name (loaded from the `baselines` store) or a baseline
    dict from [`save_baseline()`](#ek.save_baseline).
  * **tolerance** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Allowed drift before a change counts as a regression.
  * **higher_is_better** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Override metric-direction inference (CER/WER/graph are
    lower-is-better; F1/similarity are higher-is-better).
  * **rootdir** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – data root (when `baseline` is a name).
* **Return type:**
  [`GateResult`](ek.harness.html.md#ek.harness.GateResult)
* **Returns:**
  A [`GateResult`](#ek.GateResult) (falsy if any regression was found).

### ek.reliability_curve(probs, correct, , n_bins=10)

Per-bin `{confidence, accuracy, count}` for a reliability diagram.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.requires_extra(extra, , packages=None)

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

### ek.resolve(namespace, ref, , default=None)

Coerce a strategy reference to a callable.

`ref` may be a registered name (`str`), an already-resolved object, or
`None` (in which case `default`, itself a name or object, is used).

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ek.resolve_table_parser(ref)

Resolve a table parser reference (a registered name or a callable) to a callable.

Mirrors how other ek strategies resolve from the registry (see
[`ek.registry.resolve()`](ek.registry.html.md#ek.registry.resolve)). A `str` is looked up under the `"table_parsers"`
namespace; an already-callable parser is returned unchanged.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ek.risk_coverage_curve(probs, correct, , thresholds=None)

The risk-coverage trade-off: `{threshold, coverage, selective_risk}` per point.

Coverage is the fraction auto-accepted at each threshold; selective risk is the
error rate *among the accepted*. Lower-left is better; pick the operating point
against a target risk or coverage.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ek.rover(hypotheses, , use_confidence=True, conf_weight=0.5, null_conf=0.7, tokenize=None, max_tokens=5000)

Align N hypotheses, vote per slot, and emit consensus + per-position agreement.

* **Parameters:**
  * **hypotheses** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The recognizer/extractor outputs to fuse. Each may be a string,
    an `OcrResult`-shaped object (`.text`/`.mean_confidence`), a list of
    token strings, or a list of `(token, confidence)` pairs.
  * **use_confidence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Blend confidence into the vote (`True`) or vote purely by
    frequency (`False`, which forces `conf_weight` to `0`).
  * **conf_weight** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Weight on average confidence vs vote frequency in the slot score
    (Fiscus’s `1 - alpha`); `0` is frequency-only, `1` is
    confidence-only. Ignored when `use_confidence` is `False`.
  * **null_conf** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Confidence credited to a NULL vote (an engine that produced no
    token at a slot) – the lever for how readily a deletion wins.
  * **tokenize** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]]) – Optional `hypothesis -> [(token, confidence), ...]` override; by
    default `_as_units()` handles strings/OcrResults/token lists.
  * **max_tokens** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Reject any hypothesis longer than this (the aligner is
    `O(N*l*L*L')`; an unbounded input is a quadratic time/memory DoS).
    `None` disables the guard.
* **Return type:**
  [`RoverConsensus`](ek.qe.html.md#ek.qe.RoverConsensus)
* **Returns:**
  A [`RoverConsensus`](#ek.RoverConsensus) with the consensus `tokens`/`text`, the per-slot
  breakdown, and the per-consensus-token `agreement` usable as a raw signal.

### ek.run_suite(agent, tasks, , k=1, check=None, metrics=None, price=None, prices=None, run=None, seed=None, persist=False, run_id=None, rootdir=None)

Run `agent` over a task suite, `k` trials per task, and report reliability + cost.

* **Parameters:**
  * **agent** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`TaskSpec`](ek.agents.base.html.md#ek.agents.base.TaskSpec)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The system under test: `TaskSpec -> Episode` (a bare answer is wrapped). It
    receives the whole spec – `.input`, `.tools`, `.gold` – not just the input.
    If it accepts a keyword `seed`, the per-trial seed is passed to it.
  * **tasks** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – [`TaskSpec`](ek.agents.base.html.md#ek.agents.base.TaskSpec) s (or a `{task_id: spec}` mapping).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Trials per task. `k > 1` is what makes `pass^k` meaningful – and it only means
    anything if the agent is genuinely stochastic (the report warns if it is not).
  * **check** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The success oracle – a registered checker name or an `(episode, gold) -> bool`
    callable. Defaults to `"output"`; **inject a state-based/executable oracle** for a
    real suite, and run it isolated from the agent.
  * **metrics** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – Optional `{name: Metric}` scored on every episode against `task.gold`, with
    the **suite’s Layer-A grammar injected** – this is how the tool/argument cost
    weights reach a `tool_call`/`trajectory` metric. Results land in
    `report.detail["metrics"]`.
  * **price/prices** – Rates for costing the episodes (see [`ek.agents.cost`](ek.agents.cost.html.md#module-ek.agents.cost)).
  * **run** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance)]) – [`RunProvenance`](ek.agents.base.html.md#ek.agents.base.RunProvenance) for this run (model, simulator, suite
    version, scaffold). Recorded on every episode and on any baseline saved from it.
  * **seed** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Base RNG seed. Trial *i* of each task uses `seed + i`: it is recorded on the
    episode’s provenance **and passed to the agent** when the agent accepts a `seed`
    keyword.
  * **persist/run_id/rootdir** – Persist the run to the `runs`/`results` stores.
* **Return type:**
  [`ReliabilityReport`](ek.agents.html.md#ek.agents.ReliabilityReport)
* **Returns:**
  A [`ReliabilityReport`](ek.agents.html.md#ek.agents.ReliabilityReport) carrying `pass@k`, `pass^k` (with
  a bootstrap CI), the success rate (with a Wilson CI), the per-slice cuts, and the cost
  report – because reliability without cost is only half the answer.

### ek.save_baseline(report, name, , rootdir=None)

Freeze a report’s aggregate + per-slice scores as a named baseline.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.save_calibrator(calibrator, name, , rootdir=None)

Persist a fitted calibrator’s parameters to the `calibrators` store.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ek.schema_validator(value, , spec=None)

A spec-driven [`Validator`](ek.base.html.md#ek.base.Validator): type and `domain` checks from a FieldSpec.

Reads the [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec) `type` and `domain` – a `(lo, hi)`
numeric range, an enum tuple, or a regex string – and applies the matching
check. With no spec it is a no-op, so it composes safely in any validator list.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### ek.score(pred, gold, , grammar=None, metric=None, normalize=None, weights=None)

Score one prediction against one gold reference (reference-based).

* **Parameters:**
  * **pred** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The predicted output (string, record dict, or anything with `.text`).
  * **gold** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The gold reference, same shape as `pred`.
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Optional Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) (carries cost weights).
  * **metric** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Metric`](ek.base.html.md#ek.base.Metric)]) – A registered metric name (`"cer"`, `"wer"`, `"fields"`, …), a
    callable [`Metric`](ek.base.html.md#ek.base.Metric), or `None` to dispatch by type.
  * **normalize** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional canonicalizer (name, callable, step list, or
    [`Canonicalizer`](ek.canonicalize.html.md#ek.canonicalize.Canonicalizer)) applied before comparison.
  * **weights** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A [`CostWeight`](ek.base.html.md#ek.base.CostWeight) for cost-weighted metrics (the
    typed-graph distance); overrides the schema’s importance weights.
* **Return type:**
  [`Score`](ek.base.html.md#ek.base.Score)
* **Returns:**
  A [`Score`](ek.base.html.md#ek.base.Score).

### ek.split_conformal_quantile(scores, alpha)

The split-conformal threshold: the `ceil((n+1)(1-alpha)) / n` empirical quantile.

`scores` are nonconformity scores on an exchangeable calibration set. Returns
`+inf` when `n` is too small to guarantee `1 - alpha` (so nothing is
flagged – the honest behaviour at that sample size), and `-inf` at the
`alpha -> 1` limit (flag everything).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.stop_on_correction(findings)

Stop policy: end the pass as soon as any layer proposes a correction.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.stop_on_flag(findings)

Stop policy: end the pass at the first FLAG (fail fast).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.table_from_ocr(ocr_result, , parser=None)

Extract a normalized [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table) from an `OcrResult`.

The single seam ek’s table metrics (TEDS / GriTS) use to score OCR table output
uniformly. Returns the recovered Table, or `None` when the result carries no
recoverable table structure.

* **Parameters:**
  * **ocr_result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Any `OcrResult`-shaped object. ek reads only its `.raw` (the
    engine-specific payload), `.markdown` (VLM/markdown engines), and
    `.meta` – never ocracy types, so any `image -> OcrResult` callable’s
    output works.
  * **parser** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – 

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

:param [`Table.coerce()`](#ek.Table.coerce):
:param so a parser may yield a Table:
:param a 2-D grid:
:param or HTML.:

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table)]
* **Returns:**
  A non-empty [`Table`](ek.metrics.tables.html.md#ek.metrics.tables.Table), or `None` when no table
  structure is recoverable. An empty Table (no rows/cells) is normalized to
  `None` so the predicate [`has_table_structure()`](#ek.has_table_structure) and downstream metrics
  see “no table”, not a degenerate one.

### ek.totals_consistent(record, , total_key, item_keys, tol=0.01, layer='cross_field')

Flag when `record[total_key]` does not equal the sum of `item_keys` (±tol).

The canonical cross-field check (invoice total = Σ line items). Skipped (rather
than raising or flagging) when the total or *all* item keys are absent or
non-numeric, so it is safe on partial records.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Finding`](ek.base.html.md#ek.base.Finding)]

### ek.validation_pipeline(\*layers, apply_corrections=True, stop_when=<function \_never_stop>)

Compose validators/correctors into a cheapest -> most-expensive pipeline.

`layers` run in the given order (the order encodes the cost spine). Each layer
is a `(value, *, spec) -> Iterable[Finding]` callable. When `apply_corrections`
(default), the first `CORRECT` finding a layer emits is applied to the value
before the next layer runs – so a cheap deterministic fix is what the expensive
layers then see (the noisy-channel chain). `stop_when` is an injectable
early-exit policy over the findings accumulated so far (see
[`stop_on_correction()`](#ek.stop_on_correction) / [`stop_on_flag()`](#ek.stop_on_flag)); the default runs every layer.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`ValidationResult`](ek.validate.html.md#ek.validate.ValidationResult)]

### ek.zscore_anomaly_findings(numbers, , field='', threshold=3.5, min_n=30, layer='anomaly')

FLAG numeric values that are robust-z-score outliers (median + MAD based).

A reference-free, **dependency-free** anomaly check that complements
[`benford_findings()`](#ek.benford_findings) (which checks the leading-digit *distribution*; this
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

### Modules

| [`ocr`](ek.ocr.html.md#module-ek.ocr)                   | OCR evaluation -- `ek`'s first concrete instance, built on `ocracy`.                                                                                                            |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`agents`](ek.agents.html.md#module-ek.agents)             | `ek.agents` -- evaluating AI agents and assistants, in cost per successful task.                                                                                                |
| [`metrics`](ek.metrics.html.md#module-ek.metrics)           | Reference-based (offline) metrics: the `score()` side of `ek`.                                                                                                                  |
| [`canonicalize`](ek.canonicalize.html.md#module-ek.canonicalize) | Versioned, composable canonicalization applied *before* scoring.                                                                                                                |
| [`qe`](ek.qe.html.md#module-ek.qe)                     | Reference-free quality estimation: the `estimate_quality()` side of `ek`.                                                                                                       |
| [`base`](ek.base.html.md#module-ek.base)                 | Foundational data structures and strategy protocols for `ek`.                                                                                                                   |
| [`facade`](ek.facade.html.md#module-ek.facade)             | The two top-level facades: [`score()`](#ek.score) (offline) and [`estimate_quality()`](#ek.estimate_quality) (online). |
| [`harness`](ek.harness.html.md#module-ek.harness)           | The offline evaluation harness: run a predictor over gold, gate regressions.                                                                                                    |
| [`registry`](ek.registry.html.md#module-ek.registry)         | Strategy registry, optional-dependency guarding, and requirement checks.                                                                                                        |
| [`stores`](ek.stores.html.md#module-ek.stores)             | Local-file persistence for `ek`, as plain `MutableMapping` stores.                                                                                                              |
| [`tools`](ek.tools.html.md#module-ek.tools)               | Small, CLI-friendly functions over the `ek` core (the dispatch surface).                                                                                                        |
| [`validate`](ek.validate.html.md#module-ek.validate)         | Reference-free post-extraction validation & correction (issue #7).                                                                                                              |
