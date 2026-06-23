# R4 — Post-Extraction Validation & Correction (incl. Post-OCR)

*Author: Thor Whalen*

## TL;DR
- Treat reference-free validation as a **layered noisy-channel pipeline** running cheapest→most-expensive: canonicalize first, then deterministic schema/range/cross-field checks, then lexicon priors, then LM-prior anomaly flags, then (for generative extractors) constrained generation, and only last reach for seq2seq/LLM correction.
- The single most important architectural distinction is **FLAG vs. CORRECT**: Layers 0–4 are overwhelmingly deterministic flag-or-coerce (auditable, cheap); only Layer 5 (neural/LLM correction) actually rewrites content, and it is stochastic, needs its own evaluation, and can degrade text it was meant to fix.
- Most of the stack is production-ready in 2026 (pydantic 2.13, pandera 0.31, jsonschema 4.26, outlines 1.3, XGrammar 0.2, instructor 1.14, rapidfuzz 3.14, symspellpy, pyod 3.6); LLM post-OCR correction remains research-grade and inconsistent ("no free lunches").

## Key Findings
1. **Post-OCR correction is a mature research field with exactly two canonical benchmarks.** The ICDAR Competition on Post-OCR Text Correction ran only twice — ICDAR2017 and ICDAR2019; there has been no later edition. Both split the problem into Task 1 (error detection) and Task 2 (error correction). The state of the art moved from weighted finite-state transducers + character SMT/NMT (2017) → fine-tuned BERT detector + character-level seq2seq corrector (2019, Clova AI's "CCC") → fine-tuned ByT5/BART and prompted LLMs (2024–2025).
2. **LM priors (perplexity / pseudo-log-likelihood) are excellent FLAGS, weak correctors alone.** n-gram perplexity (KenLM) and masked-LM pseudo-log-likelihood (Salazar et al. 2020, via `minicons`) reliably surface low-probability substrings and rank correction candidates, but to actually correct you must pair them with a candidate generator (edit-distance, FST, seq2seq).
3. **Constrained generation guarantees well-formedness, not correctness — and can hurt content quality.** outlines/XGrammar/llguidance/lm-format-enforcer plus native OpenAI/Anthropic structured outputs all guarantee schema-valid JSON, but the "Let Me Speak Freely?" line of work shows format constraints cause domain-dependent reasoning degradation. Constrain structure, leave room to reason.
4. **Deterministic declarative validation stacks are the cheap workhorse and are all healthy in 2026.** pydantic (records/API boundary), pandera (DataFrames), jsonschema/fastjsonschema (interop), Great Expectations (pipeline data-quality), plus lighter Cerberus/voluptuous/marshmallow and attrs/cattrs. These FLAG (or coerce); they don't invent correct values.
5. **Cross-source triangulation is the most powerful reference-free signal you have.** Agreement between independent extractors/engines (ROVER-style voting for sequences, self-consistency for stochastic LLMs) is positive evidence; disagreement is a high-value flag. Combine with statistical anomaly detection (Benford's law, isolation forests via pyod, robust z-scores) on extracted numeric fields.

## Details

### The governing frame: a noisy channel with a layered prior
Every method below is a way of estimating `argmax_v P(v|o) ∝ P(o|v)·P(v)` without a reference. The error model `P(o|v)` is whatever you know about how your source corrupts values (OCR confusion matrices, ASR phonetic confusions, LLM "silent autocorrect"). The prior `P(v)` strengthens in layers — schema/type/range (cheapest, strongest constraints), lexicon/gazetteer, then language-model surprisal (softest). Two further reference-free validators sit on top: cross-field/cross-record consistency and cross-source corroboration. OCR is just the noisiest special case of a source-agnostic problem; the closest twin transducer is ASR, and when an LLM is the extractor it too is a noisy source needing reference-free quality estimation.

A crucial connection to the R1 capability inventory: OCR engines expose raw, **uncalibrated** signals that feed these layers. Tesseract's built-in DAWG dictionaries (`load_system_dawg`, `load_freq_dawg`), `language_model_penalty_non_dict_word`, `--user-words`/`--user-patterns`, and per-symbol/word/line confidence + `GetChoiceIterator` N-best lists are exactly the `P(o|v)` and candidate-set inputs an external correction layer wants — but they are uncalibrated, so treat them as features, not probabilities. Classical neural engines (EasyOCR, RapidOCR, PaddleOCR classic) give you only a line score, so the external prior layers carry more weight. VLM/LLM extractors silently autocorrect (a verbatim-transcription hazard) and mostly hide logprobs (OpenAI's are flaky on images/empty under strict json_schema; Claude Vision returns none; Mistral OCR exposes opt-in word/page confidence), which is precisely why you cannot rely on the extractor's own confidence and must add an independent validation layer.

---

### The layered decision framework (cheapest → most expensive)

#### Layer 0 — Canonicalization / normalization
**What it does:** deterministically maps surface variants to a canonical form before any validation. **CORRECTS** (a narrow, safe, deterministic class of "errors").
- Unicode normalization (NFC/NFKC via `unicodedata.normalize`), whitespace folding, case folding, quote/dash normalization.
- Number and date parsing to typed values (`babel`, `dateutil`, `pint` for units, `python-stdnum` for checksum-bearing identifiers like IBAN/VAT/ISBN).
- **Tradeoffs:** runs first, is near-free, and removes huge classes of spurious downstream "errors." Risk: NFKC and aggressive folding can destroy genuine distinctions (e.g., ﬁ ligature vs. "fi", or a deliberately stylized identifier). Keep canonicalization lossless-where-possible and record the original.
- **Status:** all stdlib or mature libraries; production-ready indefinitely.

#### Layer 1 — Deterministic validators (type / range / enum / regex / cross-field / referential integrity)
**What it does:** declaratively asserts structural and business constraints. **FLAGS** (and coerces where you opt in); does not invent correct content.
- **pydantic v2 (2.13.x; pydantic-core in Rust; ~1B downloads/month; actively maintained):** the default for typed records and the IE boundary. Use `Annotated` constraints (`Annotated[int, Field(ge=0, le=1_000_000)]`), `field_validator`/`model_validator` for cross-field rules, `StrictInt`/`StrictStr` to disable silent coercion where you want hard failure. Single-source-of-truth: the same `BaseModel` defines the LLM's response schema (via instructor / native structured outputs) AND validates the parsed record.
- **jsonschema (4.26.0, actively maintained, Julian Berman) / fastjsonschema (2.21.x):** use Draft 2020-12 for new work. jsonschema is the interpreted reference implementation (10–100× slower than pydantic); fastjsonschema compiles schemas to Python code for ~Ajv-class throughput — precompile once, reuse. Caveat: jsonschema does **not** validate `format` by default, and `format` is an annotation not an assertion unless explicitly enabled.
- **pandera (0.31.x; maintained by Union.ai / `unionai-oss/pandera`):** DataFrame-native schema validation across pandas/Polars/Dask/PySpark/Ibis. Class-based `DataFrameModel` mirrors pydantic; supports `Check` objects with arbitrary vectorized/element-wise/grouped logic, statistical hypothesis checks, and reuse of pydantic models for row-wise validation. Faster than row-wise pydantic on large frames because it validates columns natively.
- **Great Expectations / GX Core (1.18.x; Apache-2.0; active — note GX Cloud was sunset, GX Core remains free):** heavier, pipeline-oriented data-quality platform with human-readable Data Docs and checkpoint/action triggers across pandas/Spark/SQL. Worth its setup cost at data-product boundaries ("gold" tables); overkill for in-process record validation, where pydantic/pandera win.
- **Lighter alternatives:** Cerberus (rule-dict validation, flexible/non-typed), voluptuous (functional schema), marshmallow (validation + (de)serialization), Frictionless / Table Schema (tabular data contracts), pointblank / soda-core / dbt tests (data-quality at the warehouse layer). attrs (26.1) + cattrs (26.1) are the typed-record alternative to pydantic when you want plain classes with structure/unstructure hooks and minimal runtime magic; pydantic wins when you want coercion + JSON-schema emission + ecosystem (instructor/FastAPI).
- **Cross-field & referential integrity:** express as pydantic `model_validator`s (line items sum to total; `start_date <= end_date`) or pandera dataframe checks / wide-form checks. Foreign-key resolution = set-membership checks against the referenced key set.
- **Tradeoffs:** cheap, fully deterministic, auditable, and the single highest-ROI layer. They never produce a corrected value beyond coercion/canonicalization. **Worth it: always.**

```python
# SSOT schema reused for extraction AND validation
from typing import Annotated, Literal
from pydantic import BaseModel, Field, model_validator

class Donation(BaseModel):
    donor: Annotated[str, Field(min_length=1)]
    amount_usd: Annotated[float, Field(gt=0, le=1_000_000)]
    country: Literal["US", "CA", "GB", "FR", "DE"]   # enum prior
    pledged: float
    received: float

    @model_validator(mode="after")
    def _received_not_over_pledged(self):
        if self.received > self.pledged:
            raise ValueError("received exceeds pledged")  # cross-field FLAG
        return self
```

#### Layer 2 — Lexicon / gazetteer priors
**What it does:** matches values against dictionaries, controlled vocabularies, name lists, and enum sets; fuzzy-matches near-misses. **FLAGS**, and **CORRECTS** when the candidate set is closed and a single high-confidence match exists.
- **rapidfuzz (3.14.x; MIT; ~83M downloads/month; very active):** the go-to for fuzzy matching against a known set (`process.extractOne` against a gazetteer/enum). C++ backed, fast, drop-in for the older `fuzzywuzzy`.
- **symspellpy (6.9.0; port of SymSpell v6.7.2):** Symmetric-Delete algorithm — O(1)-ish lookup via precomputed deletes, "1000× faster" than Norvig-style generate-and-test; supports compound splitting (`LookupCompound`) and word segmentation. Ideal when you have a frequency dictionary and want cheap deterministic spelling/segmentation correction.
- **hunspell/aspell:** classical morphological spellcheckers; good when you need affix-aware dictionaries for a natural language.
- **Mechanism note:** matching against a **closed enum** (country codes, product SKUs, ICD codes) is the strongest, safest corrector in the whole stack — if "Frnace" fuzzy-matches "France" at distance 1 and nothing else is close, correct it deterministically. Against an **open** vocabulary, prefer FLAG-and-review.
- **Tradeoffs:** cheap, deterministic, auditable; correction safety scales with how closed the vocabulary is. Failure mode: aggressive autocorrect on legitimately rare/OOV tokens (names, neologisms) — the same hazard VLMs have.

#### Layer 3 — Language-model priors (surprisal / perplexity / pseudo-log-likelihood)
**What it does:** scores how "expected" a substring is under a language model; low probability ⇒ anomaly flag and/or candidate re-ranking. **FLAGS** on its own; **CORRECTS** only when combined with a candidate generator (Layers 2/5).
- **n-gram / KenLM (`kpu/kenlm`, MIT):** fast, deterministic, cheap to train in-domain with `lmplz` (modified Kneser-Ney). `model.score(text)` and `full_scores` give per-token log-probs; perplexity = `10**(-Σlog10 p / N)`. Facebook released 5-gram KN models for 100 languages. KenLM is the workhorse for perplexity-based filtering (CCNet/BERTIN-style) and integrates with FST/seq2seq correctors. Character-level KenLM is especially apt for OCR (errors are sub-word).
- **Neural autoregressive surprisal & masked-LM pseudo-log-likelihood (PLL):** Salazar et al. (2020), "Masked Language Model Scoring" — mask each token in turn, sum log-probs to get PLL; pseudo-perplexity (PPPL) is the MLM analogue of perplexity. Per the paper's abstract, "By rescoring ASR and NMT hypotheses, RoBERTa reduces an end-to-end LibriSpeech model's WER by 30% relative and adds up to +1.7 BLEU on state-of-the-art baselines for low-resource translation pairs." Kauf & Ivanova (2023) give a "better" within-word-masking variant that fixes inflated OOV scores. Compute both in Python with **`minicons`** (`scorer.IncrementalLMScorer`, `scorer.MaskedLMScorer`, `scorer.Seq2SeqScorer`) — it wraps any HuggingFace LM and returns token surprisals/sequence scores with selectable reductions; it has a CLI too.
- **Thresholding:** calibrate on in-domain clean text — compute the perplexity/PLL distribution and flag the upper tail (e.g., per-field z-score or quantile boundary, as the BERTIN mc4-sampling code does with Gaussian/step functions around the median). Don't use a global magic number; thresholds are domain- and length-sensitive.
- **Domain-adapting the prior (critical):** an out-of-domain LM flags style, not errors. Options: train an in-domain n-gram model (cheapest), fine-tune the neural LM, or interpolate in-domain + general n-gram models. Choose **character/byte-level** for OCR/typo work and subword/word-level for semantic anomalies.
- **Tradeoffs:** flagging is cheap and effective; neural PLL is slower (one forward pass per masked token unless you fine-tune for single-pass scoring). Failure modes: penalizes rare-but-correct domain terms; conflates "unusual" with "wrong."

#### Layer 4 — Constrained / structured generation (for generative extractors only)
**What it does:** at decode time, masks tokens that would violate a grammar/schema so the extractor *cannot* emit malformed output. **Prevents-on-generate** (a structural form of correction); guarantees **well-formedness, not semantic correctness**.
- **Mechanism:** compile schema → regex/CFG → finite-state machine (or pushdown automaton); at each step the FSM yields the legal next-token set and the sampler sets illegal logits to −∞. Outlines precomputes an FSM/index over the vocabulary (≈O(1) per step, model-size-independent overhead); XGrammar/llguidance use byte-level pushdown automata with token-mask precomputation (~50µs/token).
- **Library landscape (2026):**
  - **outlines (dottxt-ai, 1.3.0; very active):** regex/JSON-schema/CFG/Pydantic; FSM-based; strong vLLM integration; broad local-model support. Limited support for closed APIs (needs logits).
  - **XGrammar (mlc-ai, 0.2.x; very active; default backend in vLLM/SGLang/TensorRT-LLM/MLC-LLM):** best on complex nested grammars; near-zero overhead.
  - **guidance (guidance-ai, 0.3.x) + llguidance (Rust core):** programmatic interleaving of generation and control; "guidance acceleration" skips forced tokens.
  - **lm-format-enforcer (0.11.x):** token-filtering, integrated in vLLM; lower maintenance cadence; can under-enforce on some long-context cases.
  - **jsonformer (0.12.0):** effectively dormant/minimally maintained — fills values into a fixed JSON skeleton; fine for simple flat schemas, but prefer outlines/XGrammar for new work.
  - **instructor (567-labs, 1.14.x; very active):** not a decoder constraint — it wraps provider APIs, validates against a pydantic model, and **retries on validation failure** (post-hoc + auto-repair loop). 15+ providers via `from_provider`, modes `TOOLS`/`JSON_SCHEMA`/`MD_JSON`.
  - **Native provider structured outputs:** OpenAI `response_format={"type":"json_schema", strict:true}` guarantees structurally schema-valid JSON ("will reliably produce valid JSON matching the supplied schema") but **constrains structure, not content** — `pattern`/`format`/`minimum` are NOT enforced, and refusals/length-truncation still occur. Anthropic shipped Structured Outputs (beta header `structured-outputs-2025-11-13`, transitioning to an `output_config` shape) for Claude Sonnet 4.5 / Opus 4.1 via constrained decoding, also a structure-not-accuracy guarantee.
- **Key tradeoffs:** (1) constraining structure does NOT guarantee correct content — a schema-valid record can be entirely wrong. (2) Over-constraining degrades quality: "Let Me Speak Freely?" (Tam et al., 2024, EMNLP Industry Track) reports domain-dependent degradation, with "minimal impact on mathematical reasoning (14.6% drop) but severe effects on tasks requiring uncertainty handling like business ethics (48.1% drop)," concluding "we observe a significant decline in LLMs' reasoning abilities under format restrictions." Mitigations: let the model reason in free text first then emit structure (2-turn), and grammar-aligned decoding (ASAp) to undo the sampling-distribution distortion that masking introduces. (3) Tokenization artifacts (the `89,000` vs `89000` problem) can push the model out of distribution. (4) Constraining interacts badly with logprobs (some APIs return empty logprobs under strict mode). **Distinguish constrain-on-generate (outlines/XGrammar/native) from post-hoc validation (instructor's validate+retry, or Layer 1 on the parsed output).**

#### Layer 5 — Learned / seq2seq / LLM correction
**What it does:** rewrites the observed value toward the inferred true value. **CORRECTS** — and is the only layer that freely invents content. Most expensive, stochastic, needs its own reference-based or reference-free evaluation.
- **Lineage (post-OCR):** lexical/dictionary + heuristics → statistical & neural MT framing (treat OCR'd text → clean text as translation) → character-level seq2seq with copy/attention → pretrained encoder-decoders (BART, ByT5, mT5; ByT5 byte-level is well-suited to OCR noise) → prompted/fine-tuned LLMs.
- **ICDAR competition results to anchor expectations:**
  - **ICDAR2017** (Chiron, Doucet, Coustaty, Moreux): 12M characters, English+French, 11 teams; best detector **WFST-PostOCR** (Nguyen et al. — probabilistic char error models compiled into weighted FST edit transducers + LM), best corrector **Char-SMT/NMT** (Amrhein & Clematide — ensembles of char-based MT over token windows). Metrics: F-measure (detection); weighted sum of Levenshtein distances / % improvement (correction). Only half the methods improved the data on average — correction is genuinely hard.
  - **ICDAR2019** (Rigaud et al.): "An original dataset of 22M OCR-ed symbols along with an aligned ground truth was provided… covering 10 European languages (Bulgarian, Czech, Dutch, English, Finnish…)" — newspapers, historical books, receipts; 34 registrations, 5 submissions; tasks now dependent (Task 2 conditioned on Task 1). Winner **Clova AI "CCC"** — multilingual BERT fine-tuned (with conv + FC layers) for detection, then a character-level seq2seq with attention for correction; best detection (95% on German) and best correction on 8/10 languages. Detection scores ranged 41–95%; best correction improvement ~44%.
  - **After 2019:** Ramirez-Orta et al. (2022) char-seq2seq ensembles reached SOTA on 5/9 ICDAR2019 languages; Beshirov et al. (2024) report that "The Seq2Seq Final achieves 25.4% of improvement on the ICDAR 2019 dataset, which improves the results of the best-performing model from the competition (CCC) by +16.4% of improvement." Synthetic-noise pretraining (inject empirical OCR error distributions into clean corpora) is the dominant trick for data scarcity.
- **LLM-as-corrector (current, mixed evidence):**
  - **Positive:** Thomas et al. (2024, LT4HALA) instruction-tuned Llama 2 vs. fine-tuned BART on BLN600 (19th-c. British newspapers): "Llama 2 7B achieves a 43.26% reduction in CER, whilst Llama 2 13B achieves a 54.51% reduction in CER… against BART's 23.30%." Bourne's **CLOCR-C** (2024) used infilling/context-adaptive LMs (GPT-4, GPT-3.5, Claude 3, Llama 3, Gemma, Mixtral): "GPT-4 and Opus, the top performing LMs, reduced the CER by an error reduction percentage of over 60% on the NCSE dataset (CER from 0.18 to 0.1)… over 51% on SMH… and 48% on the CA dataset," with downstream NER gains and a measurable benefit from supplying socio-cultural context in the prompt.
  - **Negative / cautionary:** Kanerva et al. (2025), "OCR Error Post-Correction with LLMs in Historical Documents: **No Free Lunches**": open-weight LLMs helped English CER but failed to reach practically useful performance for Finnish. Boros et al. (2024, "Post-Correction of Historical Text Transcripts with LLMs: An Exploratory Study," LaTeCH-CLfL) found "the results of the study were mostly negative, concluding that LLMs (including the commercial GPT-4 model) are not effective at correcting transcriptions of historical documents, in many cases the LLM actually decreasing the quality instead of improving it." LLMs also cannot reliably self-correct (Huang et al. 2024; Stechly et al. 2023) and can introduce hallucinations / "fix" genuine source misspellings (the verbatim-transcription hazard).
  - **Multimodal:** feeding the page image + OCR text to an mLLM (Greif et al. 2025) markedly improves correction on hard historical sources (Fraktur, mixed fonts).
- **Tooling reality:** classical baselines are production-ready and deterministic — Norvig's corrector (toy baseline), symspellpy, hunspell/aspell, KenLM+FST. Neural seq2seq correctors (ByT5/BART fine-tunes) are reproducible but require training data and GPUs. **LLM post-OCR correction is research-grade**: powerful in the best cases, unreliable across languages/domains, and always needs evaluation because it can make text worse.
- **Tradeoffs:** highest correction ceiling, highest cost and risk. Gate it behind Layers 0–3 (only send flagged spans), constrain or verify its output, and never let it silently rewrite high-stakes fields without an audit trail.

---

### Cross-cutting: anomaly detection & cross-source triangulation

**Cross-source corroboration (the strongest reference-free signal).** If two independent extractors/engines (or two source documents) yield the same value, that agreement is positive evidence; disagreement is a high-value flag. For sequences, **ROVER** (Recognizer Output Voting Error Reduction; Fiscus 1997) aligns multiple hypotheses into word-transition/confusion networks and picks each slot by (optionally confidence-weighted) majority vote — directly applicable to combining multiple OCR/ASR engines. For stochastic LLM extractors, **self-consistency** (Wang et al. 2022 — sample N times, majority-vote the answer) is the same idea; ranked-voting and self-certainty-weighted variants exist. ROVER's known limitations (skeleton-ordering sensitivity; needs confidence scores for weighting; assumes white-box systems) motivate LLM-based fusion of N-best lists where available.

**Statistical anomaly detection on extracted values (reference-free FLAGS).**
- **Benford's law** for naturally-occurring numeric fields (amounts, populations): first-digit distribution should be logarithmic; deviation flags fabricated/erroneous data. Standard in forensic accounting (Nigrini); easy in numpy/scipy, with packages like `benfordpy`.
- **Isolation forests / robust statistics / distribution flags** via **pyod (3.6.x; 60+ detectors; very active, Yue Zhao)** — train on "normal" extracted values, score outliers; or simpler robust z-scores / IQR via `scipy.stats`. For streaming extraction, **river** provides online outlier detectors.
- **Tradeoffs:** these surface implausible values cheaply and without references, but they only FLAG and can false-positive on legitimately skewed distributions. Use them to route records to review or to higher layers, not to auto-edit.

---

### Architectural sketch: a pluggable, declarative validation/correction pipeline
Consistent with a functional/declarative Python style (composition over inheritance, SSOT schemas, dependency-injected validators, progressive disclosure):

```python
from dataclasses import dataclass
from typing import Callable, Protocol

@dataclass(frozen=True)
class Finding:
    field: str; layer: str; severity: str; message: str
    suggestion: str | None = None   # present iff the layer can CORRECT

class Validator(Protocol):
    layer: str
    def __call__(self, record: dict) -> list[Finding]: ...

# Each layer is a plain function (or a closure capturing its config/deps).
# Compose them declaratively; order encodes the cheap->expensive spine.
def pipeline(*validators: Validator) -> Callable[[dict], list[Finding]]:
    def run(record: dict) -> list[Finding]:
        findings: list[Finding] = []
        for v in validators:                 # short-circuit policy is injectable
            findings.extend(v(record))
        return findings
    return run

validate = pipeline(
    canonicalize,            # Layer 0  (CORRECTS, deterministic)
    schema_checks,           # Layer 1  (FLAG/coerce; pydantic/pandera)
    enum_fuzzy_resolve,      # Layer 2  (CORRECT on closed sets; rapidfuzz/symspell)
    lm_surprisal_flags,      # Layer 3  (FLAG; kenlm/minicons, in-domain-tuned)
    cross_source_vote,       # cross-cutting (FLAG/CORRECT by agreement; ROVER/self-consistency)
    benford_outlier_flags,   # cross-cutting (FLAG; pyod/scipy)
    # llm_correct,           # Layer 5  (CORRECT; gated to flagged spans only)
)
```
Key properties: validators are independently testable pure functions; the schema is a single pydantic model reused for generation, parsing, and validation; expensive layers (LM, LLM) are dependency-injected and invoked only on records the cheap layers flagged; every `Finding` carries provenance (which layer, correct-or-flag) so the whole pipeline is auditable.

## FLAG vs. CORRECT — summary table

| Layer | Method | FLAG / CORRECT | Determinism | Key 2026 tooling (status) | Worth the cost when… |
|---|---|---|---|---|---|
| 0 | Canonicalization | CORRECT (narrow) | Deterministic | stdlib, `dateutil`, `babel`, `python-stdnum` | Always — run first |
| 1 | Type/range/enum/regex/cross-field/FK | FLAG (+coerce) | Deterministic | pydantic 2.13, pandera 0.31, jsonschema 4.26 / fastjsonschema, Great Expectations 1.18 | Always |
| 2 | Lexicon/gazetteer/fuzzy | FLAG; CORRECT on closed sets | Deterministic | rapidfuzz 3.14, symspellpy 6.9, hunspell | You have a controlled vocabulary/enum |
| 3 | LM surprisal / PLL | FLAG (CORRECT only with a generator) | Deterministic (given model) | KenLM, minicons (PLL) | Free-text fields; you can build an in-domain prior |
| 4 | Constrained generation | Prevent-on-generate (well-formedness only) | Deterministic structure | outlines 1.3, XGrammar 0.2, llguidance, lm-format-enforcer, instructor 1.14, native OpenAI/Anthropic | Extractor is generative; you control decoding |
| 5 | seq2seq / LLM correction | CORRECT (free) | Stochastic | ByT5/BART fine-tunes; LLMs (research-grade) | Cheap layers can't resolve and the field is high-value |
| ✶ | Cross-source / anomaly detection | FLAG (CORRECT by vote) | Deterministic | ROVER/self-consistency; pyod 3.6, scipy, river; Benford | You have ≥2 sources/engines, or numeric distributions |

## Recommendations
1. **Build the pipeline bottom-up and stop as early as it works.** Implement Layers 0–1 first (canonicalization + pydantic/pandera). For a large fraction of extraction fields (codes, dates, amounts, enums) this plus Layer 2 closed-set resolution catches and fixes most errors deterministically and auditably — no LM needed. Benchmark: if deterministic layers already drive your field-level error rate below your acceptance threshold, do not add LM/LLM layers.
2. **Add Layer 3 LM-flagging only for free-text fields, and always domain-adapt the prior.** Train an in-domain (character-level for OCR) KenLM first; reach for neural PLL via minicons only if n-gram flagging under-performs. Calibrate thresholds on in-domain clean text (quantile/z-score), never a global constant. This layer routes records to review — it does not edit.
3. **For generative (LLM/VLM) extractors, prefer native structured outputs or outlines/XGrammar to guarantee well-formedness — but let the model reason before it emits structure.** Use a two-turn pattern (reason free-text → emit constrained JSON) to avoid the reasoning tax (which, per Tam et al., ranges from ~15% on math to ~48% on uncertainty-handling tasks). Use instructor's validate-and-retry as the post-hoc safety net. Remember structure ≠ correctness: still run Layer 1 on the parsed record.
4. **Exploit cross-source agreement wherever you have it.** Running two OCR engines (or sampling an LLM N times) and voting (ROVER / self-consistency) is often a higher-ROI accuracy gain than a fancier single corrector, and it is reference-free. Treat disagreement as the flag that triggers Layer 5.
5. **Gate Layer 5 (LLM/seq2seq correction) tightly and evaluate it continuously.** Send only flagged spans, supply context (and the page image to an mLLM for hard OCR), constrain or verify the output, keep the original + an audit trail, and measure CER/field-accuracy on a held-out set every release — because the literature shows LLM correction can silently make text worse, especially in non-English / low-resource settings. **Threshold to change course:** if measured post-correction error on any language/domain slice is not strictly better than the pre-correction baseline, disable Layer 5 for that slice.

## Caveats
- **"No reference" is the whole point and the whole risk.** Every method here estimates plausibility, not truth. A schema-valid, low-perplexity, multi-source-agreed value can still be wrong (e.g., a systematic OCR error reproduced by both engines). Reserve human review for high-stakes flagged records.
- **Confidence signals from extractors are uncalibrated** (Tesseract's per-symbol scores, LLM logprobs where present). Use them as features feeding the prior layers, not as probabilities.
- **The constrained-generation quality literature is still settling.** "Let Me Speak Freely?" (degradation) and dottxt's rebuttal ("Say What You Mean," attributing degradation to poor prompts/schemas) disagree; treat the reasoning tax as real but mitigable, and benchmark on your own task.
- **LLM post-OCR results are dataset-specific and not yet production-trustworthy as an unsupervised step.** Headline CER reductions (54.51%, 60%+) coexist with mostly-negative findings (Boros et al.: LLMs including GPT-4 "not effective," often decreasing quality; Kanerva et al.: Finnish unusable). Do not generalize a single paper's numbers to your corpus.
- **Library currency:** versions cited are as of mid-2026 (pydantic 2.13, pandera 0.31, jsonschema 4.26, outlines 1.3, XGrammar 0.2, guidance 0.3, instructor 1.14, rapidfuzz 3.14, symspellpy 6.9, pyod 3.6, attrs/cattrs 26.1, Great Expectations 1.18). jsonformer is effectively dormant — prefer outlines/XGrammar. Anthropic's structured-outputs beta header is transitioning to a newer `output_config` API; verify before pinning.

## REFERENCES
1. Chiron G, Doucet A, Coustaty M, Moreux J-P. [ICDAR2017 Competition on Post-OCR Text Correction](https://hal.science/hal-03025499/document). Proc. 14th IAPR ICDAR; 2017.
2. Rigaud C, Doucet A, Coustaty M, Moreux J-P. [ICDAR 2019 Competition on Post-OCR Text Correction](https://hal.science/hal-02304334v1/document). Proc. 15th ICDAR; 2019.
3. Nguyen TTH, Jatowt A, Coustaty M, Doucet A. [Survey of Post-OCR Processing Approaches](https://dl.acm.org/doi/fullHtml/10.1145/3453476). ACM Comput Surv. 2021;54(6).
4. Salazar J, Liang D, Nguyen TQ, Kirchhoff K. [Masked Language Model Scoring](https://aclanthology.org/2020.acl-main.240/). Proc. ACL; 2020.
5. Kauf C, Ivanova AA. [A Better Way to Do Masked Language Model Scoring](https://aclanthology.org/2023.acl-short.80/). Proc. ACL; 2023.
6. Misra K. [minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models](https://arxiv.org/abs/2203.13112). arXiv:2203.13112; 2022. Repo: [kanishkamisra/minicons](https://github.com/kanishkamisra/minicons).
7. Heafield K. [KenLM: Faster and Smaller Language Model Queries](https://kheafield.com/papers/avenue/kenlm.pdf). Proc. WMT; 2011. Repo: [kpu/kenlm](https://github.com/kpu/kenlm).
8. Willard BT, Louf R. [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) (Outlines). arXiv:2307.09702; 2023. Repo: [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines).
9. Dong Y, et al. [XGrammar: Flexible and Efficient Structured Generation Engine for LLMs](https://arxiv.org/abs/2411.15100). arXiv:2411.15100; 2024. Repo: [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar).
10. guidance-ai. [llguidance: Super-fast Structured Outputs](https://github.com/guidance-ai/llguidance). GitHub; 2026.
11. Tam ZR, Wu C-K, Tsai Y-L, Lin C-Y, Lee H, Chen Y-N. [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of LLMs](https://arxiv.org/abs/2408.02442). arXiv:2408.02442; 2024.
12. Kurt W. [Say What You Mean: A Response to 'Let Me Speak Freely'](https://blog.dottxt.co/say-what-you-mean.html). dottxt blog; 2024.
13. Geng S, et al. [JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models](https://arxiv.org/abs/2501.10868). arXiv:2501.10868; 2025.
14. OpenAI. [Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/). 2024.
15. Anthropic. [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). Claude Developer Platform docs; 2025–2026.
16. instructor (567-labs). [Instructor documentation](https://python.useinstructor.com/). 2026.
17. Pydantic. [Pydantic v2 documentation](https://docs.pydantic.dev/). 2026. Repo: [pydantic/pydantic](https://github.com/pydantic/pydantic).
18. Pandera (Union.ai). [Pandera documentation](https://pandera.readthedocs.io/). 2026. Repo: [unionai-oss/pandera](https://github.com/unionai-oss/pandera).
19. Berman J. [jsonschema for Python](https://python-jsonschema.readthedocs.io/). 2025. PyPI: [jsonschema](https://pypi.org/project/jsonschema/). Also [fastjsonschema](https://horejsek.github.io/python-fastjsonschema/).
20. Great Expectations. [Introducing GX Core 1.0](https://greatexpectations.io/blog/the-future-of-gx-os-rises-with-gx-core/). 2024.
21. Garbe W. [SymSpell](https://github.com/wolfgarbe/SymSpell); Python port [symspellpy](https://github.com/mammothb/symspellpy).
22. [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz). PyPI: [rapidfuzz](https://pypi.org/project/RapidFuzz/).
23. Zhao Y, Nasrullah Z, Li Z. [PyOD: A Python Toolbox for Scalable Outlier Detection](https://www.jmlr.org/papers/v20/19-011.html). JMLR; 2019. Repo: [yzhao062/pyod](https://github.com/yzhao062/pyod).
24. Fiscus JG. [A Post-Processing System to Yield Reduced Word Error Rates: Recognizer Output Voting Error Reduction (ROVER)](https://www.researchgate.net/publication/2397671). Proc. IEEE ASRU; 1997.
25. Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D. [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). arXiv:2203.11171; 2022.
26. Bourne J. [CLOCR-C: Context Leveraging OCR Correction with Pre-trained Language Models](https://arxiv.org/abs/2408.17428). arXiv:2408.17428; 2024. Repo: [JonnoB/clocrc](https://github.com/JonnoB/clocrc).
27. Thomas A, Gaizauskas R, Lu H. [Leveraging LLMs for Post-OCR Correction of Historical Newspapers](https://aclanthology.org/2024.lt4hala-1.14/). Proc. LT4HALA @ LREC-COLING; 2024.
28. Kanerva J, Ledins C, Käpyaho S, Ginter F. [OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches](https://aclanthology.org/2025.resourceful-1.8/). Proc. RESOURCEFUL; 2025.
29. Ramirez-Orta J, Xamena E, Maguitman A, Milios E, Soto AJ. [Post-OCR Document Correction with Large Ensembles of Character Sequence-to-Sequence Models](https://arxiv.org/abs/2109.06264). arXiv:2109.06264; 2021.
30. Beshirov A, et al. [Post-OCR Text Correction for Bulgarian Historical Documents](https://arxiv.org/abs/2409.00527). arXiv:2409.00527; 2024.
31. Boros E, et al. [Post-Correction of Historical Text Transcripts with Large Language Models: An Exploratory Study](https://aclanthology.org/2024.latechclfl-1.14/). Proc. LaTeCH-CLfL; 2024.
32. Nigrini MJ. Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection. Wiley; 2012.

---

*This report is provided as Markdown. To save as a `.md` file: copy the content above into a file named `R4_Post-Extraction_Validation_and_Correction.md`.*