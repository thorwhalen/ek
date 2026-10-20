# ek.agents.wrappers

Optional WRAP adapters: Inspect AI, uqlm, Ragas and DeepEval as ek strategies.

`ek.agents` scores episodes in pure python; this module is the batteries-included layer for
teams who already run these libraries. Every adapter imports its backend **lazily** (at call
time, guarded by [`requires_extra()`](ek.registry.html.md#ek.registry.requires_extra)), so `import ek` stays light, and each
takes an injectable backend object so it can be tested – and used – without network access.
The dependency direction stays `ek -> backend`, never the reverse. Inspect AI, Ragas and
DeepEval come with `ek[agents]`; uqlm (heavy: torch/transformers) with `ek[agreement]`.

- [`run_inspect()`](#ek.agents.wrappers.run_inspect) – run an Inspect AI eval and get graded [`Episode`](ek.agents.html.md#ek.agents.Episode) s
  back (Inspect’s `epochs` are ek’s trials, so the result feeds `reliability` directly).
- [`UqlmConsistencySignal`](#ek.agents.wrappers.UqlmConsistencySignal) – black-box sample-and-vote self-consistency from `uqlm`
  (a `[0, 1]` confidence), a [`Signal`](ek.base.html.md#ek.base.Signal) (expensive: `cost_tier=3`).
- [`RagasSignal`](#ek.agents.wrappers.RagasSignal) – reference-free Ragas metrics (faithfulness, answer relevancy,
  context precision) as a [`Signal`](ek.base.html.md#ek.base.Signal) over a RAG sample.
- [`DeepEvalMetric`](#ek.agents.wrappers.DeepEvalMetric) – DeepEval’s `ToolCorrectnessMetric` / `TaskCompletionMetric` /
  `GEval` as ek `metrics`. Confident-AI telemetry is **disabled** before the import.
  Note `ToolCorrectnessMetric` divides by `len(expected_tools)` – it is
  *recall-flavored*, not F1 (compare `ek`’s own cost-aware `tool_call`).

Example (a fake backend, no extra needed):

```pycon
>>> class FakeUQ:
...     def score(self, prompt):
...         return 0.75
>>> UqlmConsistencySignal(scorer=FakeUQ())("What is 2+2?")
0.75
```

### Module Attributes

| [`DEEPEVAL_TELEMETRY_ENV`](#ek.agents.wrappers.DEEPEVAL_TELEMETRY_ENV)   | Env var that opts out of DeepEval's Confident-AI telemetry (set before importing it).                                                                                                                                    |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DEFAULT_NUM_RESPONSES`](#ek.agents.wrappers.DEFAULT_NUM_RESPONSES)    | Sample count for uqlm's black-box scorers (a keyword default, not a magic number).                                                                                                                                       |
| [`DEFAULT_UQLM_SCORER`](#ek.agents.wrappers.DEFAULT_UQLM_SCORER)      | uqlm black-box scorer used when none is named.                                                                                                                                                                           |
| [`DEFAULT_RAGAS_METRIC`](#ek.agents.wrappers.DEFAULT_RAGAS_METRIC)     | Ragas metric used when none is named.                                                                                                                                                                                    |
| [`RAGAS_METRICS`](#ek.agents.wrappers.RAGAS_METRICS)            | ek's metric name -> Ragas class name in the legacy `ragas.metrics` API (deprecated since Ragas 0.4) -- metrics take a `BaseRagasLLM` and expose `single_turn_ascore(sample)`.                                            |
| [`RAGAS_COLLECTION_METRICS`](#ek.agents.wrappers.RAGAS_COLLECTION_METRICS) | ek's metric name -> Ragas class name in `ragas.metrics.collections` (Ragas >= 0.4, the API that survives 1.0) -- metrics take an `InstructorBaseRagasLLM` (`llm_factory`) and expose `ascore(**fields) -> MetricResult`. |
| [`DEEPEVAL_KINDS`](#ek.agents.wrappers.DEEPEVAL_KINDS)           | ek name -> (DeepEval class name, needs an LLM judge).                                                                                                                                                                    |
| [`DEFAULT_GEVAL_PARAMS`](#ek.agents.wrappers.DEFAULT_GEVAL_PARAMS)     | Test-case fields a GEval judge reads when `evaluation_params` is not given.                                                                                                                                              |

### Functions

| [`disable_deepeval_telemetry`](#ek.agents.wrappers.disable_deepeval_telemetry)()                 | Opt out of Confident-AI telemetry (idempotent; never overrides an explicit choice).                                                   |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [`run_inspect`](#ek.agents.wrappers.run_inspect)(tasks, \*[, model, epochs, ...]) | Run an Inspect AI eval and return its samples as graded [`Episode`](ek.agents.html.md#ek.agents.Episode) s. |

### Classes

| [`DeepEvalMetric`](#ek.agents.wrappers.DeepEvalMetric)([kind, metric])              | A DeepEval metric as an ek `Metric` over an [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode).   |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [`RagasSignal`](#ek.agents.wrappers.RagasSignal)([metric, name, llm, cost_tier]) | A reference-free Ragas metric as a [`Signal`](ek.base.html.md#ek.base.Signal) over a RAG sample.   |
| [`UqlmConsistencySignal`](#ek.agents.wrappers.UqlmConsistencySignal)([scorer, llm, ...])   | Self-consistency confidence in `[0, 1]` from `uqlm`'s black-box scorers.                                                       |

### ek.agents.wrappers.DEEPEVAL_KINDS *= {'geval': 'GEval', 'task_completion': 'TaskCompletionMetric', 'tool_correctness': 'ToolCorrectnessMetric'}*

ek name -> (DeepEval class name, needs an LLM judge).

### ek.agents.wrappers.DEEPEVAL_TELEMETRY_ENV *= 'DEEPEVAL_TELEMETRY_OPT_OUT'*

Env var that opts out of DeepEval’s Confident-AI telemetry (set before importing it).

### ek.agents.wrappers.DEFAULT_GEVAL_PARAMS *= ('input', 'actual_output')*

Test-case fields a GEval judge reads when `evaluation_params` is not given. DeepEval >= 4
refuses a plain `LLMTestCase` without them (it would otherwise evaluate a *trace*).

### ek.agents.wrappers.DEFAULT_NUM_RESPONSES *= 5*

Sample count for uqlm’s black-box scorers (a keyword default, not a magic number).

### ek.agents.wrappers.DEFAULT_RAGAS_METRIC *= 'faithfulness'*

Ragas metric used when none is named.

### ek.agents.wrappers.DEFAULT_UQLM_SCORER *= 'noncontradiction'*

uqlm black-box scorer used when none is named.

### *class* ek.agents.wrappers.DeepEvalMetric(kind='tool_correctness', , metric=None, \*\*metric_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A DeepEval metric as an ek `Metric` over an [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode).

`score(episode, gold)` builds a DeepEval `LLMTestCase` from the episode’s output and
tool calls (`gold` is the expected output string, or a mapping with `expected_output`
/ `expected_tools`) and returns DeepEval’s score as an ek `Score`.

`tool_correctness` is **recall-flavored** (divides by `len(expected_tools)`), not F1:
the label says so, and `detail['flavor']` records it.

* **Parameters:**
  * **kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `tool_correctness` | `task_completion` | `geval`.
  * **metric** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An injected DeepEval-shaped metric (`.measure(case)`, `.score`); `None`
    builds the real one lazily (telemetry disabled first).
  * **\*\*metric_kwargs** – Passed to the DeepEval constructor (e.g. `criteria=` for GEval, and
    `model=` – a model name or a `DeepEvalBaseLLM` – for any of them: DeepEval >= 4
    builds its default OpenAI judge at construction, so without `model=` it needs
    `OPENAI_API_KEY` even for `tool_correctness`). For GEval, `name` defaults to
    `"ek_geval"` and `evaluation_params` to `DEFAULT_GEVAL_PARAMS` (strings are
    accepted).

#### aggregate(scores)

Mean of the per-episode scores (DeepEval scores are per-case, not count-based).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.wrappers.RAGAS_COLLECTION_METRICS *= {'answer_relevancy': 'AnswerRelevancy', 'context_precision': 'ContextPrecisionWithoutReference', 'faithfulness': 'Faithfulness'}*

ek’s metric name -> Ragas class name in `ragas.metrics.collections` (Ragas >= 0.4, the API
that survives 1.0) – metrics take an `InstructorBaseRagasLLM` (`llm_factory`) and expose
`ascore(**fields) -> MetricResult`. `answer_relevancy` also needs `embeddings=`.

### ek.agents.wrappers.RAGAS_METRICS *= {'answer_relevancy': 'ResponseRelevancy', 'context_precision': 'LLMContextPrecisionWithoutReference', 'faithfulness': 'Faithfulness'}*

ek’s metric name -> Ragas class name in the legacy `ragas.metrics` API (deprecated since
Ragas 0.4) – metrics take a `BaseRagasLLM` and expose `single_turn_ascore(sample)`.

### *class* ek.agents.wrappers.RagasSignal(metric=None, , name='faithfulness', llm=None, cost_tier=3, \*\*metric_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reference-free Ragas metric as a [`Signal`](ek.base.html.md#ek.base.Signal) over a RAG sample.

The sample is a mapping/object with `user_input` (or `question`), `response` (or
`answer`) and `retrieved_contexts` (or `contexts`). The score is in `[0, 1]`.
Ragas metrics call an LLM, so pass a configured `metric` (or `llm=` to build one).

Both Ragas metric APIs are accepted: the collections API (`ascore(**fields)` returning a
`MetricResult`; `ragas.metrics.collections`, the one that survives Ragas 1.0) and the
legacy one (`single_turn_ascore(SingleTurnSample)`; deprecated since Ragas 0.4).

* **Parameters:**
  * **metric** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A Ragas metric object (either API), or `None` to build `name` from `llm`.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `faithfulness` | `answer_relevancy` | `context_precision`.
  * **llm** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The judge, used when `metric` is `None`. An `InstructorBaseRagasLLM` (from
    `ragas.llms.llm_factory`) builds the collections metric; any other Ragas LLM
    wrapper (e.g. `LangchainLLMWrapper`) builds the legacy one.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `3` – an LLM call per sample.
  * **\*\*metric_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Passed to the Ragas constructor (e.g. `embeddings=` for
    `answer_relevancy`).

### Example

```pycon
>>> class Fake:
...     async def single_turn_ascore(self, sample): return 0.5
>>> RagasSignal(Fake())({"question": "q", "answer": "a", "contexts": ["c"]})
0.5
```

### *class* ek.agents.wrappers.UqlmConsistencySignal(scorer=None, , llm=None, scorer_name='noncontradiction', num_responses=5, cost_tier=3)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Self-consistency confidence in `[0, 1]` from `uqlm`’s black-box scorers.

Samples `num_responses` generations for the prompt and measures how much they agree
(non-contradiction, exact match, semantic negentropy, …). Raw and **uncalibrated**:
calibrate before gating (Hard Rule 1).

* **Parameters:**
  * **scorer** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A backend with `score(prompt) -> float` (tests / custom sampling), or
    `None` to build uqlm’s `BlackBoxUQ` from `llm`.
  * **llm** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A LangChain chat model, required when `scorer` is `None`. uqlm reads its
    `temperature` attribute, so it must have one.
  * **scorer_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The uqlm scorer column to read (default `"noncontradiction"`, which
    downloads an NLI model on first use; `"exact_match"` needs none).
  * **num_responses** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Samples per prompt.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `3` – it pays for `num_responses` extra generations.

### Example

```pycon
>>> class Fake:
...     def score(self, prompt): return 2.0   # out-of-range is clamped
>>> UqlmConsistencySignal(scorer=Fake())("q")
1.0
```

### ek.agents.wrappers.disable_deepeval_telemetry()

Opt out of Confident-AI telemetry (idempotent; never overrides an explicit choice).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ek.agents.wrappers.run_inspect(tasks, , model=None, epochs=None, scorer=None, success_threshold=1.0, log_dir=None, \*\*eval_kwargs)

Run an Inspect AI eval and return its samples as graded [`Episode`](ek.agents.html.md#ek.agents.Episode) s.

A thin convenience over `inspect_ai.eval` + [`episodes_from_inspect_log()`](ek.agents.html.md#ek.agents.episodes_from_inspect_log):
one episode per sample per epoch, graded by the sample’s score (`scorer`, else the first)
reaching `success_threshold`. Run with `epochs=k` and the result is a k-trial run –
hand it to [`reliability()`](ek.agents.html.md#ek.agents.reliability) for `pass^k`, or to
[`cost_report()`](ek.agents.html.md#ek.agents.cost_report) for cost per successful task.

* **Parameters:**
  * **tasks** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Anything `inspect_ai.eval` accepts (a `Task`, a task function, a registry
    name, a list of them).
  * **model** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The model under test (e.g. `"openai/gpt-4o"`); `None` lets Inspect use
    `INSPECT_EVAL_MODEL`. `"mockllm/model"` runs offline at no cost.
  * **epochs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Trials per sample (Inspect’s `epochs`); `None` keeps the task’s own setting.
  * **scorer** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Which scorer’s value grades success (default: each sample’s first score).
  * **success_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – A sample succeeds when its score value is `>=` this.
  * **log_dir** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Where Inspect writes its `.eval` logs. `None` uses a temporary directory
    that is removed once the episodes are read – pass a path to keep the logs.
  * **\*\*eval_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Passed to `inspect_ai.eval` (`limit=`, `solver=`, …). `display`
    defaults to `"none"`.
* **Raises:**
  [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – if any eval did not finish with status `success` (a partial run is not
      a k-trial run, and grading it as one would understate failures).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

#### NOTE
`inspect_ai.eval` is synchronous and refuses to run inside an event loop; from async
code, call `inspect_ai.eval_async` and pass each log to `episodes_from_inspect_log`.
