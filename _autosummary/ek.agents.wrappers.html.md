# ek.agents.wrappers

Optional WRAP adapters behind `ek[agents]`: uqlm, Ragas and DeepEval as ek strategies.

`ek.agents` scores episodes in pure python; this module is the batteries-included layer for
teams who already run these libraries. Every adapter imports its backend **lazily** (at call
time, guarded by [`requires_extra()`](ek.registry.html.md#ek.registry.requires_extra)), so `import ek` stays light, and each
takes an injectable backend object so it can be tested – and used – without network access.

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

| [`DEEPEVAL_TELEMETRY_ENV`](#ek.agents.wrappers.DEEPEVAL_TELEMETRY_ENV)   | Env var that opts out of DeepEval's Confident-AI telemetry (set before importing it).   |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`DEFAULT_NUM_RESPONSES`](#ek.agents.wrappers.DEFAULT_NUM_RESPONSES)    | Sample count for uqlm's black-box scorers (a keyword default, not a magic number).      |
| [`DEFAULT_UQLM_SCORER`](#ek.agents.wrappers.DEFAULT_UQLM_SCORER)      | uqlm black-box scorer used when none is named.                                          |
| [`DEFAULT_RAGAS_METRIC`](#ek.agents.wrappers.DEFAULT_RAGAS_METRIC)     | Ragas metric used when none is named.                                                   |
| [`RAGAS_METRICS`](#ek.agents.wrappers.RAGAS_METRICS)            | ek's metric name -> Ragas class name.                                                   |
| [`DEEPEVAL_KINDS`](#ek.agents.wrappers.DEEPEVAL_KINDS)           | ek name -> (DeepEval class name, needs an LLM judge).                                   |

### Functions

| [`disable_deepeval_telemetry`](#ek.agents.wrappers.disable_deepeval_telemetry)()   | Opt out of Confident-AI telemetry (idempotent; never overrides an explicit choice).   |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|

### Classes

| [`DeepEvalMetric`](#ek.agents.wrappers.DeepEvalMetric)([kind, metric])              | A DeepEval metric as an ek `Metric` over an [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode).   |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [`RagasSignal`](#ek.agents.wrappers.RagasSignal)([metric, name, llm, cost_tier]) | A reference-free Ragas metric as a [`Signal`](ek.base.html.md#ek.base.Signal) over a RAG sample.   |
| [`UqlmConsistencySignal`](#ek.agents.wrappers.UqlmConsistencySignal)([scorer, llm, ...])   | Self-consistency confidence in `[0, 1]` from `uqlm`'s black-box scorers.                                                       |

### ek.agents.wrappers.DEEPEVAL_KINDS *= {'geval': 'GEval', 'task_completion': 'TaskCompletionMetric', 'tool_correctness': 'ToolCorrectnessMetric'}*

ek name -> (DeepEval class name, needs an LLM judge).

### ek.agents.wrappers.DEEPEVAL_TELEMETRY_ENV *= 'DEEPEVAL_TELEMETRY_OPT_OUT'*

Env var that opts out of DeepEval’s Confident-AI telemetry (set before importing it).

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
  * **\*\*metric_kwargs** – Passed to the DeepEval constructor (e.g. `criteria=` for GEval).

#### aggregate(scores)

Mean of the per-episode scores (DeepEval scores are per-case, not count-based).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.wrappers.RAGAS_METRICS *= {'answer_relevancy': 'ResponseRelevancy', 'context_precision': 'LLMContextPrecisionWithoutReference', 'faithfulness': 'Faithfulness'}*

ek’s metric name -> Ragas class name.

### *class* ek.agents.wrappers.RagasSignal(metric=None, , name='faithfulness', llm=None, cost_tier=3)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reference-free Ragas metric as a [`Signal`](ek.base.html.md#ek.base.Signal) over a RAG sample.

The sample is a mapping/object with `user_input` (or `question`), `response` (or
`answer`) and `retrieved_contexts` (or `contexts`). The score is in `[0, 1]`.
Ragas metrics call an LLM, so pass a configured `metric` (or `llm=` to build one).

* **Parameters:**
  * **metric** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A Ragas metric object exposing `single_turn_ascore(sample)` (injected), or
    `None` to build `name` from `llm`.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `faithfulness` | `answer_relevancy` | `context_precision`.
  * **llm** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A `ragas` LLM wrapper, used when `metric` is `None`.
  * **cost_tier** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `3` – an LLM call per sample.

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
  * **llm** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A LangChain chat model, required when `scorer` is `None`.
  * **scorer_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The uqlm scorer column to read (default `"noncontradiction"`).
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
