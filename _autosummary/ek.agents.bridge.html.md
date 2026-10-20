# ek.agents.bridge

One-way adapters: turn whatever your agent already emits into an `ek` `Episode`.

The dependency direction is the `ek -> ocracy` rule restated, and it is a **hard rule**:
`ek -> inspect_ai / deepeval / ragas` (via the `ek[agents]` extra), **never the reverse**.

Note what that buys: `ek` core depends only on the *shape* of what those tools emit, so the
adapters here **duck-type** and import nothing. You need the extra to *run* Inspect or DeepEval;
you do not need it to *score* what they produced – exactly as `ek` evaluates any
`OcrResult`-shaped object without importing an OCR engine.

The workhorse is [`trajectory_from_messages()`](#ek.agents.bridge.trajectory_from_messages): provider-shaped chat transcripts (OpenAI- and
Anthropic-style `tool_calls` / `tool_use` blocks) are the universal wire format for a
tool-using agent, and parsing them needs no SDK at all.

### Example

```pycon
>>> messages = [
...     {"role": "user", "content": "weather in Paris?"},
...     {"role": "assistant", "tool_calls": [
...         {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
...     {"role": "tool", "content": "18C"},
...     {"role": "assistant", "content": "It is 18C in Paris."},
... ]
>>> traj = trajectory_from_messages(messages)
>>> traj.tools
('get_weather',)
>>> traj.steps[0].args, traj.steps[0].observation
({'city': 'Paris'}, '18C')
```

### Functions

| [`as_agent`](#ek.agents.bridge.as_agent)(fn)                                 | Wrap a plain `input -> answer` function into the harness's `TaskSpec -> Episode` shape.                                                  |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| [`cost_from_usage`](#ek.agents.bridge.cost_from_usage)(usage, \*[, latency_s])      | Build a [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) from a provider `usage` object or dict.                |
| [`episode_from_messages`](#ek.agents.bridge.episode_from_messages)(messages, \*[, ...])   | Build a full [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode) from a transcript (+ optional usage/state). |
| [`from_deepeval_test_case`](#ek.agents.bridge.from_deepeval_test_case)(case, \*[, task_id]) | Adapt a DeepEval `LLMTestCase`-shaped object into an `Episode` (duck-typed).                                                             |
| [`from_inspect_sample`](#ek.agents.bridge.from_inspect_sample)(sample, \*[, task_id])   | Adapt an Inspect AI `EvalSample`-shaped object into an `Episode` (duck-typed).                                                           |
| [`trajectory_from_messages`](#ek.agents.bridge.trajectory_from_messages)(messages)           | Parse a chat transcript into a [`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory).                    |

### ek.agents.bridge.as_agent(fn)

Wrap a plain `input -> answer` function into the harness’s `TaskSpec -> Episode` shape.

Progressive disclosure: the trivial agent should not have to learn the Episode type.

### Example

```pycon
>>> from ek.agents.base import TaskSpec
>>> agent = as_agent(str.upper)
>>> agent(TaskSpec("t1", input="hi")).output
'HI'
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### ek.agents.bridge.cost_from_usage(usage, , latency_s=None)

Build a [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) from a provider `usage` object or dict.

Understands the OpenAI (`prompt_tokens` / `completion_tokens`) and Anthropic
(`input_tokens` / `output_tokens` / `cache_read_input_tokens`) spellings.

### Example

```pycon
>>> c = cost_from_usage({"prompt_tokens": 100, "completion_tokens": 20})
>>> c.input_tokens, c.output_tokens
(100, 20)
```

* **Return type:**
  [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)

### ek.agents.bridge.episode_from_messages(messages, , task_id='', usage=None, output=None, final_state=None, latency_s=None)

Build a full [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode) from a transcript (+ optional usage/state).

`output` defaults to the last assistant text – the agent’s final answer.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### ek.agents.bridge.from_deepeval_test_case(case, , task_id='')

Adapt a DeepEval `LLMTestCase`-shaped object into an `Episode` (duck-typed).

Reads `.actual_output` and `.tools_called`. Note DeepEval phones home by default
(Confident-AI telemetry) – disable it before use if that matters to you; `ek` never
enables it.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### ek.agents.bridge.from_inspect_sample(sample, , task_id='')

Adapt an Inspect AI `EvalSample`-shaped object into an `Episode` (duck-typed).

Reads `.messages`, `.output` and `.id` – no `inspect_ai` import required, so scoring
an Inspect log never drags the harness into `ek`’s dependency closure.

* **Return type:**
  [`Episode`](ek.agents.base.html.md#ek.agents.base.Episode)

### ek.agents.bridge.trajectory_from_messages(messages)

Parse a chat transcript into a [`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory).

Each assistant tool call becomes a [`Step`](ek.agents.base.html.md#ek.agents.base.Step), and the *next* tool/result
message becomes that step’s `observation`. A tool message carrying an error marks the step
as errored – error *recovery* across later steps is itself an evaluable signal.

* **Return type:**
  [`Trajectory`](ek.agents.base.html.md#ek.agents.base.Trajectory)
