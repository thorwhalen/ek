# ek.agents.base

Agent-evaluation data shapes: the episode (Layer B) and the task/tool grammar (Layer A).

Evaluating an *agent* is the same two-layer problem `ek` already solves for
information extraction – only the evaluated object changes. An extraction is a
static payload; an **episode** is a *process*: a sequence of tool calls and
observations ending in a final state. So:

- **Layer A – the task/tool grammar.** [`ToolSpec`](#ek.agents.base.ToolSpec) and [`TaskSpec`](#ek.agents.base.TaskSpec) are
  *builders* over the existing [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar): a tool becomes a
  [`NodeType`](ek.base.html.md#ek.base.NodeType) whose fields are its argument [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec) s,
  > and `importance` carries the **cost of getting that argument wrong**. A wrong
  > argument to a *destructive* tool is not one unit of error. This reuses the Layer-A
  > cost-weight SSOT rather than inventing a parallel schema.
- **Layer B – the episode.** [`Episode`](#ek.agents.base.Episode) carries the [`Trajectory`](#ek.agents.base.Trajectory), the
  [`Cost`](#ek.agents.base.Cost), the outcome, and the [`RunProvenance`](#ek.agents.base.RunProvenance) – riding *alongside* the
  > frozen grammar, never mutating it, exactly as `AnnotatedExtraction` does.

These shapes live here, **not** in [`ek.base`](ek.base.html.md#module-ek.base): that module is the zero-dependency,
IE-schema-only SSOT, and the OCR instance sets the precedent that an instance’s concrete
shapes stay out of core (core duck-types them). Nothing here is imported by `ek` core.

### Example

```pycon
>>> from ek.base import FieldSpec
>>> book = ToolSpec("book_flight", params={"flight": FieldSpec("flight", "string"),
...                                        "seat": FieldSpec("seat", "string")})
>>> refund = ToolSpec("refund", params={"amount": FieldSpec("amount", "number",
...                                     importance=20.0)}, destructive=True)
>>> g = tool_grammar(book, refund)
>>> g.field_cost("refund", "amount")        # a wrong refund amount is expensive
20.0
>>> g.field_cost("book_flight", "seat")     # a wrong seat is not
1.0
```

See `misc/docs/ek_07` (the map) and `misc/docs/ek_12` (the integration report).

### Module Attributes

| [`DEFAULT_TASK_VALUE`](#ek.agents.base.DEFAULT_TASK_VALUE)   | Default value of a completed task (the task-level analog of `FieldSpec.importance`).   |
|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`DESTRUCTIVE_WEIGHT`](#ek.agents.base.DESTRUCTIVE_WEIGHT)   | Multiplier applied to a destructive tool's importance when it is not set explicitly.   |

### Functions

| [`as_call`](#ek.agents.base.as_call)(x)            | Coerce a [`Step`](#ek.agents.base.Step), `(tool, args)` tuple, or `{"tool":…, "args":…}` dict.                                                  |
|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`is_episode`](#ek.agents.base.is_episode)(x)         | Duck-typed episode check (core never imports this module -- it checks the *shape*).                                                                                    |
| [`suite_grammar`](#ek.agents.base.suite_grammar)(tasks)  | The union grammar over a whole task suite (later tools win on name collision).                                                                                         |
| [`tool_grammar`](#ek.agents.base.tool_grammar)(\*tools) | Build the Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) from [`ToolSpec`](#ek.agents.base.ToolSpec) s. |

### Classes

| [`Cost`](#ek.agents.base.Cost)([input_tokens, output_tokens, ...])          | What one episode consumed: tokens (by kind), retries, and wall-clock.                          |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`Episode`](#ek.agents.base.Episode)([task_id, trajectory, output, ...])       | Layer B: one agent run over one task -- the object agent evaluation scores.                    |
| [`RunProvenance`](#ek.agents.base.RunProvenance)([seed, temperature, model, ...])    | The *hidden eval variables* that decide whether two runs are even comparable.                  |
| [`Step`](#ek.agents.base.Step)(tool[, args, observation, error])            | One step of a trajectory: a tool call and what came back.                                      |
| [`TaskSpec`](#ek.agents.base.TaskSpec)(task_id[, input, gold, value, ...])      | One task in a suite: its input, its oracle, its value, and its allowed tools.                  |
| [`ToolSpec`](#ek.agents.base.ToolSpec)(name[, params, importance, destructive]) | One tool the agent may call: its argument schema and its error costs.                          |
| [`Trajectory`](#ek.agents.base.Trajectory)([steps])                               | The ordered sequence of [`Step`](#ek.agents.base.Step) s an agent took. |

### *class* ek.agents.base.Cost(input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0, retries=0, latency_s=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What one episode consumed: tokens (by kind), retries, and wall-clock.

Token *kinds* are tracked separately because they are **priced asymmetrically** –
output typically costs ~5x input, cached input bills at a fraction, and reasoning
tokens are billed. This is precisely why the objective is *dollars* per successful
task and never a raw token count (`misc/docs/ek_11`). Pricing lives in
[`ek.agents.cost`](ek.agents.cost.html.md#module-ek.agents.cost); this dataclass is the provider-neutral tally.

### Example

```pycon
>>> a = Cost(input_tokens=100, output_tokens=50)
>>> b = Cost(input_tokens=10, output_tokens=5, retries=1)
>>> c = a + b
>>> c.input_tokens, c.output_tokens, c.retries
(110, 55, 1)
```

#### *property* total_tokens *: [int](https://docs.python.org/3/builtins/functions.html#int)*

All billed tokens (a rough size proxy – *not* a cost; see `dollars()`).

### ek.agents.base.DEFAULT_TASK_VALUE *= 1.0*

Default value of a completed task (the task-level analog of `FieldSpec.importance`).

### ek.agents.base.DESTRUCTIVE_WEIGHT *= 10.0*

Multiplier applied to a destructive tool’s importance when it is not set explicitly.
Calling a destructive tool wrongly is categorically worse than calling a read-only one;
a keyword-tunable default, never a magic number buried in a metric.

### *class* ek.agents.base.Episode(task_id='', trajectory=<factory>, output=None, final_state=None, cost=None, success=None, run=None, provenance=None, meta=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Layer B: one agent run over one task – the object agent evaluation scores.

The direct analog of `AnnotatedExtraction`: it rides alongside the frozen Layer-A
grammar and carries the run’s metadata. It is scored *offline* against gold
([`ek.score()`](ek.html.md#ek.score) / [`ek.evaluate()`](ek.html.md#ek.evaluate)) and estimated *online* against a judge or a
consensus ([`ek.estimate_quality()`](ek.html.md#ek.estimate_quality)) – the same object, both halves.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Which task this episode answers (the k-trial grouping key).
  * **trajectory** ([`Trajectory`](#ek.agents.base.Trajectory)) – The tool calls taken.
  * **output** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The agent’s final user-facing answer.
  * **final_state** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The end state of the world (the DB, the filesystem) – what a
    state-based oracle actually grades. Success is a *final-state* check, not a
    surface-text check.
  * **cost** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Cost`](#ek.agents.base.Cost)]) – The [`Cost`](#ek.agents.base.Cost) consumed.
  * **success** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Filled in by a checker; `None` until graded.
  * **run** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`RunProvenance`](#ek.agents.base.RunProvenance)]) – The [`RunProvenance`](#ek.agents.base.RunProvenance) (seed, model, simulator, suite version).
  * **provenance** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Provenance`](ek.base.html.md#ek.base.Provenance)]) – Optional core [`Provenance`](ek.base.html.md#ek.base.Provenance) for click-to-source audit.
  * **meta** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Anything else the bridge captured.

#### *property* calls *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pairs taken, in order.

#### graded(success)

A copy with `success` set (never mutates – grading stays idempotent).

`meta` is copied too: `dataclasses.replace` is a *shallow* copy, so without this
the “copy” would share the original’s mutable `meta` dict.

* **Return type:**
  [`Episode`](#ek.agents.base.Episode)

### *class* ek.agents.base.RunProvenance(seed=None, temperature=None, model='', simulator='', suite_version='', scaffold='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The *hidden eval variables* that decide whether two runs are even comparable.

Agent evaluation is stochastic, and several of its inputs are themselves models. In a
tau-bench-style suite the **user simulator is an LLM agent**, so its model/prompt/version
silently changes results run to run. Recording these is what lets the harness *refuse*
to compare a baseline against a run that changed the simulator or the suite – the same
discipline `ek`’s offline harness already applies when it refuses to compare across
two different metrics (`misc/docs/ek_08`, `misc/docs/ek_11`).

* **Parameters:**
  * **seed** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – RNG seed for the trial (`None` = unseeded/nondeterministic).
  * **temperature** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Sampling temperature of the agent under test.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The agent’s model id.
  * **simulator** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The **user-simulator** id/version – an eval variable, not a constant.
  * **suite_version** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Version of the task suite (contamination and task fixes move scores).
  * **scaffold** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The agent scaffold/harness id (the same model moves points on scaffold alone).

#### comparability_key()

The fields that must match for two runs to be comparable (seed excluded).

Seed and temperature are *expected* to vary across trials; the simulator, the suite
version and the scaffold are not – changing them invalidates a baseline.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### *class* ek.agents.base.Step(tool, args=<factory>, observation=None, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One step of a trajectory: a tool call and what came back.

`error` is set when the call itself failed (a bad argument, a tool exception) –
error *recovery* across subsequent steps is itself an evaluable signal.

#### *property* call *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pair – what tool-call correctness actually compares.

### *class* ek.agents.base.TaskSpec(task_id, input=None, gold=None, value=1.0, tools=(), slice=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One task in a suite: its input, its oracle, its value, and its allowed tools.

* **Parameters:**
  * **task_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Stable identifier (the grouping key for k-trial reliability).
  * **input** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Whatever the agent under test is called with.
  * **gold** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The reference the checker grades against (a goal state, an expected answer,
    a gold trajectory – the checker decides how to read it).
  * **value** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Task-level cost weight: how much a completed task is worth. The task-level
    analog of `FieldSpec.importance`; feeds value-weighted cost accounting.
  * **tools** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`ToolSpec`](#ek.agents.base.ToolSpec)]) – The tools this task permits (the Layer-A grammar for the task).
  * **slice** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional slice label (domain, difficulty, language) – per-slice cuts are
    mandatory, not optional (a low pass^k concentrated in one hard slice is a
    different problem from uniform flakiness).

#### grammar()

The Layer-A grammar for this task’s tool set.

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### *class* ek.agents.base.ToolSpec(name, params=<factory>, importance=None, destructive=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One tool the agent may call: its argument schema and its error costs.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The tool/function name (the AST-match ground truth).
  * **params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec)]) – Argument name -> [`FieldSpec`](ek.base.html.md#ek.base.FieldSpec). Each spec’s `importance`
    is the cost of getting *that argument* wrong, and its `domain` feeds
    validators – the same contract as an extracted field.
  * **importance** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Cost weight of the *call itself* (a missing or spurious call).
    Defaults to [`DESTRUCTIVE_WEIGHT`](#ek.agents.base.DESTRUCTIVE_WEIGHT) when `destructive` is set, else 1.0.
  * **destructive** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether calling this tool mutates the world irreversibly (a refund,
    a delete, a send). Drives the default weight and the safety validators.

#### to_node_type()

Render this tool as a Layer-A [`NodeType`](ek.base.html.md#ek.base.NodeType) (args as fields).

* **Return type:**
  [`NodeType`](ek.base.html.md#ek.base.NodeType)

#### *property* weight *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The call-level cost weight (explicit, else destructive-aware default).

### *class* ek.agents.base.Trajectory(steps=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The ordered sequence of [`Step`](#ek.agents.base.Step) s an agent took.

A trajectory is *linear* – which is why it is scored with a sequence edit distance,
not a graph edit distance (see [`ek.agents.trajectory`](ek.agents.trajectory.html.md#module-ek.agents.trajectory)).

### Example

```pycon
>>> t = Trajectory([Step("search", {"q": "cat"}), Step("answer", {"a": "meow"})])
>>> len(t), t.tools
(2, ('search', 'answer'))
```

#### *property* calls *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The `(tool, args)` pairs in call order.

#### *property* tools *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

The tool names in call order.

### ek.agents.base.as_call(x)

Coerce a [`Step`](#ek.agents.base.Step), `(tool, args)` tuple, or `{"tool":…, "args":…}` dict.

Gold tool calls are usually written as bare pairs or dicts (no observation), while a
prediction arrives as a [`Step`](#ek.agents.base.Step); both must compare. Progressive disclosure: the
simple literal works, the rich object works.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> as_call(Step("search", {"q": "cat"}))
('search', {'q': 'cat'})
>>> as_call(("search", {"q": "cat"}))
('search', {'q': 'cat'})
>>> as_call({"tool": "search", "args": {"q": "cat"}})
('search', {'q': 'cat'})
>>> as_call("search")
('search', {})
```

### ek.agents.base.is_episode(x)

Duck-typed episode check (core never imports this module – it checks the *shape*).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ek.agents.base.suite_grammar(tasks)

The union grammar over a whole task suite (later tools win on name collision).

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### ek.agents.base.tool_grammar(\*tools)

Build the Layer-A [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar) from [`ToolSpec`](#ek.agents.base.ToolSpec) s.

The resulting grammar is the frozen cost SSOT every agent metric reads: node cost =
the tool’s call weight, field cost = the argument’s importance.

* **Return type:**
  [`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)

### Example

```pycon
>>> from ek.base import FieldSpec
>>> g = tool_grammar(ToolSpec("send", params={"to": FieldSpec("to")},
...                           destructive=True))
>>> g.node_cost("send")
10.0
```
