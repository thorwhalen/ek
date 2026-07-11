"""Agent-evaluation data shapes: the episode (Layer B) and the task/tool grammar (Layer A).

Evaluating an *agent* is the same two-layer problem ``ek`` already solves for
information extraction -- only the evaluated object changes. An extraction is a
static payload; an **episode** is a *process*: a sequence of tool calls and
observations ending in a final state. So:

- **Layer A -- the task/tool grammar.** :class:`ToolSpec` and :class:`TaskSpec` are
  *builders* over the existing :class:`~ek.base.GraphGrammar`: a tool becomes a
  :class:`~ek.base.NodeType` whose fields are its argument :class:`~ek.base.FieldSpec` s,
  and ``importance`` carries the **cost of getting that argument wrong**. A wrong
  argument to a *destructive* tool is not one unit of error. This reuses the Layer-A
  cost-weight SSOT rather than inventing a parallel schema.
- **Layer B -- the episode.** :class:`Episode` carries the :class:`Trajectory`, the
  :class:`Cost`, the outcome, and the :class:`RunProvenance` -- riding *alongside* the
  frozen grammar, never mutating it, exactly as ``AnnotatedExtraction`` does.

These shapes live here, **not** in :mod:`ek.base`: that module is the zero-dependency,
IE-schema-only SSOT, and the OCR instance sets the precedent that an instance's concrete
shapes stay out of core (core duck-types them). Nothing here is imported by ``ek`` core.

Example:
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

See ``misc/docs/ek_07`` (the map) and ``misc/docs/ek_12`` (the integration report).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Optional

from ..base import FieldSpec, GraphGrammar, NodeType, Provenance

#: Default value of a completed task (the task-level analog of ``FieldSpec.importance``).
DEFAULT_TASK_VALUE = 1.0

#: Multiplier applied to a destructive tool's importance when it is not set explicitly.
#: Calling a destructive tool wrongly is categorically worse than calling a read-only one;
#: a keyword-tunable default, never a magic number buried in a metric.
DESTRUCTIVE_WEIGHT = 10.0


# ---------------------------------------------------------------------------
# Layer A -- the task/tool grammar (builders over GraphGrammar)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool the agent may call: its argument schema and its error costs.

    Args:
        name: The tool/function name (the AST-match ground truth).
        params: Argument name -> :class:`~ek.base.FieldSpec`. Each spec's ``importance``
            is the cost of getting *that argument* wrong, and its ``domain`` feeds
            validators -- the same contract as an extracted field.
        importance: Cost weight of the *call itself* (a missing or spurious call).
            Defaults to :data:`DESTRUCTIVE_WEIGHT` when ``destructive`` is set, else 1.0.
        destructive: Whether calling this tool mutates the world irreversibly (a refund,
            a delete, a send). Drives the default weight and the safety validators.
    """

    name: str
    params: Mapping[str, FieldSpec] = field(default_factory=dict)
    importance: Optional[float] = None
    destructive: bool = False

    @property
    def weight(self) -> float:
        """The call-level cost weight (explicit, else destructive-aware default)."""
        if self.importance is not None:
            return self.importance
        return DESTRUCTIVE_WEIGHT if self.destructive else 1.0

    def to_node_type(self) -> NodeType:
        """Render this tool as a Layer-A :class:`~ek.base.NodeType` (args as fields)."""
        return NodeType(self.name, fields=dict(self.params), importance=self.weight)


def tool_grammar(*tools: ToolSpec) -> GraphGrammar:
    """Build the Layer-A :class:`~ek.base.GraphGrammar` from :class:`ToolSpec` s.

    The resulting grammar is the frozen cost SSOT every agent metric reads: node cost =
    the tool's call weight, field cost = the argument's importance.

    Example:
        >>> from ek.base import FieldSpec
        >>> g = tool_grammar(ToolSpec("send", params={"to": FieldSpec("to")},
        ...                           destructive=True))
        >>> g.node_cost("send")
        10.0
    """
    return GraphGrammar(node_types={t.name: t.to_node_type() for t in tools})


@dataclass(frozen=True)
class TaskSpec:
    """One task in a suite: its input, its oracle, its value, and its allowed tools.

    Args:
        task_id: Stable identifier (the grouping key for k-trial reliability).
        input: Whatever the agent under test is called with.
        gold: The reference the checker grades against (a goal state, an expected answer,
            a gold trajectory -- the checker decides how to read it).
        value: Task-level cost weight: how much a completed task is worth. The task-level
            analog of ``FieldSpec.importance``; feeds value-weighted cost accounting.
        tools: The tools this task permits (the Layer-A grammar for the task).
        slice: Optional slice label (domain, difficulty, language) -- per-slice cuts are
            mandatory, not optional (a low pass^k concentrated in one hard slice is a
            different problem from uniform flakiness).
    """

    task_id: str
    input: Any = None
    gold: Any = None
    value: float = DEFAULT_TASK_VALUE
    tools: Sequence[ToolSpec] = ()
    slice: Optional[str] = None

    def grammar(self) -> GraphGrammar:
        """The Layer-A grammar for this task's tool set."""
        return tool_grammar(*self.tools)


def suite_grammar(tasks: Sequence[TaskSpec]) -> GraphGrammar:
    """The union grammar over a whole task suite (later tools win on name collision)."""
    node_types: dict = {}
    for task in tasks:
        for tool in task.tools:
            node_types[tool.name] = tool.to_node_type()
    return GraphGrammar(node_types=node_types)


# ---------------------------------------------------------------------------
# Layer B -- the episode (trajectory + cost + outcome + provenance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One step of a trajectory: a tool call and what came back.

    ``error`` is set when the call itself failed (a bad argument, a tool exception) --
    error *recovery* across subsequent steps is itself an evaluable signal.
    """

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    observation: Any = None
    error: Optional[str] = None

    @property
    def call(self) -> tuple:
        """The ``(tool, args)`` pair -- what tool-call correctness actually compares."""
        return (self.tool, dict(self.args))


def as_call(x: Any) -> tuple:
    """Coerce a :class:`Step`, ``(tool, args)`` tuple, or ``{"tool":…, "args":…}`` dict.

    Gold tool calls are usually written as bare pairs or dicts (no observation), while a
    prediction arrives as a :class:`Step`; both must compare. Progressive disclosure: the
    simple literal works, the rich object works.

    Example:
        >>> as_call(Step("search", {"q": "cat"}))
        ('search', {'q': 'cat'})
        >>> as_call(("search", {"q": "cat"}))
        ('search', {'q': 'cat'})
        >>> as_call({"tool": "search", "args": {"q": "cat"}})
        ('search', {'q': 'cat'})
        >>> as_call("search")
        ('search', {})
    """
    if isinstance(x, Step):
        return x.call
    if isinstance(x, str):
        return (x, {})
    if isinstance(x, Mapping):
        name = x.get("tool", x.get("name", x.get("function", "")))
        args = x.get("args", x.get("arguments", x.get("parameters", {}))) or {}
        return (str(name), dict(args))
    if isinstance(x, Sequence) and len(x) >= 1:
        name = x[0]
        args = x[1] if len(x) > 1 and isinstance(x[1], Mapping) else {}
        return (str(name), dict(args))
    raise TypeError(
        f"Cannot read a tool call from {type(x).__name__}; pass a Step, a "
        "(tool, args) pair, or a {'tool':…, 'args':…} mapping."
    )


@dataclass(frozen=True)
class Trajectory:
    """The ordered sequence of :class:`Step` s an agent took.

    A trajectory is *linear* -- which is why it is scored with a sequence edit distance,
    not a graph edit distance (see :mod:`ek.agents.trajectory`).

    Example:
        >>> t = Trajectory([Step("search", {"q": "cat"}), Step("answer", {"a": "meow"})])
        >>> len(t), t.tools
        (2, ('search', 'answer'))
    """

    steps: Sequence[Step] = ()

    def __post_init__(self) -> None:
        # Freeze the caller's sequence: a frozen dataclass holding a caller-owned *list*
        # is not actually immutable (appending to that list grows this "frozen" trajectory).
        object.__setattr__(self, "steps", tuple(self.steps))

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, i):
        return self.steps[i]

    @property
    def tools(self) -> tuple:
        """The tool names in call order."""
        return tuple(s.tool for s in self.steps)

    @property
    def calls(self) -> tuple:
        """The ``(tool, args)`` pairs in call order."""
        return tuple(s.call for s in self.steps)


@dataclass(frozen=True)
class Cost:
    """What one episode consumed: tokens (by kind), retries, and wall-clock.

    Token *kinds* are tracked separately because they are **priced asymmetrically** --
    output typically costs ~5x input, cached input bills at a fraction, and reasoning
    tokens are billed. This is precisely why the objective is *dollars* per successful
    task and never a raw token count (``misc/docs/ek_11``). Pricing lives in
    :mod:`ek.agents.cost`; this dataclass is the provider-neutral tally.

    Example:
        >>> a = Cost(input_tokens=100, output_tokens=50)
        >>> b = Cost(input_tokens=10, output_tokens=5, retries=1)
        >>> c = a + b
        >>> c.input_tokens, c.output_tokens, c.retries
        (110, 55, 1)
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    retries: int = 0
    latency_s: Optional[float] = None

    def __add__(self, other: "Cost") -> "Cost":
        if not isinstance(other, Cost):
            return NotImplemented
        lat = (self.latency_s or 0.0) + (other.latency_s or 0.0)
        return Cost(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            retries=self.retries + other.retries,
            latency_s=lat
            if (self.latency_s is not None or other.latency_s is not None)
            else None,
        )

    def __radd__(self, other: Any) -> "Cost":
        # ``sum()`` starts from the int 0, so a bare ``__radd__ = __add__`` would return
        # NotImplemented and make ``sum(costs)`` raise.
        if other == 0 or other is None:
            return self
        return self.__add__(other)

    @property
    def total_tokens(self) -> int:
        """All billed tokens (a rough size proxy -- *not* a cost; see ``dollars()``)."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cached_input_tokens
            + self.reasoning_tokens
        )


@dataclass(frozen=True)
class RunProvenance:
    """The *hidden eval variables* that decide whether two runs are even comparable.

    Agent evaluation is stochastic, and several of its inputs are themselves models. In a
    tau-bench-style suite the **user simulator is an LLM agent**, so its model/prompt/version
    silently changes results run to run. Recording these is what lets the harness *refuse*
    to compare a baseline against a run that changed the simulator or the suite -- the same
    discipline ``ek``'s offline harness already applies when it refuses to compare across
    two different metrics (``misc/docs/ek_08``, ``misc/docs/ek_11``).

    Args:
        seed: RNG seed for the trial (``None`` = unseeded/nondeterministic).
        temperature: Sampling temperature of the agent under test.
        model: The agent's model id.
        simulator: The **user-simulator** id/version -- an eval variable, not a constant.
        suite_version: Version of the task suite (contamination and task fixes move scores).
        scaffold: The agent scaffold/harness id (the same model moves points on scaffold alone).
    """

    seed: Optional[int] = None
    temperature: Optional[float] = None
    model: str = ""
    simulator: str = ""
    suite_version: str = ""
    scaffold: str = ""

    def comparability_key(self) -> tuple:
        """The fields that must match for two runs to be comparable (seed excluded).

        Seed and temperature are *expected* to vary across trials; the simulator, the suite
        version and the scaffold are not -- changing them invalidates a baseline.
        """
        return (self.simulator, self.suite_version, self.scaffold)


@dataclass
class Episode:
    """Layer B: one agent run over one task -- the object agent evaluation scores.

    The direct analog of ``AnnotatedExtraction``: it rides alongside the frozen Layer-A
    grammar and carries the run's metadata. It is scored *offline* against gold
    (:func:`ek.score` / :func:`ek.evaluate`) and estimated *online* against a judge or a
    consensus (:func:`ek.estimate_quality`) -- the same object, both halves.

    Args:
        task_id: Which task this episode answers (the k-trial grouping key).
        trajectory: The tool calls taken.
        output: The agent's final user-facing answer.
        final_state: The end state of the world (the DB, the filesystem) -- what a
            state-based oracle actually grades. Success is a *final-state* check, not a
            surface-text check.
        cost: The :class:`Cost` consumed.
        success: Filled in by a checker; ``None`` until graded.
        run: The :class:`RunProvenance` (seed, model, simulator, suite version).
        provenance: Optional core :class:`~ek.base.Provenance` for click-to-source audit.
        meta: Anything else the bridge captured.
    """

    task_id: str = ""
    trajectory: Trajectory = field(default_factory=Trajectory)
    output: Any = None
    final_state: Any = None
    cost: Optional[Cost] = None
    success: Optional[bool] = None
    run: Optional[RunProvenance] = None
    provenance: Optional[Provenance] = None
    meta: dict = field(default_factory=dict)

    @property
    def calls(self) -> tuple:
        """The ``(tool, args)`` pairs taken, in order."""
        return self.trajectory.calls

    def graded(self, success: bool) -> "Episode":
        """A copy with ``success`` set (never mutates -- grading stays idempotent).

        ``meta`` is copied too: ``dataclasses.replace`` is a *shallow* copy, so without this
        the "copy" would share the original's mutable ``meta`` dict.
        """
        return replace(self, success=success, meta=dict(self.meta))


def is_episode(x: Any) -> bool:
    """Duck-typed episode check (core never imports this module -- it checks the *shape*)."""
    return isinstance(x, Episode) or (
        hasattr(x, "trajectory") and hasattr(x, "task_id") and hasattr(x, "cost")
    )
