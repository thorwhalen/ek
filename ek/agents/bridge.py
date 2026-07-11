"""One-way adapters: turn whatever your agent already emits into an ``ek`` :class:`Episode`.

The dependency direction is the ``ek -> ocracy`` rule restated, and it is a **hard rule**:
``ek -> inspect_ai / deepeval / ragas`` (via the ``ek[agents]`` extra), **never the reverse**.

Note what that buys: ``ek`` core depends only on the *shape* of what those tools emit, so the
adapters here **duck-type** and import nothing. You need the extra to *run* Inspect or DeepEval;
you do not need it to *score* what they produced -- exactly as ``ek`` evaluates any
``OcrResult``-shaped object without importing an OCR engine.

The workhorse is :func:`trajectory_from_messages`: provider-shaped chat transcripts (OpenAI- and
Anthropic-style ``tool_calls`` / ``tool_use`` blocks) are the universal wire format for a
tool-using agent, and parsing them needs no SDK at all.

Example:
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
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from .base import Cost, Episode, Step, TaskSpec, Trajectory


def _loads(args: Any) -> dict:
    """Tool arguments arrive as a JSON *string* on the wire; as a dict in-process."""
    if isinstance(args, Mapping):
        return dict(args)
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            return dict(parsed) if isinstance(parsed, Mapping) else {"_": parsed}
        except json.JSONDecodeError:
            # A malformed tool-call payload is itself an evaluable failure, not a crash.
            return {"_malformed": args}
    return {}


def _read_call(call: Any) -> Optional[tuple]:
    """Read ``(name, args)`` from ONE tool call in any of the shapes the ecosystem emits.

    There are three, and getting this wrong is a *silent* wrong answer (the call is dropped and
    the episode scores as though the agent never called anything):

    - **OpenAI wire / dict**: ``{"function": {"name": …, "arguments": "<json string>"}}`` -- the
      name is *nested* and the arguments are a JSON **string**.
    - **Inspect AI object**: ``ToolCall(function="get_weather", arguments={...})`` -- ``.function``
      IS the name (a plain string), and the arguments are already a dict.
    - **OpenAI SDK object**: ``call.function.name`` / ``call.function.arguments``.
    """
    if isinstance(call, Mapping):
        fn = call.get("function", call)
        if isinstance(fn, Mapping):  # nested: OpenAI wire format
            name = fn.get("name") or ""
            raw = fn.get("arguments", fn.get("args", fn.get("input", {})))
        else:  # flat: {"function": "get_weather", "arguments": {...}}
            name = fn if isinstance(fn, str) else call.get("name", "")
            raw = call.get("arguments", call.get("args", call.get("input", {})))
        return (str(name), _loads(raw)) if name else None

    # Object shapes (Inspect's ToolCall, the OpenAI SDK's objects).
    fn = getattr(call, "function", None)
    if fn is not None and not isinstance(fn, str):  # nested object: call.function.name
        name = getattr(fn, "name", "") or ""
        raw = getattr(fn, "arguments", getattr(fn, "args", {}))
    else:  # flat object: call.function is the name (Inspect)
        name = fn if isinstance(fn, str) else (getattr(call, "name", "") or "")
        raw = getattr(
            call, "arguments", getattr(call, "args", getattr(call, "input", {}))
        )
    return (str(name), _loads(raw)) if name else None


def _tool_use_block(block: Any) -> Optional[tuple]:
    """Read an Anthropic ``tool_use`` content block (dict- or object-shaped)."""
    kind = (
        block.get("type")
        if isinstance(block, Mapping)
        else getattr(block, "type", None)
    )
    if kind != "tool_use":
        return None
    if isinstance(block, Mapping):
        name, raw = block.get("name", ""), block.get("input", {})
    else:
        name, raw = getattr(block, "name", ""), getattr(block, "input", {})
    return (str(name), _loads(raw)) if name else None


def _tool_calls_of(message: Mapping) -> list:
    """Extract ``(name, args)`` pairs from an assistant message, in any provider's shape."""
    calls: list = []
    for call in message.get("tool_calls") or ():
        got = _read_call(call)
        if got:
            calls.append(got)
    # Anthropic: content is a list of blocks, one of which may be a tool_use.
    content = message.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for block in content:
            got = _tool_use_block(block)
            if got:
                calls.append(got)
    return calls


def _text_of(message: Mapping) -> str:
    """The plain text of a message (flattening content blocks, dict- or object-shaped)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)
    return ""


def trajectory_from_messages(messages: Sequence[Mapping]) -> Trajectory:
    """Parse a chat transcript into a :class:`~ek.agents.base.Trajectory`.

    Each assistant tool call becomes a :class:`~ek.agents.base.Step`, and the *next* tool/result
    message becomes that step's ``observation``. A tool message carrying an error marks the step
    as errored -- error *recovery* across later steps is itself an evaluable signal.
    """
    steps: list = []
    pending: list = []
    for message in messages:
        role = message.get("role", "")
        if role == "assistant":
            for name, args in _tool_calls_of(message):
                pending.append(Step(tool=name, args=args))
        elif role in ("tool", "function", "user") and pending:
            # Attach results to the outstanding calls, in order.
            observation = _text_of(message) or message.get("content")
            is_error = bool(message.get("is_error"))
            step = pending.pop(0)
            steps.append(
                Step(
                    tool=step.tool,
                    args=step.args,
                    observation=observation,
                    error=str(observation) if is_error else None,
                )
            )
    steps.extend(pending)  # calls that never got a result (the agent stopped)
    return Trajectory(steps=steps)


def cost_from_usage(usage: Any, *, latency_s: Optional[float] = None) -> Cost:
    """Build a :class:`~ek.agents.base.Cost` from a provider ``usage`` object or dict.

    Understands the OpenAI (``prompt_tokens`` / ``completion_tokens``) and Anthropic
    (``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens``) spellings.

    Example:
        >>> c = cost_from_usage({"prompt_tokens": 100, "completion_tokens": 20})
        >>> c.input_tokens, c.output_tokens
        (100, 20)
    """

    def get(*names, default=0):
        for name in names:
            if isinstance(usage, Mapping):
                if name in usage and usage[name] is not None:
                    return usage[name]
            else:
                got = getattr(usage, name, None)
                if got is not None:
                    return got
        return default

    return Cost(
        input_tokens=int(get("input_tokens", "prompt_tokens")),
        output_tokens=int(get("output_tokens", "completion_tokens")),
        cached_input_tokens=int(
            get("cache_read_input_tokens", "cached_tokens", "cache_read_tokens")
        ),
        reasoning_tokens=int(get("reasoning_tokens")),
        latency_s=latency_s,
    )


def episode_from_messages(
    messages: Sequence[Mapping],
    *,
    task_id: str = "",
    usage: Any = None,
    output: Any = None,
    final_state: Any = None,
    latency_s: Optional[float] = None,
) -> Episode:
    """Build a full :class:`~ek.agents.base.Episode` from a transcript (+ optional usage/state).

    ``output`` defaults to the last assistant text -- the agent's final answer.
    """
    trajectory = trajectory_from_messages(messages)
    if output is None:
        assistant_texts = [
            _text_of(m) for m in messages if m.get("role") == "assistant"
        ]
        output = next((t for t in reversed(assistant_texts) if t), None)
    return Episode(
        task_id=task_id,
        trajectory=trajectory,
        output=output,
        final_state=final_state,
        cost=cost_from_usage(usage, latency_s=latency_s) if usage is not None else None,
    )


def from_inspect_sample(sample: Any, *, task_id: str = "") -> Episode:
    """Adapt an Inspect AI ``EvalSample``-shaped object into an :class:`Episode` (duck-typed).

    Reads ``.messages``, ``.output`` and ``.id`` -- no ``inspect_ai`` import required, so scoring
    an Inspect log never drags the harness into ``ek``'s dependency closure.
    """
    messages = getattr(sample, "messages", None) or []
    raw = [m if isinstance(m, Mapping) else _message_dict(m) for m in messages]
    output = getattr(sample, "output", None)
    completion = getattr(output, "completion", None) if output is not None else None
    usage = getattr(output, "usage", None) if output is not None else None
    return episode_from_messages(
        raw,
        task_id=task_id or str(getattr(sample, "id", "") or ""),
        usage=usage,
        output=completion,
    )


def _message_dict(message: Any) -> dict:
    """Best-effort dict view of a message object (Inspect/DeepEval carry dataclass-ish shapes)."""
    return {
        "role": getattr(message, "role", ""),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def from_deepeval_test_case(case: Any, *, task_id: str = "") -> Episode:
    """Adapt a DeepEval ``LLMTestCase``-shaped object into an :class:`Episode` (duck-typed).

    Reads ``.actual_output`` and ``.tools_called``. Note DeepEval phones home by default
    (Confident-AI telemetry) -- disable it before use if that matters to you; ``ek`` never
    enables it.
    """
    tools = getattr(case, "tools_called", None) or []
    steps = [
        Step(
            tool=getattr(t, "name", t if isinstance(t, str) else ""),
            args=dict(getattr(t, "input_parameters", None) or {}),
        )
        for t in tools
    ]
    return Episode(
        task_id=task_id,
        trajectory=Trajectory(steps=steps),
        output=getattr(case, "actual_output", None),
    )


def as_agent(fn: Callable) -> Callable:
    """Wrap a plain ``input -> answer`` function into the harness's ``TaskSpec -> Episode`` shape.

    Progressive disclosure: the trivial agent should not have to learn the Episode type.

    Example:
        >>> from ek.agents.base import TaskSpec
        >>> agent = as_agent(str.upper)
        >>> agent(TaskSpec("t1", input="hi")).output
        'HI'
    """

    def run(task: TaskSpec) -> Episode:
        return Episode(task_id=task.task_id, output=fn(task.input))

    run.__name__ = getattr(fn, "__name__", "agent")
    return run
