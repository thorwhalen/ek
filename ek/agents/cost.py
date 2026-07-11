"""The cost model and the price SSOT: dollars per successfully completed task.

The unit that matters for an agent is **cost per successfully completed task**, not cost
per token. Two facts force it (``misc/docs/ek_11``):

1. **Tokens are not dollars.** Input, output, cached-input and reasoning tokens are priced
   *asymmetrically* (output typically ~5x input; cached input bills at a fraction). A
   "tokens per success" figure is at best a rough proxy.
2. **Failure is not free.** Tokens spent on an episode that failed are pure waste, so the
   denominator must be *successes*, not attempts. This is **Cost-of-Pass**: the expected
   monetary cost of obtaining a *correct* solution, ``E[cost] / P(success)`` -- which
   diverges to infinity when the agent never succeeds (a model that cannot solve a task at
   any price is not cheap, it is unusable; a per-token metric that reports it as
   "$0.003/call" is actively misleading).

**No hardcoded prices.** Rates go stale monthly, so this module ships *no* built-in price
table: a :class:`ModelPrice` is either passed directly or resolved from an injected
catalog (:func:`load_prices` reads LiteLLM's MIT-licensed
``model_prices_and_context_window.json`` -- **the data file, never the SDK**, whose
``enterprise/`` subtree is a proprietary carve-out). Missing rates raise an *actionable*
error rather than silently guessing.

Example:
    >>> from ek.agents.base import Cost
    >>> price = ModelPrice(input=1e-6, output=5e-6)      # $1 / $5 per 1M tokens
    >>> c = Cost(input_tokens=1_000_000, output_tokens=200_000)
    >>> round(dollars(c, price), 4)
    2.0
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

from .base import Cost, Episode

#: Cached input is billed at a fraction of the normal input rate (~10% across the major
#: providers). Used only when a model's price entry declares no explicit cached rate.
DEFAULT_CACHE_DISCOUNT = 0.1

#: Batch/deferred APIs are ~50% off. A keyword multiplier, never assumed.
DEFAULT_BATCH_DISCOUNT = 0.5


class UnknownModelPrice(KeyError):
    """Raised when a model's rates are not in the injected price catalog."""


@dataclass(frozen=True)
class ModelPrice:
    """Per-token rates for one model, in **USD per token**.

    Args:
        input: USD per input (prompt) token.
        output: USD per output (completion) token.
        cached_input: USD per cached-input token; defaults to
            ``DEFAULT_CACHE_DISCOUNT * input`` when not declared.
        reasoning: USD per reasoning token; defaults to the ``output`` rate (reasoning
            tokens are billed as output by the major providers).
    """

    input: float = 0.0
    output: float = 0.0
    cached_input: Optional[float] = None
    reasoning: Optional[float] = None

    @property
    def cached_rate(self) -> float:
        """The cached-input rate (explicit, else the discounted input rate)."""
        if self.cached_input is not None:
            return self.cached_input
        return DEFAULT_CACHE_DISCOUNT * self.input

    @property
    def reasoning_rate(self) -> float:
        """The reasoning-token rate (explicit, else the output rate)."""
        return self.reasoning if self.reasoning is not None else self.output


def per_million(
    input: float, output: float, *, cached_input=None, reasoning=None
) -> ModelPrice:
    """Build a :class:`ModelPrice` from the human-facing **USD per 1M tokens** figures.

    Example:
        >>> p = per_million(3.0, 15.0)          # $3 in / $15 out per 1M tokens
        >>> round(p.input, 9), round(p.output, 9)
        (3e-06, 1.5e-05)
    """
    scale = 1e-6
    return ModelPrice(
        input=input * scale,
        output=output * scale,
        cached_input=None if cached_input is None else cached_input * scale,
        reasoning=None if reasoning is None else reasoning * scale,
    )


def dollars(
    cost: Cost,
    price: Union[ModelPrice, str],
    *,
    prices: Optional[Mapping[str, ModelPrice]] = None,
    batch: bool = False,
) -> float:
    """Monetary cost of one :class:`~ek.agents.base.Cost` tally, in USD.

    Args:
        cost: The token/retry tally.
        price: A :class:`ModelPrice`, or a model name to resolve against ``prices``.
        prices: The injected price catalog (required when ``price`` is a name).
        batch: Apply the batch-API discount (:data:`DEFAULT_BATCH_DISCOUNT`).

    Raises:
        UnknownModelPrice: if a model name is given with no rates for it.
    """
    if isinstance(price, str):
        price = price_of(price, prices=prices)
    total = (
        cost.input_tokens * price.input
        + cost.output_tokens * price.output
        + cost.cached_input_tokens * price.cached_rate
        + cost.reasoning_tokens * price.reasoning_rate
    )
    return total * (DEFAULT_BATCH_DISCOUNT if batch else 1.0)


def price_of(
    model: str, *, prices: Optional[Mapping[str, ModelPrice]] = None
) -> ModelPrice:
    """Resolve a model's rates from an injected catalog, or fail *actionably*.

    ``ek`` deliberately ships no built-in price table (rates go stale monthly and would be
    magic numbers). Supply one with ``prices=``, or load the LiteLLM catalog with
    :func:`load_prices`.
    """
    if prices and model in prices:
        return prices[model]
    raise UnknownModelPrice(
        f"No price for model {model!r}. ek ships no built-in price table (rates go stale). "
        "Pass prices={'<model>': ModelPrice(...)} (see ek.agents.per_million), or load a "
        "catalog with ek.agents.load_prices(<path to LiteLLM "
        "model_prices_and_context_window.json>)."
    )


def load_prices(source: Any) -> dict:
    """Parse a LiteLLM-format price catalog into a ``{model: ModelPrice}`` table.

    ``source`` is a path to (or an already-parsed mapping of) LiteLLM's MIT-licensed
    ``model_prices_and_context_window.json``. We read the **data file only** and never
    import the ``litellm`` SDK -- its ``enterprise/`` subtree is a proprietary carve-out,
    and the SDK drags a large transitive tree in just to look up a rate.

    Entries lacking token rates (embeddings, unpriced models) are skipped.

    Example:
        >>> table = load_prices({"m": {"input_cost_per_token": 1e-6,
        ...                            "output_cost_per_token": 2e-6}})
        >>> table["m"].output
        2e-06
    """
    if isinstance(source, Mapping):
        raw = source
    else:
        with open(source, "r", encoding="utf-8") as f:
            raw = json.load(f)

    table: dict = {}
    for model, entry in raw.items():
        if not isinstance(entry, Mapping):
            continue
        inp = entry.get("input_cost_per_token")
        out = entry.get("output_cost_per_token")
        if inp is None and out is None:
            continue  # not a priced completion model
        table[model] = ModelPrice(
            input=float(inp or 0.0),
            output=float(out or 0.0),
            cached_input=_opt_float(entry.get("cache_read_input_token_cost")),
            reasoning=_opt_float(entry.get("output_cost_per_reasoning_token")),
        )
    return table


def _opt_float(x: Any) -> Optional[float]:
    return None if x is None else float(x)


# ---------------------------------------------------------------------------
# Cost-of-Pass: the aggregation
# ---------------------------------------------------------------------------


def cost_of_pass(total_cost: float, n_success: int) -> float:
    """Expected monetary cost of one *successful* task: ``total_cost / n_success``.

    Returns ``inf`` when nothing succeeded -- infeasibility is the honest answer, not a
    cheap-looking zero (this divergence is the point of the metric).

    Example:
        >>> cost_of_pass(10.0, 4)
        2.5
        >>> cost_of_pass(10.0, 0)
        inf
    """
    if n_success <= 0:
        return math.inf
    return total_cost / n_success


def episode_dollars(
    episode: Episode,
    *,
    prices: Optional[Mapping[str, ModelPrice]] = None,
    price: Optional[ModelPrice] = None,
    batch: bool = False,
) -> float:
    """Monetary cost of one episode (0.0 if it carries no :class:`~ek.agents.base.Cost`)."""
    if episode.cost is None:
        return 0.0
    resolved = price
    if resolved is None:
        model = (episode.run.model if episode.run else "") or ""
        resolved = price_of(model, prices=prices)
    return dollars(episode.cost, resolved, batch=batch)


def cost_report(
    episodes: Iterable[Episode],
    *,
    prices: Optional[Mapping[str, ModelPrice]] = None,
    price: Optional[ModelPrice] = None,
) -> dict:
    """The quality x cost x latency triple, reported **together**.

    Scoring accuracy *without* cost lets an agent chase tiny gains with unbounded API calls, so
    a report that omits cost is not a report. Returns ``n``, ``n_success``, ``success_rate``,
    ``total_dollars``, ``cost_per_success`` (Cost-of-Pass), ``mean_latency_s`` and ``total_tokens``.

    With **no** ``price``/``prices``, dollar figures are ``None`` rather than a fabricated zero:
    ``ek`` will not invent rates, and a silent 0.0 would make a costly agent look free. Tokens
    and latency are still reported.

    Example:
        >>> from ek.agents.base import Cost, Episode
        >>> p = per_million(1.0, 1.0)
        >>> eps = [Episode(task_id="a", cost=Cost(input_tokens=1_000_000), success=True),
        ...        Episode(task_id="b", cost=Cost(input_tokens=1_000_000), success=False)]
        >>> r = cost_report(eps, price=p)
        >>> r["success_rate"], round(r["cost_per_success"], 2)
        (0.5, 2.0)
        >>> cost_report(eps)["cost_per_success"] is None      # no rates -> no dollars invented
        True
    """
    episodes = list(episodes)
    n = len(episodes)
    n_success = sum(1 for e in episodes if e.success)
    priced = price is not None or prices is not None
    total = (
        sum(episode_dollars(e, prices=prices, price=price) for e in episodes)
        if priced
        else None
    )
    latencies = [
        e.cost.latency_s
        for e in episodes
        if e.cost is not None and e.cost.latency_s is not None
    ]
    tokens = sum(e.cost.total_tokens for e in episodes if e.cost is not None)
    return {
        "n": n,
        "n_success": n_success,
        "success_rate": (n_success / n) if n else 0.0,
        "total_dollars": total,
        "cost_per_success": cost_of_pass(total, n_success) if priced else None,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "total_tokens": tokens,
        "priced": priced,
    }
