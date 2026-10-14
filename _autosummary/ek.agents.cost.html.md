# ek.agents.cost

The cost model and the price SSOT: dollars per successfully completed task.

The unit that matters for an agent is **cost per successfully completed task**, not cost
per token. Two facts force it (`misc/docs/ek_11`):

1. **Tokens are not dollars.** Input, output, cached-input and reasoning tokens are priced
   *asymmetrically* (output typically ~5x input; cached input bills at a fraction). A
   “tokens per success” figure is at best a rough proxy.
2. **Failure is not free.** Tokens spent on an episode that failed are pure waste, so the
   denominator must be *successes*, not attempts. This is **Cost-of-Pass**: the expected
   monetary cost of obtaining a *correct* solution, `E[cost] / P(success)` – which
   diverges to infinity when the agent never succeeds (a model that cannot solve a task at
   any price is not cheap, it is unusable; a per-token metric that reports it as
   “$0.003/call” is actively misleading).

**No hardcoded prices.** Rates go stale monthly, so this module ships *no* built-in price
table: a [`ModelPrice`](#ek.agents.cost.ModelPrice) is either passed directly or resolved from an injected
catalog ([`load_prices()`](#ek.agents.cost.load_prices) reads LiteLLM’s MIT-licensed
`model_prices_and_context_window.json` – **the data file, never the SDK**, whose
`enterprise/` subtree is a proprietary carve-out). Missing rates raise an *actionable*
error rather than silently guessing.

### Example

```pycon
>>> from ek.agents.base import Cost
>>> price = ModelPrice(input=1e-6, output=5e-6)      # $1 / $5 per 1M tokens
>>> c = Cost(input_tokens=1_000_000, output_tokens=200_000)
>>> round(dollars(c, price), 4)
2.0
```

### Module Attributes

| [`DEFAULT_CACHE_DISCOUNT`](#ek.agents.cost.DEFAULT_CACHE_DISCOUNT)   | Cached input is billed at a fraction of the normal input rate (~10% across the major providers).   |
|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [`DEFAULT_BATCH_DISCOUNT`](#ek.agents.cost.DEFAULT_BATCH_DISCOUNT)   | Batch/deferred APIs are ~50% off.                                                                  |

### Functions

| [`cost_of_pass`](#ek.agents.cost.cost_of_pass)(total_cost, n_success)                | Expected monetary cost of one *successful* task: `total_cost / n_success`.                                                     |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [`cost_report`](#ek.agents.cost.cost_report)(episodes, \*[, prices, price])         | The quality x cost x latency triple, reported **together**.                                                                    |
| [`dollars`](#ek.agents.cost.dollars)(cost, price, \*[, prices, batch])          | Monetary cost of one [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) tally, in USD.                  |
| [`episode_dollars`](#ek.agents.cost.episode_dollars)(episode, \*[, prices, price, ...]) | Monetary cost of one episode (0.0 if it carries no [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)). |
| [`load_prices`](#ek.agents.cost.load_prices)(source)                                | Parse a LiteLLM-format price catalog into a `{model: ModelPrice}` table.                                                       |
| [`per_million`](#ek.agents.cost.per_million)(input, output, \*[, ...])              | Build a [`ModelPrice`](#ek.agents.cost.ModelPrice) from the human-facing **USD per 1M tokens** figures.       |
| [`price_of`](#ek.agents.cost.price_of)(model, \*[, prices])                      | Resolve a model's rates from an injected catalog, or fail *actionably*.                                                        |

### Classes

| [`ModelPrice`](#ek.agents.cost.ModelPrice)([input, output, cached_input, ...])   | Per-token rates for one model, in **USD per token**.   |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------|

### Exceptions

| [`UnknownModelPrice`](#ek.agents.cost.UnknownModelPrice)   | Raised when a model's rates are not in the injected price catalog.   |
|----------------------------------------------------------------------|----------------------------------------------------------------------|

### ek.agents.cost.DEFAULT_BATCH_DISCOUNT *= 0.5*

Batch/deferred APIs are ~50% off. A keyword multiplier, never assumed.

### ek.agents.cost.DEFAULT_CACHE_DISCOUNT *= 0.1*

Cached input is billed at a fraction of the normal input rate (~10% across the major
providers). Used only when a model’s price entry declares no explicit cached rate.

### *class* ek.agents.cost.ModelPrice(input=0.0, output=0.0, cached_input=None, reasoning=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-token rates for one model, in **USD per token**.

* **Parameters:**
  * **input** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – USD per input (prompt) token.
  * **output** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – USD per output (completion) token.
  * **cached_input** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – USD per cached-input token; defaults to
    `DEFAULT_CACHE_DISCOUNT * input` when not declared.
  * **reasoning** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – USD per reasoning token; defaults to the `output` rate (reasoning
    tokens are billed as output by the major providers).

#### *property* cached_rate *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The cached-input rate (explicit, else the discounted input rate).

#### *property* reasoning_rate *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The reasoning-token rate (explicit, else the output rate).

### *exception* ek.agents.cost.UnknownModelPrice

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

Raised when a model’s rates are not in the injected price catalog.

### ek.agents.cost.cost_of_pass(total_cost, n_success)

Expected monetary cost of one *successful* task: `total_cost / n_success`.

Returns `inf` when nothing succeeded – infeasibility is the honest answer, not a
cheap-looking zero (this divergence is the point of the metric).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Example

```pycon
>>> cost_of_pass(10.0, 4)
2.5
>>> cost_of_pass(10.0, 0)
inf
```

### ek.agents.cost.cost_report(episodes, , prices=None, price=None)

The quality x cost x latency triple, reported **together**.

Scoring accuracy *without* cost lets an agent chase tiny gains with unbounded API calls, so
a report that omits cost is not a report. Returns `n`, `n_success`, `success_rate`,
`total_dollars`, `cost_per_success` (Cost-of-Pass), `mean_latency_s` and `total_tokens`.

With **no** `price`/`prices`, dollar figures are `None` rather than a fabricated zero:
`ek` will not invent rates, and a silent 0.0 would make a costly agent look free. Tokens
and latency are still reported.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Example

```pycon
>>> from ek.agents.base import Cost, Episode
>>> p = per_million(1.0, 1.0)
>>> eps = [Episode(task_id="a", cost=Cost(input_tokens=1_000_000), success=True),
...        Episode(task_id="b", cost=Cost(input_tokens=1_000_000), success=False)]
>>> r = cost_report(eps, price=p)
>>> r["success_rate"], round(r["cost_per_success"], 2)
(0.5, 2.0)
>>> cost_report(eps)["cost_per_success"] is None      # no rates -> no dollars invented
True
```

### ek.agents.cost.dollars(cost, price, , prices=None, batch=False)

Monetary cost of one [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost) tally, in USD.

* **Parameters:**
  * **cost** ([`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)) – The token/retry tally.
  * **price** (`Union`[[`ModelPrice`](#ek.agents.cost.ModelPrice), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A [`ModelPrice`](#ek.agents.cost.ModelPrice), or a model name to resolve against `prices`.
  * **prices** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`ModelPrice`](#ek.agents.cost.ModelPrice)]]) – The injected price catalog (required when `price` is a name).
  * **batch** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Apply the batch-API discount ([`DEFAULT_BATCH_DISCOUNT`](#ek.agents.cost.DEFAULT_BATCH_DISCOUNT)).
* **Raises:**
  [**UnknownModelPrice**](#ek.agents.cost.UnknownModelPrice) – if a model name is given with no rates for it.
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.cost.episode_dollars(episode, , prices=None, price=None, batch=False)

Monetary cost of one episode (0.0 if it carries no [`Cost`](ek.agents.base.html.md#ek.agents.base.Cost)).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ek.agents.cost.load_prices(source)

Parse a LiteLLM-format price catalog into a `{model: ModelPrice}` table.

`source` is a path to (or an already-parsed mapping of) LiteLLM’s MIT-licensed
`model_prices_and_context_window.json`. We read the **data file only** and never
import the `litellm` SDK – its `enterprise/` subtree is a proprietary carve-out,
and the SDK drags a large transitive tree in just to look up a rate.

Entries lacking token rates (embeddings, unpriced models) are skipped.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Example

```pycon
>>> table = load_prices({"m": {"input_cost_per_token": 1e-6,
...                            "output_cost_per_token": 2e-6}})
>>> table["m"].output
2e-06
```

### ek.agents.cost.per_million(input, output, , cached_input=None, reasoning=None)

Build a [`ModelPrice`](#ek.agents.cost.ModelPrice) from the human-facing **USD per 1M tokens** figures.

* **Return type:**
  [`ModelPrice`](#ek.agents.cost.ModelPrice)

### Example

```pycon
>>> p = per_million(3.0, 15.0)          # $3 in / $15 out per 1M tokens
>>> round(p.input, 9), round(p.output, 9)
(3e-06, 1.5e-05)
```

### ek.agents.cost.price_of(model, , prices=None)

Resolve a model’s rates from an injected catalog, or fail *actionably*.

`ek` deliberately ships no built-in price table (rates go stale monthly and would be
magic numbers). Supply one with `prices=`, or load the LiteLLM catalog with
[`load_prices()`](#ek.agents.cost.load_prices).

* **Return type:**
  [`ModelPrice`](#ek.agents.cost.ModelPrice)
