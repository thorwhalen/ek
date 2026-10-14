# ek.agents.trajectory

Trajectory (process) evaluation: a cost-weighted **sequence** edit distance over steps.

Outcome grading tells you *whether* the agent arrived; trajectory evaluation tells you *how*.
This scores the ordered sequence of tool calls against a reference trajectory, with per-step
and per-argument costs read from the Layer-A grammar – so taking a needless detour through a
*destructive* tool costs more than a harmless one.

**Why not reuse the flagship** [`TypedGraphMetric`](ek.metrics.graphs.html.md#ek.metrics.graphs.TypedGraphMetric)? It is tempting
(a trajectory *is* a typed graph), but the graph-edit-distance engine is the wrong tool for a
**linear** object, on four counts:

1. **It ignores order.** `networkx.graph_edit_distance` is an isomorphism search; node
   substitution matches on type + fields and ignores node identity. Chain edges add
   substitutable mass; they do not impose an ordering constraint. Two identical calls at
   different positions become interchangeable – which is exactly the error a trajectory metric
   exists to catch.
2. **It is capped.** `DEFAULT_MAX_NODES = 60` raises on a 100-step episode.
3. **It is nondeterministic.** The NP-hard search runs under a wall-clock `timeout` and
   returns a best-so-far bound – a wall-clock-dependent score for what is a *linear* alignment.
4. **Its denominator is polluted** by the synthetic ordering edges.

So we reuse the **cost model** (the grammar’s importance weights) and not the **engine**: an
O(n\*m) Needleman-Wunsch alignment, which is order-honoring, exact, deterministic and unbounded
in length. (A genuine DAG mode – for trajectories carrying explicit step *dependencies* –
would justify the graph engine; `Step` does not model dependency edges yet, so it is
deliberately not offered rather than shipped as a mode that quietly ignores order.)

The four schemes **disagree with each other** – pick one deliberately, exactly as with the
partial-match schemes on the IE side (`misc/docs/ek_02`, `misc/docs/ek_10`).

### Example

```pycon
>>> from ek.agents.base import Step
>>> gold = [Step("search", {"q": "cat"}), Step("answer", {"a": "meow"})]
>>> TrajectoryMetric()(gold, gold).value                      # identical -> distance 0
0.0
>>> detour = [Step("search", {"q": "cat"}), Step("search", {"q": "cat"}),
...           Step("answer", {"a": "meow"})]
>>> round(TrajectoryMetric()(detour, gold).value, 3) > 0      # a needless extra step costs
True
>>> TrajectoryMetric(scheme="superset")(detour, gold).value   # but it IS a superset of gold
0.0
```

### Module Attributes

| [`SCHEMES`](#ek.agents.trajectory.SCHEMES)   | The trajectory-match schemes.   |
|------------------------------------------------------------|---------------------------------|

### Classes

| [`TrajectoryMetric`](#ek.agents.trajectory.TrajectoryMetric)([scheme, grammar, ...])   | Cost-weighted trajectory distance as a [`Metric`](ek.base.html.md#ek.base.Metric) (**lower is better**).   |
|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|

### ek.agents.trajectory.SCHEMES *= ('in_order', 'exact', 'any_order', 'superset')*

The trajectory-match schemes. They disagree; choose deliberately.

### *class* ek.agents.trajectory.TrajectoryMetric(scheme='in_order', , grammar=None, canonicalizer=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cost-weighted trajectory distance as a [`Metric`](ek.base.html.md#ek.base.Metric) (**lower is better**).

* **Parameters:**
  * **scheme** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – One of [`SCHEMES`](#ek.agents.trajectory.SCHEMES).
    `in_order` (default) – Needleman-Wunsch edit distance; order matters, gaps allowed.
    `exact` – 0 iff the call sequences are identical, else the full mass.
    `any_order` – multiset distance; order ignored (the calls, not the path).
    `superset` – 0 iff every gold call occurs in the prediction; extra steps are free
    (use when the reference lists *required* calls rather than the whole path).
  * **grammar** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`GraphGrammar`](ek.base.html.md#ek.base.GraphGrammar)]) – Layer-A grammar supplying step/argument cost weights (may be passed per call).
  * **canonicalizer** – `str -> str` applied to string arguments before comparison.

#### aggregate(scores)

Corpus distance = total raw distance / total maximum distance (never a mean).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
