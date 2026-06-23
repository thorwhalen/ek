"""Foundational data structures and strategy protocols for ``ek``.

This module is the single source of truth (SSOT) for ``ek``'s *two-layer* data
model and for the small set of ``typing.Protocol`` strategy interfaces that every
swappable behaviour in the framework implements. It contains **data structures
and contracts only** -- no business logic, no heavy imports. Importing it is
cheap and pulls in nothing beyond the standard library.

The two layers
--------------
Evaluating an information-extraction (IE) output splits into two halves -- scoring
against a gold reference (*offline*) and estimating quality with no reference
(*online*) -- and both operate on the **same** object:

- **Layer A -- the schema** (:class:`GraphGrammar` and friends): a *frozen*
  description of the typed graph you extract, carrying both the **types** and the
  **importance/cost weights** per field/node/edge. It is the SSOT consulted by
  cost-sensitive metrics, by validators, and (eventually) by constrained decoders.
- **Layer B -- the extraction metadata** (:class:`AnnotatedExtraction` and
  friends): per-extraction runtime metadata -- provenance, raw signals,
  (calibrated) confidence, validator findings, and the accept/flag/block decision
  -- keyed by a :class:`NodePath` into the Layer-A schema, riding *alongside* the
  grammar without ever mutating it.

The strategy protocols
----------------------
:class:`Metric`, :class:`Validator`, :class:`Calibrator`, :class:`DecisionPolicy`
and :class:`Signal` are structural (duck-typed) contracts. Concrete
implementations are resolved from :mod:`ek.registry` by name and injected as
keyword-only arguments with smart defaults, so the simple path Just Works while
every layer stays replaceable (open-closed).

Example:
    >>> g = GraphGrammar(
    ...     node_types={"donation": NodeType(
    ...         "donation",
    ...         fields={
    ...             "amount": FieldSpec("amount", "number", importance=10.0),
    ...             "city": FieldSpec("city", "string", importance=1.0),
    ...         },
    ...     )},
    ... )
    >>> g.field_cost("donation", "amount")
    10.0
    >>> g.field_cost("donation", "city")
    1.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Small value enums (str-subclassed so they serialize transparently to JSON)
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    """Terminal action a selective-prediction gate emits for an extracted value."""

    ACCEPT = "accept"
    FLAG = "flag"
    BLOCK = "block"


class Severity(str, Enum):
    """Whether a :class:`Finding` merely flags or can also correct a value."""

    CORRECT = "correct"
    FLAG = "flag"


# ---------------------------------------------------------------------------
# Layer A -- the typed-graph schema (frozen SSOT, carries cost weights)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """A leaf attribute of a node: its type, validation domain and cost weight.

    Args:
        name: Attribute name.
        type: A free-form type tag (e.g. ``"string"``, ``"number"``, ``"date"``);
            ``ek`` does not impose a type system -- the tag drives metric/validator
            dispatch and is yours to define.
        importance: Per-field cost weight; the lever that makes "two extra digits
            on a donation amount" outweigh "a misspelled city". Defaults to ``1.0``.
        domain: Optional allowed values: an enum tuple, a ``(lo, hi)`` range, or a
            regex string -- consumed by validators, never by metrics.
        normalizer: Optional key naming a canonicalizer (resolved from the registry)
            to apply to this field before comparison.
    """

    name: str
    type: str = "string"
    importance: float = 1.0
    domain: tuple = ()
    normalizer: Optional[str] = None


@dataclass(frozen=True)
class NodeType:
    """A node kind in the typed graph (e.g. an "invoice" or a "line item")."""

    name: str
    fields: Mapping[str, FieldSpec] = field(default_factory=dict)
    importance: float = 1.0


@dataclass(frozen=True)
class EdgeType:
    """A directed relation kind between two :class:`NodeType` names."""

    name: str
    src: str
    dst: str
    importance: float = 1.0


@dataclass(frozen=True)
class TypeRef:
    """A reference into the schema, so an injected cost function can read weights.

    ``kind`` is one of ``"node"``, ``"edge"`` or ``"field"``; for a field, ``name``
    is the node-type name and ``field`` the field name.
    """

    kind: str
    name: str
    field: Optional[str] = None


@dataclass(frozen=True)
class GraphGrammar:
    """The frozen schema SSOT: typed nodes/edges plus their importance weights.

    Methods return the cost weight for a referenced type, defaulting to ``1.0`` for
    anything not declared, so a partial grammar is always usable.
    """

    node_types: Mapping[str, NodeType] = field(default_factory=dict)
    edge_types: Mapping[str, EdgeType] = field(default_factory=dict)

    def node_cost(self, name: str) -> float:
        """Importance weight of a node type (``1.0`` if undeclared)."""
        nt = self.node_types.get(name)
        return nt.importance if nt is not None else 1.0

    def edge_cost(self, name: str) -> float:
        """Importance weight of an edge type (``1.0`` if undeclared)."""
        et = self.edge_types.get(name)
        return et.importance if et is not None else 1.0

    def field_cost(self, node: str, field_name: str) -> float:
        """Importance weight of a field on a node type (``1.0`` if undeclared)."""
        nt = self.node_types.get(node)
        if nt is None:
            return 1.0
        fs = nt.fields.get(field_name)
        return fs.importance if fs is not None else 1.0

    def cost(self, ref: TypeRef) -> float:
        """Importance weight for any :class:`TypeRef` (the default ``CostWeight``)."""
        if ref.kind == "node":
            return self.node_cost(ref.name)
        if ref.kind == "edge":
            return self.edge_cost(ref.name)
        if ref.kind == "field":
            return self.field_cost(ref.name, ref.field or "")
        return 1.0


# ---------------------------------------------------------------------------
# Layer B -- per-extraction verification metadata (rides alongside Layer A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodePath:
    """Addresses a node (or one of its fields) in an extracted graph; keys Layer B."""

    node_id: str
    node_type: str
    field: Optional[str] = None


@dataclass(frozen=True)
class Finding:
    """One validator observation about a value: a flag, optionally a correction.

    A ``suggestion`` is present iff the producing layer can *correct* (only the
    deterministic canonicalizer and the gated LLM corrector freely do so). The
    original value is always retained elsewhere; a Finding never mutates.
    """

    field: str
    layer: str
    severity: Severity = Severity.FLAG
    message: str = ""
    suggestion: Optional[Any] = None


@dataclass
class Provenance:
    """Where an extracted value came from, for audit and click-to-source review.

    ``bbox`` is intentionally typed ``Any`` so ``ek`` core need not import any OCR
    package; the OCR instance stores normalized ``[0, 1]`` top-left-origin geometry.
    """

    engine: str = ""
    source_span: Optional[tuple] = None
    bbox: Any = None
    raw_transcripts: Sequence[str] = ()


@dataclass
class FieldEstimate:
    """One extracted value plus all of its reference-free verification metadata."""

    value: Any
    raw_signals: dict = field(default_factory=dict)
    confidence: Optional[float] = None
    findings: tuple = ()
    provenance: Optional[Provenance] = None
    decision: Optional[Decision] = None


@dataclass
class AnnotatedExtraction:
    """Layer B: the verification metadata, referencing a frozen Layer-A grammar.

    The grammar is held *by reference* and never mutated; ``estimates`` maps a
    :class:`NodePath` to the :class:`FieldEstimate` for that node/field.
    """

    grammar: GraphGrammar
    estimates: Mapping[NodePath, FieldEstimate] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result objects returned by the two facades
# ---------------------------------------------------------------------------


@dataclass
class Score:
    """The result of comparing one prediction against one reference.

    ``value`` is the headline number (higher is better unless a metric documents
    otherwise). The optional decomposition fields let structured metrics carry
    precision/recall/F1, an alignment, and raw counts (e.g. edit operations) so
    that corpus aggregation can be done *correctly* (e.g. global WER accumulation)
    rather than by averaging per-item scores. ``float(score)`` returns ``value``.
    """

    value: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    metric: str = ""
    detail: dict = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)


@dataclass
class Report:
    """Aggregate of many :class:`Score` s over a corpus, with optional per-slice cuts.

    ``aggregate`` is the corpus-level headline (computed by the metric's own
    aggregator -- e.g. globally accumulated WER -- not a naive mean of per-item
    values). ``per_slice`` maps a slice label to its own aggregate.
    """

    metric: str = ""
    aggregate: Optional[float] = None
    n: int = 0
    scores: list = field(default_factory=list)
    per_slice: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.aggregate) if self.aggregate is not None else float("nan")


@dataclass
class QualityReport:
    """The result of a reference-free quality estimate for one extraction."""

    calibrated_confidence: Optional[float] = None
    decision: Optional[Decision] = None
    findings: tuple = ()
    raw_signals: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy protocols (structural contracts; concrete impls live in submodules)
# ---------------------------------------------------------------------------


@runtime_checkable
class Metric(Protocol):
    """Compares one prediction to one gold reference, returning a :class:`Score`."""

    def __call__(self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None) -> Score:
        ...


@runtime_checkable
class Validator(Protocol):
    """Reference-free check on a value, yielding zero or more :class:`Finding` s."""

    def __call__(self, value: Any, *, spec: Optional[FieldSpec] = None) -> Iterable[Finding]:
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Maps a raw scalar signal to a probability after being ``fit`` on labels."""

    def fit(self, scores: Sequence[float], correct: Sequence[bool]) -> "Calibrator":
        ...

    def __call__(self, raw_score: float) -> float:
        ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """Turns a (calibrated) confidence into an accept/flag/block :class:`Decision`."""

    def __call__(self, confidence: float) -> Decision:
        ...


@runtime_checkable
class Signal(Protocol):
    """A reference-free quality signal: extractor output -> raw scalar score.

    The ``cost_tier`` attribute (lower is cheaper) lets an escalation policy run
    cheap signals first and pay for expensive ones only on residual uncertainty.
    """

    cost_tier: int

    def __call__(self, extractor_output: Any) -> float:
        ...


# ---------------------------------------------------------------------------
# Callable type aliases (functions-as-strategies)
# ---------------------------------------------------------------------------

#: Canonicalize a string before comparison (lowercase, fold, normalize, ...).
Normalizer = Callable[[str], str]

#: Read named raw signals off a :class:`FieldEstimate` (e.g. for calibration).
ConfidenceSource = Callable[[FieldEstimate], Mapping[str, float]]

#: Cost of an edit on a typed element; defaults to reading ``*.importance``.
CostWeight = Callable[[GraphGrammar, TypeRef], float]

#: Anything that turns an image (path/bytes/array) into an ``OcrResult``-shaped
#: object (with ``.text`` and ``.blocks``). Typed loosely so ``ek`` core never
#: imports an OCR package.
OcrBackend = Callable[[Any], Any]


def default_cost_weight(grammar: GraphGrammar, ref: TypeRef) -> float:
    """The default :data:`CostWeight`: read the importance weight off the schema."""
    return grammar.cost(ref)
