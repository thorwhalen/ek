"""``ek`` -- a framework for building Knowledge Evaluation systems.

``ek`` evaluates the outputs of information-extraction systems. OCR is treated as
the noisiest *special case* of a general problem, so the core is source-agnostic
(PDF, DOCX, tables, DB responses, LLM extractors) and the OCR specifics live in
:mod:`ek.ocr`.

Two facades cover the two halves of evaluation, both operating on the same typed
schema:

- :func:`score` / :func:`evaluate` -- *reference-based* (offline): compare against a
  gold answer, one item or a whole corpus, the metric chosen by output type.
- :func:`estimate_quality` -- *reference-free* (online): gather signals, calibrate,
  validate, and decide accept/flag/block with no gold answer.

Everything swappable is a :mod:`typing.Protocol` strategy resolved from
:mod:`ek.registry` and injected as a keyword-only argument with a smart default --
so the simple path Just Works and every layer stays replaceable (open-closed).

Quickstart:
    >>> import ek
    >>> round(ek.score("hello wrld", "hello world").value, 3)   # CER by default
    0.091
    >>> ek.evaluate([("ct", "cat"), ("dg", "dog")], metric="cer").n
    2

The two-layer data model (:class:`GraphGrammar` for the schema + cost weights,
:class:`AnnotatedExtraction` for per-extraction verification metadata) is the SSOT
every component plugs into -- see :mod:`ek.base`. Persistence is local-file ``dol``
stores under ``~/.local/share/ek/`` -- see :mod:`ek.stores`.
"""

from __future__ import annotations

# Importing these registers built-in strategies (normalizers, metrics) by name.
from . import canonicalize as canonicalize
from . import metrics as metrics
from . import ocr as ocr
from .base import (
    AnnotatedExtraction,
    Calibrator,
    ConfidenceSource,
    CostWeight,
    Decision,
    DecisionPolicy,
    EdgeType,
    FieldEstimate,
    FieldSpec,
    Finding,
    GraphGrammar,
    Metric,
    NodePath,
    NodeType,
    Normalizer,
    OcrBackend,
    Provenance,
    QualityReport,
    Report,
    Score,
    Severity,
    Signal,
    TypeRef,
    Validator,
    default_cost_weight,
)
from .canonicalize import Canonicalizer, default_canonicalizer
from .facade import estimate_quality, evaluate, score
from .harness import (
    GateResult,
    cohen_kappa,
    evaluate_store,
    krippendorff_alpha,
    load_baseline,
    percent_agreement,
    regression_gate,
    save_baseline,
)
from .metrics import (
    FieldMetric,
    StringMetric,
    TypedEdge,
    TypedGraph,
    TypedGraphMetric,
    TypedNode,
)
from .registry import (
    check_requirements,
    get,
    names,
    register,
    requires_extra,
    resolve,
)
from .stores import app_folder, cache_this, json_store, mall, persistent_cache

__all__ = [
    # facades
    "score",
    "evaluate",
    "estimate_quality",
    # offline harness
    "evaluate_store",
    "save_baseline",
    "load_baseline",
    "regression_gate",
    "GateResult",
    "cohen_kappa",
    "percent_agreement",
    "krippendorff_alpha",
    # Layer A (schema SSOT)
    "GraphGrammar",
    "FieldSpec",
    "NodeType",
    "EdgeType",
    "TypeRef",
    # Layer B (extraction metadata)
    "AnnotatedExtraction",
    "FieldEstimate",
    "NodePath",
    "Provenance",
    "Finding",
    "Decision",
    "Severity",
    # results
    "Score",
    "Report",
    "QualityReport",
    # strategy protocols + aliases
    "Metric",
    "Validator",
    "Calibrator",
    "DecisionPolicy",
    "Signal",
    "Normalizer",
    "ConfidenceSource",
    "CostWeight",
    "OcrBackend",
    "default_cost_weight",
    # canonicalization
    "Canonicalizer",
    "default_canonicalizer",
    # metrics
    "StringMetric",
    "FieldMetric",
    "TypedGraphMetric",
    "TypedGraph",
    "TypedNode",
    "TypedEdge",
    # registry
    "register",
    "get",
    "resolve",
    "names",
    "requires_extra",
    "check_requirements",
    # persistence
    "json_store",
    "mall",
    "app_folder",
    "persistent_cache",
    "cache_this",
    # subpackages
    "ocr",
    "metrics",
    "canonicalize",
    "__version__",
]

# Version from installed metadata (pyproject is the SSOT; CI auto-bumps it).
from importlib.metadata import PackageNotFoundError as _PNFE, version as _version

try:
    __version__ = _version("ek")
except _PNFE:  # running from a source tree without install metadata
    __version__ = "0.0.0+source"
del _version, _PNFE
