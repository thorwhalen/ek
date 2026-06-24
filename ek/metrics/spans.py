"""Span / slot precision-recall-F1 over two backends that legitimately disagree.

Entity/slot F1 is the workhorse for fields-and-spans extraction, but a *single* F1
is meaningless without naming its **match scheme** -- and the two standard backends
diverge widely *and disagree with each other* (``misc/docs/ek_02`` §2.2):

- **seqeval** (MIT, ``chakki-works``) -- CoNLL ``conlleval``-compatible entity F1
  over BIO/IOBES tag *sequences*. Its ``conll`` (default) and ``strict`` (span+type,
  keyed to a tagging scheme) modes are the reference for sequence-labelling NER.
- **nervaluate** (MIT, ``MantisAI``) -- the SemEval-2013 Task 9.1 scheme over typed
  *spans*, with four schemas: **strict** (span + type), **exact** (span, any type),
  **partial** (boundary overlap, any type), **type** (type with span overlap). It
  counts COR/INC/PAR/MIS/SPU and gives partial spans half credit.

A model that finds "Electric" instead of "General Electric" scores F1=0 under
exact-span matching but ~0.667 under a partial/token scheme -- so the scheme is *not*
a default you can hide. This module makes :class:`MatchScheme` a **required keyword**
with no default: you must say which contract you are scoring under.

Both backends ship behind the ``[metrics]`` extra and are imported lazily, so
importing ``ek`` never pulls them in. Each :class:`~ek.base.Score` carries the raw
TP/FP/FN (or COR/INC/PAR/MIS/SPU) counts in its ``detail`` so
:meth:`SpanF1Metric.aggregate` micro-averages over a corpus -- never a naive mean of
per-document F1s.

Input shapes
------------
- ``seqeval`` schemes (:attr:`MatchScheme.SEQEVAL_CONLL`,
  :attr:`MatchScheme.SEQEVAL_STRICT`): ``pred`` and ``gold`` are **tag sequences** --
  a list of BIO/IOBES tags (e.g. ``["B-PER", "I-PER", "O"]``).
- ``nervaluate`` schemes (:attr:`MatchScheme.STRICT`, ``EXACT``, ``PARTIAL``,
  ``TYPE``): ``pred`` and ``gold`` are **span lists** -- a list of
  ``{"label", "start", "end"}`` dicts.

Example:
    >>> from ek.metrics.spans import SpanF1Metric, MatchScheme
    >>> m = SpanF1Metric(scheme=MatchScheme.SEQEVAL_CONLL)
    >>> s = m(["B-PER", "I-PER", "O"], ["B-PER", "I-PER", "O"])
    >>> s.f1
    1.0
    >>> SpanF1Metric()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: SpanF1Metric requires an explicit scheme=MatchScheme.<...>
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Sequence

from ..base import GraphGrammar, Score
from ..registry import requires_extra


def _f1(precision: float, recall: float) -> float:
    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )


class MatchScheme(str, Enum):
    """The (backend, scheme) contract a span/slot F1 is computed under.

    The two backends legitimately disagree (seqeval ignores other-type tags at the
    tag level; nervaluate includes them), so the scheme is always explicit. The
    ``SEQEVAL_*`` members consume BIO/IOBES *tag sequences*; the others consume
    SemEval-2013 *span lists* via nervaluate.
    """

    #: seqeval, CoNLL ``conlleval``-compatible entity F1 (the default seqeval mode).
    SEQEVAL_CONLL = "seqeval_conll"
    #: seqeval strict mode: span boundaries AND type must match, keyed to a scheme.
    SEQEVAL_STRICT = "seqeval_strict"
    #: nervaluate SemEval-2013 strict: exact span boundaries AND correct type.
    STRICT = "strict"
    #: nervaluate exact: exact span boundaries, type ignored.
    EXACT = "exact"
    #: nervaluate partial: boundary *overlap* (half credit), type ignored.
    PARTIAL = "partial"
    #: nervaluate type: correct type with at least some span overlap.
    TYPE = "type"


_SEQEVAL_SCHEMES = (MatchScheme.SEQEVAL_CONLL, MatchScheme.SEQEVAL_STRICT)

# Tagging scheme name -> seqeval scheme class (resolved lazily inside the call).
_TAGGING_SCHEMES = ("IOB2", "IOBES", "BILOU", "IOB1", "IOE1", "IOE2")

# nervaluate's result keys differ from our enum values: its "type" scheme is keyed
# "ent_type" (using "type" raises KeyError on every TYPE-scheme call).
_NERV_KEY = {
    MatchScheme.STRICT: "strict",
    MatchScheme.EXACT: "exact",
    MatchScheme.PARTIAL: "partial",
    MatchScheme.TYPE: "ent_type",
}


class SpanF1Metric:
    """Entity/slot precision/recall/F1 under an **explicit** :class:`MatchScheme`.

    Args:
        scheme: REQUIRED. The (backend, scheme) contract -- there is deliberately no
            default, because seqeval and nervaluate disagree and a bare F1 is
            ambiguous (``misc/docs/ek_02`` §2.2). Pass a :class:`MatchScheme`.
        tagging_scheme: For :attr:`MatchScheme.SEQEVAL_STRICT` only: the BIO/IOBES
            tagging scheme name (e.g. ``"IOB2"``, ``"IOBES"``, ``"BILOU"``) that
            strict-mode entity decoding is keyed to. Defaults to ``"IOB2"``.

    Raises:
        TypeError: if ``scheme`` is omitted (the whole point of this metric).
    """

    name = "span_f1"

    def __init__(
        self, scheme: Optional[MatchScheme] = None, *, tagging_scheme: str = "IOB2"
    ):
        if scheme is None:
            raise TypeError(
                "SpanF1Metric requires an explicit scheme=MatchScheme.<...>; "
                "seqeval and nervaluate disagree, so a bare F1 is ambiguous. "
                f"Choose one of: {[s.name for s in MatchScheme]}."
            )
        self.scheme = MatchScheme(scheme)
        if tagging_scheme not in _TAGGING_SCHEMES:
            raise ValueError(
                f"tagging_scheme must be one of {_TAGGING_SCHEMES}, got "
                f"{tagging_scheme!r}"
            )
        self.tagging_scheme = tagging_scheme

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        if self.scheme in _SEQEVAL_SCHEMES:
            return self._seqeval(pred, gold)
        return self._nervaluate(pred, gold)

    @requires_extra("metrics", packages=["seqeval"])
    def _seqeval(self, pred: Sequence[str], gold: Sequence[str]) -> Score:
        # CONLL = lenient conlleval decoding; STRICT = scheme-validated entities
        # (a malformed sequence for the chosen tagging scheme yields no entity), so
        # the two genuinely differ -- which is the whole point of the strict mode.
        if self.scheme is MatchScheme.SEQEVAL_STRICT:
            from seqeval import scheme as _seqeval_scheme

            scheme_cls = getattr(_seqeval_scheme, self.tagging_scheme)
            gold_ents = self._strict_entities(gold, scheme_cls)
            pred_ents = self._strict_entities(pred, scheme_cls)
        else:
            from seqeval.metrics.sequence_labeling import get_entities

            gold_ents = set(get_entities(list(gold)))
            pred_ents = set(get_entities(list(pred)))
        tp = len(gold_ents & pred_ents)
        fp = len(pred_ents - gold_ents)
        fn = len(gold_ents - pred_ents)
        return self._score_from_counts(
            tp=tp, fp=fp, fn=fn, detail={"backend": "seqeval"}
        )

    @staticmethod
    def _strict_entities(tags: Sequence[str], scheme_cls) -> set:
        """Scheme-validated ``(tag, start, end)`` entities for one tag sequence.

        A sequence that does not conform to the chosen tagging scheme (e.g. an ``E-``
        prefix under IOB2) has **no** valid strict entities -- seqeval raises, so we
        treat it as empty rather than crashing a corpus evaluation on one malformed
        document (invalid predictions then earn nothing, as strict scoring intends).
        """
        from seqeval.scheme import Entities

        try:
            decoded = Entities([list(tags)], scheme_cls).entities[0]
        except ValueError:
            return set()
        return {(e.tag, e.start, e.end) for e in decoded}

    @requires_extra("metrics", packages=["nervaluate"])
    def _nervaluate(self, pred: Sequence, gold: Sequence) -> Score:
        gold, pred = list(gold), list(pred)
        # nervaluate is ill-defined / raises on empty inputs, so handle them here:
        # empty/empty is perfect; empty gold -> all preds spurious; empty pred -> all
        # gold missed.
        if not gold and not pred:
            return self._nervaluate_score(cor=0, inc=0, par=0, mis=0, spu=0, possible=0, actual=0)
        if not gold:
            return self._nervaluate_score(
                cor=0, inc=0, par=0, mis=0, spu=len(pred), possible=0, actual=len(pred)
            )
        if not pred:
            return self._nervaluate_score(
                cor=0, inc=0, par=0, mis=len(gold), spu=0, possible=len(gold), actual=0
            )

        from nervaluate import Evaluator

        tags = sorted({s["label"] for s in gold + pred})
        ev = Evaluator([gold], [pred], tags=tags).evaluate()
        result = ev[0] if isinstance(ev, tuple) else ev  # some versions return a tuple
        overall = result.get("overall", result)
        r = overall[_NERV_KEY[self.scheme]]
        # version-robust: nervaluate returns an EvaluationResult object (newer) or a
        # plain dict (older) of the COR/INC/PAR/MIS/SPU counts.
        get = (lambda k: r[k]) if isinstance(r, dict) else (lambda k: getattr(r, k))
        return self._nervaluate_score(
            cor=get("correct"), inc=get("incorrect"), par=get("partial"),
            mis=get("missed"), spu=get("spurious"),
            possible=get("possible"), actual=get("actual"),
        )

    def _nervaluate_score(
        self, *, cor, inc, par, mis, spu, possible, actual
    ) -> Score:
        # nervaluate counts: COR/INC/PAR/MIS/SPU, with PAR worth half credit.
        precision = (cor + 0.5 * par) / actual if actual else (1.0 if possible == 0 else 0.0)
        recall = (cor + 0.5 * par) / possible if possible else (1.0 if actual == 0 else 0.0)
        f1 = _f1(precision, recall)
        return Score(
            value=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            metric=f"span_f1[{self.scheme.value}]",
            detail={
                "backend": "nervaluate",
                "scheme": self.scheme.value,
                "correct": cor,
                "incorrect": inc,
                "partial": par,
                "missed": mis,
                "spurious": spu,
                "possible": possible,
                "actual": actual,
                "higher_is_better": True,
            },
        )

    def _score_from_counts(self, *, tp: int, fp: int, fn: int, detail: dict) -> Score:
        precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
        f1 = _f1(precision, recall)
        return Score(
            value=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            metric=f"span_f1[{self.scheme.value}]",
            detail={
                "scheme": self.scheme.value,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "higher_is_better": True,
                **detail,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Micro-averaged F1 over the corpus (sum the raw counts, then divide once).

        seqeval schemes carry TP/FP/FN; nervaluate schemes carry COR/INC/PAR/MIS/SPU
        (PAR worth half). Both micro-average by summing counts globally -- never a
        mean of per-document F1s. The two backends' counts are not commensurable, so
        a corpus mixing them is rejected rather than silently dropping one.
        """
        backends = {s.detail.get("backend") for s in scores if s.detail.get("backend")}
        if len(backends) > 1:
            raise ValueError(
                f"Cannot micro-aggregate mixed span-F1 backends {sorted(backends)}; "
                "their counts (TP/FP/FN vs COR/INC/PAR/MIS/SPU) are not commensurable. "
                "Aggregate each backend/scheme separately."
            )
        if any(s.detail.get("backend") == "nervaluate" for s in scores):
            cor = sum(s.detail.get("correct", 0) for s in scores)
            par = sum(s.detail.get("partial", 0) for s in scores)
            possible = sum(s.detail.get("possible", 0) for s in scores)
            actual = sum(s.detail.get("actual", 0) for s in scores)
            if possible == 0 and actual == 0:
                return 1.0
            precision = (cor + 0.5 * par) / actual if actual else 0.0
            recall = (cor + 0.5 * par) / possible if possible else 0.0
            return _f1(precision, recall)
        tp = sum(s.detail.get("tp", 0) for s in scores)
        fp = sum(s.detail.get("fp", 0) for s in scores)
        fn = sum(s.detail.get("fn", 0) for s in scores)
        if tp + fp + fn == 0:
            return 1.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return _f1(precision, recall)
