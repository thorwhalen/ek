"""ANLS / ANLS* (nested-JSON) similarity, wrapping ``anls_star``.

ANLS (Average Normalized Levenshtein Similarity) is the DocVQA-family metric for a
short answer against acceptable variants: a normalized edit similarity thresholded
at ~0.5 so minor OCR/spelling differences do not zero a correct answer. **ANLS\\***
(Peer et al., 2024, `arXiv 2402.03848`_) generalizes it to *arbitrarily nested*
JSON -- strings, tuples (best match), lists (Hungarian matching, penalizing
missing/hallucinated items), and dicts (key-value with penalties for missing or
hallucinated keys) -- by mapping both sides to a tree and comparing them. It is the
closest thing to an off-the-shelf nested-JSON IE metric (``misc/docs/ek_02`` §3.2).

This wraps the ``anls_star`` package (Apache-2.0; its only dependency is ``munkres``
for the Hungarian matching, and it carries its own pure-Python Levenshtein -- no GPL
``Levenshtein`` is pulled in). It ships behind the ``[metrics]`` extra and is
imported lazily, so importing ``ek`` never requires it.

Because ``anls_star`` already returns a ``[0, 1]`` higher-is-better similarity, the
wrapper is thin: it flips ``ek``'s ``(pred, gold)`` argument order to the library's
``(gt, pred)`` order, records the raw similarity in :attr:`~ek.base.Score.detail`,
and -- for a corpus -- :meth:`AnlsMetric.aggregate` returns the *mean* ANLS (the "A"
in ANLS: the score is already length-normalized per item, so the corpus statistic is
the average of per-item similarities, unlike CER/WER which accumulate edit counts).

.. _arXiv 2402.03848: https://arxiv.org/abs/2402.03848

Example:
    >>> from ek.metrics.anls import AnlsMetric
    >>> m = AnlsMetric()
    >>> round(m("Hello Wrld", "Hello World").value, 3)   # one-char OCR slip
    0.909
    >>> m({"name": "Acme", "city": "Paris"},
    ...   {"name": "Acme", "city": "Paris"}).value         # nested, exact
    1.0
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..base import GraphGrammar, Score
from ..registry import requires_extra


class AnlsMetric:
    """ANLS / ANLS* (nested-JSON) similarity as a :class:`~ek.base.Metric`.

    Handles a bare string (classic ANLS) or an arbitrarily nested
    dict/list/tuple/``None`` structure (ANLS*); the backend dispatches on type.

    Args:
        threshold: ANLS zeroes a per-leaf similarity below this value before
            averaging (the classic ANLS 0.5 cut tolerates minor OCR/spelling
            noise). ``anls_star`` exposes no parameter for this -- it reads a
            class constant ``ANLSTree.THRESHOLD`` (default ``0.5``) -- so the metric
            applies the threshold by setting that constant around the call and
            restoring it. **Caveat:** this mutates a process-global, so it is not
            thread-safe; concurrent ``AnlsMetric`` calls with different thresholds in
            the same process can interfere. Defaults to ``0.5``.
    """

    name = "anls"

    def __init__(self, *, threshold: float = 0.5):
        self.threshold = threshold

    @requires_extra("metrics", packages=["anls_star"])
    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        from anls_star import anls_score
        from anls_star import anls_star as _anls_star

        # anls_star has no threshold argument; it reads ANLSTree.THRESHOLD. Apply our
        # threshold by setting that class constant for the duration of the call, then
        # restore it (so the documented knob actually does something).
        _tree = _anls_star.ANLSTree
        _saved = _tree.THRESHOLD
        _tree.THRESHOLD = self.threshold
        try:
            # anls_star takes (ground_truth, prediction); ek's Metric is (pred, gold).
            value = float(
                anls_score(gold, pred, return_gt=False, return_key_scores=False)
            )
        finally:
            _tree.THRESHOLD = _saved
        return Score(
            value=value,
            metric="anls",
            detail={
                "anls": value,
                "threshold": self.threshold,
                "higher_is_better": True,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Corpus ANLS = mean of per-item ANLS (each is already length-normalized).

        Unlike CER/WER (which accumulate global edit counts) or field-F1 (micro-TP/
        FP/FN), ANLS is defined as the *average* normalized similarity over the
        evaluation set -- the "A" in ANLS -- so the corpus statistic is the plain
        mean of the per-item scores.
        """
        vals = [s.detail.get("anls", s.value) for s in scores]
        return (sum(vals) / len(vals)) if vals else float("nan")
