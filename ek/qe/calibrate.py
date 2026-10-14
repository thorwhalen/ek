"""Calibration: make a raw score *mean* a probability (the non-optional stage).

A confidence of 0.9 should mean "correct 90% of the time." Modern models violate
this badly -- they are systematically **overconfident** (Guo et al. 2017), and RLHF
makes LLMs worse. So **calibration is non-optional: never gate a raw posterior or a
raw logprob** (``misc/docs/ek_03`` §2, Hard Rule 1). A :class:`~ek.base.Calibrator`
is fit on a labelled holdout of ``(raw_score, field_correct?)`` pairs and maps any
later raw score to a calibrated probability.

Three methods, by what input you have:

- :class:`PlattCalibrator` -- logistic fit on any scalar score (no logits needed):
  **the default** for aggregated OCR confidence or aggregated logprobs.
- :class:`IsotonicCalibrator` -- non-parametric monotonic fit; more flexible, needs
  more data, can overfit small sets.
- :class:`TemperatureCalibrator` -- a single scalar ``T`` on **logits**; use only
  when you have logits (it does not change the argmax).

All three are pure-Python (stdlib only) so the calibration stage works with zero
extra dependencies; :func:`sklearn_calibrator` / :func:`netcal_calibrator` offer the
library-backed equivalents behind the ``ek[calibration]`` extra. Measure calibration
with :func:`expected_calibration_error` (+ a reliability curve). For per-field-type
validity, wrap per group with :class:`GroupCalibrator` (Mondrian / class-conditional)
-- distribution-free *conditional* coverage is otherwise impossible (Hard Rule 2).
When the raw signals are *heterogeneous* (a judge score next to a logprob next to a
ROVER agreement), calibrate each signal family on its own with
:class:`SignalFamilyCalibrator` and fuse the calibrated probabilities with a
documented combiner (:func:`mean_probability` by default, :func:`min_probability`
for a conservative gate), rather than calibrating the mean of incomparable scores.

Calibrate at the **granularity of the decision** (gate on fields -> calibrate a
"field-correct?" target), and persist the fit (:func:`save_calibrator`); calibration
is dataset-specific and decays, so re-fit on drift.

Example:
    >>> # An overconfident raw signal, calibrated against ground truth.
    >>> raw =     [0.95, 0.93, 0.92, 0.90, 0.55, 0.52, 0.51, 0.50]
    >>> correct = [True, True, False, False, True, False, False, False]
    >>> cal = PlattCalibrator().fit(raw, correct)
    >>> cal(0.95) < 0.95            # overconfidence pulled down
    True
    >>> 0.0 <= cal(0.5) <= 1.0
    True
"""

from __future__ import annotations

import bisect
import math
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..registry import register, requires_extra
from ..stores import json_store

#: Default number of bins for :func:`expected_calibration_error` / reliability curves.
DEFAULT_N_BINS = 10


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _clip01(p: float) -> float:
    if math.isnan(p):
        raise ValueError(
            "calibrated probability is NaN (was the raw score non-finite?)"
        )
    return 0.0 if p < 0.0 else 1.0 if p > 1.0 else p


def _finite_pairs(scores: Sequence[float], correct: Sequence[bool]) -> list:
    """``(float(score), correct)`` pairs, dropping rows whose score is non-finite.

    A single ``NaN``/``inf`` raw score would otherwise corrupt a fit silently (Platt
    diverges, isotonic gets a junk knot); drop them rather than poison the model.
    """
    out = []
    for s, c in zip(scores, correct):
        s = float(s)
        if math.isfinite(s):
            out.append((s, c))
    return out


# ---------------------------------------------------------------------------
# Platt scaling (logistic on any scalar score) -- the default
# ---------------------------------------------------------------------------


@dataclass
class PlattCalibrator:
    """Platt scaling: ``sigmoid(a * score + b)``, fit by Newton/IRLS on labels.

    The default calibrator: works on any scalar (aggregated confidence/logprob), no
    logits required. Uses Platt's target smoothing so it does not overfit small
    calibration sets.

    Args:
        max_iter: Newton iterations (converges in a handful for a 2-parameter model).
    """

    a: float = 1.0
    b: float = 0.0
    max_iter: int = 100
    kind: str = "platt"

    def fit(
        self, scores: Sequence[float], correct: Sequence[bool]
    ) -> "PlattCalibrator":
        """Fit ``a, b`` to maximise the likelihood of ``correct`` given ``scores``."""
        pairs = _finite_pairs(scores, correct)
        xs = [x for x, _ in pairs]
        cs = [c for _, c in pairs]
        n_pos = sum(1 for c in cs if c)
        n_neg = len(cs) - n_pos
        # Platt target smoothing (avoids 0/1 targets driving |a| to infinity).
        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        ts = [hi if c else lo for c in cs]
        a, b = 0.0, math.log((n_pos + 1.0) / (n_neg + 1.0)) if n_neg else 0.0
        for _ in range(self.max_iter):
            g0 = g1 = h00 = h01 = h11 = 0.0
            for x, t in zip(xs, ts):
                p = _sigmoid(a * x + b)
                d = p - t
                w = max(p * (1.0 - p), 1e-12)
                g0 += d * x
                g1 += d
                h00 += w * x * x
                h01 += w * x
                h11 += w
            h00 += 1e-10
            h11 += 1e-10
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-18:
                break
            da = (h11 * g0 - h01 * g1) / det
            db = (h00 * g1 - h01 * g0) / det
            a -= da
            b -= db
            if abs(da) < 1e-9 and abs(db) < 1e-9:
                break
        self.a, self.b = a, b
        return self

    def __call__(self, raw_score: float) -> float:
        return _clip01(_sigmoid(self.a * float(raw_score) + self.b))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        return cls(a=d["a"], b=d["b"])


# ---------------------------------------------------------------------------
# Isotonic regression (non-parametric monotonic) via pool-adjacent-violators
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Isotonic (monotonic non-decreasing) calibration via pool-adjacent-violators.

    More flexible than Platt; needs more calibration data and can overfit small
    sets. Predicts by linear interpolation between fitted points, clipped at the
    ends.
    """

    x: List[float] = field(default_factory=list)  # sorted score breakpoints
    y: List[float] = field(default_factory=list)  # calibrated probability at each
    kind: str = "isotonic"

    def fit(
        self, scores: Sequence[float], correct: Sequence[bool]
    ) -> "IsotonicCalibrator":
        """Fit the monotonic step function to ``(score, correct)`` pairs."""
        # Aggregate ALL observations at each distinct x into one (sum, weight) block
        # FIRST -- isotonic enforces monotonicity only across distinct x. Pooling
        # incrementally as samples arrive is wrong: a partial block can trigger a
        # spurious merge with its predecessor before all its observations are seen.
        agg: dict = {}
        for s, c in zip(scores, correct):
            xv = float(s)
            if not math.isfinite(xv):  # a NaN/inf score would corrupt the knots
                continue
            sw = agg.setdefault(xv, [0.0, 0.0])
            sw[0] += 1.0 if c else 0.0
            sw[1] += 1.0
        if not agg:
            self.x, self.y = [], []
            return self
        # Pool-adjacent-violators over distinct-x blocks, each tracking its member
        # x's so EVERY distinct x stays a knot -- this matches sklearn's linear
        # interpolation between unique-x points (not just the block right-edges).
        blocks: List[list] = []  # [sum, weight, [x, ...]]
        for xv in sorted(agg):
            s_sum, w = agg[xv]
            blocks.append([s_sum, w, [xv]])
            while (
                len(blocks) > 1
                and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
            ):
                s2, w2, xs2 = blocks.pop()
                blocks[-1][0] += s2
                blocks[-1][1] += w2
                blocks[-1][2].extend(xs2)
        self.x, self.y = [], []
        for s_sum, w, xlist in blocks:
            mean = s_sum / w
            for xv in xlist:
                self.x.append(xv)
                self.y.append(mean)
        return self

    def __call__(self, raw_score: float) -> float:
        if not self.x:
            return _clip01(float(raw_score))
        s = float(raw_score)
        if s <= self.x[0]:
            return _clip01(self.y[0])
        if s >= self.x[-1]:
            return _clip01(self.y[-1])
        i = bisect.bisect_right(self.x, s)
        x0, x1, y0, y1 = self.x[i - 1], self.x[i], self.y[i - 1], self.y[i]
        frac = (s - x0) / (x1 - x0) if x1 > x0 else 0.0
        return _clip01(y0 + frac * (y1 - y0))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        return cls(x=list(d["x"]), y=list(d["y"]))


# ---------------------------------------------------------------------------
# Temperature scaling (single scalar on logits)
# ---------------------------------------------------------------------------


@dataclass
class TemperatureCalibrator:
    """Temperature scaling: ``sigmoid(logit / T)`` with one ``T`` fit on a holdout.

    Use **only when you have logits**: ``__call__`` expects a *logit*, not a
    probability. ``T > 1`` softens overconfidence; the argmax is unchanged.
    """

    T: float = 1.0
    kind: str = "temperature"
    #: Golden-section search bounds and iteration count (config, not magic numbers).
    t_min: float = 0.05
    t_max: float = 10.0
    max_iter: int = 60

    def fit(
        self, logits: Sequence[float], correct: Sequence[bool]
    ) -> "TemperatureCalibrator":
        """Fit ``T`` by minimising NLL with a bounded 1-D search."""
        pairs = _finite_pairs(logits, correct)
        zs = [z for z, _ in pairs]
        ys = [1.0 if c else 0.0 for _, c in pairs]

        def nll(t: float) -> float:
            total = 0.0
            for z, y in zip(zs, ys):
                p = min(max(_sigmoid(z / t), 1e-12), 1 - 1e-12)
                total -= y * math.log(p) + (1 - y) * math.log(1 - p)
            return total

        lo, hi = self.t_min, self.t_max  # golden-section search over the temperature
        gr = (math.sqrt(5) - 1) / 2
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(self.max_iter):
            if nll(c) < nll(d):
                hi, d = d, c
                c = hi - gr * (hi - lo)
            else:
                lo, c = c, d
                d = lo + gr * (hi - lo)
        self.T = (lo + hi) / 2
        return self

    def __call__(self, logit: float) -> float:
        return _clip01(_sigmoid(float(logit) / self.T))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "T": self.T}

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureCalibrator":
        return cls(T=d["T"])


# ---------------------------------------------------------------------------
# Mondrian / class-conditional calibration (restores per-group validity)
# ---------------------------------------------------------------------------


@dataclass
class GroupCalibrator:
    """Per-group (Mondrian) calibration: one calibrator per ``NodeType``/``FieldSpec``.

    Distribution-free *conditional* (per-field-type) coverage is impossible
    in general (Barber et al. 2019); calibrating separately per group restores it
    approximately. ``fit`` takes a parallel ``groups`` sequence; ``__call__`` routes
    by group key, falling back to a pooled calibrator for unseen groups.

    Args:
        factory: Zero-arg callable producing a fresh per-group calibrator (default
            :class:`PlattCalibrator`).
    """

    factory: Callable[[], Any] = PlattCalibrator
    by_group: dict = field(default_factory=dict)
    pooled: Any = None
    kind: str = "group"

    def fit(
        self, scores: Sequence[float], correct: Sequence[bool], *, groups: Sequence[Any]
    ) -> "GroupCalibrator":
        """Fit one calibrator per distinct group key, plus a pooled fallback."""
        buckets: dict = {}
        for s, c, g in zip(scores, correct, groups):
            buckets.setdefault(g, ([], []))
            buckets[g][0].append(s)
            buckets[g][1].append(c)
        self.by_group = {
            g: self.factory().fit(ss, cc) for g, (ss, cc) in buckets.items()
        }
        self.pooled = self.factory().fit(list(scores), list(correct))
        return self

    def __call__(self, raw_score: float, *, group: Any = None) -> float:
        cal = self.by_group.get(group, self.pooled)
        return cal(raw_score) if cal is not None else _clip01(float(raw_score))

    def to_dict(self) -> dict:
        """Serialize each per-group calibrator (keyed by stringified group) plus the
        pooled fallback, so a fitted Mondrian calibrator round-trips like the others.
        Group keys are stringified (they are ``NodeType``/``FieldSpec`` names)."""
        return {
            "kind": self.kind,
            "by_group": {str(g): c.to_dict() for g, c in self.by_group.items()},
            "pooled": self.pooled.to_dict() if self.pooled is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GroupCalibrator":
        obj = cls()
        obj.by_group = {
            g: _calibrator_from_record(rec)
            for g, rec in (d.get("by_group") or {}).items()
        }
        pooled = d.get("pooled")
        obj.pooled = _calibrator_from_record(pooled) if pooled else None
        return obj


# ---------------------------------------------------------------------------
# Per-signal-family calibration (heterogeneous signals)
# ---------------------------------------------------------------------------


def mean_probability(probs: Mapping[str, float]) -> float:
    """Fuse calibrated per-family probabilities by their mean (the default combiner).

    Each family's calibrated value estimates the *same* event ("this field is
    correct"), so the mean is a defensible, assumption-light pooling rule.

    >>> mean_probability({"judge": 0.9, "logprob": 0.7})
    0.8
    """
    if not probs:
        raise ValueError("mean_probability: no calibrated probabilities to combine")
    return sum(probs.values()) / len(probs)


def min_probability(probs: Mapping[str, float]) -> float:
    """Fuse calibrated per-family probabilities by their minimum (conservative).

    Use when any one family flagging a problem should be enough to lower the gate.

    >>> min_probability({"judge": 0.9, "logprob": 0.7})
    0.7
    """
    if not probs:
        raise ValueError("min_probability: no calibrated probabilities to combine")
    return min(probs.values())


#: Registered combiners: ``name -> (Mapping[str, float] -> float)``. A learned stacker
#: is any callable with the same shape (pass it as ``combine=``).
_COMBINERS = {"mean": mean_probability, "min": min_probability}
for _k, _fn in _COMBINERS.items():
    register("combiners", _k, _fn)


def _apply_calibrator(cal: Any, raw: float, group: Any) -> float:
    """Apply ``cal`` to ``raw``, passing ``group`` only to a Mondrian-aware one."""
    if group is not None:
        try:
            accepts_group = "group" in inspect.signature(cal).parameters
        except (TypeError, ValueError):
            accepts_group = False
        if accepts_group:
            return cal(raw, group=group)
    return cal(raw)


@dataclass
class SignalFamilyCalibrator:
    """One calibrator per signal family, fused by a documented combination rule.

    Averaging heterogeneous raw signals (an LLM-judge score, a token logprob, a ROVER
    agreement) and calibrating the mean discards exactly what a calibrator needs:
    each family has its own scale, reliability and direction of miscalibration. This
    calibrates every signal *by name* with its own calibrator, then fuses the
    calibrated probabilities with ``combine``.

    Pass it (or simply a ``Mapping[str, Calibrator]``) as
    ``estimate_quality(..., calibrator=...)``.

    **The fused value must itself be calibrated before it is gated.** Fusing
    calibrated probabilities (by mean, min, ...) does not yield a calibrated
    probability: agreeing families make the mean under-confident, a weak family
    drags it toward the base rate. So :meth:`fit` also learns a second-stage
    ``fusion`` calibrator on ``combine``'s output, and ``__call__`` applies it.
    Without a ``fusion`` calibrator (e.g. a hand-built mapping),
    :func:`ek.estimate_quality` refuses to gate a value fused from two or more
    families unless told ``assume_calibrated=True`` (Hard Rule 1). A value from a
    single family is that family's calibrated probability and gates as is.

    A field's own upstream ``confidence`` is just one more family here, named
    ``"confidence"``: include it in the training rows under that name.

    Args:
        calibrators: ``signal name -> Calibrator``. A ``"*"`` entry is the fallback
            for any signal without its own entry. A signal with neither is refused
            (``ValueError``) rather than silently dropped or passed through raw
            (Hard Rule 1).
        combine: ``Mapping[str, float] -> float`` fusing the calibrated values;
            :func:`mean_probability` (default) or :func:`min_probability`, or any
            callable of that shape (e.g. a learned stacker).
        factory: Zero-arg callable producing a fresh calibrator, used by :meth:`fit`.
        fusion: Second-stage calibrator applied to ``combine``'s output (learned by
            :meth:`fit`); ``None`` leaves the fused value uncalibrated.

    Example:
        >>> rows = [{"judge": j, "logprob": l} for j, l in
        ...         [(0.9, -0.1), (0.8, -0.2), (0.85, -2.0), (0.3, -0.1), (0.2, -2.5), (0.7, -1.8)]]
        >>> correct = [True, True, False, True, False, False]
        >>> cal = SignalFamilyCalibrator().fit(rows, correct)
        >>> sorted(cal.calibrators)
        ['judge', 'logprob']
        >>> 0.0 <= cal({"judge": 0.9, "logprob": -0.1}) <= 1.0
        True
    """

    calibrators: dict = field(default_factory=dict)
    combine: Callable[[Mapping[str, float]], float] = mean_probability
    factory: Callable[[], Any] = PlattCalibrator
    fusion: Any = None
    kind: str = "signal_family"

    def fit(
        self, signals: Sequence[Mapping[str, float]], correct: Sequence[bool]
    ) -> "SignalFamilyCalibrator":
        """Fit one calibrator per signal name, then the ``fusion`` calibrator.

        Each family is fit over the rows where it appears with a finite value. A
        family with fewer than two such rows, or only one label class, cannot be
        calibrated and is refused (``ValueError``) rather than turned into a
        constant that would then dilute the fusion.
        """
        if len(signals) != len(correct):
            raise ValueError(
                f"fit: {len(signals)} signal rows but {len(correct)} labels"
            )
        if not signals:
            raise ValueError("fit: no labelled rows to fit on")
        buckets: dict = {}
        for row, c in zip(signals, correct):
            for name, raw in row.items():
                buckets.setdefault(name, [])
                buckets[name].append((raw, c))
        unfittable = []
        for name, pairs in buckets.items():
            finite = _finite_pairs([s for s, _ in pairs], [c for _, c in pairs])
            if len(finite) < 2 or len({bool(c) for _, c in finite}) < 2:
                unfittable.append(name)
        if unfittable:
            raise ValueError(
                f"fit: cannot calibrate signal(s) {sorted(unfittable)}: each needs at "
                "least two finite scores covering both outcomes (correct and not). "
                "Drop them or label more rows."
            )
        self.calibrators = {
            name: self.factory().fit([s for s, _ in pairs], [c for _, c in pairs])
            for name, pairs in buckets.items()
        }
        # The fusion stage only ever sees values fused from 2+ families (a single
        # family bypasses it), so it is learned on exactly those rows. Too few of
        # them leaves it unset, and estimate_quality then refuses to gate them.
        fused, labels = [], []
        for row, c in zip(signals, correct):
            finite = {n: v for n, v in row.items() if math.isfinite(float(v))}
            if len(finite) >= 2:
                fused.append(self.combine(self.calibrate_signals(finite)))
                labels.append(c)
        fittable = len(fused) >= 2 and len({bool(c) for c in labels}) == 2
        self.fusion = self.factory().fit(fused, labels) if fittable else None
        return self

    def _calibrator_for(self, name: str) -> Any:
        cal = self.calibrators.get(name, self.calibrators.get("*"))
        if cal is None:
            raise ValueError(
                f"SignalFamilyCalibrator: no calibrator for signal {name!r} (have "
                f"{sorted(self.calibrators)}). Add one for it, or a '*' fallback -- "
                "an uncalibrated signal must not be fused (Hard Rule 1)."
            )
        return cal

    def calibrate_signals(
        self, raw_signals: Mapping[str, float], *, group: Any = None
    ) -> dict:
        """``{name: calibrated probability}`` for each raw signal (before fusion)."""
        return {
            name: _apply_calibrator(self._calibrator_for(name), raw, group)
            for name, raw in raw_signals.items()
        }

    def fuse(self, calibrated: Mapping[str, float]) -> float:
        """Fuse per-family calibrated probabilities: ``combine``, then ``fusion``.

        A single family's value is returned as is: it is already calibrated, and
        ``fusion`` was learned on fused values.
        """
        if len(calibrated) == 1:
            return next(iter(calibrated.values()))
        fused = self.combine(calibrated)
        return self.fusion(fused) if self.fusion is not None else fused

    def __call__(self, raw_signals: Mapping[str, float], *, group: Any = None) -> float:
        """Calibrate each signal family, then :meth:`fuse` them."""
        return self.fuse(self.calibrate_signals(raw_signals, group=group))

    def to_dict(self) -> dict:
        """Serialize every family's calibrator and the (registered) combiner's name."""
        names = [k for k, fn in _COMBINERS.items() if fn is self.combine]
        if not names:
            raise ValueError(
                "SignalFamilyCalibrator.to_dict: only a registered combiner "
                f"({sorted(_COMBINERS)}) can be persisted; got {self.combine!r}"
            )
        return {
            "kind": self.kind,
            "calibrators": {n: c.to_dict() for n, c in self.calibrators.items()},
            "combine": names[0],
            "fusion": self.fusion.to_dict() if self.fusion is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignalFamilyCalibrator":
        combine = d.get("combine", "mean")
        if combine not in _COMBINERS:
            raise ValueError(
                f"unknown combiner {combine!r}; known: {sorted(_COMBINERS)}"
            )
        return cls(
            calibrators={
                n: _calibrator_from_record(rec)
                for n, rec in (d.get("calibrators") or {}).items()
            },
            combine=_COMBINERS[combine],
            fusion=_calibrator_from_record(d["fusion"]) if d.get("fusion") else None,
        )


# ---------------------------------------------------------------------------
# Calibration measurement
# ---------------------------------------------------------------------------


def _binned(probs: Sequence[float], correct: Sequence[bool], n_bins: int):
    """Bin ``(prob, correct)`` into ``n_bins`` equal-width bins of ``[0, 1]``.

    Validates ``n_bins >= 1``; skips non-finite probs and clamps the rest into
    ``[0, 1]`` before binning. Returns ``(bins, hits, n)``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    bins: List[List[float]] = [[] for _ in range(n_bins)]
    hits: List[List[float]] = [[] for _ in range(n_bins)]
    n = 0
    for p, c in zip(probs, correct):
        p = float(p)
        if not math.isfinite(p):
            continue
        pc = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p
        idx = min(int(pc * n_bins), n_bins - 1)
        bins[idx].append(pc)
        hits[idx].append(1.0 if c else 0.0)
        n += 1
    return bins, hits, n


def expected_calibration_error(
    probs: Sequence[float], correct: Sequence[bool], *, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Expected Calibration Error: weighted mean gap between confidence and accuracy.

    Bins predictions by confidence into ``n_bins`` equal-width bins and averages
    ``|mean_confidence - accuracy|`` weighted by bin population. ``0`` is perfect.
    Non-finite probs are skipped; out-of-range probs are clamped into ``[0, 1]``.
    """
    bins, hits, n = _binned(probs, correct, n_bins)
    if n == 0:
        return 0.0
    ece = 0.0
    for b, h in zip(bins, hits):
        if b:
            conf = sum(b) / len(b)
            acc = sum(h) / len(h)
            ece += (len(b) / n) * abs(conf - acc)
    return ece


def reliability_curve(
    probs: Sequence[float], correct: Sequence[bool], *, n_bins: int = DEFAULT_N_BINS
) -> List[dict]:
    """Per-bin ``{confidence, accuracy, count}`` for a reliability diagram."""
    bins, hits, _ = _binned(probs, correct, n_bins)
    out = []
    for b, h in zip(bins, hits):
        if b:
            out.append(
                {
                    "confidence": sum(b) / len(b),
                    "accuracy": sum(h) / len(h),
                    "count": len(b),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Optional library-backed calibrators (ek[calibration]) -- opt-in
# ---------------------------------------------------------------------------


@requires_extra("calibration", packages=["sklearn"])
def sklearn_calibrator(method: str = "sigmoid"):
    """A calibrator backed by scikit-learn (``method='sigmoid'`` Platt or ``'isotonic'``).

    Behind ``ek[calibration]``. Returns an object satisfying the
    :class:`~ek.base.Calibrator` protocol that wraps sklearn's calibration. The
    pure-Python :class:`PlattCalibrator`/:class:`IsotonicCalibrator` are the
    dependency-free defaults; use this for parity with an sklearn-centric stack.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    @dataclass
    class _SklearnCalibrator:
        method: str
        model: Any = None

        def fit(self, scores, correct):
            xs = [[float(s)] for s in scores]
            ys = [1 if c else 0 for c in correct]
            if self.method == "isotonic":
                self.model = IsotonicRegression(out_of_bounds="clip")
                self.model.fit([s[0] for s in xs], ys)
            else:
                self.model = LogisticRegression()
                self.model.fit(xs, ys)
            return self

        def __call__(self, raw_score):
            if self.method == "isotonic":
                return _clip01(float(self.model.predict([float(raw_score)])[0]))
            return _clip01(float(self.model.predict_proba([[float(raw_score)]])[0][1]))

    return _SklearnCalibrator(method=method)


@requires_extra("calibration", packages=["netcal"])
def netcal_ece(
    probs: Sequence[float], correct: Sequence[bool], *, bins: int = DEFAULT_N_BINS
) -> float:
    """ECE via ``netcal`` (behind ``ek[calibration]``); D-ECE for localized outputs lives there too."""
    import numpy as np
    from netcal.metrics import ECE

    # netcal>=1.4 requires numpy arrays (a Python list raises); coerce explicitly.
    p = np.asarray([float(x) for x in probs], dtype=float)
    y = np.asarray([1 if c else 0 for c in correct], dtype=int)
    return float(ECE(bins).measure(p, y))


# ---------------------------------------------------------------------------
# Persistence (fitted calibrators are first-class assets; they decay -> re-fit)
# ---------------------------------------------------------------------------

_CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
    "group": GroupCalibrator,
    "signal_family": SignalFamilyCalibrator,
}
for _k, _cls in _CALIBRATORS.items():
    register("calibrators", _k, _cls)


def _calibrator_from_record(record: Any) -> Any:
    """Reconstruct a calibrator from a stored record, dispatched on its ``kind``."""
    if not isinstance(record, dict) or "kind" not in record:
        raise ValueError(f"calibrator record is malformed (no 'kind'): {record!r}")
    kind = record["kind"]
    if kind not in _CALIBRATORS:
        raise ValueError(
            f"unknown calibrator kind {kind!r}; known: {sorted(_CALIBRATORS)}"
        )
    return _CALIBRATORS[kind].from_dict(record)


def save_calibrator(
    calibrator: Any, name: str, *, rootdir: Optional[str] = None
) -> dict:
    """Persist a fitted calibrator's parameters to the ``calibrators`` store."""
    record = calibrator.to_dict()
    json_store("calibrators", rootdir=rootdir)[name] = record
    return record


def load_calibrator(name: str, *, rootdir: Optional[str] = None) -> Any:
    """Reconstruct a persisted calibrator by name (dispatched on its ``kind``).

    Validates the stored record so a malformed or unknown ``kind`` fails with an
    actionable error rather than a raw ``KeyError`` or a load-then-crash-later.
    """
    record = json_store("calibrators", rootdir=rootdir)[name]
    if not isinstance(record, dict) or "kind" not in record:
        raise ValueError(
            f"calibrator record {name!r} is malformed (no 'kind'): {record!r}"
        )
    kind = record["kind"]
    if kind not in _CALIBRATORS:
        raise ValueError(
            f"unknown calibrator kind {kind!r} in {name!r}; known: {sorted(_CALIBRATORS)}"
        )
    try:
        return _CALIBRATORS[kind].from_dict(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"calibrator record {name!r} is missing/invalid fields: {exc}"
        ) from exc
