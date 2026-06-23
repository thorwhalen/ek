"""ROVER: multi-engine consensus + per-position agreement (the online flagship).

NIST's **Recognizer Output Voting Error Reduction** (Fiscus, 1997) combines the
outputs of several recognizers/extractors when *no reference is available*. It runs
in two stages: an **alignment** stage that merges the hypotheses into a single word
transition network (WTN) via iterative dynamic-programming alignments, and a
**voting** stage that picks, per branch point, the best-scoring word -- by vote
frequency alone, or blended with confidence. There is no maintained, permissively
licensed Python ROVER to depend on, so ``ek`` builds it on the edit-distance
primitives it already ships (pure-Python here for null-safety and zero new deps).

Two outputs come out of one pass:

- a **consensus** transcription (often lower error than any single engine), and
- a **per-position agreement** score in ``[0, 1]`` -- the fraction of engines that
  voted for the winning token at each slot. Positions where engines *disagree* are
  exactly the positions to flag, so this doubles as a **reference-free confidence
  signal** (an :class:`AgreementSignal` for :func:`ek.estimate_quality`).

``ek`` depends only on the ``OcrResult`` *shape* (``.text`` / ``.blocks`` /
``.mean_confidence``), so ROVER fuses **any** ``image -> OcrResult`` callable -- or
plain strings, token lists, or ``(token, confidence)`` lists. The alignment is
``O(N * l * L * L')`` in the engine count and sequence lengths, hence designed for a
handful of engines (its historical regime). Alignment is incremental and therefore
*order-dependent* -- a documented property of ROVER, not a bug.

See ``misc/docs/ek_03`` (and ``ek_04`` for cross-source corroboration).

Example:
    >>> c = rover(["the cat sat", "the cat sit", "the bat sat"])
    >>> c.text
    'the cat sat'
    >>> [round(a, 3) for a in c.agreement]            # per consensus token
    [1.0, 0.667, 0.667]
    >>> round(c.mean_agreement, 3)
    0.778
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Tuple

# A slot entry is (token, confidence); a NULL (engine produced nothing here) is
# represented by a ``None`` token.
_Entry = Tuple[Optional[str], float]
_NULL: _Entry = (None, 0.0)
_SENTINEL = object()

#: Confidence assigned to a present token whose source reports no usable confidence
#: (e.g. a VLM/markdown OcrResult with ``confidence is None``).
DEFAULT_TOKEN_CONF = 1.0

#: Default blend of average confidence vs vote frequency in the slot score
#: (Fiscus's ``1 - alpha``); ``0`` is frequency-only, ``1`` is confidence-only.
DEFAULT_CONF_WEIGHT = 0.5

#: Default confidence credited to a NULL vote (an engine that emitted no token at a
#: slot) -- the lever for how readily a deletion wins.
DEFAULT_NULL_CONF = 0.7

#: Default per-hypothesis token cap for the O(N*l*L*L') aligner (DoS guard).
DEFAULT_MAX_TOKENS = 5000


def _as_units(hyp: Any, *, default_conf: float = DEFAULT_TOKEN_CONF) -> List[_Entry]:
    """Coerce one hypothesis to a list of ``(token, confidence)`` units.

    Accepts a plain string, an ``OcrResult``-shaped object (uses ``.text`` with a
    broadcast ``.mean_confidence``), a list of token strings, or a list of
    ``(token, confidence)`` pairs. Null-safe: a ``None`` text/confidence degrades to
    an empty unit list / the ``default_conf`` rather than crashing.
    """
    if hyp is None:
        return []
    if isinstance(hyp, str):
        return [(tok, default_conf) for tok in hyp.split()]
    # OcrResult-shaped: prefer per-block units when blocks are word-level (each
    # block's text is a single whitespace-free token), so each token carries its own
    # confidence; else fall back to splitting .text with a broadcast confidence.
    text = getattr(hyp, "text", _SENTINEL)
    if text is not _SENTINEL:
        units = _word_level_block_units(hyp, default_conf=default_conf)
        if units is not None:
            return units
        conf = getattr(hyp, "mean_confidence", None)
        broadcast = default_conf if conf is None else float(conf)
        words = (text or "").split()
        return [(w, broadcast) for w in words]
    # A bare sequence of tokens or (token, conf) pairs.
    units: List[_Entry] = []
    for item in hyp:
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and not isinstance(item, str)
        ):
            tok, conf = item
            units.append((str(tok), default_conf if conf is None else float(conf)))
        else:
            units.append((str(item), default_conf))
    return units


def _word_level_block_units(hyp: Any, *, default_conf: float) -> Optional[List[_Entry]]:
    """Per-block ``(token, confidence)`` units, but only if blocks are word-level.

    Returns ``None`` (caller falls back to ``.text``) unless every block carries a
    single whitespace-free token -- the case where per-block confidence is a genuine
    per-word signal rather than a line/paragraph score diluted across many words.
    """
    blocks = getattr(hyp, "blocks", None)
    if not blocks:
        return None
    units: List[_Entry] = []
    for b in blocks:
        bt = getattr(b, "text", None)
        if not bt or any(ch.isspace() for ch in bt):
            return None  # a line/paragraph block: fall back to .text splitting
        bc = getattr(b, "confidence", None)
        units.append((bt, default_conf if bc is None else float(bc)))
    return units or None


def _align(
    spine: Sequence[Optional[str]], hyp: Sequence[str]
) -> List[Tuple[str, int, int]]:
    """Needleman-Wunsch alignment of a network ``spine`` to a hypothesis token list.

    Returns operations in left-to-right order, each ``(op, s_idx, h_idx)`` where
    ``op`` is ``"match"`` (tokens equal or substituted onto the same slot),
    ``"del"`` (a spine slot the hypothesis skips -> NULL), or ``"ins"`` (a
    hypothesis token with no spine slot -> a new slot). Pure-Python so the engine
    runs in a bare environment and over arbitrary (non-string) tokens.
    """
    s, t = len(spine), len(hyp)
    # dp[i][j] = min edit cost aligning spine[:i] with hyp[:j].
    dp = [[0] * (t + 1) for _ in range(s + 1)]
    for i in range(1, s + 1):
        dp[i][0] = i
    for j in range(1, t + 1):
        dp[0][j] = j
    for i in range(1, s + 1):
        for j in range(1, t + 1):
            sub = dp[i - 1][j - 1] + (0 if spine[i - 1] == hyp[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    ops: List[Tuple[str, int, int]] = []
    i, j = s, t
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if spine[i - 1] == hyp[j - 1] else 1)
        ):
            ops.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))  # spine slot, hypothesis has no token
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))  # hypothesis token, no spine slot
            j -= 1
    ops.reverse()
    return ops


@dataclass
class RoverSlot:
    """One branch point of the transition network: each engine's token (or NULL)."""

    entries: List[_Entry] = field(default_factory=list)  # one per engine, in order
    winner: Optional[str] = None
    score: float = 0.0
    vote_share: float = 0.0  # fraction of all engines voting for the winner ([0, 1])


@dataclass
class RoverConsensus:
    """The result of a ROVER pass over N hypotheses."""

    tokens: List[str] = field(default_factory=list)  # winning non-NULL tokens, in order
    slots: List[RoverSlot] = field(default_factory=list)
    agreement: List[float] = field(
        default_factory=list
    )  # vote share per consensus token
    n_engines: int = 0

    @property
    def text(self) -> str:
        """The consensus transcription (space-joined winning tokens)."""
        return " ".join(self.tokens)

    @property
    def mean_agreement(self) -> float:
        """Mean per-position agreement over consensus tokens (``1.0`` if degenerate).

        A reference-free confidence in ``[0, 1]``: ``1.0`` means every engine agreed
        at every emitted position; lower means at least one engine dissented.
        """
        return sum(self.agreement) / len(self.agreement) if self.agreement else 1.0


def _vote(
    entries: Sequence[_Entry],
    *,
    n_engines: int,
    conf_weight: float,
    null_conf: float,
) -> Tuple[Optional[str], float, float]:
    """Score the candidates in one slot; return ``(winner, score, vote_share)``.

    ``score(w) = (1 - conf_weight) * (votes_w / N) + conf_weight * avg_conf_w`` --
    Fiscus's frequency/confidence blend. NULL is a candidate (engines that produced
    nothing here vote for it, at ``null_conf``). Ties prefer a real token over NULL,
    then higher vote count, then first appearance.
    """
    votes: dict = {}
    conf_sum: dict = {}
    order: dict = {}
    for idx, (tok, conf) in enumerate(entries):
        votes[tok] = votes.get(tok, 0) + 1
        conf_sum[tok] = conf_sum.get(tok, 0.0) + (null_conf if tok is None else conf)
        order.setdefault(tok, idx)

    # NULL is a legitimate winner, so the "unset" marker must be distinct from None.
    best_tok: Any = _SENTINEL
    best_key: Tuple = ()
    best_score = 0.0
    for tok, v in votes.items():
        avg_conf = conf_sum[tok] / v
        score = (1.0 - conf_weight) * (v / n_engines) + conf_weight * avg_conf
        # sort key: maximise score, prefer real over NULL, then votes, then earliest.
        key = (score, tok is not None, v, -order[tok])
        if best_tok is _SENTINEL or key > best_key:
            best_tok, best_key, best_score = tok, key, score

    winner_votes = votes[best_tok]
    return best_tok, best_score, winner_votes / n_engines


def rover(
    hypotheses: Iterable[Any],
    *,
    use_confidence: bool = True,
    conf_weight: float = DEFAULT_CONF_WEIGHT,
    null_conf: float = DEFAULT_NULL_CONF,
    tokenize: Optional[Callable[[Any], List[_Entry]]] = None,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
) -> RoverConsensus:
    """Align N hypotheses, vote per slot, and emit consensus + per-position agreement.

    Args:
        hypotheses: The recognizer/extractor outputs to fuse. Each may be a string,
            an ``OcrResult``-shaped object (``.text``/``.mean_confidence``), a list of
            token strings, or a list of ``(token, confidence)`` pairs.
        use_confidence: Blend confidence into the vote (``True``) or vote purely by
            frequency (``False``, which forces ``conf_weight`` to ``0``).
        conf_weight: Weight on average confidence vs vote frequency in the slot score
            (Fiscus's ``1 - alpha``); ``0`` is frequency-only, ``1`` is
            confidence-only. Ignored when ``use_confidence`` is ``False``.
        null_conf: Confidence credited to a NULL vote (an engine that produced no
            token at a slot) -- the lever for how readily a deletion wins.
        tokenize: Optional ``hypothesis -> [(token, confidence), ...]`` override; by
            default :func:`_as_units` handles strings/OcrResults/token lists.
        max_tokens: Reject any hypothesis longer than this (the aligner is
            ``O(N*l*L*L')``; an unbounded input is a quadratic time/memory DoS).
            ``None`` disables the guard.

    Returns:
        A :class:`RoverConsensus` with the consensus ``tokens``/``text``, the per-slot
        breakdown, and the per-consensus-token ``agreement`` usable as a raw signal.
    """
    to_units = tokenize or _as_units
    hyps = [to_units(h) for h in hypotheses]
    n = len(hyps)
    if n == 0:
        return RoverConsensus(n_engines=0)

    if max_tokens is not None:
        longest = max(len(h) for h in hyps)
        if longest > max_tokens:
            raise ValueError(
                f"ROVER hypothesis has {longest} tokens (> max_tokens={max_tokens}); "
                "the O(N*l*L*L') aligner is designed for a handful of short-to-medium "
                "sequences. Pass max_tokens=<larger> (or None) to override deliberately."
            )

    effective_conf_weight = conf_weight if use_confidence else 0.0

    # Incrementally merge each hypothesis into the network of slots. A slot is a list
    # of (token, conf) entries, one per engine processed so far (NULL where absent).
    slots: List[List[_Entry]] = [[unit] for unit in hyps[0]]
    for i in range(1, n):
        units = hyps[i]
        spine = [_slot_majority(slot) for slot in slots]
        ops = _align(spine, [tok for tok, _ in units])
        merged: List[List[_Entry]] = []
        for op, s_idx, h_idx in ops:
            if op == "match":
                slot = slots[s_idx]
                slot.append(units[h_idx])
                merged.append(slot)
            elif op == "del":
                slot = slots[s_idx]
                slot.append(_NULL)
                merged.append(slot)
            else:  # insertion: a brand-new slot, NULL for every prior engine
                merged.append([_NULL] * i + [units[h_idx]])
        slots = merged

    rover_slots: List[RoverSlot] = []
    tokens: List[str] = []
    agreement: List[float] = []
    for slot in slots:
        winner, score, share = _vote(
            slot, n_engines=n, conf_weight=effective_conf_weight, null_conf=null_conf
        )
        rover_slots.append(
            RoverSlot(entries=list(slot), winner=winner, score=score, vote_share=share)
        )
        if winner is not None:
            tokens.append(winner)
            agreement.append(share)

    return RoverConsensus(
        tokens=tokens, slots=rover_slots, agreement=agreement, n_engines=n
    )


def _slot_majority(slot: Sequence[_Entry]) -> Optional[str]:
    """The most common non-NULL token in a slot (alignment spine representative)."""
    counts: dict = {}
    order: dict = {}
    for idx, (tok, _) in enumerate(slot):
        if tok is None:
            continue
        counts[tok] = counts.get(tok, 0) + 1
        order.setdefault(tok, idx)
    if not counts:
        return None
    return max(counts, key=lambda tok: (counts[tok], -order[tok]))


@dataclass
class AgreementSignal:
    """ROVER multi-engine agreement as a reference-free :class:`~ek.base.Signal`.

    Cost tier 3 (``N`` engine runs): try the deterministic verifier layer and any
    free intrinsic confidence first. Called on a collection of hypotheses, it runs
    :func:`rover` and returns the mean per-position agreement as the raw signal --
    *uncalibrated*, like every signal, so it must pass through a
    :class:`~ek.base.Calibrator` before any gate reads it.

    Example:
        >>> sig = AgreementSignal()
        >>> round(sig(["the cat sat", "the cat sit", "the bat sat"]), 3)
        0.778
    """

    cost_tier: int = 3
    use_confidence: bool = True
    conf_weight: float = DEFAULT_CONF_WEIGHT
    null_conf: float = DEFAULT_NULL_CONF
    tokenize: Optional[Callable[[Any], List[_Entry]]] = None
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS

    def __call__(self, hypotheses: Iterable[Any]) -> float:
        return rover(
            hypotheses,
            use_confidence=self.use_confidence,
            conf_weight=self.conf_weight,
            null_conf=self.null_conf,
            tokenize=self.tokenize,
            max_tokens=self.max_tokens,
        ).mean_agreement
