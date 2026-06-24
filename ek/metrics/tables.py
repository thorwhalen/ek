"""Table-structure metrics: TEDS / TEDS-Struct and GriTS (structure-only toggle).

Two complementary table-recognition metrics from ``misc/docs/ek_02`` §3.1:

- **TEDS** (Tree-Edit-Distance-based Similarity; Zhong et al., PubTabNet, IBM,
  `arXiv 1911.10683`_) renders a table as an HTML tree and scores
  ``1 - EditDist(Ta, Tb) / max(|Ta|, |Tb|)``. It captures both *structure* and
  *cell content*. **TEDS-Struct** (the ``structure_only=True`` toggle) ignores cell
  text to isolate structure, since "taking OCR errors into account may lead to an
  unfair comparison due to the different OCR models used by various TSR methods".
- **GriTS** (Grid Table Similarity; Smock et al., Microsoft, `arXiv 2203.12555`_)
  scores the predicted table directly as a 2-D matrix and yields **precision/recall**
  over the most-similar common sub-grid. Its **GriTS-Top** (topology, structure-only)
  and **GriTS-Con** (content) variants share the ``structure_only`` toggle.

Licensing note (``skills/ek-dev-licensing``): the obvious wrapper target,
``table-recognition-metric``, pulls in the **GPL** ``Levenshtein`` package as a hard
runtime dependency (``from Levenshtein import distance`` in its source) -- a
scanner-invisible copyleft landmine that would fail the CI license gate. So TEDS is
implemented **clean-room** here on the MIT ``apted`` engine (already in
``[metrics]``), and GriTS is a dependency-free pure-Python implementation of the
2D-MSS heuristic. Both follow the precedent of reimplementing Krippendorff's alpha in
pure Python rather than quarantining a copyleft dependency.

Input shapes
------------
``pred`` and ``gold`` are :class:`Table` objects, or anything the constructors
accept: an HTML ``str`` (``<table>...</table>``; via :meth:`Table.from_html`) or a
2-D grid -- a list of rows of cell strings, where ``None`` marks a cell spanned over
by a neighbour (via :meth:`Table.from_grid`). The facade and registry coerce HTML
strings automatically.

.. _arXiv 1911.10683: https://arxiv.org/abs/1911.10683
.. _arXiv 2203.12555: https://arxiv.org/abs/2203.12555

Example:
    >>> from ek.metrics.tables import TedsMetric, GritsMetric, Table
    >>> a = "<table><tr><td>a</td><td>b</td></tr></table>"
    >>> b = "<table><tr><td>a</td><td>c</td></tr></table>"
    >>> round(TedsMetric()(a, b).value, 3)            # one cell-content edit
    0.75
    >>> TedsMetric(structure_only=True)(a, b).value   # same structure -> 1.0
    1.0
    >>> GritsMetric(structure_only=True)(a, b).value  # identical grid topology
    1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Optional, Sequence

from ..base import GraphGrammar, Score
from ..registry import requires_extra

_DATA_CELLS = ("td", "th")


@dataclass(frozen=True)
class Cell:
    """One logical table cell: its text plus its row/column span (>=1 each)."""

    text: str = ""
    rowspan: int = 1
    colspan: int = 1


@dataclass(frozen=True)
class Table:
    """A table as an ordered list of rows of :class:`Cell` s (the logical cells).

    ``rows[i]`` is the i-th row's logical cells in reading order; spans are carried
    on each :class:`Cell`. Use :meth:`from_html` / :meth:`from_grid` to build one,
    and :meth:`as_grid` for the dense 2-D occupancy view GriTS scores.
    """

    rows: Sequence[Sequence[Cell]] = ()

    @classmethod
    def coerce(cls, obj: Any) -> "Table":
        """Coerce a :class:`Table`, an HTML string, or a 2-D grid into a Table."""
        if isinstance(obj, Table):
            return obj
        if isinstance(obj, str):
            return cls.from_html(obj)
        return cls.from_grid(obj)

    @classmethod
    def from_html(cls, html: str) -> "Table":
        """Parse an HTML ``<table>`` (with ``colspan``/``rowspan``) into a Table."""
        parser = _TableHTMLParser()
        parser.feed(html)
        return cls(rows=tuple(tuple(r) for r in parser.rows))

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[Any]]) -> "Table":
        """Build a Table from a 2-D grid of cell texts (``None`` = spanned-over)."""
        rows = tuple(
            tuple(Cell(text=str(c)) for c in row if c is not None) for row in grid
        )
        return cls(rows=rows)

    def as_grid(self) -> list[list[Optional[Cell]]]:
        """Expand spans into a dense 2-D grid; a spanned position repeats its cell.

        Each physical grid position holds the (shared) :class:`Cell` covering it, so
        two tables with the same topology have grids of the same shape -- the view
        GriTS aligns position-by-position.
        """
        grid: list[list[Optional[Cell]]] = []
        for r, row in enumerate(self.rows):
            while len(grid) <= r:
                grid.append([])
            col = 0
            for cell in row:
                while col < len(grid[r]) and grid[r][col] is not None:
                    col += 1
                for dr in range(cell.rowspan):
                    rr = r + dr
                    while len(grid) <= rr:
                        grid.append([])
                    for dc in range(cell.colspan):
                        cc = col + dc
                        while len(grid[rr]) <= cc:
                            grid[rr].append(None)
                        grid[rr][cc] = cell
                col += cell.colspan
        width = max((len(row) for row in grid), default=0)
        for row in grid:
            row.extend([None] * (width - len(row)))
        return grid


def _pos_int(value: Any) -> int:
    """Parse a span attribute to an int ``>= 1`` (default 1 on missing/garbage)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return n if n >= 1 else 1


class _TableHTMLParser(HTMLParser):
    """Span-aware HTML ``<table>`` parser (stdlib only), tolerant of real-world HTML.

    Flushes the current cell on any new ``td``/``th`` start (implicit close) and on
    ``</tr>``, appends a pending row on ``</table>`` (missing ``</tr>``), and only
    treats the **outermost** table's structure -- a nested table's text flows into
    the enclosing cell rather than corrupting the outer row/cell stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[Cell]] = []
        self._cur_row: Optional[list[Cell]] = None
        self._cur_text: list[str] = []
        self._cur_attrs: dict = {}
        self._in_cell = False
        self._table_depth = 0

    def _flush_cell(self) -> None:
        if self._in_cell:
            cell = Cell(
                text="".join(self._cur_text).strip(),
                rowspan=_pos_int(self._cur_attrs.get("rowspan")),
                colspan=_pos_int(self._cur_attrs.get("colspan")),
            )
            if self._cur_row is None:
                self._cur_row = []
            self._cur_row.append(cell)
            self._in_cell = False
            self._cur_text = []

    def _flush_row(self) -> None:
        self._flush_cell()
        if self._cur_row:
            self.rows.append(self._cur_row)
        self._cur_row = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return  # nested-table structure is ignored; its text still flows into the cell
        if tag == "tr":
            self._flush_row()
            self._cur_row = []
        elif tag in _DATA_CELLS:
            self._flush_cell()  # implicit close of an unclosed previous cell
            self._in_cell = True
            self._cur_text = []
            self._cur_attrs = dict(attrs)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._table_depth == 1:
                self._flush_row()  # flush a row left open by a missing </tr>
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if tag in _DATA_CELLS:
            self._flush_cell()
        elif tag == "tr":
            self._flush_row()


# ---------------------------------------------------------------------------
# TEDS -- tree-edit-distance similarity over the table tree (apted engine)
# ---------------------------------------------------------------------------


@dataclass
class _TreeNode:
    """A node in the table tree apted scores: a structural label, optional cell text,
    and ordered children. Keeping ``text`` OFF the ``name`` lets the rename cost be
    *graded* by content similarity (canonical TEDS) rather than a flat 0/1."""

    name: str
    text: str = ""
    children: list = field(default_factory=list)


def _table_tree(table: Table, *, structure_only: bool) -> _TreeNode:
    """Map a :class:`Table` to the HTML-style tree TEDS edits (``table -> tr* -> td*``).

    The node ``name`` encodes only the *structure* (the tag and span ``td:RxC``); the
    cell text rides on ``.text`` so a content difference is scored by the normalized
    string-edit distance of the texts (graded, like canonical TEDS), while a span/
    shape change is a full-cost structural rename. ``structure_only`` drops the text.
    """
    root = _TreeNode("table")
    for row in table.rows:
        tr = _TreeNode("tr")
        for cell in row:
            shape = f"td:{cell.rowspan}x{cell.colspan}"
            tr.children.append(
                _TreeNode(shape, text="" if structure_only else cell.text)
            )
        root.children.append(tr)
    return root


def _tree_size(node: _TreeNode) -> int:
    return 1 + sum(_tree_size(c) for c in node.children)


class TedsMetric:
    """TEDS / TEDS-Struct as a :class:`~ek.base.Metric` (clean-room on ``apted``).

    Args:
        structure_only: When ``True``, ignore cell text and score structure alone
            (TEDS-Struct), isolating table-structure accuracy from OCR noise.
            Defaults to ``False`` (full TEDS: structure + content).
    """

    name = "teds"

    def __init__(self, *, structure_only: bool = False):
        self.structure_only = structure_only

    @requires_extra("metrics", packages=["apted"])
    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        from apted import APTED, Config

        class _Config(Config):
            def rename(self, n1: _TreeNode, n2: _TreeNode) -> float:
                # A structural difference (tag/span) is a full-cost rename; same
                # structure with different text costs the NORMALIZED string-edit
                # distance of the cell text (canonical TEDS), so a one-char slip in a
                # long cell costs a small fraction, not a full 1.
                if n1.name != n2.name:
                    return 1.0
                if n1.text == n2.text:
                    return 0.0
                return 1.0 - _string_sim(n1.text, n2.text)

            def children(self, node: _TreeNode) -> list:
                return node.children

        tp = _table_tree(Table.coerce(pred), structure_only=self.structure_only)
        tg = _table_tree(Table.coerce(gold), structure_only=self.structure_only)
        dist = APTED(tp, tg, _Config()).compute_edit_distance()
        denom = max(_tree_size(tp), _tree_size(tg))
        similarity = 1.0 - (dist / denom) if denom else 1.0
        return Score(
            value=similarity,
            metric="teds_struct" if self.structure_only else "teds",
            detail={
                "edit_distance": dist,
                "max_nodes": denom,
                "structure_only": self.structure_only,
                "higher_is_better": True,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Corpus TEDS = **mean** of per-table TEDS (the PubTabNet/OmniDocBench
        convention). Each per-table TEDS is already node-count-normalized, so the
        reported corpus statistic is the average, not a pooled edit-rate. The pooled
        ``1 - sum(dist)/sum(nodes)`` is available from the per-item ``detail`` if a
        size-weighted variant is wanted instead.
        """
        vals = [s.value for s in scores]
        return (sum(vals) / len(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# GriTS -- 2-D grid similarity with precision/recall (pure-Python, no deps)
# ---------------------------------------------------------------------------


def _cell_match(a: Cell, b: Cell, *, structure_only: bool) -> float:
    """Per-position similarity of two aligned cells: span match (and text, if scored).

    GriTS-Top (``structure_only``) scores only span/topology agreement; GriTS-Con
    averages topology with a normalized cell-text similarity, so partial content
    matches earn partial credit.
    """
    span_ok = a.rowspan == b.rowspan and a.colspan == b.colspan
    if structure_only:
        return 1.0 if span_ok else 0.0
    text_sim = _string_sim(a.text, b.text)
    return 0.5 * (1.0 if span_ok else 0.0) + 0.5 * text_sim


def _string_sim(a: str, b: str) -> float:
    """Normalized edit similarity ``1 - lev/max(len)`` (pure-Python; no GPL dep)."""
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    dist = prev[-1]
    return 1.0 - dist / max(len(a), len(b))


def _align_value(n_a: int, n_b: int, score) -> float:
    """Max total ``score(i, j)`` over a monotonic alignment of ``[0,n_a)`` to
    ``[0,n_b)`` (whole-element insert/delete are free). The 1-D building block."""
    if n_a == 0 or n_b == 0:
        return 0.0
    prev = [0.0] * (n_b + 1)
    for i in range(1, n_a + 1):
        cur = [0.0] * (n_b + 1)
        for j in range(1, n_b + 1):
            cur[j] = max(prev[j], cur[j - 1], prev[j - 1] + score(i - 1, j - 1))
        prev = cur
    return prev[n_b]


def _align_pairs(score_matrix: list) -> list:
    """Matched ``(i, j)`` pairs of the optimal monotonic alignment of a precomputed
    score matrix (rows = first sequence, cols = second), insert/delete free."""
    n_a = len(score_matrix)
    n_b = len(score_matrix[0]) if n_a else 0
    if n_a == 0 or n_b == 0:
        return []
    dp = [[0.0] * (n_b + 1) for _ in range(n_a + 1)]
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            dp[i][j] = max(
                dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + score_matrix[i - 1][j - 1]
            )
    pairs = []
    i, j = n_a, n_b
    while i > 0 and j > 0:
        diag = dp[i - 1][j - 1] + score_matrix[i - 1][j - 1]
        if dp[i][j] == diag and diag >= dp[i - 1][j] and diag >= dp[i][j - 1]:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _grits_overlap(grid_p: list, grid_g: list, *, structure_only: bool) -> float:
    """Factored 2-D Most-Similar-Substructure (Smock et al., GriTS).

    The exact 2D-MSS is NP-hard, so GriTS uses the **factored** polynomial
    approximation: align the row-sequences and the column-sequences *independently*
    (each a 1-D alignment that allows whole-row / whole-column insert and delete),
    then sum the per-cell similarity over the Cartesian product of the matched row
    pairs and matched column pairs. Unlike a fixed top-left-origin overlap, this
    stays meaningful when a row or column is inserted, deleted, or shifted -- the case
    the metric exists to score. Complexity is polynomial (~O(R^2 C^2)) in the grid
    dimensions, fine for offline evaluation.
    """
    pr, pc = len(grid_p), max((len(r) for r in grid_p), default=0)
    gr, gc = len(grid_g), max((len(r) for r in grid_g), default=0)

    def at(grid: list, r: int, c: int):
        return grid[r][c] if (r < len(grid) and c < len(grid[r])) else None

    def m(a, b) -> float:
        return 0.0 if (a is None or b is None) else _cell_match(a, b, structure_only=structure_only)

    # row-similarity[i][j] = best cell alignment of pred row i vs gold row j (over cols)
    row_sim = [
        [_align_value(pc, gc, lambda c1, c2, i=i, j=j: m(at(grid_p, i, c1), at(grid_g, j, c2)))
         for j in range(gr)]
        for i in range(pr)
    ]
    # col-similarity[i][j] = best cell alignment of pred col i vs gold col j (over rows)
    col_sim = [
        [_align_value(pr, gr, lambda r1, r2, i=i, j=j: m(at(grid_p, r1, i), at(grid_g, r2, j)))
         for j in range(gc)]
        for i in range(pc)
    ]
    row_pairs = _align_pairs(row_sim) if row_sim else []
    col_pairs = _align_pairs(col_sim) if col_sim else []

    total = 0.0
    for ip, ig in row_pairs:
        for jp, jg in col_pairs:
            total += m(at(grid_p, ip, jp), at(grid_g, ig, jg))
    return total


def _occupied(grid: list) -> int:
    return sum(1 for row in grid for cell in row if cell is not None)


class GritsMetric:
    """GriTS-Top / GriTS-Con as a :class:`~ek.base.Metric` (pure-Python, P/R aware).

    Args:
        structure_only: When ``True``, score cell topology only (GriTS-Top); when
            ``False`` (default), average topology with cell-content similarity
            (GriTS-Con).
    """

    name = "grits"

    def __init__(self, *, structure_only: bool = False):
        self.structure_only = structure_only

    def __call__(
        self, pred: Any, gold: Any, *, grammar: Optional[GraphGrammar] = None
    ) -> Score:
        grid_p = Table.coerce(pred).as_grid()
        grid_g = Table.coerce(gold).as_grid()
        overlap = _grits_overlap(
            grid_p, grid_g, structure_only=self.structure_only
        )
        n_pred = _occupied(grid_p)
        n_gold = _occupied(grid_g)
        precision = overlap / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
        recall = overlap / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return Score(
            value=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            metric="grits_top" if self.structure_only else "grits_con",
            detail={
                "overlap": overlap,
                "n_pred": n_pred,
                "n_gold": n_gold,
                "structure_only": self.structure_only,
                "higher_is_better": True,
            },
        )

    def aggregate(self, scores: Sequence[Score]) -> float:
        """Corpus GriTS F1: pool overlap and per-table cell counts, then divide once.

        GriTS precision = overlap / |pred cells| and recall = overlap / |gold cells|;
        the corpus statistic micro-averages by summing the numerators and
        denominators globally -- never a mean of per-table F1s.
        """
        overlap = sum(s.detail.get("overlap", 0.0) for s in scores)
        n_pred = sum(s.detail.get("n_pred", 0) for s in scores)
        n_gold = sum(s.detail.get("n_gold", 0) for s in scores)
        if n_pred == 0 and n_gold == 0:
            return 1.0
        precision = overlap / n_pred if n_pred else 0.0
        recall = overlap / n_gold if n_gold else 0.0
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
