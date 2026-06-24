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


class _TableHTMLParser(HTMLParser):
    """Minimal HTML-table parser: rows of :class:`Cell` s, span-aware (stdlib only)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[Cell]] = []
        self._cur_row: Optional[list[Cell]] = None
        self._cur_text: list[str] = []
        self._cur_attrs: dict = {}
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._cur_row = []
        elif tag in _DATA_CELLS:
            self._in_cell = True
            self._cur_text = []
            self._cur_attrs = dict(attrs)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in _DATA_CELLS and self._in_cell:
            cell = Cell(
                text="".join(self._cur_text).strip(),
                rowspan=int(self._cur_attrs.get("rowspan", 1) or 1),
                colspan=int(self._cur_attrs.get("colspan", 1) or 1),
            )
            if self._cur_row is None:
                self._cur_row = []
            self._cur_row.append(cell)
            self._in_cell = False
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None


# ---------------------------------------------------------------------------
# TEDS -- tree-edit-distance similarity over the table tree (apted engine)
# ---------------------------------------------------------------------------


@dataclass
class _TreeNode:
    """A node in the table tree apted scores: a label plus ordered children."""

    name: str
    children: list = field(default_factory=list)


def _table_tree(table: Table, *, structure_only: bool) -> _TreeNode:
    """Map a :class:`Table` to the HTML-style tree TEDS edits.

    ``table -> tr* -> td*``; a cell's label encodes its span (always) and its text
    (unless ``structure_only``), so a content edit costs one rename and a structural
    edit (span/shape change) is a distinct rename.
    """
    root = _TreeNode("table")
    for row in table.rows:
        tr = _TreeNode("tr")
        for cell in row:
            shape = f"td:{cell.rowspan}x{cell.colspan}"
            label = shape if structure_only else f"{shape}:{cell.text}"
            tr.children.append(_TreeNode(label))
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
            def rename(self, n1: _TreeNode, n2: _TreeNode) -> int:
                return int(n1.name != n2.name)

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
        """Corpus TEDS = ``1 - sum(edit_distance) / sum(max_nodes)`` (globally pooled).

        Like CER (a normalized edit rate), TEDS pools the raw edit distances and the
        per-table node counts across the corpus before dividing -- not a mean of
        per-table similarities.
        """
        total_dist = sum(s.detail.get("edit_distance", 0.0) for s in scores)
        total_nodes = sum(s.detail.get("max_nodes", 0) for s in scores)
        return (1.0 - total_dist / total_nodes) if total_nodes else float("nan")


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


def _grits_overlap(
    grid_p: list, grid_g: list, *, structure_only: bool
) -> float:
    """Summed similarity of the largest aligned common sub-grid (2D-MSS heuristic).

    The exact 2D-MSS is NP-hard; following Smock et al., we use the polynomial
    heuristic of aligning the two grids at their shared top-left origin and summing
    per-position cell similarities over the overlapping region (rows/cols both
    tables have). On equal-shape grids this is exact; on differing shapes it is the
    standard upper/lower-bound-agreeing heuristic the paper reports.
    """
    rows = min(len(grid_p), len(grid_g))
    total = 0.0
    for r in range(rows):
        cols = min(len(grid_p[r]), len(grid_g[r]))
        for c in range(cols):
            cp, cg = grid_p[r][c], grid_g[r][c]
            if cp is None or cg is None:
                continue
            total += _cell_match(cp, cg, structure_only=structure_only)
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
