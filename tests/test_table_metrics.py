"""Tests for table metrics: TEDS / TEDS-Struct and GriTS (structure-only toggle)."""

import math

import pytest

from ek import score
from ek.facade import evaluate
from ek.metrics.tables import Cell, GritsMetric, Table, TedsMetric

pytest.importorskip("apted")  # TEDS needs apted; GriTS is pure-Python


# --- table model parsing ----------------------------------------------------------


def test_html_parsing_with_spans():
    t = Table.from_html(
        "<table><tr><td colspan='2'>a</td></tr>"
        "<tr><td>b</td><td>c</td></tr></table>"
    )
    assert t.rows[0][0] == Cell(text="a", rowspan=1, colspan=2)
    assert [c.text for c in t.rows[1]] == ["b", "c"]


def test_grid_construction_and_dense_expansion():
    t = Table.from_grid([["a", "b"], ["c", "d"]])
    grid = t.as_grid()
    assert len(grid) == 2 and len(grid[0]) == 2
    assert grid[0][0].text == "a" and grid[1][1].text == "d"


def test_rowspan_expands_into_grid():
    t = Table(rows=((Cell("a", rowspan=2), Cell("b")), (Cell("c"),)))
    grid = t.as_grid()
    # 'a' occupies (0,0) and (1,0); 'b' at (0,1); 'c' at (1,1)
    assert grid[0][0].text == "a" and grid[1][0].text == "a"
    assert grid[0][1].text == "b" and grid[1][1].text == "c"


# --- TEDS / TEDS-Struct -----------------------------------------------------------


def test_teds_identical_is_one():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert math.isclose(score(a, a, metric="teds").value, 1.0, abs_tol=1e-9)


def test_teds_content_edit_drops_below_one():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    b = "<table><tr><td>a</td><td>c</td></tr></table>"
    s = score(a, b, metric="teds")
    assert s.metric == "teds"
    assert 0.0 < s.value < 1.0
    assert s.detail["edit_distance"] == 1  # one cell rename
    assert s.detail["max_nodes"] == 4  # table + tr + 2 td


def test_teds_struct_ignores_content():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    b = "<table><tr><td>a</td><td>c</td></tr></table>"  # same structure, diff content
    s = score(a, b, metric="teds_struct")
    assert s.value == 1.0
    assert s.detail["structure_only"] is True


def test_teds_struct_catches_structural_difference():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    b = "<table><tr><td>a</td></tr></table>"  # one fewer cell
    s = score(a, b, metric="teds_struct")
    assert s.value < 1.0


def test_teds_grades_multichar_content():
    # Regression: full-TEDS cell cost was binary (a full 1 for any content diff). It
    # must be GRADED by normalized string-edit distance: a 1-char slip < a full diff.
    one = "<table><tr><td>tahle</td></tr></table>"
    near = score(one, "<table><tr><td>table</td></tr></table>", metric="teds").value
    far = score(one, "<table><tr><td>zzzzz</td></tr></table>", metric="teds").value
    assert near > far          # graded, not binary
    assert 0.0 < far < near < 1.0


def test_teds_corpus_is_mean_of_per_sample():
    # Use tables of DIFFERENT node counts so the per-sample mean != the pooled
    # edit-rate (otherwise the test cannot tell mean from pooled).
    a2 = "<table><tr><td>a</td><td>b</td></tr></table>"       # 4 nodes; edit 1 -> 0.75
    a3 = "<table><tr><td>a</td><td>b</td><td>c</td></tr></table>"  # 5 nodes
    b3 = "<table><tr><td>a</td><td>b</td><td>x</td></tr></table>"  # edit 1 -> 1 - 1/5 = 0.8
    report = evaluate([(a2, "<table><tr><td>a</td><td>z</td></tr></table>"), (a3, b3)], metric="teds")
    per_item = [s.value for s in report.scores]
    mean = sum(per_item) / len(per_item)
    pooled = 1 - (sum(s.detail["edit_distance"] for s in report.scores)
                  / sum(s.detail["max_nodes"] for s in report.scores))
    assert math.isclose(report.aggregate, mean, rel_tol=1e-9)
    assert not math.isclose(mean, pooled, rel_tol=1e-6)  # mean genuinely != pooled


# --- GriTS ------------------------------------------------------------------------


def test_grits_identical_is_one():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    s = score(a, a, metric="grits")
    assert s.value == 1.0
    assert s.precision == 1.0 and s.recall == 1.0


def test_grits_con_partial_credit_for_content():
    a = "<table><tr><td>cat</td></tr></table>"
    b = "<table><tr><td>car</td></tr></table>"  # 1 char off -> partial content credit
    s = score(a, b, metric="grits_con")
    # span matches (1.0) averaged with text sim (1 - 1/3) -> 0.5*(1) + 0.5*(2/3)
    assert math.isclose(s.value, 0.5 * 1.0 + 0.5 * (2 / 3), rel_tol=1e-6)


def test_grits_top_ignores_content():
    a = "<table><tr><td>cat</td></tr></table>"
    b = "<table><tr><td>dog</td></tr></table>"  # same topology
    s = score(a, b, metric="grits_top")
    assert s.value == 1.0
    assert s.detail["structure_only"] is True


def test_grits_precision_recall_differ_on_size_mismatch():
    big = Table.from_grid([["a", "b"], ["c", "d"]])  # 4 gold cells
    small = Table.from_grid([["a", "b"]])  # 2 pred cells, both correct
    s = GritsMetric(structure_only=True)(small, big)
    # all predicted cells match -> precision 1.0; only half of gold recovered -> 0.5
    assert math.isclose(s.precision, 1.0, abs_tol=1e-9)
    assert math.isclose(s.recall, 0.5, abs_tol=1e-9)


def test_grits_handles_inserted_row_and_column():
    # Regression (CRITICAL): the old top-left-origin overlap collapsed on any shift.
    # Canonical GriTS (factored 2D-MSS) deletes the inserted blank row/col and matches
    # the 3x3 substructure: 2*9/(9+12) = 0.857.
    gold = Table.from_grid([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]])
    ins_row = Table.from_grid([["", "", ""], ["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]])
    ins_col = Table.from_grid([["", "a", "b", "c"], ["", "d", "e", "f"], ["", "g", "h", "i"]])
    assert math.isclose(GritsMetric()(ins_row, gold).value, 2 * 9 / (9 + 12), rel_tol=1e-6)
    assert math.isclose(GritsMetric()(ins_col, gold).value, 2 * 9 / (9 + 12), rel_tol=1e-6)


def test_grits_handles_row_rotation():
    gold = Table.from_grid([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]])
    rotated = Table.from_grid([["d", "e", "f"], ["g", "h", "i"], ["a", "b", "c"]])
    # best monotonic alignment matches 2 of 3 rows -> 2*6/(9+9) = 0.667
    assert math.isclose(GritsMetric()(rotated, gold).value, 2 / 3, rel_tol=1e-6)


def test_grits_corpus_micro_aggregation_is_not_mean():
    # Different-shape tables so pooled overlap/counts diverge from the per-table mean.
    a = Table.from_grid([["a", "b"], ["c", "d"]])
    half = Table.from_grid([["a", "b"]])  # half the gold recovered
    report = evaluate([(a, a), (half, a)], metric="grits_top")
    per_item = [s.value for s in report.scores]
    mean = sum(per_item) / len(per_item)
    assert not math.isclose(report.aggregate, mean, rel_tol=1e-6)  # micro != mean


def test_empty_tables_are_perfect():
    s = GritsMetric()(Table(), Table())
    assert s.value == 1.0


# --- HTML parser robustness (regression for silent cell/row drops) ----------------


def test_html_parser_tolerates_missing_close_tags():
    # No </td> / </tr> closes -- a real-world messy table. Cells must not be dropped.
    t = Table.from_html("<table><tr><td>a<td>b<tr><td>c<td>d</table>")
    assert [[c.text for c in row] for row in t.rows] == [["a", "b"], ["c", "d"]]


def test_html_parser_ignores_nested_table_structure():
    # A nested table's rows/cells must not corrupt the outer row stream.
    t = Table.from_html(
        "<table><tr><td>outer<table><tr><td>inner</td></tr></table></td>"
        "<td>z</td></tr></table>"
    )
    assert len(t.rows) == 1
    assert len(t.rows[0]) == 2  # exactly the two OUTER cells
    assert t.rows[0][1].text == "z"
