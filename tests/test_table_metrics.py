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


def test_teds_corpus_is_globally_pooled():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    b = "<table><tr><td>a</td><td>c</td></tr></table>"
    report = evaluate([(a, a), (a, b)], metric="teds")
    # total edits = 0 + 1, total max_nodes = 4 + 4 = 8 -> 1 - 1/8
    assert math.isclose(report.aggregate, 1 - 1 / 8, rel_tol=1e-9)


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


def test_grits_corpus_micro_aggregation():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    b = "<table><tr><td>a</td><td>c</td></tr></table>"
    report = evaluate([(a, a), (a, b)], metric="grits_top")
    # both structurally identical -> overlap pooled = full; aggregate 1.0
    assert math.isclose(report.aggregate, 1.0, abs_tol=1e-9)


def test_empty_tables_are_perfect():
    s = GritsMetric()(Table(), Table())
    assert s.value == 1.0
