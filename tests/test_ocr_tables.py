"""Tests for :mod:`ek.ocr.tables`: recovering a scoreable ``Table`` from an ``OcrResult``.

ocracy has no normalized table type, so OCR table structure lives only in
``OcrResult.raw`` (engine-specific) or ``.markdown``. These tests cover the ek-side
seam that turns each of those forms into the existing :class:`ek.metrics.tables.Table`
the table metrics already score, plus the registry/injection seam and import purity.
"""

import subprocess
import sys

import pytest

from ek import has_table_structure, score, table_from_ocr
from ek.metrics.tables import Cell, Table
from ek.ocr.tables import engine_yields_tables, resolve_table_parser
from ek.registry import register


class FakeResult:
    """Minimal OcrResult-shaped object: ek reads only ``.raw`` / ``.markdown`` / ``.meta``."""

    def __init__(self, *, raw=None, markdown=None, meta=None):
        self.raw = raw
        self.markdown = markdown
        self.meta = meta


# --- the documented default heuristics -------------------------------------------


def test_raw_is_a_2d_grid():
    res = FakeResult(raw=[["a", "b"], ["c", "d"]])
    table = table_from_ocr(res)
    assert isinstance(table, Table)
    assert [[c.text for c in row] for row in table.rows] == [["a", "b"], ["c", "d"]]
    assert has_table_structure(res)


def test_raw_is_an_html_table_string():
    html = "<table><tr><td>a</td><td>b</td></tr></table>"
    res = FakeResult(raw=html)
    table = table_from_ocr(res)
    assert [c.text for c in table.rows[0]] == ["a", "b"]


def test_raw_is_a_prebuilt_table():
    prebuilt = Table(rows=((Cell("x"), Cell("y")),))
    res = FakeResult(raw=prebuilt)
    assert table_from_ocr(res) is prebuilt


def test_raw_mapping_under_tables_key_is_a_list_of_tables():
    # The engine-agnostic `raw['tables']` convention: a list of tables; first is used.
    res = FakeResult(raw={"tables": [[["a", "b"], ["c", "d"]]]})
    table = table_from_ocr(res)
    assert [[c.text for c in row] for row in table.rows] == [["a", "b"], ["c", "d"]]


def test_raw_mapping_under_cells_key():
    res = FakeResult(raw={"cells": [["1", "2"]]})
    assert [c.text for c in table_from_ocr(res).rows[0]] == ["1", "2"]


def test_raw_mapping_under_html_key():
    res = FakeResult(raw={"html": "<table><tr><td>z</td></tr></table>"})
    assert table_from_ocr(res).rows[0][0].text == "z"


def test_markdown_html_table_is_recovered():
    res = FakeResult(markdown="see below\n<table><tr><td>m</td></tr></table>\n")
    assert table_from_ocr(res).rows[0][0].text == "m"


def test_meta_markdown_html_table_is_recovered():
    res = FakeResult(meta={"markdown": "<table><tr><td>q</td></tr></table>"})
    assert table_from_ocr(res).rows[0][0].text == "q"


# --- the "no table present" path -> None -----------------------------------------


def test_raw_with_no_table_returns_none():
    assert table_from_ocr(FakeResult(raw={"text": "just prose, no table"})) is None
    assert has_table_structure(FakeResult(raw={"text": "x"})) is False


def test_plain_text_string_in_raw_is_not_a_table():
    # A bare string without a <table> tag must not be guessed into a table.
    assert table_from_ocr(FakeResult(raw="hello world, plain text")) is None


def test_empty_grid_is_normalized_to_none():
    assert table_from_ocr(FakeResult(raw=[])) is None
    # an HTML string with no actual cells -> empty Table -> None
    assert table_from_ocr(FakeResult(raw="<table></table>")) is None


def test_no_raw_attribute_at_all_returns_none():
    class Bare:
        pass

    assert table_from_ocr(Bare()) is None


# --- the injected / registered parser seam (open-closed) --------------------------


def test_injected_callable_parser_receives_raw():
    seen = {}

    def parser(raw):
        seen["raw"] = raw
        return raw["mytable"]  # this fake engine stores a grid under raw['mytable']

    res = FakeResult(raw={"mytable": [["p", "q"]]})
    table = table_from_ocr(res, parser=parser)
    assert [c.text for c in table.rows[0]] == ["p", "q"]
    assert seen["raw"] == {"mytable": [["p", "q"]]}


def test_registered_parser_resolved_by_name():
    @register("table_parsers", "test_engine")
    def _parse(raw):
        return raw.get("grid")

    res = FakeResult(raw={"grid": [["r"]]})
    assert table_from_ocr(res, parser="test_engine").rows[0][0].text == "r"
    # and it resolves through the same registry seam the facade uses
    assert resolve_table_parser("test_engine") is _parse


def test_injected_parser_returning_none_yields_none():
    assert table_from_ocr(FakeResult(raw={"x": 1}), parser=lambda raw: None) is None


def test_parser_output_html_and_grid_are_coerced():
    res = FakeResult(raw={"x": 1})
    html_table = table_from_ocr(
        res, parser=lambda raw: "<table><tr><td>h</td></tr></table>"
    )
    assert html_table.rows[0][0].text == "h"
    grid_table = table_from_ocr(res, parser=lambda raw: [["g"]])
    assert grid_table.rows[0][0].text == "g"


# --- the recovered Table scores via the table metrics -----------------------------


def test_recovered_table_scores_via_teds_and_grits():
    pytest.importorskip("apted")  # TEDS needs apted; GriTS is pure-Python
    pred = table_from_ocr(FakeResult(raw=[["a", "b"], ["c", "d"]]))
    gold = Table.from_grid([["a", "b"], ["c", "x"]])  # one cell differs

    teds = score(pred, gold, metric="teds")
    assert teds.metric == "teds"
    assert 0.0 < teds.value < 1.0  # close but not identical

    grits = score(pred, gold, metric="grits")
    assert 0.0 < grits.value <= 1.0

    # identical tables -> perfect structure score
    same = table_from_ocr(FakeResult(raw=[["a", "b"], ["c", "d"]]))
    assert score(pred, same, metric="teds_struct").value == 1.0


# --- profile convenience -----------------------------------------------------------


def test_engine_yields_tables_reads_the_profile():
    assert engine_yields_tables("aws-textract") is True
    assert engine_yields_tables("google-vision") is False  # TABLE = tag only
    assert engine_yields_tables("totally-unknown-engine") is False


# --- import purity: recovering tables must not pull ocracy / an engine ------------


def test_importing_ek_ocr_tables_pulls_no_engine():
    code = (
        "import sys, ek.ocr.tables\n"
        "from ek.ocr.tables import table_from_ocr\n"
        # exercise the default heuristic path too, to be sure no lazy engine import fires
        "class R:\n"
        "    raw = [['a', 'b']]\n"
        "table_from_ocr(R())\n"
        "engines = ['ocracy', 'pytesseract', 'easyocr', 'paddleocr', 'boto3']\n"
        "bad = [m for m in engines if m in sys.modules]\n"
        "assert not bad, 'engine/ocracy imported by ek.ocr.tables: ' + repr(bad)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
