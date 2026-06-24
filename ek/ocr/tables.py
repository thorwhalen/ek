"""Recover a normalized table from an ``OcrResult`` so ek's table metrics can score it.

``ocracy`` exposes **no** normalized table/cell type: ``TextBlock.level`` stops at
``block``, with no cell/row/column model, so OCR table structure survives only inside
the engine-specific ``OcrResult.raw`` (Textract ``Block`` objects, PaddleOCR
``pred_html``, Azure ``tables[]``) or in the markdown/HTML an engine emits
(``res.markdown`` for VLM engines). ek therefore can't uniformly score table
structure (TEDS / GriTS) off the ``OcrResult`` shape alone.

This module is the **ek-side seam** that closes that gap *without touching ocracy*
(the dependency arrow stays ``ek -> ocracy``, never the reverse). It turns an
``OcrResult``-shaped object into the existing :class:`ek.metrics.tables.Table` (the
one TEDS/GriTS already score) -- it does **not** define a new table type.

Because every engine stores tables differently, the per-engine extractor is an
**injected** callable (open-closed / dependency-injection): pass ``parser=`` a
``(raw) -> Table | grid | html | None`` function, or register one under the
``"table_parsers"`` namespace and pass its name. With no parser,
:func:`table_from_ocr` falls back to a few **safe, documented** heuristics over the
result's ``raw``/``markdown``/``meta`` and returns ``None`` when no table structure is
present -- it never guesses wildly.

Nothing here imports ``ocracy`` or any OCR engine: ek depends only on the
``OcrResult`` *shape* (``.text`` / ``.raw`` / ``.markdown`` / ``.meta``), so this works
on any ``image -> OcrResult`` callable's output.

Example:
    >>> from ek.metrics.tables import Table
    >>> class R:  # a minimal OcrResult-shaped object carrying a 2-D grid in .raw
    ...     raw = {"tables": [[["a", "b"], ["c", "d"]]]}
    >>> t = table_from_ocr(R())
    >>> [[c.text for c in row] for row in t.rows]
    [['a', 'b'], ['c', 'd']]
    >>> has_table_structure(R())
    True
    >>> class Empty:  # no table anywhere in the result
    ...     raw = {"text": "just prose"}
    >>> table_from_ocr(Empty()) is None
    True

A registered third-party engine parser, used by name:
    >>> from ek.registry import register
    >>> @register("table_parsers", "demo")
    ... def _demo(raw):
    ...     return raw.get("cells")  # this engine stores a grid under raw['cells']
    >>> class D:
    ...     raw = {"cells": [["x"]]}
    >>> table_from_ocr(D(), parser="demo").rows[0][0].text
    'x'
"""

from __future__ import annotations

from collections.abc import Sequence as _SequenceABC
from typing import Any, Callable, Optional, Union

from ..metrics.tables import Table
from ..registry import resolve

#: Registry namespace under which per-engine table parsers live. Register a parser
#: with ``ek.registry.register("table_parsers", "<engine>", fn)`` (or an
#: ``ek.table_parsers`` entry point) and resolve it by name via the ``parser=`` arg.
TABLE_PARSERS_NAMESPACE = "table_parsers"

#: Keys ek looks under, in order, for a table payload nested in a ``raw``/``meta`` dict
#: (engine-agnostic conventions). First non-empty hit wins.
_RAW_TABLE_KEYS = ("tables", "table", "cells", "grid", "html")


def table_from_ocr(
    ocr_result: Any, *, parser: Union[None, str, Callable[[Any], Any]] = None
) -> Optional[Table]:
    """Extract a normalized :class:`~ek.metrics.tables.Table` from an ``OcrResult``.

    The single seam ek's table metrics (TEDS / GriTS) use to score OCR table output
    uniformly. Returns the recovered Table, or ``None`` when the result carries no
    recoverable table structure.

    Args:
        ocr_result: Any ``OcrResult``-shaped object. ek reads only its ``.raw`` (the
            engine-specific payload), ``.markdown`` (VLM/markdown engines), and
            ``.meta`` -- never ocracy types, so any ``image -> OcrResult`` callable's
            output works.
        parser: How to turn the raw payload into a table (dependency injection /
            open-closed). One of:

            - ``None`` (default): try the built-in **safe heuristics** -- ``.raw`` is
              already a :class:`~ek.metrics.tables.Table`, a 2-D grid, or a
              ``<table>...</table>`` HTML string; or ``.raw``/``.meta`` is a mapping
              carrying one of ``{tables, table, cells, grid, html}``; or ``.markdown``
              contains an HTML ``<table>``. Returns ``None`` if none match -- it never
              guesses wildly.
            - a ``str``: the name of a parser registered under ``"table_parsers"``
              (per-engine extractor), resolved from :mod:`ek.registry`.
            - a callable ``(raw) -> Table | grid | html | None``: an inline per-engine
              extractor. It receives the result's ``.raw`` (falling back to the result
              itself when there is no ``.raw``).

        Any non-``None`` value an injected parser returns is fed through
        :meth:`Table.coerce`, so a parser may yield a Table, a 2-D grid, or HTML.

    Returns:
        A non-empty :class:`~ek.metrics.tables.Table`, or ``None`` when no table
        structure is recoverable. An empty Table (no rows/cells) is normalized to
        ``None`` so the predicate :func:`has_table_structure` and downstream metrics
        see "no table", not a degenerate one.
    """
    if parser is not None:
        candidate = _run_injected_parser(ocr_result, parser)
        candidate = _coerce_parser_output(candidate)
    else:
        candidate = _heuristic_table(ocr_result)
    return _nonempty_table(candidate)


def has_table_structure(
    ocr_result: Any, *, parser: Union[None, str, Callable[[Any], Any]] = None
) -> bool:
    """Whether :func:`table_from_ocr` recovers a non-empty table from ``ocr_result``.

    A thin predicate over :func:`table_from_ocr` (same ``parser`` semantics): true iff
    a table with at least one cell can be recovered. Use it to route an OCR result to
    the table metrics vs the text-only CER/WER fallback.
    """
    return table_from_ocr(ocr_result, parser=parser) is not None


def resolve_table_parser(ref: Union[str, Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """Resolve a table parser reference (a registered name or a callable) to a callable.

    Mirrors how other ek strategies resolve from the registry (see
    :func:`ek.registry.resolve`). A ``str`` is looked up under the ``"table_parsers"``
    namespace; an already-callable parser is returned unchanged.
    """
    return resolve(TABLE_PARSERS_NAMESPACE, ref)


def engine_yields_tables(engine: str) -> bool:
    """Whether ``engine``'s capability profile claims scoreable table structure.

    A convenience over :func:`ek.ocr.profile`: returns the engine profile's ``tables``
    flag. ek treats the profile as a *prior* (Azure DI, AWS Textract, PaddleOCR and the
    markdown engines are expected to yield scoreable tables; Google Vision is not), to
    be verified empirically by actually running :func:`table_from_ocr`. Use it to skip
    table scoring for engines that cannot emit cells in the first place.
    """
    from .profiles import profile

    return bool(profile(engine).get("tables", False))


# ---------------------------------------------------------------------------
# Internals: injected-parser dispatch + the default safe heuristics
# ---------------------------------------------------------------------------


def _run_injected_parser(
    ocr_result: Any, parser: Union[str, Callable[[Any], Any]]
) -> Any:
    """Run an injected/registered parser on the result's ``raw`` payload."""
    fn = resolve_table_parser(parser)
    raw = getattr(ocr_result, "raw", None)
    return fn(raw if raw is not None else ocr_result)


def _coerce_parser_output(candidate: Any) -> Optional[Table]:
    """Coerce a parser's return (``None`` / Table / grid / HTML) to a Table or ``None``."""
    if candidate is None:
        return None
    return Table.coerce(candidate)


def _heuristic_table(ocr_result: Any) -> Optional[Table]:
    """The default, no-parser heuristics: a few safe, documented shapes only.

    Order: (1) ``.raw`` is already a table-ish payload (Table / grid / ``<table>``
    HTML); (2) ``.raw``/``.meta`` is a mapping carrying a known table key; (3)
    ``.markdown`` (or a ``.meta['markdown']``) contains an HTML ``<table>``. Anything
    else -> ``None`` (no wild guessing).
    """
    raw = getattr(ocr_result, "raw", None)

    direct = _coerce_table_like(raw)
    if direct is not None:
        return direct

    if isinstance(raw, dict):
        nested = _table_from_mapping(raw)
        if nested is not None:
            return nested

    meta = getattr(ocr_result, "meta", None)
    if isinstance(meta, dict):
        nested = _table_from_mapping(meta)
        if nested is not None:
            return nested

    return _table_from_markdown(ocr_result)


def _coerce_table_like(obj: Any) -> Optional[Table]:
    """Coerce ``obj`` to a Table iff it is unambiguously table-shaped, else ``None``.

    "Table-shaped" means: an existing :class:`Table`, an HTML string containing a
    ``<table>`` tag, or a 2-D grid (a sequence of row-sequences). A bare string without
    a ``<table>`` tag, a mapping, or a flat list is **not** table-shaped here (mappings
    are handled separately, by key).
    """
    if isinstance(obj, Table):
        return obj
    if isinstance(obj, str):
        return Table.from_html(obj) if _looks_like_html_table(obj) else None
    if _is_2d_grid(obj):
        return Table.coerce(obj)
    return None


def _table_from_mapping(mapping: dict) -> Optional[Table]:
    """First table recoverable from a known key of a ``raw``/``meta`` mapping.

    Looks under ``{tables, table, cells, grid, html}``. A ``tables`` value may be a
    *list of tables*; the first non-empty one is used (single-table convention -- a
    multi-table seam can be added open-closed via an injected parser).
    """
    for key in _RAW_TABLE_KEYS:
        if key not in mapping:
            continue
        value = mapping[key]
        for candidate in _iter_table_candidates(value):
            table = _nonempty_table(_coerce_table_like(candidate))
            if table is not None:
                return table
    return None


def _iter_table_candidates(value: Any):
    """Yield plausible single-table payloads from a mapping value.

    A ``list of tables`` (e.g. ``raw['tables'] == [grid_a, grid_b]``) yields each
    table; a single table-shaped value yields just itself. Strings, mappings and
    Tables are always yielded whole (never iterated into characters/items).
    """
    if isinstance(value, (str, dict, Table)):
        yield value
        return
    if _is_list_of_tables(value):
        yield from value
        return
    yield value


def _table_from_markdown(ocr_result: Any) -> Optional[Table]:
    """Recover a table from a result's markdown HTML ``<table>`` (VLM/markdown engines).

    Reads ``.markdown`` (ocracy surfaces ``meta['markdown']`` there) or, failing that,
    ``.meta['markdown']``. Only an HTML ``<table>`` is parsed -- pipe-style markdown
    tables are intentionally out of scope for the *safe* default (register a parser for
    those). Returns ``None`` if there is no HTML table.
    """
    markdown = getattr(ocr_result, "markdown", None)
    if not isinstance(markdown, str):
        meta = getattr(ocr_result, "meta", None)
        markdown = meta.get("markdown") if isinstance(meta, dict) else None
    if isinstance(markdown, str) and _looks_like_html_table(markdown):
        return Table.from_html(markdown)
    return None


def _looks_like_html_table(text: str) -> bool:
    """Whether ``text`` contains an HTML ``<table>`` opening tag (case-insensitive)."""
    return "<table" in text.lower()


def _is_2d_grid(obj: Any) -> bool:
    """Whether ``obj`` is a 2-D grid: a non-empty non-string sequence whose items are
    all non-string sequences (rows). Empty rows are allowed; an empty outer sequence is
    not a grid (nothing to score)."""
    if not _is_sequence(obj) or not obj:
        return False
    return all(_is_sequence(row) for row in obj)


def _is_list_of_tables(obj: Any) -> bool:
    """Whether ``obj`` is a sequence whose items are themselves table-shaped (each a
    Table, an HTML/``<table>`` string, or a 2-D grid) -- i.e. a list *of tables*."""
    if not _is_sequence(obj) or not obj:
        return False
    return all(
        isinstance(item, Table)
        or (isinstance(item, str) and _looks_like_html_table(item))
        or _is_2d_grid(item)
        for item in obj
    )


def _is_sequence(obj: Any) -> bool:
    """A non-string, non-bytes :class:`collections.abc.Sequence` (list/tuple-like)."""
    return isinstance(obj, _SequenceABC) and not isinstance(obj, (str, bytes))


def _nonempty_table(table: Optional[Table]) -> Optional[Table]:
    """Normalize an empty Table (no cells) to ``None``; pass a non-empty one through."""
    if table is None:
        return None
    has_cells = any(len(row) > 0 for row in table.rows)
    return table if has_cells else None
