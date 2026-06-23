"""Local-file persistence for ``ek``, as plain ``MutableMapping`` stores.

Everything ``ek`` persists -- gold corpora, evaluation results, fitted
calibrators, human corrections, run logs -- lives under the per-user data
directory ``~/.local/share/ek/`` (resolved cross-platform by
:class:`config2py.AppData`), one subfolder per *kind*. Each store is a JSON
key-value store from :mod:`dol`: ``store[key] = value`` writes
``<kind>/<key>.json``, ``store[key]`` reads it back, iteration lists keys -- so
callers get a dict-like facade rather than a bespoke god-class.

Progressive disclosure:
    >>> import tempfile; root = tempfile.mkdtemp()
    >>> s = json_store("gold", rootdir=root)
    >>> s["doc-001"] = {"reference_text": "hello world"}
    >>> s["doc-001"]["reference_text"]
    'hello world'
    >>> list(s)
    ['doc-001']

A :func:`mall` groups every kind into one stores-of-stores mapping, with optional
``mall["gold", "doc-001"]`` tuple-key access.

Tests and ephemeral use can redirect the root via the ``rootdir`` argument or the
``EK_DATA_HOME`` environment variable, so nothing touches the real user folder.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Callable, Optional

import dol
from config2py import AppData

#: Re-export for the idiomatic class-property persisted cache, e.g.
#: ``@cache_this(cache=lambda self: json_store("runs"), key=...)`` on a method.
cache_this = dol.cache_this

#: The kinds of artifact ``ek`` persists; each becomes a subfolder + a store.
KINDS = ("gold", "results", "calibrators", "corrections", "runs", "baselines")

_APP = AppData("ek")

# A JSON file store that creates missing parent directories on write.
_JsonStore = dol.mk_dirs_if_missing(dol.Jsons)


def app_folder() -> Path:
    """The root ``ek`` data folder (e.g. ``~/.local/share/ek``)."""
    return Path(_APP.app_folder())


def _resolve_root(rootdir: Optional[str]) -> Optional[Path]:
    if rootdir is not None:
        return Path(rootdir)
    env = os.environ.get("EK_DATA_HOME")
    return Path(env) if env else None


def data_dir(kind: str, *, rootdir: Optional[str] = None) -> Path:
    """Filesystem directory backing ``kind`` (under the data root or ``rootdir``)."""
    root = _resolve_root(rootdir)
    if root is None:
        return Path(_APP.get_artifact_dir(kind))
    return root / kind


def json_store(kind: str, *, rootdir: Optional[str] = None) -> MutableMapping:
    """A JSON ``MutableMapping`` for one ``kind`` of artifact.

    Keys are plain names (no extension); values are JSON-serializable objects.
    """
    return _JsonStore(str(data_dir(kind, rootdir=rootdir)))


class Mall(Mapping):
    """A read-only mapping of ``kind -> json_store(kind)`` (a dol "mall").

    Supports both ``mall["gold"]`` (the store) and ``mall["gold", key]`` (the
    value), the latter being sugar for ``mall["gold"][key]``.
    """

    def __init__(self, *, kinds=KINDS, rootdir: Optional[str] = None):
        self._kinds = tuple(kinds)
        self._rootdir = rootdir
        self._cache: dict[str, MutableMapping] = {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            collection, inner = key
            return self[collection][inner]
        if key not in self._kinds:
            raise KeyError(f"Unknown store kind {key!r}; known: {self._kinds}")
        if key not in self._cache:
            self._cache[key] = json_store(key, rootdir=self._rootdir)
        return self._cache[key]

    def __iter__(self):
        return iter(self._kinds)

    def __len__(self) -> int:
        return len(self._kinds)


def mall(*, rootdir: Optional[str] = None) -> Mall:
    """The full stores-of-stores mall (one JSON store per :data:`KINDS`)."""
    return Mall(rootdir=rootdir)


def persistent_cache(
    func: Optional[Callable] = None,
    *,
    kind: str = "runs",
    key: Any = None,
    rootdir: Optional[str] = None,
):
    """Memoize a plain function's result in a ``kind`` JSON store, across sessions.

    Use as ``@persistent_cache(kind="runs", key=...)``. Returned values must be
    JSON-serializable. ``key`` may be a constant or a callable receiving the call
    arguments; it defaults to the function's qualified name (so provide a ``key``
    for functions that take arguments). For caching a *class property/method*,
    prefer :data:`cache_this` (dol's descriptor) instead.

    Example:
        >>> import tempfile; root = tempfile.mkdtemp()
        >>> calls = []
        >>> @persistent_cache(kind="runs", key="answer", rootdir=root)
        ... def compute():
        ...     calls.append(1)
        ...     return {"answer": 42}
        >>> compute(), compute(), len(calls)
        ({'answer': 42}, {'answer': 42}, 1)
    """

    def deco(fn: Callable) -> Callable:
        store = json_store(kind, rootdir=rootdir)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if callable(key):
                cache_key = key(*args, **kwargs)
            elif key is not None:
                cache_key = key
            else:
                cache_key = fn.__qualname__
            if cache_key in store:
                return store[cache_key]
            value = fn(*args, **kwargs)
            store[cache_key] = value
            return value

        wrapper.store = store  # exposed for inspection/invalidation
        return wrapper

    return deco(func) if func is not None else deco
