"""Strategy registry, optional-dependency guarding, and requirement checks.

``ek`` is open-closed: every swappable behaviour (a :class:`~ek.base.Metric`, a
:class:`~ek.base.Signal`, a calibrator, an OCR backend, ...) is registered under a
*namespace* and resolved by name. Built-in strategies register themselves at
import time; third parties add their own either by calling :func:`register` or by
declaring an entry point in the ``ek.<namespace>`` group -- no edit to ``ek`` needed.

Two more concerns live here so they stay out of the core logic:

- :func:`requires_extra` -- a decorator that turns a missing optional dependency
  into an *actionable* error (``pip install ek[ocr]``) instead of an opaque
  ``ImportError`` raised deep inside a call.
- :func:`check_requirements` -- a helper that reports, without raising, what an
  engine/feature needs (Python packages, system binaries, credentials), and
  defers to ``ocracy``'s own doctor for OCR engines when it is installed.

Example:
    >>> @register("metrics", "shout")
    ... def shout(pred, gold, *, grammar=None):
    ...     return pred == gold
    >>> get("metrics", "shout")("a", "a")
    True
    >>> "shout" in names("metrics")
    True
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Callable, Iterable, Optional

# namespace -> {name: object}. Populated by register() and entry-point discovery.
_REGISTRY: dict[str, dict[str, Any]] = {}
_DISCOVERED: set[str] = set()


def register(namespace: str, name: str, obj: Optional[Any] = None) -> Any:
    """Register ``obj`` under ``namespace``/``name`` (usable as a decorator).

    Args:
        namespace: A strategy family, e.g. ``"metrics"``, ``"signals"``,
            ``"calibrators"``, ``"policies"``, ``"normalizers"``, ``"ocr"``.
        name: The lookup key within the namespace.
        obj: The object to register; omit to use as a decorator.

    Returns:
        ``obj`` (so it can be used as a decorator).
    """
    bucket = _REGISTRY.setdefault(namespace, {})

    def _do(target: Any) -> Any:
        bucket[name] = target
        return target

    return _do if obj is None else _do(obj)


def get(namespace: str, name: str) -> Any:
    """Resolve a registered strategy by name (discovering entry points first).

    Raises:
        KeyError: with the list of available names, if ``name`` is unknown.
    """
    _discover(namespace)
    bucket = _REGISTRY.get(namespace, {})
    if name not in bucket:
        available = sorted(bucket)
        raise KeyError(
            f"No {namespace!r} strategy named {name!r}. Available: {available}"
        )
    return bucket[name]


def names(namespace: str) -> Iterable[str]:
    """All registered names in a namespace (after entry-point discovery)."""
    _discover(namespace)
    return list(_REGISTRY.get(namespace, {}))


def resolve(namespace: str, ref: Any, *, default: Optional[Any] = None) -> Any:
    """Coerce a strategy reference to a callable.

    ``ref`` may be a registered name (``str``), an already-resolved object, or
    ``None`` (in which case ``default``, itself a name or object, is used).
    """
    if ref is None:
        ref = default
    if isinstance(ref, str):
        return get(namespace, ref)
    return ref


def _discover(namespace: str) -> None:
    """Lazily import third-party strategies declared in the ``ek.<ns>`` group."""
    if namespace in _DISCOVERED:
        return
    _DISCOVERED.add(namespace)
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = f"ek.{namespace}"
        # Python >=3.10 always exposes EntryPoints.select(); the .get fallback is
        # only defensive for unusual backports.
        selected = (
            eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        )
        for ep in selected:
            try:
                register(namespace, ep.name, ep.load())
            except Exception:  # a broken third-party plugin must not break ek
                continue
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Optional-dependency guarding
# ---------------------------------------------------------------------------


class MissingExtraError(ImportError):
    """Raised when an optional feature is used without its extra installed."""


def requires_extra(extra: str, *, packages: Optional[Iterable[str]] = None) -> Callable:
    """Decorator: fail with an actionable install hint if an extra is missing.

    The wrapped callable runs only once every package in ``packages`` imports; the
    common case (everything installed) adds a single cheap import check.

    Args:
        extra: The extra name, used in the hint ``pip install ek[<extra>]``.
        packages: Import names to probe (defaults to ``[extra]``).

    Example:
        >>> @requires_extra("ocr", packages=["definitely_not_installed_pkg"])
        ... def run():
        ...     return "ran"
        >>> try:
        ...     run()
        ... except MissingExtraError as e:
        ...     print("ek[ocr]" in str(e))
        True
    """
    probe = list(packages) if packages is not None else [extra]

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            missing = [p for p in probe if not _can_import(p)]
            if missing:
                raise MissingExtraError(
                    f"{fn.__qualname__} needs optional dependencies "
                    f"{missing} -- install them with:  pip install ek[{extra}]"
                )
            return fn(*args, **kwargs)

        return wrapper

    return deco


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Requirement checks (report, don't raise)
# ---------------------------------------------------------------------------

# Extra name -> representative import name(s) to probe (the extra name is rarely
# the import name; e.g. ek[ocr] installs `ocracy`).
_EXTRA_PROBES: dict[str, list[str]] = {
    "ocr": ["ocracy"],
    "metrics": ["networkx", "anls_star", "seqeval", "nervaluate", "apted"],
    "calibration": ["netcal", "sklearn"],
    "validation": ["pydantic", "jsonschema"],
    "constrained": ["outlines"],
    "agreement": ["uqlm"],
    "hitl": ["label_studio_sdk"],
    "monitoring": ["nannyml"],
}


def check_requirements(
    *, engine: Optional[str] = None, extra: Optional[str] = None
) -> dict:
    """Report what an OCR engine or feature extra needs, without raising.

    For an OCR ``engine`` and an installed ``ocracy``, defers to ``ocracy``'s own
    doctor (which knows each backend's pip package, system binaries and
    credentials). Otherwise returns a generic hint.

    Returns:
        A dict with at least ``{"ok": bool, "hint": str}``.
    """
    if engine is not None:
        try:
            import ocracy  # type: ignore

            ok = bool(ocracy.check(engine))
            doctor = ocracy.doctor()
            hint = "" if ok else doctor.get("missing", {}).get(engine, "")
            return {"ok": ok, "engine": engine, "hint": hint, "via": "ocracy"}
        except ImportError:
            return {
                "ok": False,
                "engine": engine,
                "hint": "Install the OCR fleet:  pip install ek[ocr]",
                "via": "none",
            }
    if extra is not None:
        probe = _EXTRA_PROBES.get(extra, [extra])
        missing = [p for p in probe if not _can_import(p)]
        ok = not missing
        return {
            "ok": ok,
            "extra": extra,
            "missing": missing,
            "hint": "" if ok else f"pip install ek[{extra}]  (missing: {', '.join(missing)})",
        }
    return {"ok": True, "hint": ""}
