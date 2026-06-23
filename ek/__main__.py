# PYTHON_ARGCOMPLETE_OK
"""Command-line entry point for ``ek`` (``python -m ek`` / the ``ek`` script).

Commands come from :data:`ek.tools._dispatch_funcs` (the SSOT), dispatched with
``argh`` -- which reads each function's signature to build its flags -- and
optionally completed with ``argcomplete``. The ``# PYTHON_ARGCOMPLETE_OK`` marker
on line 1 enables shell tab-completion (after the user activates it).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional


def dispatch_with_namespaces(
    functions: Iterable, namespaced_funcs: Optional[Mapping[str, Iterable]] = None
) -> None:
    """Build an ``argh`` parser from top-level and namespaced functions and dispatch."""
    import argh

    parser = argh.ArghParser()
    argh.add_commands(parser, list(functions))
    for namespace, funcs in (namespaced_funcs or {}).items():
        argh.add_commands(parser, list(funcs), namespace=namespace)
    try:  # tab completion is best-effort
        import argcomplete

        argcomplete.autocomplete(parser)
    except Exception:
        pass
    parser.dispatch()


def main() -> None:
    """Entry point registered as the ``ek`` console script."""
    from . import tools

    dispatch_with_namespaces(tools._dispatch_funcs)


if __name__ == "__main__":
    main()
