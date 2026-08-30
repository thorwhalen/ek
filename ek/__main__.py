# PYTHON_ARGCOMPLETE_OK
"""Command-line entry point for ``ek`` (``python -m ek`` / the ``ek`` script).

Commands come from :data:`ek.tools._dispatch_funcs` (the SSOT). A function's
*signature* is its command-line interface, so there is no duplicated command
registration and no argument declarations to keep in sync with the code:

===============================  =========================================
Python parameter                 Command-line form
===============================  =========================================
``f(x)``                         positional ``x``
``f(*xs)``                       repeatable positional ``[xs ...]``
``f(*, x=None)`` / ``f(x=None)`` option ``-x``/``--x`` (default in --help)
``f(*, flag=False)``             switch ``--flag``
``f(*, flag=True)``              switch ``--no-flag`` (``--flag`` still ok)
===============================  =========================================

Underscores become dashes (``pass_k`` -> ``pass-k``); a command's docstring is
its ``--help`` text. Return values are printed: a list/tuple/iterator one item
per line, anything else on a single line.

This is built on :mod:`argparse` from the standard library, so ek's core carries
no CLI dependency at all. It replaces ``argh``, which is LGPL-3.0-or-later --
ek is MIT and its licence gate (``.github/scripts/check_licenses.py``) now
rejects the whole GPL family, LGPL included.

The ``# PYTHON_ARGCOMPLETE_OK`` marker on line 1 enables shell tab-completion
via ``argcomplete`` (after the user activates it).
"""

from __future__ import annotations

import argparse
import inspect
from typing import Any, Callable, Iterable, Mapping, Optional

#: Where the parser stashes the function a parsed command line resolved to.
_FUNC_DEST = "_ek_command_func"

#: Where it stashes the parser whose usage to show when no command was given.
_PARSER_DEST = "_ek_command_parser"

_EMPTY = inspect.Parameter.empty

#: Short flags that are never auto-assigned to a parameter (argparse owns ``-h``).
_RESERVED_SHORT_FLAGS = frozenset("h")


def _cli_name(python_name: str) -> str:
    """Command/argument name as it is spelled on the command line.

    >>> _cli_name('pass_k')
    'pass-k'
    """
    return python_name.replace("_", "-")


def _help_for_default(default: Any) -> str:
    """The help column: a default rendered with ``repr``, ``None`` as ``'-'``.

    >>> _help_for_default(1), _help_for_default(None), _help_for_default(True)
    ('1', '-', 'True')
    """
    return "-" if default is None else repr(default)


def _summary(func: Callable) -> str:
    """First docstring paragraph, as one line, safe to hand to argparse.

    argparse ``%``-formats help strings, so a literal ``%`` (``"90%-reliable"``)
    must be escaped or it is read as a format spec.
    """
    doc = inspect.getdoc(func) or ""
    first_paragraph = doc.split("\n\n", 1)[0]
    return " ".join(first_paragraph.split()).replace("%", "%%")


def _add_positional(parser: argparse.ArgumentParser, param: inspect.Parameter) -> None:
    """Add ``param`` as a positional argument (``*args`` becomes ``[name ...]``)."""
    nargs = {"nargs": "*"} if param.kind is param.VAR_POSITIONAL else {}
    parser.add_argument(param.name, metavar=_cli_name(param.name), help="-", **nargs)


def _add_option(
    parser: argparse.ArgumentParser,
    param: inspect.Parameter,
    taken_short_flags: set,
) -> None:
    """Add ``param`` as an option, with a short flag when its initial is free."""
    long_name = _cli_name(param.name)
    flags = []
    if (initial := param.name[0]) not in taken_short_flags:
        taken_short_flags.add(initial)
        flags.append(f"-{initial}")
    flags.append(f"--{long_name}")

    default = param.default
    if isinstance(default, bool):
        parser.add_argument(
            *flags,
            dest=param.name,
            action="store_true",
            default=default,
            help=_help_for_default(default),
        )
        if default is True:
            # Without this a default-True switch could never be turned off.
            parser.add_argument(
                f"--no-{long_name}",
                dest=param.name,
                action="store_false",
                help=f"unset --{long_name}",
            )
    else:
        # No ``type=``: commands take strings and do their own coercion, so a
        # command's Python signature stays the SSOT of what it accepts.
        parser.add_argument(
            *flags, dest=param.name, default=default, help=_help_for_default(default)
        )


def _is_option(param: inspect.Parameter) -> bool:
    """Whether ``param`` maps to an option rather than a positional argument."""
    return param.default is not _EMPTY


def add_command(subparsers: Any, func: Callable) -> argparse.ArgumentParser:
    """Register ``func`` as a subcommand, deriving its arguments from its signature."""
    parser = subparsers.add_parser(
        _cli_name(func.__name__),
        help=_summary(func),
        description=inspect.getdoc(func) or "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    taken_short_flags = set(_RESERVED_SHORT_FLAGS)
    for param in inspect.signature(func).parameters.values():
        if param.kind is param.VAR_KEYWORD:
            continue  # ``**kwargs`` has no command-line spelling
        if _is_option(param):
            _add_option(parser, param, taken_short_flags)
        else:
            _add_positional(parser, param)
    parser.set_defaults(**{_FUNC_DEST: func})
    return parser


def make_parser(
    functions: Iterable[Callable],
    namespaced_funcs: Optional[Mapping[str, Iterable[Callable]]] = None,
    **parser_kwargs,
) -> argparse.ArgumentParser:
    """Build the ``ek`` parser from top-level functions and namespaced groups."""
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.set_defaults(**{_FUNC_DEST: None, _PARSER_DEST: parser})
    subparsers = parser.add_subparsers()
    for func in functions:
        add_command(subparsers, func)
    for namespace, funcs in (namespaced_funcs or {}).items():
        group = subparsers.add_parser(_cli_name(namespace))
        group_subparsers = group.add_subparsers()
        for func in funcs:
            add_command(group_subparsers, func)
        group.set_defaults(**{_FUNC_DEST: None, _PARSER_DEST: group})
    return parser


def call_with(func: Callable, namespace: argparse.Namespace) -> Any:
    """Call ``func`` with the parsed ``namespace``, honouring ``*args`` and kwonly."""
    parsed = vars(namespace)
    args, kwargs = [], {}
    for name, param in inspect.signature(func).parameters.items():
        if param.kind is param.VAR_KEYWORD:
            continue
        value = parsed.get(name, param.default)
        if param.kind is param.VAR_POSITIONAL:
            args.extend(value or ())
        elif _is_option(param) or param.kind is param.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    return func(*args, **kwargs)


def print_result(result: Any) -> None:
    """Print a command's return value: a sequence one item per line, else one line."""
    if result is None:
        return
    if isinstance(result, (list, tuple)) or inspect.isgenerator(result):
        for line in result:
            print(line)
    else:
        print(result)


def dispatch_with_namespaces(
    functions: Iterable[Callable],
    namespaced_funcs: Optional[Mapping[str, Iterable[Callable]]] = None,
    *,
    argv: Optional[Iterable[str]] = None,
) -> None:
    """Build a parser from top-level and namespaced functions and dispatch ``argv``."""
    parser = make_parser(functions, namespaced_funcs)
    try:  # tab completion is best-effort
        import argcomplete

        argcomplete.autocomplete(parser)
    except Exception:
        pass
    namespace = parser.parse_args(None if argv is None else list(argv))
    func = getattr(namespace, _FUNC_DEST, None)
    if func is None:  # no (sub)command given -- same as ``argh``: show usage, exit 0
        getattr(namespace, _PARSER_DEST, parser).print_usage()
        return
    print_result(call_with(func, namespace))


def main() -> None:
    """Entry point registered as the ``ek`` console script."""
    from . import tools

    dispatch_with_namespaces(tools._dispatch_funcs)


if __name__ == "__main__":
    main()
