"""Tests for the stdlib-argparse CLI dispatcher (:mod:`ek.__main__`).

``ek`` used to get this from ``argh``; it now derives the same command line from the
same function signatures with nothing but the standard library, so the mapping rules
(dashes, positionals vs options, ``*args``, switches, how a return value is printed)
need tests of their own -- there is no upstream to trust for them any more.
"""

import argparse
import subprocess
import sys

import pytest

from ek.__main__ import (
    _cli_name,
    _help_for_default,
    _summary,
    call_with,
    dispatch_with_namespaces,
    make_parser,
    print_result,
)


# --- the signature -> command-line mapping ------------------------------------------


def plain(alpha, beta):
    """A one-liner summary.

    And a second paragraph that is not the summary.
    """
    return f"{alpha}|{beta}"


def with_options(alpha, *, beta=None, gamma=3):
    """Options come from parameters that have defaults."""
    return f"{alpha}|{beta}|{gamma}"


def var_args(*items, joiner=","):
    """``*args`` becomes a repeatable positional."""
    return joiner.join(items)


def switches(*, off=False, on=True):
    """Bool defaults become switches."""
    return f"{off}|{on}"


def under_scored(some_arg, *, other_arg=None):
    """Underscores become dashes on the command line."""
    return f"{some_arg}|{other_arg}"


ALL_FUNCS = [plain, with_options, var_args, switches, under_scored]


def _run(argv):
    """Parse ``argv`` with the full parser and call whatever it resolved to."""
    parser = make_parser(ALL_FUNCS)
    namespace = parser.parse_args(argv)
    return call_with(getattr(namespace, "_ek_command_func"), namespace)


def test_positional_parameters_are_positional():
    assert _run(["plain", "a", "b"]) == "a|b"


def test_parameters_with_defaults_become_options():
    assert _run(["with-options", "a"]) == "a|None|3"
    assert _run(["with-options", "a", "--beta", "B", "--gamma", "9"]) == "a|B|9"


def test_short_flags_are_assigned_from_the_initial():
    assert _run(["with-options", "a", "-b", "B", "-g", "9"]) == "a|B|9"


def test_short_flag_is_skipped_when_the_initial_is_taken():
    # ``-h`` belongs to argparse, so a ``h*`` parameter gets only its long form.
    def hidden(*, host="local"):
        """Doc."""
        return host

    parser = make_parser([hidden])
    assert parser.parse_args(["hidden", "--host", "x"]).host == "x"
    with pytest.raises(SystemExit):
        parser.parse_args(["hidden", "-h", "x"])


def test_var_positional_collects_the_rest():
    assert _run(["var-args", "x", "y", "z"]) == "x,y,z"
    assert _run(["var-args"]) == ""
    assert _run(["var-args", "x", "y", "--joiner", "-"]) == "x-y"


def test_bool_default_false_is_a_store_true_switch():
    assert _run(["switches"]) == "False|True"
    assert _run(["switches", "--off"]) == "True|True"


def test_bool_default_true_gets_a_no_switch_that_can_turn_it_off():
    # Without ``--no-on`` a default-True switch could never be unset -- the flag
    # would be a no-op, which is what the previous (argh) CLI actually shipped.
    assert _run(["switches", "--no-on"]) == "False|False"
    assert _run(["switches", "--on"]) == "False|True"


def test_underscores_become_dashes_in_commands_and_options():
    assert _run(["under-scored", "a", "--other-arg", "b"]) == "a|b"


def test_cli_name():
    assert _cli_name("pass_k") == "pass-k"
    assert _cli_name("where") == "where"


def test_help_column_renders_defaults_with_repr_and_none_as_dash():
    assert _help_for_default(None) == "-"
    assert _help_for_default(1) == "1"
    assert _help_for_default(True) == "True"
    assert _help_for_default("x") == "'x'"


def test_summary_is_the_first_paragraph_with_percent_escaped():
    assert _summary(plain) == "A one-liner summary."

    def pct():
        """A 90%-reliable agent."""

    # argparse ``%``-formats help strings; an unescaped ``%`` corrupts the listing.
    assert _summary(pct) == "A 90%%-reliable agent."
    parser = make_parser([pct])
    assert "90%-reliable" in parser.format_help()


# --- how a return value reaches stdout ----------------------------------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        (0.5, "0.5\n"),
        ("text", "text\n"),
        ({"a": 1}, "{'a': 1}\n"),  # a dict is one line, not one line per key
        (["a", "b"], "a\nb\n"),
        (("a", "b"), "a\nb\n"),
        (None, ""),  # nothing to say prints nothing
    ],
)
def test_print_result(result, expected, capsys):
    print_result(result)
    assert capsys.readouterr().out == expected


def test_print_result_consumes_a_generator(capsys):
    print_result(x for x in "ab")
    assert capsys.readouterr().out == "a\nb\n"


# --- namespaces and the no-command case ---------------------------------------------


def test_namespaced_commands_dispatch():
    parser = make_parser([plain], {"grouped": [var_args]})
    namespace = parser.parse_args(["grouped", "var-args", "x", "y"])
    assert call_with(getattr(namespace, "_ek_command_func"), namespace) == "x,y"


def test_no_command_prints_usage_and_returns(capsys):
    dispatch_with_namespaces(ALL_FUNCS, argv=[])
    out = capsys.readouterr().out
    assert out.startswith("usage:")
    assert "plain" in out


def test_namespace_without_a_subcommand_prints_that_namespace_usage(capsys):
    dispatch_with_namespaces([plain], {"grouped": [var_args]}, argv=["grouped"])
    assert "grouped" in capsys.readouterr().out


def test_dispatch_prints_the_return_value(capsys):
    dispatch_with_namespaces(ALL_FUNCS, argv=["plain", "a", "b"])
    assert capsys.readouterr().out == "a|b\n"


# --- the real ek CLI, end to end -----------------------------------------------------


def _ek(*args):
    # Deliberately NOT run under `-W error`: that would make this suite fail on any
    # third-party deprecation warning, which says nothing about ek's CLI.
    proc = subprocess.run(
        [sys.executable, "-m", "ek", *args], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize(
    "argv,expected",
    [
        (("cer", "hello wrld", "hello world"), "0.09090909090909091\n"),
        (("wer", "the cat", "the bat"), "0.5\n"),
        (
            ("pass-k", "10", "9", "--k", "8"),
            "{'n': 10, 'c': 9, 'k': 8, 'pass_at_k': 1.0, 'pass_hat_k': 0.2}\n",
        ),
        (
            ("cost-per-success", "12.50", "5", "--attempts", "20"),
            "{'total_dollars': 12.5, 'successes': 5, "
            "'cost_per_success': 2.5, 'success_rate': 0.25}\n",
        ),
    ],
)
def test_real_commands_produce_the_documented_output(argv, expected):
    assert _ek(*argv) == expected


def test_every_dispatch_func_is_reachable_and_documents_itself():
    from ek import tools

    listing = _ek("--help")
    for func in tools._dispatch_funcs:
        name = _cli_name(func.__name__)
        assert name in listing, f"{name} missing from `ek --help`"
        assert _ek(name, "--help").startswith("usage:")


def test_the_cli_says_nothing_on_stderr():
    # argh 0.30+ printed a DeprecationWarning about `pass_k(n, c, k=1)` on every
    # single run; a CLI that greets its user with a stack trace is a bug.
    proc = subprocess.run(
        [sys.executable, "-m", "ek", "pass-k", "10", "9", "--k", "8"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


def test_the_cli_does_not_import_argh():
    # The point of the migration: `argh` is LGPL-3.0-or-later and ek is MIT.
    code = (
        "import sys; sys.argv = ['ek', 'version']\n"
        "import ek.__main__ as m; m.main()\n"
        "assert 'argh' not in sys.modules, 'the CLI still imports argh'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --- signature shapes argh handled that the replacement must keep handling -----------


def test_positional_only_parameters_stay_positional():
    # A positional-only parameter cannot be passed by keyword, so a default must not
    # promote it to an option -- doing so made `call_with` raise TypeError.
    def posonly(alpha=1, /):
        """Doc."""
        return f"alpha={alpha}"

    parser = make_parser([posonly])
    for argv, expected in ((["posonly"], "alpha=1"), (["posonly", "7"], "alpha=7")):
        namespace = parser.parse_args(argv)
        assert call_with(posonly, namespace) == expected


def test_keyword_only_without_a_default_is_a_required_option():
    # argh spelled this `-m MUST`; a positional would be unreachable by name.
    def kwonly(*, must):
        """Doc."""
        return f"must={must}"

    parser = make_parser([kwonly])
    assert call_with(kwonly, parser.parse_args(["kwonly", "--must", "X"])) == "must=X"
    assert call_with(kwonly, parser.parse_args(["kwonly", "-m", "Y"])) == "must=Y"
    with pytest.raises(SystemExit):  # required: omitting it is an error
        parser.parse_args(["kwonly"])


def test_a_contested_initial_gives_neither_option_a_short_flag():
    # argh's rule, and the reason for it: first-declared-wins would make the command
    # line depend on parameter *order*, so reordering a signature would silently move
    # `-c` from one option to another.
    def two_c(*, cost=1, count=2):
        """Doc."""
        return f"{cost}|{count}"

    parser = make_parser([two_c])
    assert call_with(two_c, parser.parse_args(["two-c", "--cost", "9"])) == "9|2"
    with pytest.raises(SystemExit):
        parser.parse_args(["two-c", "-c", "9"])


# --- the real ek CLI's short flags are part of its published surface ----------------

#: Every short flag ek's shipping commands had under argh. Pinned so that editing a
#: signature cannot silently move or drop one.
EK_SHORT_FLAGS = {
    "cer": {"-n"},
    "wer": {"-n"},
    "rover": {"-c"},
    "pass-k": {"-k"},
    "cost-per-success": {"-a"},
    "where": set(),
    "check": set(),
    "engines": set(),
    "version": set(),
}


def test_the_real_commands_short_flags_are_unchanged():
    from ek import tools

    parser = make_parser(tools._dispatch_funcs)
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == set(EK_SHORT_FLAGS), "command set changed"
    for command, expected in EK_SHORT_FLAGS.items():
        found = {
            flag
            for action in subparsers.choices[command]._actions
            for flag in action.option_strings
            if len(flag) == 2 and flag != "-h"
        }
        assert found == expected, f"{command}: short flags moved {found} != {expected}"
