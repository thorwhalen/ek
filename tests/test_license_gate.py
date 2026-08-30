"""Tests for the CI license gate logic (.github/scripts/check_licenses.py)."""

import csv
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_licenses.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_licenses", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cl = _load()


def _csv(tmp_path, rows):
    path = tmp_path / "licenses.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Version", "License"])
        for name, lic in rows:
            w.writerow([name, "1.0", lic])
    return str(path)


def test_nvidia_cuda_runtime_is_allowlisted(tmp_path):
    # torch's transitive GPU runtime wheels (proprietary) are cleared by prefix.
    path = _csv(tmp_path, [
        ("nvidia-cublas-cu13", "LicenseRef-NVIDIA-Proprietary"),
        ("nvidia-cuda-runtime-cu13", "Other/Proprietary License"),
        ("nvidia_cudnn_cu13", "NVIDIA Proprietary Software"),
        ("torch", "BSD-3-Clause"),
    ])
    assert cl.main(path) == 0


def test_gpl_is_still_rejected(tmp_path):
    # the gate must still catch a real copyleft dep (e.g. the dropped krippendorff).
    assert cl.main(_csv(tmp_path, [("krippendorff", "GPL-3.0-or-later")])) == 1


def test_lgpl_is_rejected():
    # The LGPL escape hatch is gone: ek is MIT and pure-Python, so the
    # "dynamically-linked library" rationale for tolerating LGPL never applied.
    # These are the real strings seen in the wild, not invented ones.
    assert cl._is_violation("LGPL-3.0")  # argh's SPDX expression
    assert cl._is_violation("GNU Lesser General Public License v3")
    # argh's and PyGithub's actual trove classifier -- caught via "GENERAL PUBLIC".
    assert cl._is_violation(
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)"
    )
    assert cl._is_violation("LGPL-2.1-or-later")  # soxr's actual expression


def test_spelled_out_gpl_family_is_rejected():
    # Regression guard for a hole the LGPL allowance was hiding: the fully spelled-out
    # names contain no "GPL" substring AND do not say "GNU General Public" (they say
    # "GNU *Lesser* / *Affero* General Public"), so the old pattern pair missed them
    # entirely -- AGPL, the most viral licence of all, included.
    assert cl._is_violation("GNU Lesser General Public License v3")
    assert cl._is_violation("GNU Affero General Public License v3")
    assert cl._is_violation("GNU General Public License v2 or later")


def test_argh_the_dependency_this_gate_used_to_let_through(tmp_path):
    # Regression guard for the hole this gate was built with: ek's own CLI dep.
    assert cl.main(
        _csv(
            tmp_path,
            [
                (
                    "argh",
                    "GNU Library or Lesser General Public License (LGPL)",
                )
            ],
        )
    ) == 1


def test_non_commercial_and_proprietary_rejected():
    assert cl._is_violation("CC-BY-NC-4.0")
    assert cl._is_violation("Apache-2.0 with RAIL-M restriction")
    assert cl._is_violation("Business Source License")


def test_permissive_closure_passes(tmp_path):
    path = _csv(tmp_path, [
        ("rapidfuzz", "MIT"),
        ("jiwer", "Apache-2.0"),
        ("networkx", "BSD-3-Clause"),
        ("nvidia-cufft-cu13", "Other/Proprietary License"),  # allowlisted
    ])
    assert cl.main(path) == 0


def test_packages_with_no_declared_license_fail_unless_audited(tmp_path):
    # A blank License field is *unaudited*, not fine -- it is how copyleft hides.
    assert cl.main(_csv(tmp_path, [("mystery-pkg", "UNKNOWN")])) == 1
    # ...unless it is in the audited allowlist. ragas 0.4.x inlines the whole
    # Apache-2.0 text into its License field with no classifier, which some
    # pip-licenses versions render as UNKNOWN.
    assert cl.main(_csv(tmp_path, [("ragas", "UNKNOWN")])) == 0


def test_mpl_and_epl_full_text_are_not_read_as_gpl():
    # MPL-2.0 and EPL-2.0 both name the GPL family in their "Secondary Licenses"
    # clause. A package that inlines the full licence text into its `License` field
    # (the `license = {file = "LICENSE"}` packaging mistake -- ragas does exactly
    # this with Apache-2.0) would otherwise be failed by the broad "GENERAL PUBLIC"
    # pattern for a licence ek accepts.
    mpl = (
        "Mozilla Public License Version 2.0 ... 3.3. Distribution of a Larger Work ... "
        'under the terms of a Secondary License: the GNU General Public License, '
        "Version 2.0, the GNU Lesser General Public License, Version 2.1, the GNU "
        "Affero General Public License, Version 3.0 ..."
    )
    epl = (
        "Eclipse Public License - v 2.0 ... Secondary License means either the GNU "
        "General Public License, Version 2.0, or any later versions ..."
    )
    assert cl._is_violation(mpl) == ""
    assert cl._is_violation(epl) == ""
    # short declarations of the same licences were always fine, and still are
    assert cl._is_violation("MPL-2.0") == ""
    assert cl._is_violation("MPL-2.0 AND MIT") == ""  # tqdm's actual expression
    # and the clearance is title-gated, not a hole: real copyleft still fails
    assert cl._is_violation("GNU General Public License v3") != ""


def test_unknown_clearance_does_not_also_clear_copyleft(tmp_path):
    # `ragas` is cleared for a *blank* licence field only. If it ever starts
    # declaring copyleft, the gate must still fail -- an audited-unknown row is not
    # a blanket exemption from every rule.
    assert cl.main(_csv(tmp_path, [("ragas", "UNKNOWN")])) == 0
    assert cl.main(_csv(tmp_path, [("ragas", "AGPL-3.0")])) == 1
    assert cl.main(_csv(tmp_path, [("zss", "GPL-3.0-or-later")])) == 1
    # the strong override (NVIDIA runtime wheels, cuda-toolkit) still bypasses all rules
    assert cl.main(_csv(tmp_path, [("cuda-toolkit", "Other/Proprietary License")])) == 0


def test_the_gate_scripts_doctests_run():
    # .github/scripts/ is outside pytest's testpaths, so --doctest-modules never
    # reaches check_licenses.py. Run its doctests from inside the suite instead, so
    # the examples in _is_violation stay executable rather than decorative.
    import doctest

    results = doctest.testmod(cl, verbose=False)
    assert results.attempted > 0, "no doctests found in check_licenses.py"
    assert results.failed == 0
