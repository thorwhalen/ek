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


def test_lgpl_is_allowed():
    assert cl._is_violation("LGPL-3.0") == ""
    assert cl._is_violation("GNU Lesser General Public License v3") == ""


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
