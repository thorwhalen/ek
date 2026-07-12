"""Fail the build if a copyleft / non-commercial license is in ek's dep closure.

License landmines in this ecosystem hide their terms in repo files, invisible to
PyPI metadata scanners (e.g. TorchCP is LGPL with a blank PyPI license field;
surya-ocr ships non-commercial RAIL-M weights behind an "Apache-2.0" classifier).
This gate reads a ``pip-licenses`` CSV of the *installed* closure and rejects:

- GPL / AGPL (but allows LGPL -- acceptable for dynamically-linked libraries)
- Any non-commercial / source-available restriction (RAIL, CC-BY-NC, BUSL, SSPL,
  **Elastic-2.0**, ...)

Note on Elastic License 2.0 (the agent-eval-era trap, see ``misc/docs/ek_12``): Arize
Phoenix is ELv2 -- source-available, *not* OSI-approved, and it forbids offering the
software "to third parties as a hosted or managed service". It is not copyleft, but it
would pollute ek's permissive-core story, so it is quarantined (HTTP-only, never a
default dependency). Crucially, Phoenix declares ELv2 in its ``License`` metadata field
but ships **no** ``License ::`` trove classifier -- so a gate keyed only off classifiers
sails straight past it. This gate reads the ``License`` field, which is why it catches it.

Usage:
    pip-licenses --format=csv --with-system > licenses.csv
    python .github/scripts/check_licenses.py licenses.csv

Exit code 1 (with the offending rows printed) on any violation.
"""

from __future__ import annotations

import csv
import sys

# Substrings that mark a forbidden license (matched case-insensitively).
_GPL = ("GPL", "GNU GENERAL PUBLIC")
_GPL_ALLOW = ("LGPL", "LESSER")  # LGPL is permitted
_NON_COMMERCIAL = (
    "NON-COMMERCIAL",
    "NONCOMMERCIAL",
    "NON COMMERCIAL",
    "CC-BY-NC",
    "CC BY-NC",
    "RAIL",
    "BUSL",
    "BUSINESS SOURCE",
    "PROPRIETARY",
    "SSPL",
    # Source-available, not OSI: forbids offering the software as a hosted/managed
    # service. Arize Phoenix ships this in its License field with NO trove classifier,
    # so a classifier-only gate misses it entirely (see the module docstring).
    "ELASTIC-2.0",
    "ELASTIC LICENSE",
    "ELASTICV2",
)

# Packages explicitly cleared despite a scary-looking or blank license field.
# Keep this list short and justified; it is the audited override.
_ALLOWLIST: set[str] = {
    # Pulled transitively by inspect-ai (the ek[agents] task-suite runner). Its metadata
    # `License` field is EMPTY, so pip-licenses reports "UNKNOWN" -- the classic
    # scanner-invisible pattern. Audited 2026-07: the wheel ships the full Apache-2.0 text at
    # dist-info/licenses/LICENSE (Zed Industries' Agent Client Protocol). Permissive; cleared.
    "agent-client-protocol",
    # Tree edit distance, used by the TEDS table metric (ek[metrics]). Declares no license in
    # its PyPI metadata; its terms live in a repo file -- the scanner-invisible case ek's own
    # licensing register already names. Audited 2026-07: BSD-3-Clause. Permissive; cleared.
    "zss",
    # NVIDIA's redistributable CUDA *runtime*, pulled transitively by torch (BSD) when an extra
    # needs it. Same audited justification as the nvidia-* prefixes below: a hardware-driver
    # runtime the end user installs for acceleration, not a copyleft/non-commercial library ek
    # ships. A CPU-only install omits it entirely. Audited 2026-07; cleared.
    "cuda-toolkit",
}

# A blank/UNKNOWN license field is not "fine", it is *unaudited* -- the terms may live in a
# repo file the scanner never reads (this is exactly how TorchCP's LGPL and surya-ocr's
# non-commercial weights hide). This is a HARD FAILURE, not a notice: a warning that still
# exits 0 is precisely the hiding place we are trying to close -- nobody reads a green build's
# log. Clear a package by reading its actual LICENSE file and adding it to _ALLOWLIST above
# with a dated justification.
_UNKNOWN = ("UNKNOWN", "", "NONE")

# Name *prefixes* cleared as an audited override. The NVIDIA CUDA runtime wheels
# (nvidia-cublas, nvidia-cudnn, nvidia-cuda-*, ...) are pulled transitively by the
# permissive `torch` (BSD) when an extra needs it (e.g. uqlm in [agreement]). They
# carry an NVIDIA "proprietary" license field, but they are NVIDIA's redistributable
# GPU *runtime* -- hardware-driver libraries the end user installs for acceleration,
# not a copyleft/non-commercial library ek ships. A CPU-only install omits them
# entirely. They are not a redistribution-license risk, so they are cleared here.
_ALLOWLIST_PREFIXES: tuple[str, ...] = ("nvidia-", "nvidia_")


def _is_violation(license_text: str) -> str:
    up = license_text.upper()
    if any(nc in up for nc in _NON_COMMERCIAL):
        return "non-commercial / source-available"
    if any(g in up for g in _GPL) and not any(a in up for a in _GPL_ALLOW):
        return "GPL/AGPL copyleft"
    return ""


def main(path: str) -> int:
    violations = []
    unaudited = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or "").strip()
            license_text = (row.get("License") or "").strip()
            if name in _ALLOWLIST or name.lower().startswith(_ALLOWLIST_PREFIXES):
                continue
            reason = _is_violation(license_text)
            if reason:
                violations.append((name, license_text, reason))
            elif license_text.upper() in _UNKNOWN:
                unaudited.append(name)

    if violations:
        print("License gate FAILED -- forbidden licenses in the dependency closure:")
        for name, lic, reason in violations:
            print(f"  - {name}: {lic}  [{reason}]")
        print(
            "\nQuarantine these behind an explicit, opt-in install (never a default "
            "extra). See skills/ek-dev-licensing."
        )

    if unaudited:
        print(
            "License gate FAILED -- packages declaring NO license in their metadata. The terms "
            "may live in a repo/wheel file the scanner cannot see, which is exactly how a "
            "copyleft or non-commercial dependency hides:"
        )
        for name in unaudited:
            print(f"  - {name}")
        print(
            "\nRead each one's actual LICENSE file. If permissive, add it to _ALLOWLIST in this "
            "script with a dated justification; if not, quarantine it behind an opt-in extra."
        )

    if violations or unaudited:
        return 1
    print("License gate passed: no copyleft/non-commercial/unaudited licenses in the closure.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "licenses.csv"))
