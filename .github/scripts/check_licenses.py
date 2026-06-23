"""Fail the build if a copyleft / non-commercial license is in ek's dep closure.

License landmines in this ecosystem hide their terms in repo files, invisible to
PyPI metadata scanners (e.g. TorchCP is LGPL with a blank PyPI license field;
surya-ocr ships non-commercial RAIL-M weights behind an "Apache-2.0" classifier).
This gate reads a ``pip-licenses`` CSV of the *installed* closure and rejects:

- GPL / AGPL (but allows LGPL -- acceptable for dynamically-linked libraries)
- Any non-commercial / source-available restriction (RAIL, CC-BY-NC, BUSL, ...)

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
)

# Packages explicitly cleared despite a scary-looking or blank license field.
# Keep this list short and justified; it is the audited override.
_ALLOWLIST: set[str] = set()

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
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or "").strip()
            license_text = (row.get("License") or "").strip()
            if name in _ALLOWLIST or name.lower().startswith(_ALLOWLIST_PREFIXES):
                continue
            reason = _is_violation(license_text)
            if reason:
                violations.append((name, license_text, reason))

    if violations:
        print("License gate FAILED -- forbidden licenses in the dependency closure:")
        for name, lic, reason in violations:
            print(f"  - {name}: {lic}  [{reason}]")
        print(
            "\nQuarantine these behind an explicit, opt-in install (never a default "
            "extra). See skills/ek-dev-licensing."
        )
        return 1
    print("License gate passed: no copyleft/non-commercial licenses in the closure.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "licenses.csv"))
