"""Fail the build if a copyleft / non-commercial license is in ek's dep closure.

License landmines in this ecosystem hide their terms in repo files, invisible to
PyPI metadata scanners (e.g. TorchCP is LGPL with a blank PyPI license field;
surya-ocr ships non-commercial RAIL-M weights behind an "Apache-2.0" classifier).
This gate reads a ``pip-licenses`` CSV of the *installed* closure and rejects:

- The whole GPL family -- GPL, AGPL **and LGPL**. The gate used to allow LGPL on the
  "acceptable for a dynamically-linked library" argument, but that argument does not
  describe a pure-Python import: a `pip install` of an LGPL package puts its source in
  the same interpreter as ek's, and the relinking right the LGPL trades for is
  meaningless there. ek is MIT and its closure stays permissive, so LGPL is a violation
  like any other copyleft (this is what removing the `argh` dependency bought).
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
# The whole GPL family, in every spelling seen in the wild. There is no LGPL escape
# hatch (there used to be, and ek's own `argh` dependency was what fit through it).
#
# Two patterns, because neither alone is enough:
#   "GPL"            catches the abbreviations -- GPL-3.0, LGPL-2.1-or-later, AGPL-3.0,
#                    and the "(LGPL)" tail of the trove classifiers.
#   "GENERAL PUBLIC" catches the spelled-out names, which contain no "GPL" substring at
#                    all: "GNU Lesser General Public License v3" and "GNU Affero General
#                    Public License v3" both sailed straight past the previous
#                    "GNU GENERAL PUBLIC" pattern, since neither says *GNU General*.
#
# The broad "GENERAL PUBLIC" pattern DOES have one false-positive class, so do not trust
# it unqualified: MPL-2.0 and EPL-2.0 both name the GPL/LGPL/AGPL in their
# "Secondary Licenses" clause, so a package that inlines either licence's *full text*
# into its `License` metadata field (the `license = {file = "LICENSE"}` packaging
# mistake -- see `ragas` below for a real instance of that shape) matches it. Those two
# are permissive-enough weak-copyleft licences ek accepts, so they are recognised and
# cleared before the GPL patterns run. See ``_SECONDARY_LICENCE_TEXTS``.
_GPL = ("GPL", "GENERAL PUBLIC")

# Licences whose own text *mentions* the GPL family without being it. Matching any of
# these titles means the field holds a full licence text, not a declaration, and the
# GPL patterns below would be reading that licence's compatibility clause.
_SECONDARY_LICENCE_TEXTS = (
    "MOZILLA PUBLIC LICENSE",
    "ECLIPSE PUBLIC LICENSE",
)
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

# Packages cleared for the *blank/UNKNOWN* license field only. Each was audited by
# reading the LICENSE file the wheel actually ships. A name here is NOT cleared for
# copyleft or non-commercial terms: if one of these ever starts declaring a forbidden
# licence, the gate still fails on it, which is the point -- a blanket override would
# silence the very rule the audit was about.
_CLEARED_UNKNOWN: set[str] = {
    # Pulled transitively by inspect-ai (the ek[agents] task-suite runner). Its metadata
    # `License` field is EMPTY, so pip-licenses reports "UNKNOWN" -- the classic
    # scanner-invisible pattern. Audited 2026-07: the wheel ships the full Apache-2.0 text at
    # dist-info/licenses/LICENSE (Zed Industries' Agent Client Protocol). Permissive; cleared.
    "agent-client-protocol",
    # Tree edit distance, used by the TEDS table metric (ek[metrics]). Declares no license in
    # its PyPI metadata; its terms live in a repo file -- the scanner-invisible case ek's own
    # licensing register already names. Audited 2026-07: BSD-3-Clause. Permissive; cleared.
    "zss",
    # The agent-eval harness in ek[agents]. Audited 2026-08 (ragas 0.4.3): it declares NO
    # `License-Expression` and NO `License ::` trove classifier, and inlines the *entire*
    # 12,921-character Apache-2.0 text into its `License` metadata field (the packaging
    # mistake `license = {file = "LICENSE"}` makes). Depending on its version,
    # pip-licenses renders that as "UNKNOWN" -- the same scanner-invisible shape as
    # agent-client-protocol above, and it is why this row started failing the gate on a
    # closure that had not otherwise changed. The wheel ships the real thing at
    # ragas-0.4.3.dist-info/licenses/LICENSE: Apache-2.0 (Copyright 2023 Vibrant Labs),
    # matching what pyproject.toml already recorded for it in 2026-07. Permissive; cleared.
    "ragas",
}

# Packages cleared for a *forbidden-looking* license field. This is the strong override
# -- it bypasses every rule -- so it stays as small as the audit allows.
_CLEARED_FORBIDDEN: set[str] = {
    # NVIDIA's redistributable CUDA *runtime*, pulled transitively by torch (BSD) when an extra
    # needs it. Same audited justification as the nvidia-* prefixes below: a hardware-driver
    # runtime the end user installs for acceleration, not a copyleft/non-commercial library ek
    # ships. A CPU-only install omits it entirely. Its license field says "proprietary", which
    # is why it needs the strong override rather than the unknown-only one.
    # Audited 2026-07; cleared.
    "cuda-toolkit",
}

# A blank/UNKNOWN license field is not "fine", it is *unaudited* -- the terms may live in a
# repo file the scanner never reads (this is exactly how TorchCP's LGPL and surya-ocr's
# non-commercial weights hide). This is a HARD FAILURE, not a notice: a warning that still
# exits 0 is precisely the hiding place we are trying to close -- nobody reads a green build's
# log. Clear a package by reading its actual LICENSE file and adding it to _CLEARED_UNKNOWN
# above with a dated justification.
_UNKNOWN = ("UNKNOWN", "", "NONE")

# Name *prefixes* cleared as an audited override. The NVIDIA CUDA runtime wheels
# (nvidia-cublas, nvidia-cudnn, nvidia-cuda-*, ...) are pulled transitively by the
# permissive `torch` (BSD) when an extra needs it (e.g. uqlm in [agreement]). They
# carry an NVIDIA "proprietary" license field, but they are NVIDIA's redistributable
# GPU *runtime* -- hardware-driver libraries the end user installs for acceleration,
# not a copyleft/non-commercial library ek ships. A CPU-only install omits them
# entirely. They are not a redistribution-license risk, so they are cleared here.
_CLEARED_FORBIDDEN_PREFIXES: tuple[str, ...] = ("nvidia-", "nvidia_")


def _is_violation(license_text: str) -> str:
    """Return why ``license_text`` is forbidden, or ``""`` if it is acceptable.

    >>> _is_violation("MIT")
    ''
    >>> _is_violation("LGPL-3.0")
    'GPL/LGPL/AGPL copyleft'
    >>> _is_violation("GNU Affero General Public License v3")
    'GPL/LGPL/AGPL copyleft'

    A field holding the *full text* of MPL-2.0 or EPL-2.0 is not a violation, even
    though both name the GPL in their "Secondary Licenses" clause:

    >>> _is_violation("Mozilla Public License Version 2.0 ... GNU General Public License")
    ''
    """
    up = license_text.upper()
    if any(nc in up for nc in _NON_COMMERCIAL):
        return "non-commercial / source-available"
    if any(title in up for title in _SECONDARY_LICENCE_TEXTS):
        return ""  # MPL/EPL full text: its "Secondary Licenses" clause names the GPL
    if any(g in up for g in _GPL):
        return "GPL/LGPL/AGPL copyleft"
    return ""


def main(path: str) -> int:
    violations = []
    unaudited = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or "").strip()
            license_text = (row.get("License") or "").strip()
            if name in _CLEARED_FORBIDDEN or name.lower().startswith(
                _CLEARED_FORBIDDEN_PREFIXES
            ):
                continue  # the strong override: every rule bypassed
            reason = _is_violation(license_text)
            if reason:
                violations.append((name, license_text, reason))
            elif license_text.upper() in _UNKNOWN and name not in _CLEARED_UNKNOWN:
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
            "\nRead each one's actual LICENSE file. If permissive, add it to _CLEARED_UNKNOWN in this "
            "script with a dated justification; if not, quarantine it behind an opt-in extra."
        )

    if violations or unaudited:
        return 1
    print(
        "License gate passed: no copyleft/non-commercial/unaudited licenses in the closure."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "licenses.csv"))
