"""Import purity: ``import ek`` (and ``ek.qe``) must not pull heavy/optional deps.

The reference-free QE pipeline ships pure-Python defaults so that calibration,
conformal, ROVER, and verifiers all work with zero extras installed. Importing the
package must therefore stay light -- optional backends (netcal/sklearn/MAPIE/crepes,
ocracy, networkx, ...) are imported lazily, only when their strategy is used.
"""

import subprocess
import sys

_HEAVY = [
    "netcal",
    "sklearn",
    "mapie",
    "crepes",
    "ocracy",
    "networkx",
    "krippendorff",
    "pydantic",
    "numpy",
    "scipy",
    "uqlm",
]


def test_importing_ek_and_qe_stays_light():
    code = (
        "import sys, ek, ek.qe\n"
        f"heavy = {_HEAVY!r}\n"
        "bad = [m for m in heavy if m in sys.modules]\n"
        "assert not bad, 'heavy modules imported on `import ek`: ' + repr(bad)\n"
    )
    # A fresh interpreter, so other tests' imports cannot mask a regression.
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
