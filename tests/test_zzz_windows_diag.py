"""TEMP diagnostic for the Windows dol-store failure (#11). Removed once fixed.

Named test_zzz_* so it runs late, and prints (via `-s`) the exact path resolution
and dol-internal key->path mapping on the Windows runner.
"""

import os
import tempfile
import traceback


def test_windows_store_diagnostic():
    import dol
    from dol.dig import inner_most_key

    from ek.stores import data_dir, json_store

    root = tempfile.mkdtemp()
    print("\n===== WINDOWS-DIAG START =====")
    print("os.sep:", repr(os.sep), "os.path.sep:", repr(os.path.sep))
    print("dol:", getattr(dol, "__version__", "?"), dol.__file__)

    d = data_dir("calibrators", rootdir=root)
    print("data_dir:", repr(str(d)), "| isdir:", os.path.isdir(str(d)))

    s = json_store("calibrators", rootdir=root)
    print("store.rootdir:", repr(getattr(s, "rootdir", "?")))

    try:
        imk = inner_most_key(s, "cal-v1")
        print("inner_most_key('cal-v1'):", repr(imk))
        print("os.path.dirname(imk):", repr(os.path.dirname(imk)))
        print("dirname isdir:", os.path.isdir(os.path.dirname(imk)))
    except Exception:
        print("inner_most_key RAISED:")
        traceback.print_exc()

    try:
        s["cal-v1"] = {"kind": "platt", "a": 1.0}
        print("WRITE OK ->", s["cal-v1"])
    except Exception:
        print("WRITE RAISED:")
        traceback.print_exc()

    print("calibrators isdir after:", os.path.isdir(str(d)))
    print("files under root:")
    for dp, _dn, fs in os.walk(root):
        for f in fs:
            print("   ", os.path.join(dp, f))
    print("===== WINDOWS-DIAG END =====")
