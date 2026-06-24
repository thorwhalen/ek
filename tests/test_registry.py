import pytest

from ek.registry import (
    MissingExtraError,
    check_requirements,
    get,
    names,
    register,
    requires_extra,
    resolve,
)


def test_register_get_and_names():
    register("metrics", "echo", lambda pred, gold, *, grammar=None: pred)
    assert "echo" in names("metrics")
    assert get("metrics", "echo")("x", "y") == "x"


def test_get_unknown_lists_available():
    register("metrics", "present", object())
    with pytest.raises(KeyError) as exc:
        get("metrics", "absent_xyz")
    assert "present" in str(exc.value)


def test_resolve_accepts_name_or_object_or_default():
    fn = lambda s: s.upper()
    register("normalizers", "up", fn)
    assert resolve("normalizers", "up") is fn
    assert resolve("normalizers", fn) is fn
    assert resolve("normalizers", None, default="up") is fn


def test_requires_extra_raises_actionable_error():
    @requires_extra("ocr", packages=["definitely_not_a_real_pkg_zzz"])
    def needs_ocr():
        return "ran"

    with pytest.raises(MissingExtraError) as exc:
        needs_ocr()
    assert "ek[ocr]" in str(exc.value)


def test_requires_extra_passes_when_present():
    @requires_extra("core", packages=["json"])  # stdlib always importable
    def fine():
        return "ran"

    assert fine() == "ran"


def test_check_requirements_generic():
    assert check_requirements()["ok"] is True
    r = check_requirements(extra="definitely_not_a_real_pkg_zzz")
    assert r["ok"] is False
    assert "ek[" in r["hint"]


def test_check_requirements_maps_extra_to_import_name():
    # ek[ocr] installs `ocracy`, not a module named `ocr`; the probe must map it.
    import importlib.util

    ocracy_present = importlib.util.find_spec("ocracy") is not None
    assert check_requirements(extra="ocr")["ok"] is ocracy_present


def test_check_requirements_reports_missing_packages():
    # A multi-package extra must report WHICH packages are missing, not a false ok.
    r = check_requirements(extra="metrics")
    assert "missing" in r
    assert r["ok"] == (not r["missing"])
    # an all-fake extra reports every probe missing
    fake = check_requirements(extra="definitely_not_a_real_pkg_zzz")
    assert fake["ok"] is False and fake["missing"]
