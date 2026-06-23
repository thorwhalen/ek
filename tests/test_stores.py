from ek.stores import KINDS, json_store, mall, persistent_cache


def test_json_store_roundtrip(tmp_path):
    s = json_store("gold", rootdir=str(tmp_path))
    s["doc-001"] = {"reference_text": "hello world", "slice": "easy"}
    assert s["doc-001"]["reference_text"] == "hello world"
    assert list(s) == ["doc-001"]
    # written under <root>/gold/doc-001.json
    assert (tmp_path / "gold" / "doc-001.json").exists()


def test_json_store_nested_keys(tmp_path):
    s = json_store("results", rootdir=str(tmp_path))
    s["run-1/item-a"] = {"score": 0.5}
    assert s["run-1/item-a"]["score"] == 0.5
    assert (tmp_path / "results" / "run-1" / "item-a.json").exists()


def test_mall_groups_kinds_with_tuple_access(tmp_path):
    m = mall(rootdir=str(tmp_path))
    assert set(m) == set(KINDS)
    m["gold"]["k"] = {"v": 1}
    assert m["gold", "k"] == {"v": 1}  # tuple-key sugar


def test_persistent_cache_memoizes_across_calls(tmp_path):
    calls = {"n": 0}

    @persistent_cache(kind="runs", key=lambda *a, **k: "the-key", rootdir=str(tmp_path))
    def expensive():
        calls["n"] += 1
        return {"answer": 42}

    assert expensive() == {"answer": 42}
    assert expensive() == {"answer": 42}
    assert calls["n"] == 1  # second call served from the JSON store
