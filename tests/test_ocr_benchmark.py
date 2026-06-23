import math

from ek.ocr import add_gold_item, evaluate_ocr, profile
from ek.ocr.profiles import DEFAULT_PROFILE
from ek.stores import json_store


class FakeResult:
    """Minimal OcrResult-shaped object: ek depends only on this shape."""

    def __init__(self, text, *, mean_confidence=None, backend="fake"):
        self.text = text
        self._mc = mean_confidence
        self.backend = backend

    @property
    def mean_confidence(self):
        return self._mc


def _echo_engine(image):
    # "reads" the image by echoing it; lets tests control predictions exactly.
    return FakeResult(text=image, mean_confidence=0.8)


def test_evaluate_ocr_end_to_end_global_accumulation(tmp_path):
    gold = {
        "d1": {"image": "cat", "reference_text": "cat", "slice": "easy"},
        "d2": {"image": "cot", "reference_text": "cat", "slice": "hard"},  # 1 sub
    }
    report = evaluate_ocr(
        _echo_engine, gold, metric="cer", persist=True, run_id="r1", rootdir=str(tmp_path)
    )
    assert report.n == 2
    # global CER = (0 + 1) / (3 + 3) = 1/6, not the mean of per-item rates
    assert math.isclose(report.aggregate, 1 / 6, rel_tol=1e-6)

    per_item = report.detail["per_item"]
    assert per_item["d1"]["score"] == 0.0
    assert math.isclose(per_item["d2"]["score"], 1 / 3, rel_tol=1e-6)
    assert per_item["d1"]["mean_confidence"] == 0.8
    assert per_item["d1"]["backend"] == "fake"

    assert set(report.per_slice) == {"easy", "hard"}
    assert report.per_slice["easy"] == 0.0


def test_evaluate_ocr_persists_runs_and_results(tmp_path):
    gold = {"d1": {"image": "cat", "reference_text": "cat"}}
    evaluate_ocr(_echo_engine, gold, persist=True, run_id="r2", rootdir=str(tmp_path))
    runs = json_store("runs", rootdir=str(tmp_path))
    results = json_store("results", rootdir=str(tmp_path))
    assert "r2" in runs
    assert runs["r2"]["n"] == 1
    assert "r2" in results
    assert results["r2"]["per_item"]["d1"]["reference"] == "cat"


def test_evaluate_ocr_reads_from_a_gold_store(tmp_path):
    add_gold_item("d1", "dog", "dog", slice="easy", rootdir=str(tmp_path))
    add_gold_item("d2", "dig", "dog", slice="hard", rootdir=str(tmp_path))
    gold = json_store("gold", rootdir=str(tmp_path))
    report = evaluate_ocr(_echo_engine, gold, metric="cer", rootdir=str(tmp_path))
    assert report.n == 2
    # d1 perfect, d2 one substitution out of 3 chars
    assert math.isclose(report.aggregate, 1 / 6, rel_tol=1e-6)


def test_engine_profiles_reflect_inventory():
    assert profile("tesseract")["confidence_grain"] == "symbol"
    assert profile("mathpix")["calibrated"] is True  # the lone calibrated emitter
    # accepts ocracy's canonical hyphenated ids...
    assert profile("claude-vision")["has_real_provenance"] is False
    assert profile("aws-textract")["tables"] is True
    # ...and underscore variants resolve to the same profile
    assert profile("google_vision")["confidence_grain"] == "symbol"
    assert profile("unknown-engine") == DEFAULT_PROFILE


def test_null_text_result_is_not_scored_as_literal_none():
    # VLM/markdown engines can return an OcrResult whose .text is None.
    class NullResult:
        text = None
        backend = "vlm"

    gold = {"d1": {"image": "x", "reference_text": "cat"}}
    report = evaluate_ocr(lambda image: NullResult(), gold, metric="cer")
    # empty prediction vs "cat" -> 3 deletions / 3 ref chars = 1.0 (NOT "None"->1.333)
    assert report.detail["per_item"]["d1"]["prediction"] == ""
    assert math.isclose(report.aggregate, 1.0, rel_tol=1e-9)
