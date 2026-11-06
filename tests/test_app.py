"""Tests for the NeuroInsight web app. Run: python tests/test_app.py   (also works with pytest if installed)

Only the VALIDATION split is used. The test split is never loaded.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

import app as appmod

CLIENT = appmod.app.test_client()


def _post(data: bytes, name: str = "x.jpg"):
    return CLIENT.post("/api/predict", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def _png_bytes(w=128, h=128, fmt="PNG") -> bytes:
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, format=fmt)
    return b.getvalue()


# ---- warning tiers ------------------------------------------------------------------------------------
def test_confidence_tiers_boundaries():
    f = appmod.confidence_warning
    assert f(1.0)["level"] == "none" and f(0.90)["level"] == "none"
    assert f(0.8999)["level"] == "review" and f(0.70)["level"] == "review"
    assert f(0.6999)["level"] == "low" and f(0.34)["level"] == "low"
    assert f(0.85)["message"] == "Review recommended – prediction has uncertainty."
    assert f(0.5)["message"] == "Low confidence – expert review strongly recommended."
    assert f(0.95)["message"] is None
    print("ok confidence tiers: >=0.90 none, [0.70,0.90) review, <0.70 low; boundaries correct")


# ---- upload validation --------------------------------------------------------------------------------
def test_rejects_missing_file():
    r = CLIENT.post("/api/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_rejects_bad_extension():
    r = _post(b"hello", "notes.txt")
    assert r.status_code == 415 and "Unsupported" in r.get_json()["error"]
    r = _post(_png_bytes(), "scan.tiff")
    assert r.status_code == 415


def test_rejects_empty_file():
    r = _post(b"", "a.png")
    assert r.status_code == 400 and "empty" in r.get_json()["error"].lower()


def test_rejects_corrupt_or_fake_image():
    for payload in (b"this is not an image", b"\xff\xd8\xff\xe0" + b"\x00" * 50, _png_bytes()[:80]):
        r = _post(payload, "scan.jpg")
        assert r.status_code in (415, 422), r.status_code
        assert r.get_json()["ok"] is False
    print("ok corrupt / fake / truncated uploads rejected with JSON errors")


def test_rejects_wrong_content_for_extension():
    b = io.BytesIO()
    Image.new("RGB", (100, 100)).save(b, format="GIF")
    r = _post(b.getvalue(), "scan.png")
    assert r.status_code == 415 and r.get_json()["ok"] is False


def test_rejects_tiny_image():
    r = _post(_png_bytes(32, 32), "tiny.png")
    assert r.status_code == 422 and "too small" in r.get_json()["error"]


def test_rejects_oversized_upload():
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (appmod.MAX_UPLOAD_BYTES + 10)
    r = _post(big, "big.png")
    assert r.status_code == 413 and r.get_json()["ok"] is False
    print("ok oversize upload -> 413 JSON")


# ---- prediction ---------------------------------------------------------------------------------------
def test_prediction_matches_bundle_pipeline_on_val_samples():
    """Every demo sample, sent through the HTTP API, equals the verified evaluate-path prediction."""
    from neuroinsight import bundle
    from neuroinsight.features import cached_features, combine
    from neuroinsight.manifest import load_split
    from inference import MODEL_BUNDLE_DIR

    meta_s = json.loads((ROOT / "static" / "samples" / "samples.json").read_text(encoding="utf-8"))
    rows = load_split("val", allow_test=False, verify_md5=False)
    pos = {r["rel"]: i for i, r in enumerate(rows)}
    feats = cached_features(rows, "resize", "val")
    est, meta = bundle.load_bundle(MODEL_BUNDLE_DIR)
    pred, proba = bundle.predict(est, meta, combine(feats, meta["flip_average"]))
    for s in meta_s["samples"]:
        i = pos[s["source"]]
        data = (ROOT / "static" / "samples" / s["file"]).read_bytes()
        r = _post(data, s["file"])
        j = r.get_json()
        assert r.status_code == 200 and j["ok"], j
        assert j["prediction"]["label"] == meta["class_names"][int(pred[i])]
        got = [j["prediction"]["probabilities"][c] for c in meta["class_names"]]
        assert np.array_equal(np.array(got), proba[i]), (s["file"], got, proba[i])
        assert abs(sum(got) - 1) < 1e-9
    print(f"ok {len(meta_s['samples'])} demo samples: API labels and probabilities identical to the evaluation pipeline")


def test_response_contract_and_gradcam_disabled():
    s = json.loads((ROOT / "static" / "samples" / "samples.json").read_text(encoding="utf-8"))["samples"][0]
    r = _post((ROOT / "static" / "samples" / s["file"]).read_bytes(), s["file"])
    j = r.get_json()
    assert j["explanation"]["available"] is False and "Grad-CAM" in j["explanation"]["reason"]
    assert not any("gradcam" in k.lower() or "heatmap" in k.lower() for k in j)
    assert j["warning"]["level"] in ("none", "review", "low")
    assert j["prediction"]["ranked"][0]["label"] == j["prediction"]["label"]
    assert "not a medical device" in j["disclaimer"]
    assert j["model"]["estimator_sha256"] == json.loads(
        (appmod.MODEL_BUNDLE_DIR / "bundle.json").read_text(encoding="utf-8"))["estimator_sha256"]
    # A confidently-low case: a random-noise image must still return a prediction (never rejected) with a warning field
    r2 = _post(_png_bytes(), "noise.png")
    j2 = r2.get_json()
    assert r2.status_code == 200 and j2["prediction"]["label"] and "message" in j2["warning"]
    print("ok response contract; Grad-CAM disabled; predictions never rejected on confidence")


# ---- pages and metadata -------------------------------------------------------------------------------
def test_index_shows_disclaimer_and_no_gradcam_ui():
    html = CLIENT.get("/").get_data(as_text=True)
    assert "Research prototype" in html and "not a medical device" in html
    assert "Grad-CAM overlay" not in html and "<canvas" not in html
    assert CLIENT.get("/static/js/app.js").status_code == 200 and CLIENT.get("/static/css/app.css").status_code == 200
    print("ok index: disclaimer present, assets served, no Grad-CAM overlay UI")


def test_model_info_matches_phase_b_report():
    j = CLIENT.get("/api/model-info").get_json()
    assert j["ok"]
    m = j["validation_metrics"]
    assert (m["n"], round(m["accuracy"], 4), round(m["macro_f1"], 4)) == (1046, 0.9216, 0.9181)
    assert round(m["tumor_vs_notumor"]["sensitivity"], 4) == 0.9845 and round(m["tumor_vs_notumor"]["specificity"], 4) == 0.9852
    assert any("not been evaluated" in t for t in j["limitations"]) and any("Grad-CAM" in t for t in j["limitations"])
    print("ok model-info equals the verified Phase B validation metrics")


def test_samples_are_validation_only():
    from neuroinsight.manifest import load_split
    meta = json.loads((ROOT / "static" / "samples" / "samples.json").read_text(encoding="utf-8"))
    val = {r["rel"] for r in load_split("val", allow_test=False, verify_md5=False)}
    for s in meta["samples"]:
        assert s["source"] in val, s
        assert (ROOT / "static" / "samples" / s["file"]).is_file()
    api = CLIENT.get("/api/samples").get_json()
    assert len(api["samples"]) == len(meta["samples"]) == 8
    assert meta["split"] == "val"
    print("ok 8 demo samples, all in the validation split")


def test_app_does_not_use_old_artifacts_or_test_split():
    for fname in ("app.py", "inference.py"):
        src = (ROOT / fname).read_text(encoding="utf-8")
        assert "artifacts" not in src, fname
        assert "allow_test=True" not in src and "cnn_classifier" not in src and "make_gradcam" not in src, fname
    from inference import Predictor
    assert Predictor.__module__ == "inference"
    print("ok app/inference reference neither artifacts/, the old CNN, Grad-CAM, nor the test split")


# ---- NEUROINSIGHT_BUNDLE_DIR override (no model split data is touched; only folders and files) ---------
class _env:
    """Set (value) or remove (None) NEUROINSIGHT_BUNDLE_DIR for a block and restore it afterwards."""
    def __init__(self, value):
        self.value, self.name = value, None

    def __enter__(self):
        import inference
        self.name, self.old = inference.BUNDLE_DIR_ENV, os.environ.get(inference.BUNDLE_DIR_ENV)
        if self.value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.value

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.old


PHASE_B_DEFAULT = ROOT / "runs" / "20261105-184406_phaseB_frozen_improvements" / "models" / "improved_svm"


def test_bundle_dir_default_when_variable_unset():
    import inference
    assert inference.BUNDLE_DIR_ENV == "NEUROINSIGHT_BUNDLE_DIR"
    assert inference.DEFAULT_BUNDLE_DIR == PHASE_B_DEFAULT
    with _env(None):
        assert inference._bundle_dir_from_env() == PHASE_B_DEFAULT
    print("ok variable unset -> default Phase B bundle path")


def test_bundle_dir_empty_variable_uses_default():
    import inference
    for value in ("", "   ", "\t"):
        with _env(value):
            assert inference._bundle_dir_from_env() == PHASE_B_DEFAULT, repr(value)
    print("ok empty / whitespace variable -> default")


def test_bundle_dir_absolute_path_used_as_is():
    import inference
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "somewhere" / "improved_svm"
        with _env(str(target)):
            got = inference._bundle_dir_from_env()
        assert got == target and got.is_absolute() and got != PHASE_B_DEFAULT
    print("ok absolute path used unchanged")


def test_bundle_dir_relative_path_is_relative_to_project_not_cwd():
    import inference
    import tempfile
    with _env("models/custom_bundle"):
        assert inference._bundle_dir_from_env() == ROOT / "models" / "custom_bundle"
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                assert inference._bundle_dir_from_env() == ROOT / "models" / "custom_bundle"
            finally:
                os.chdir(old_cwd)
    print("ok relative path resolved against the project folder, independent of the working directory")


def test_missing_or_incomplete_bundle_gives_clear_error():
    import inference
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cases = {
            "missing folder": (d / "does_not_exist", ["bundle.json", "estimator.joblib"]),
            "empty folder": (d / "empty", ["bundle.json", "estimator.joblib"]),
            "only bundle.json": (d / "only_json", ["estimator.joblib"]),
            "only estimator": (d / "only_est", ["bundle.json"]),
        }
        (d / "empty").mkdir()
        (d / "only_json").mkdir(); (d / "only_json" / "bundle.json").write_text("{}", encoding="utf-8")
        (d / "only_est").mkdir(); (d / "only_est" / "estimator.joblib").write_bytes(b"x")
        for label, (path, missing) in cases.items():
            try:
                inference.Predictor(bundle_dir=path)
            except RuntimeError as exc:
                msg = str(exc)
                assert "Model bundle not found" in msg and str(path) in msg, (label, msg)
                assert "NEUROINSIGHT_BUNDLE_DIR" in msg and "Getting the Phase B bundle" in msg, (label, msg)
                for name in missing:
                    assert name in msg.split("(missing:")[1], (label, msg)
                present = {"bundle.json", "estimator.joblib"} - set(missing)
                for name in present:
                    assert name not in msg.split("(missing:")[1], (label, msg)
            else:
                raise AssertionError(f"{label}: Predictor accepted an incomplete bundle")
    print("ok missing / empty / half-present bundle folders -> clear RuntimeError naming the missing files")


def test_variable_path_reaches_the_loader_and_verifies_the_hash():
    """The env var -> path -> Predictor chain, using the real Phase B bundle given as an absolute path."""
    import inference
    with _env(str(PHASE_B_DEFAULT)):
        path = inference._bundle_dir_from_env()
    assert path == PHASE_B_DEFAULT
    p = inference.Predictor(bundle_dir=path)
    expected = json.loads((PHASE_B_DEFAULT / "bundle.json").read_text(encoding="utf-8"))["estimator_sha256"]
    assert p.meta["estimator_sha256"] == expected
    print("ok override path loads the bundle with its estimator hash verified")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nALL PASSED" if not failed else f"\n{failed} FAILED")
    sys.exit(1 if failed else 0)
