from __future__ import annotations

import base64
import io
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input

from gradcam import make_gradcam_heatmap, overlay_heatmap_on_image
from ml_utils import load_json


ARTIFACTS_DIR = Path("artifacts")
IMAGE_SIZE = (224, 224)

app = Flask(__name__)

SVM_PIPE = None
CLASS_NAMES = None
CNN_MODEL = None
FEAT_EXTRACTOR = None
MODEL_LOAD_ERROR: str | None = None


def _img_to_base64_png(arr_uint8: np.ndarray) -> str:
    img = Image.fromarray(arr_uint8)
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def load_models():
    if not (ARTIFACTS_DIR / "svm.joblib").exists():
        raise RuntimeError("Missing artifacts/svm.joblib. Run: python train.py")
    if not (ARTIFACTS_DIR / "class_names.json").exists():
        raise RuntimeError("Missing artifacts/class_names.json. Run: python train.py")
    if not (ARTIFACTS_DIR / "cnn_classifier.keras").exists():
        raise RuntimeError("Missing artifacts/cnn_classifier.keras. Run: python train.py")

    svm_pipe = joblib.load(ARTIFACTS_DIR / "svm.joblib")
    class_names = load_json(ARTIFACTS_DIR / "class_names.json")
    cnn_model = tf.keras.models.load_model(ARTIFACTS_DIR / "cnn_classifier.keras")

    feat_extractor = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        pooling="avg",
    )
    feat_extractor.trainable = False
    return svm_pipe, class_names, cnn_model, feat_extractor


def _ensure_models_loaded() -> None:
    global SVM_PIPE, CLASS_NAMES, CNN_MODEL, FEAT_EXTRACTOR, MODEL_LOAD_ERROR
    if MODEL_LOAD_ERROR is not None:
        return
    if SVM_PIPE is not None:
        return
    try:
        SVM_PIPE, CLASS_NAMES, CNN_MODEL, FEAT_EXTRACTOR = load_models()
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)


@app.get("/")
def index():
    _ensure_models_loaded()
    if MODEL_LOAD_ERROR is not None:
        return render_template("index.html", error=MODEL_LOAD_ERROR)
    return render_template("index.html")


@app.post("/predict")
def predict():
    _ensure_models_loaded()
    if MODEL_LOAD_ERROR is not None:
        return render_template("index.html", error=MODEL_LOAD_ERROR)

    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    f = request.files["file"]
    if not f or f.filename.strip() == "":
        return render_template("index.html", error="No file selected.")

    pil = Image.open(f.stream).convert("RGB").resize(IMAGE_SIZE)
    rgb_uint8 = np.asarray(pil, dtype=np.uint8)

    x = rgb_uint8.astype(np.float32)
    x = preprocess_input(x)
    x_batch = np.expand_dims(x, axis=0)

    feat = FEAT_EXTRACTOR(x_batch, training=False).numpy()
    probs = SVM_PIPE.predict_proba(feat)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    ranked = sorted(
        [{"label": CLASS_NAMES[i], "prob": float(p)} for i, p in enumerate(probs)],
        key=lambda d: d["prob"],
        reverse=True,
    )

    gradcam_b64 = None
    gradcam_error = None
    try:
        heatmap = make_gradcam_heatmap(CNN_MODEL, x, class_index=pred_idx)
        overlay = overlay_heatmap_on_image(rgb_uint8, heatmap)
        gradcam_b64 = _img_to_base64_png(overlay)
    except Exception as exc:
        gradcam_error = f"Prediction succeeded, but Grad-CAM could not be generated: {exc}"

    return render_template(
        "index.html",
        pred_label=pred_label,
        confidence=f"{confidence*100:.2f}%",
        ranked=ranked,
        original_b64=_img_to_base64_png(rgb_uint8),
        gradcam_b64=gradcam_b64,
        gradcam_error=gradcam_error,
    )


if __name__ == "__main__":
    app.run(debug=True)

