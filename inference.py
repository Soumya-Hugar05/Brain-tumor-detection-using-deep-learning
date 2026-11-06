"""Inference for the NeuroInsight web app: the Phase B selected bundle, nothing else.

The pipeline is exactly the one used to train and evaluate the bundle:

    bytes -> tf.io.decode_image(channels=3) -> bilinear resize to 224x224 (no antialias)   [bundle "preprocessing": resize]
          -> frozen ImageNet ResNet50, GAP of conv5_block3_out (FeatureExtractor)
          -> (plain features; flip_average is read from the bundle)
          -> bundle.predict(): the bundle's decision_rule (argmax of the SVC's Platt probabilities)

Nothing here retrains or modifies a model. The estimator's SHA-256 is verified against bundle.json on load.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from neuroinsight import bundle as bundle_io
from neuroinsight.features import FeatureExtractor, combine
from neuroinsight.preprocessing import MODES, resize_baseline

ROOT = Path(__file__).resolve().parent

# The one configuration constant that selects the model served by the app. The default is the Phase B bundle in
# runs/. Set the environment variable NEUROINSIGHT_BUNDLE_DIR to load the same bundle from another folder
# (absolute path, or relative to the project folder); see the README, "Getting the Phase B bundle".
DEFAULT_BUNDLE_DIR = ROOT / "runs" / "20261105-184406_phaseB_frozen_improvements" / "models" / "improved_svm"
BUNDLE_DIR_ENV = "NEUROINSIGHT_BUNDLE_DIR"


def _bundle_dir_from_env() -> Path:
    raw = os.environ.get(BUNDLE_DIR_ENV, "").strip()
    if not raw:
        return DEFAULT_BUNDLE_DIR
    p = Path(raw).expanduser()
    return p if p.is_absolute() else ROOT / p


MODEL_BUNDLE_DIR = _bundle_dir_from_env()


@dataclass
class Prediction:
    label: str
    index: int
    confidence: float
    probabilities: dict[str, float]      # class name -> probability, in CLASS order
    ranked: list[dict]                   # [{"label", "prob"}] sorted high -> low
    image_uint8: np.ndarray              # the 224x224 RGB image the model saw (for display)
    features: np.ndarray                 # (2048,) feature vector, for verification


class Predictor:
    def __init__(self, bundle_dir: Path = MODEL_BUNDLE_DIR):
        self.bundle_dir = Path(bundle_dir)
        missing = [n for n in ("bundle.json", "estimator.joblib") if not (self.bundle_dir / n).is_file()]
        if missing:
            raise RuntimeError(
                f"Model bundle not found or incomplete at {self.bundle_dir} (missing: {', '.join(missing)}). "
                f"See the README section 'Getting the Phase B bundle', or set {BUNDLE_DIR_ENV}.")
        self.estimator, self.meta = bundle_io.load_bundle(self.bundle_dir)   # verifies the estimator hash
        if self.meta["preprocessing"] not in MODES:
            raise RuntimeError(f"Unsupported preprocessing in bundle: {self.meta['preprocessing']!r}")
        if self.meta["preprocessing"] != "resize":
            # The app implements exactly the 'resize' pipeline of the selected bundle. Refuse anything else
            # rather than silently serving different preprocessing.
            raise RuntimeError("This app only implements the 'resize' preprocessing of the Phase B bundle.")
        self.class_names: list[str] = list(self.meta["class_names"])
        self.extractor = FeatureExtractor()

    # -- pipeline -----------------------------------------------------------------------------
    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Same decoder as neuroinsight.preprocessing.read_rgb, from bytes. Returns uint8 HxWx3."""
        img = tf.io.decode_image(tf.constant(data), channels=3, expand_animations=False)
        return img.numpy()

    def features_from_bytes(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Returns (feature vector used by the model, 224x224 float32 image in [0,255])."""
        img = resize_baseline(self.decode(data))
        plain, flipped = self.extractor(img[None])
        X = combine(dict(plain=plain, flipped=flipped), self.meta["flip_average"])
        return X, img

    def predict_bytes(self, data: bytes) -> Prediction:
        X, img = self.features_from_bytes(data)
        pred, proba = bundle_io.predict(self.estimator, self.meta, X)
        idx = int(pred[0])
        p = proba[0]
        ranked = sorted(({"label": self.class_names[i], "prob": float(p[i])} for i in range(len(p))),
                        key=lambda d: d["prob"], reverse=True)
        return Prediction(
            label=self.class_names[idx], index=idx, confidence=float(p[idx]),
            probabilities={self.class_names[i]: float(p[i]) for i in range(len(p))}, ranked=ranked,
            image_uint8=np.clip(np.round(img), 0, 255).astype(np.uint8), features=X[0])

    def describe(self) -> dict:
        m = self.meta
        return dict(bundle=str(self.bundle_dir), description=m.get("description"), preprocessing=m["preprocessing"],
                    flip_average=m["flip_average"], decision_rule=m["decision_rule"],
                    estimator_sha256=m["estimator_sha256"], manifest_sha256=m["manifest_sha256"])
