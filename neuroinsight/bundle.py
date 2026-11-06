"""A model bundle = sklearn estimator (scaler + SVM [+ calibration]) + how its features are made.

runs/<run>/models/<name>/
    estimator.joblib
    bundle.json   {preprocessing, flip_average, decision_rule, class_names, sha256, ...}
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .manifest import CLASS_NAMES, sha256_file, verify_manifest

DECISION_RULES = ("argmax_proba", "predict")


def save_bundle(path: Path, estimator, *, preprocessing: str, flip_average: bool,
                decision_rule: str, description: str, hyperparameters: dict, extra: dict | None = None) -> dict:
    assert decision_rule in DECISION_RULES
    path.mkdir(parents=True, exist_ok=False)
    joblib.dump(estimator, path / "estimator.joblib")
    meta = dict(description=description, preprocessing=preprocessing, flip_average=flip_average,
                decision_rule=decision_rule, class_names=CLASS_NAMES,
                feature="ResNet50 (ImageNet, frozen) GAP of conv5_block3_out, 2048-d",
                trained_on="train split of data_splits/manifest.csv", manifest_sha256=verify_manifest(),
                hyperparameters=hyperparameters, estimator_sha256=sha256_file(path / "estimator.joblib"),
                **(extra or {}))
    (path / "bundle.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_bundle(path: Path):
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    if sha256_file(path / "estimator.joblib") != meta["estimator_sha256"]:
        raise RuntimeError(f"{path}/estimator.joblib does not match bundle.json (modified?)")
    return joblib.load(path / "estimator.joblib"), meta


def predict(estimator, meta: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    proba = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
    if meta["decision_rule"] == "argmax_proba":
        return proba.argmax(1), proba
    return estimator.predict(X), proba
