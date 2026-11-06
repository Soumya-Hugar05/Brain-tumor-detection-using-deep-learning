"""Phase A - baseline: the legacy ResNet50 + SVM pipeline retrained on the approved split.

Identical to train.py except for the data source:
  * fit on the TRAIN split of data_splits/manifest.csv only (excluded images never used)
  * VALIDATION split for reporting/model selection; the test split is never read
  * preprocessing: decode -> bilinear resize to 224x224 -> ResNet50 preprocess_input
  * features: frozen ImageNet ResNet50, global average pooling (2048-d)
  * classifier: StandardScaler + SVC(rbf, C=1, gamma='scale', class_weight='balanced',
    probability=True, random_state=1337)
  * CNN head for legacy Grad-CAM (--cnn-head): frozen ResNet50 -> GAP -> Dropout(0.2) ->
    Dense(4, softmax), Adam 1e-3, 5 epochs, same augmentation as train.py, validated on VAL.

Outputs go to runs/<timestamp>_phaseA_baseline/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neuroinsight import bundle, metrics
from neuroinsight.features import cached_features
from neuroinsight.manifest import CLASS_NAMES, load_split
from neuroinsight.runinfo import SEED, new_run, set_seeds, write_json

BASELINE_SVC = dict(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, random_state=SEED)
CNN_HEAD = dict(epochs=5, batch_size=32, learning_rate=1e-3, dropout=0.2,
                augmentation=dict(flip="horizontal", rotation=0.08, zoom=0.12, contrast=0.10))


def report_split(name, y, pred, proba, groups=None):
    rep = metrics.classification_report(y, pred)
    if proba is not None:
        rep["calibration"] = metrics.calibration_metrics(y, proba)
    return rep


def train_cnn_head(train_rows, val_rows, run_dir):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications.resnet import preprocess_input

    def ds(rows, shuffle):
        paths = [str(r["path"]) for r in rows]
        ys = [r["y"] for r in rows]
        d = tf.data.Dataset.from_tensor_slices((paths, ys))
        if shuffle:
            d = d.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

        def load(p, y):
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            return tf.image.resize(img, (224, 224), method="bilinear"), y
        return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(CNN_HEAD["batch_size"])

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.08, seed=SEED),
        layers.RandomZoom(0.12, seed=SEED),
        layers.RandomContrast(0.10, seed=SEED),
    ], name="augmentation")
    train_ds = ds(train_rows, True).map(lambda x, y: (preprocess_input(aug(x, training=True)), y)).prefetch(2)
    val_ds = ds(val_rows, False).map(lambda x, y: (preprocess_input(x), y)).prefetch(2)

    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CNN_HEAD["dropout"])(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="resnet50_classifier_for_gradcam")
    model.compile(optimizer=tf.keras.optimizers.Adam(CNN_HEAD["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=CNN_HEAD["epochs"], verbose=2)
    model.save(run_dir / "models" / "cnn_head.keras")
    val_pred = model.predict(val_ds, verbose=0)
    y_val = np.array([r["y"] for r in val_rows])
    rep = metrics.classification_report(y_val, val_pred.argmax(1))
    write_json(run_dir / "results" / "cnn_head.json", dict(history=hist.history, val=rep, config=CNN_HEAD))
    print(f"  CNN head val accuracy {rep['accuracy']:.4f}, macro-F1 {rep['macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cnn-head", action="store_true", help="also retrain the legacy Grad-CAM CNN head")
    args = ap.parse_args()

    set_seeds()
    run_dir = new_run("phaseA_baseline", dict(preprocessing="resize", flip_average=False, svc=BASELINE_SVC,
                                              cnn_head=CNN_HEAD if args.cnn_head else None))
    print(f"Run directory: {run_dir}")
    train_rows, val_rows = load_split("train"), load_split("val")
    print(f"train {len(train_rows)}  val {len(val_rows)}")

    ftr = cached_features(train_rows, "resize", "train")
    fva = cached_features(val_rows, "resize", "val")
    Xtr, Xva, ytr, yva = ftr["plain"], fva["plain"], ftr["y"], fva["y"]

    t = time.time()
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(**BASELINE_SVC))])
    pipe.fit(Xtr, ytr)
    print(f"  SVM fitted in {time.time() - t:.0f}s, {pipe['svm'].n_support_.sum()} support vectors")

    meta = bundle.save_bundle(run_dir / "models" / "baseline_svm", pipe, preprocessing="resize",
                              flip_average=False, decision_rule="argmax_proba",
                              description="Legacy pipeline (train.py hyper-parameters), retrained on approved train split",
                              hyperparameters=BASELINE_SVC)
    results = {}
    for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva)):
        pred, proba = bundle.predict(pipe, meta, X)
        pred_svm = pipe.predict(X)
        rep = report_split(split, y, pred, proba)
        rep["svm_predict_report"] = metrics.classification_report(y, pred_svm)
        rep["argmax_proba_vs_predict_agreement"] = float((pred == pred_svm).mean())
        results[split] = rep
        metrics.plot_confusion(rep["confusion_matrix"], run_dir / "results" / f"confusion_{split}.png",
                               f"Baseline ResNet50+SVM - {split} (n={len(y)})")
        print(f"  {split}: accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}  "
              f"tumor sens {rep['tumor_vs_notumor']['sensitivity']:.4f} spec {rep['tumor_vs_notumor']['specificity']:.4f}")
    metrics.plot_reliability({"baseline (SVC Platt)": results["val"]["calibration"]},
                             run_dir / "results" / "reliability_val.png", "Baseline calibration - validation")
    write_json(run_dir / "results" / "metrics.json", results)

    if args.cnn_head:
        print("Training legacy CNN head (Grad-CAM only)...", flush=True)
        train_cnn_head(train_rows, val_rows, run_dir)
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
