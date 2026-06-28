from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

from ml_utils import ensure_dir, list_class_names, save_json


DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"
ARTIFACTS_DIR = Path("artifacts")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 1337


def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.08, seed=SEED),
            layers.RandomZoom(0.12, seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="augmentation",
    )


def make_datasets():
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    class_names = train_ds_raw.class_names

    aug = build_augmentation()
    train_ds_no_aug = train_ds_raw.map(
        lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
    )
    train_ds_aug = train_ds_raw.map(
        lambda x, y: (preprocess_input(aug(tf.cast(x, tf.float32), training=True)), y)
    )
    test_ds = test_ds_raw.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))
    train_ds_no_aug = train_ds_no_aug.prefetch(tf.data.AUTOTUNE)
    train_ds_aug = train_ds_aug.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds_no_aug, train_ds_aug, test_ds, class_names


def build_feature_extractor():
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    model = tf.keras.Model(inputs, x, name="resnet50_feature_extractor")
    return model


def extract_features(model: tf.keras.Model, ds: tf.data.Dataset, *, max_batches: int | None = None):
    feats = []
    labels = []
    step = 0
    for xb, yb in ds:
        fb = model(xb, training=False).numpy()
        feats.append(fb)
        labels.append(yb.numpy())
        step += 1
        if step % 20 == 0:
            n = sum(x.shape[0] for x in feats)
            print(f"  extracted ~{n} samples...", flush=True)
        if max_batches is not None and step >= max_batches:
            break
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def build_cnn_classifier(num_classes: int):
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="resnet50_classifier_for_gradcam")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster run for demo: uses fewer batches and fewer epochs.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    ensure_dir(ARTIFACTS_DIR)

    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise RuntimeError("Expected dataset in dataset/Training and dataset/Testing")

    train_ds_no_aug, train_ds_aug, test_ds, class_names = make_datasets()
    save_json(ARTIFACTS_DIR / "class_names.json", class_names)

    if args.quick:
        args.max_train_batches = args.max_train_batches or 60
        args.max_test_batches = args.max_test_batches or 20
        args.epochs = min(args.epochs, 2)

    print("\n[1/3] Extracting ResNet50 features...", flush=True)
    feat_model = build_feature_extractor()
    X_train, y_train = extract_features(
        feat_model, train_ds_no_aug, max_batches=args.max_train_batches
    )
    X_test, y_test = extract_features(feat_model, test_ds, max_batches=args.max_test_batches)

    print("[2/3] Training SVM classifier...", flush=True)
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    pipe: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", svm),
        ]
    )
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(f"SVM test accuracy: {acc:.4f}")
    joblib.dump(pipe, ARTIFACTS_DIR / "svm.joblib")

    print("[3/3] Training CNN head (for Grad-CAM only)...", flush=True)
    cnn = build_cnn_classifier(num_classes=len(class_names))
    fit_train_ds = train_ds_aug
    fit_test_ds = test_ds
    if args.max_train_batches is not None:
        fit_train_ds = fit_train_ds.take(args.max_train_batches)
    if args.max_test_batches is not None:
        fit_test_ds = fit_test_ds.take(args.max_test_batches)

    history = cnn.fit(fit_train_ds, validation_data=fit_test_ds, epochs=args.epochs)
    cnn.save(ARTIFACTS_DIR / "cnn_classifier.keras")
    print("Saved artifacts to artifacts/")


if __name__ == "__main__":
    main()

