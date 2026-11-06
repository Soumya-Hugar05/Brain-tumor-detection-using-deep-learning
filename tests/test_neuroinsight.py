"""Quick correctness checks for the neuroinsight package. Run: python tests/test_neuroinsight.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_manifest_and_test_lock():
    from neuroinsight.manifest import TestSetLocked, load_split
    tr, va = load_split("train", verify_md5=False), load_split("val", verify_md5=False)
    assert (len(tr), len(va)) == (4877, 1046), (len(tr), len(va))
    assert not ({r["group_id"] for r in tr} & {r["group_id"] for r in va}), "group shared by train and val"
    try:
        load_split("test")
    except TestSetLocked:
        pass
    else:
        raise AssertionError("test split was not locked")
    print("ok manifest: train 4877, val 1046, disjoint groups, test locked")


def test_metrics_vs_sklearn():
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from neuroinsight import metrics
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    p = np.where(rng.random(500) < 0.8, y, rng.integers(0, 4, 500))
    m = metrics.classification_report(y, p)
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    assert np.allclose([m["per_class"][c]["precision"] for c in metrics.CLASS_NAMES], P)
    assert np.allclose([m["per_class"][c]["recall"] for c in metrics.CLASS_NAMES], R)
    assert np.allclose([m["per_class"][c]["f1"] for c in metrics.CLASS_NAMES], F)
    assert np.isclose(m["accuracy"], accuracy_score(y, p))
    assert (np.array(m["confusion_matrix"]) == confusion_matrix(y, p, labels=range(4))).all()
    cm = confusion_matrix(y, p, labels=range(4))
    spec0 = (cm.sum() - cm[0].sum() - cm[:, 0].sum() + cm[0, 0]) / (cm.sum() - cm[0].sum())
    assert np.isclose(m["per_class"]["glioma"]["specificity"], spec0)
    yt, pt = (y != 2), (p != 2)
    assert np.isclose(m["tumor_vs_notumor"]["sensitivity"], (yt & pt).sum() / yt.sum())
    assert np.isclose(m["tumor_vs_notumor"]["specificity"], (~yt & ~pt).sum() / (~yt).sum())
    ci = metrics.bootstrap_ci(y, p, groups=np.arange(500) // 2, n_boot=200)
    lo, hi = ci["ci95"]["accuracy"]
    assert lo < m["accuracy"] < hi
    print("ok metrics match sklearn (P/R/F1/acc/CM), specificity, tumor-vs-notumor, bootstrap")


def test_features_match_legacy():
    import tensorflow as tf
    from tensorflow.keras.applications.resnet import preprocess_input
    from neuroinsight.features import FeatureExtractor
    from neuroinsight.manifest import load_split
    from neuroinsight.preprocessing import load_preprocessed

    rows = load_split("train", verify_md5=False)[:6]
    imgs = np.stack([load_preprocessed(r["path"], "resize")[0] for r in rows])
    # legacy loader path (keras utils, as in image_dataset_from_directory)
    legacy = np.stack([
        tf.image.resize(tf.io.decode_image(tf.io.read_file(str(r["path"])), channels=3, expand_animations=False),
                        (224, 224), method="bilinear").numpy() for r in rows])
    assert np.array_equal(imgs, legacy), "resize preprocessing differs from legacy loader"
    fx = FeatureExtractor()
    plain, flipped = fx(imgs)
    ref = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg",
                                         input_shape=(224, 224, 3))
    ref_feat = ref(preprocess_input(imgs.copy()), training=False).numpy()
    err = np.abs(plain - ref_feat).max() / np.abs(ref_feat).max()
    assert err < 1e-4, err
    ref_flip = ref(preprocess_input(imgs[:, :, ::-1].copy()), training=False).numpy()
    assert np.abs(flipped - ref_flip).max() / np.abs(ref_flip).max() < 1e-4
    print(f"ok features equal legacy ResNet50(pooling='avg') (rel err {err:.1e}); flip branch ok")


def test_tf_svm_matches_sklearn():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from neuroinsight.svm_explain import TFSVM
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 30))
    y = np.arange(400) % 4
    X[:, :4] += np.eye(4)[y] * 2
    for kernel, kw in (("rbf", dict(gamma="scale")), ("linear", {})):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel=kernel, C=1.0, class_weight="balanced", **kw))]).fit(X, y)
        Xt = rng.normal(size=(100, 30))
        ovr = TFSVM(pipe).ovr(Xt).numpy()
        assert np.abs(ovr - pipe.decision_function(Xt)).max() < 1e-8, kernel
        pipe["svm"].decision_function_shape = "ovo"
        assert np.abs(TFSVM(pipe).ovo(Xt).numpy() - pipe.decision_function(Xt)).max() < 1e-8, kernel
    print("ok TF SVM reproduces sklearn decision_function (ovo + ovr; rbf + linear)")


if __name__ == "__main__":
    test_manifest_and_test_lock()
    test_metrics_vs_sklearn()
    test_tf_svm_matches_sklearn()
    test_features_match_legacy()
    print("ALL TESTS PASSED")
