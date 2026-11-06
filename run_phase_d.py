"""Phase D - SVM-faithful Grad-CAM: equivalence check, deletion test, randomisation sanity checks.

Usage: python run_phase_d.py --baseline-run runs/<phaseA> --improved-run runs/<phaseB>

1. Equivalence: the TensorFlow re-implementation of scaler+SVC must reproduce sklearn's
   decision_function (ovo and ovr) on all VAL features within 1e-5 (float64) - otherwise STOP.
   End-to-end (raw image -> TF graph, float32) is also compared with sklearn on cached features.
2. Legacy limitation: how often the legacy CNN head (which produced the old heatmaps) disagrees
   with the SVM's prediction on VAL.
3. Deletion test (VAL subset): pixels are removed (set to black) in order of heatmap importance;
   the SVM's continuous class margin S_c and the share of images still assigned to class c are
   tracked. Lower area under the curve = the map points at evidence the SVM actually uses.
   Maps compared: SVM-faithful Grad-CAM, legacy CNN-head Grad-CAM (baseline only), random.
4. Randomisation sanity checks (Adebayo et al., 2018): heatmaps must change when (a) the SVM is
   retrained on permuted labels and (b) the top ResNet50 blocks are re-initialised (cascading).
   Reported as Spearman rank correlation with the original heatmaps (low = passes).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.base import clone

from neuroinsight import bundle, metrics
from neuroinsight.features import FeatureExtractor, build_backbone, cached_features, combine
from neuroinsight.manifest import CLASS_NAMES, load_split
from neuroinsight.preprocessing import load_preprocessed
from neuroinsight.runinfo import SEED, new_run, set_seeds, write_json
from neuroinsight.svm_explain import SVMGradCAM, TFSVM, legacy_cnn_gradcam, upsample

TOL_FLOAT64 = 1e-5
STEPS = np.linspace(0.0, 1.0, 11)
CASCADE = [["conv5_block3"], ["conv5_block3", "conv5_block2"], ["conv5_block3", "conv5_block2", "conv5_block1"],
           ["conv5_block3", "conv5_block2", "conv5_block1", "conv4_block6", "conv4_block5"]]


def underlying_pipe(est):
    from sklearn.calibration import CalibratedClassifierCV
    return est.calibrated_classifiers_[0].estimator if isinstance(est, CalibratedClassifierCV) else est


def stratified_sample(rows, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(CLASS_NAMES)):
        cand = [i for i, r in enumerate(rows) if r["y"] == c]
        idx += sorted(rng.choice(cand, size=min(per_class, len(cand)), replace=False).tolist())
    return np.array(idx)


def verify_equivalence(pipe, X, name):
    svc = pipe["svm"]
    tfsvm = TFSVM(pipe, dtype=tf.float64)
    ref_ovr = pipe.decision_function(X)
    shape = svc.decision_function_shape
    svc.decision_function_shape = "ovo"
    ref_ovo = pipe.decision_function(X)
    svc.decision_function_shape = shape
    ovo = tfsvm.ovo(X).numpy()
    ovr = tfsvm.ovr(X).numpy()
    ovr32 = TFSVM(pipe, dtype=tf.float32).ovr(X.astype(np.float32)).numpy()
    res = dict(model=name, n=len(X), kernel=svc.kernel, n_support=int(svc.n_support_.sum()),
               max_abs_diff_ovo_float64=float(np.abs(ovo - ref_ovo).max()),
               max_abs_diff_ovr_float64=float(np.abs(ovr - ref_ovr).max()),
               max_abs_diff_ovr_float32=float(np.abs(ovr32 - ref_ovr).max()),
               argmax_agreement_float32=float((ovr32.argmax(1) == ref_ovr.argmax(1)).mean()),
               predict_agreement=float((ovr.argmax(1) == pipe.predict(X)).mean()),
               tolerance=TOL_FLOAT64)
    res["passed"] = bool(res["max_abs_diff_ovo_float64"] <= TOL_FLOAT64 and res["max_abs_diff_ovr_float64"] <= TOL_FLOAT64
                         and res["predict_agreement"] == 1.0)
    return res


def load_images(rows, mode):
    return np.stack([load_preprocessed(r["path"], mode)[0] for r in rows])


def deletion_scores(img, heat7, cls, extractor, tfsvm, flip, rng):
    heat = upsample(heat7[None])[0] + 1e-7 * rng.random((224, 224))
    order = np.argsort(-heat.ravel())
    batch = []
    for frac in STEPS:
        im = img.copy().reshape(-1, 3)
        im[order[:int(round(frac * order.size))]] = 0.0
        batch.append(im.reshape(224, 224, 3))
    a, b = extractor(np.stack(batch))
    f = (a + b) / 2.0 if flip else a
    S, _ = tfsvm.conf_sums(f.astype(np.float64))
    ovr = tfsvm.ovr(f.astype(np.float64)).numpy()
    return S.numpy()[:, cls], (ovr.argmax(1) == cls).astype(float)


def run_deletion(rows, idx, mode, flip, pipe, decided, gradcam, extractor, cnn_model, rng):
    tfsvm = TFSVM(pipe, dtype=tf.float64)
    sub = [rows[i] for i in idx]
    imgs = load_images(sub, mode)
    cls = decided[idx]
    maps = {"SVM-faithful Grad-CAM": gradcam(imgs, cls)[0]}
    if cnn_model is not None:
        maps["legacy CNN-head Grad-CAM"] = legacy_cnn_gradcam(cnn_model, imgs, cls)
    maps["random map"] = rng.random((len(sub), 7, 7))
    curves = {m: dict(margin=[], kept=[]) for m in maps}
    for i in range(len(sub)):
        for m, H in maps.items():
            s, k = deletion_scores(imgs[i], H[i], int(cls[i]), extractor, tfsvm, flip, rng)
            curves[m]["margin"].append(s)
            curves[m]["kept"].append(k)
        if (i + 1) % 20 == 0:
            print(f"    deletion {i + 1}/{len(sub)}", flush=True)
    out = {}
    for m, c in curves.items():
        margin = np.array(c["margin"])
        # normalise each image's margin curve by its starting value (S_c at 0% deleted)
        norm = margin / np.maximum(np.abs(margin[:, :1]), 1e-6)
        kept = np.array(c["kept"])
        auc = lambda y: float(np.trapezoid(y, STEPS))
        per_img = np.array([auc(r) for r in norm])
        out[m] = dict(mean_margin_curve=norm.mean(0).tolist(), mean_kept_curve=kept.mean(0).tolist(),
                      auc_margin=float(per_img.mean()), auc_margin_sd=float(per_img.std()),
                      auc_kept=auc(kept.mean(0)), per_image_auc_margin=per_img.tolist())
    ref = np.array(out["random map"]["per_image_auc_margin"])
    for m in out:
        if m != "random map":
            d = np.array(out[m]["per_image_auc_margin"]) - ref
            out[m]["share_images_better_than_random"] = float((d < 0).mean())
    return out, maps, imgs


def rank_corr(a, b):
    r = [spearmanr(x.ravel(), y.ravel()).statistic for x, y in zip(a, b)]
    r = np.nan_to_num(np.array(r, dtype=float), nan=0.0)
    return dict(mean=float(r.mean()), sd=float(r.std()), share_above_0_5=float((r > 0.5).mean()))


def randomised_backbone(backbone, prefixes, seed):
    model = tf.keras.models.clone_model(backbone)
    model.set_weights(backbone.get_weights())
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    n = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and any(layer.name.startswith(p) for p in prefixes):
            k, b = layer.get_weights()
            layer.set_weights([init(k.shape).numpy(), np.zeros_like(b)])
            n += 1
    return model, n


def sanity_checks(rows, idx, mode, flip, pipe, Xtr, ytr, backbone, decided, base_maps, rng):
    sub = [rows[i] for i in idx]
    imgs = load_images(sub, mode)
    cls = decided[idx]
    out = {}
    perm = ytr.copy()
    np.random.default_rng(SEED).shuffle(perm)
    rand_pipe = clone(pipe).fit(Xtr, perm)
    out["SVM retrained on permuted labels"] = rank_corr(base_maps, SVMGradCAM(backbone, rand_pipe, flip)(imgs, cls)[0])
    for prefixes in CASCADE:
        model, n = randomised_backbone(backbone, prefixes, SEED)
        maps = SVMGradCAM(model, pipe, flip)(imgs, cls)[0]
        out[f"backbone re-initialised: {', '.join(prefixes)} ({n} conv layers)"] = rank_corr(base_maps, maps)
        del model
    # reference: the same explanation recomputed (should be 1.0) and a different image's map (chance level)
    out["reference: different images' maps"] = rank_corr(base_maps, np.roll(base_maps, 1, axis=0))
    return out


def overlay(img, heat, alpha=0.45):
    import cv2
    h = np.uint8(255 * upsample(heat[None])[0])
    col = cv2.cvtColor(cv2.applyColorMap(h, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * img + alpha * col, 0, 255).astype(np.uint8)


def save_examples(path, imgs, maps: dict, labels, preds, title):
    plt = metrics._plt()
    n = min(8, len(imgs))
    cols = 1 + len(maps)
    fig, axes = plt.subplots(n, cols, figsize=(2.3 * cols, 2.3 * n), dpi=110)
    for i in range(n):
        axes[i, 0].imshow(imgs[i].astype(np.uint8))
        axes[i, 0].set_title(f"true {CLASS_NAMES[labels[i]]}\nSVM: {CLASS_NAMES[preds[i]]}", fontsize=7)
        for j, (m, H) in enumerate(maps.items(), start=1):
            axes[i, j].imshow(overlay(imgs[i], H[i]))
            if i == 0:
                axes[i, j].set_title(m, fontsize=7)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_deletion(path, results: dict):
    plt = metrics._plt()
    colors = {"SVM-faithful Grad-CAM": "#2a6fdb", "legacy CNN-head Grad-CAM": "#e07a1f", "random map": "#999999"}
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.8), dpi=140, squeeze=False)
    for ax, (name, res) in zip(axes[0], results.items()):
        for m, r in res.items():
            ax.plot(STEPS * 100, r["mean_margin_curve"], lw=2, marker="o", ms=3, color=colors.get(m, "#333"),
                    label=f"{m} (AUC {r['auc_margin']:.3f})")
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.set_xlabel("% of pixels deleted (most important first)")
        ax.set_ylabel("SVM class margin S_c (relative to start)")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(color="#eeeeee", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline-run", type=Path, required=True)
    ap.add_argument("--improved-run", type=Path, required=True)
    ap.add_argument("--deletion-per-class", type=int, default=30)
    ap.add_argument("--sanity-per-class", type=int, default=10)
    args = ap.parse_args()
    set_seeds()
    run_dir = new_run("phaseD_svm_gradcam", dict(baseline_run=args.baseline_run.name, improved_run=args.improved_run.name,
                                                 tolerance_float64=TOL_FLOAT64, deletion_steps=STEPS.tolist(),
                                                 deletion_fill="black (0) in raw [0,255] image space",
                                                 deletion_per_class=args.deletion_per_class,
                                                 sanity_per_class=args.sanity_per_class, cascade=CASCADE))
    res_dir = run_dir / "results"
    print(f"Run directory: {run_dir}", flush=True)
    tr_rows, va_rows = load_split("train"), load_split("val")
    ytr = np.array([r["y"] for r in tr_rows])
    yva = np.array([r["y"] for r in va_rows])
    backbone = build_backbone()
    extractor = FeatureExtractor(backbone)
    rng = np.random.default_rng(SEED)
    report = {}

    models = {}
    for key, rdir, sub in (("baseline", args.baseline_run, "baseline_svm"), ("improved", args.improved_run, "improved_svm")):
        est, meta = bundle.load_bundle(rdir / "models" / sub)
        mode, flip = meta["preprocessing"], meta["flip_average"]
        Xtr = combine(cached_features(tr_rows, mode, "train", extractor), flip)
        Xva = combine(cached_features(va_rows, mode, "val", extractor), flip)
        decided, _ = bundle.predict(est, meta, Xva)
        models[key] = dict(est=est, meta=meta, pipe=underlying_pipe(est), mode=mode, flip=flip, Xtr=Xtr, Xva=Xva,
                           decided=decided)

    # 1. equivalence
    print("[1/4] Equivalence of TF SVM vs sklearn", flush=True)
    report["equivalence"] = {}
    for key, m in models.items():
        eq = verify_equivalence(m["pipe"], m["Xva"], key)
        eq["decision_rule_vs_svm_ovr_argmax_agreement"] = float(
            (m["decided"] == m["pipe"].decision_function(m["Xva"]).argmax(1)).mean())
        idx = stratified_sample(va_rows, 8, SEED)
        imgs = load_images([va_rows[i] for i in idx], m["mode"])
        _, scores, _ = SVMGradCAM(backbone, m["pipe"], m["flip"])(imgs, m["decided"][idx])
        ref = m["pipe"].decision_function(m["Xva"][idx])
        eq["end_to_end_float32_max_abs_diff"] = float(np.abs(scores - ref).max())
        eq["end_to_end_argmax_agreement"] = float((scores.argmax(1) == ref.argmax(1)).mean())
        report["equivalence"][key] = eq
        print(f"  {key}: ovo {eq['max_abs_diff_ovo_float64']:.2e} ovr {eq['max_abs_diff_ovr_float64']:.2e} "
              f"(f32 {eq['max_abs_diff_ovr_float32']:.2e}, end-to-end {eq['end_to_end_float32_max_abs_diff']:.2e}) "
              f"-> {'PASS' if eq['passed'] else 'FAIL'}", flush=True)
    write_json(res_dir / "phase_d.json", report)
    if not all(e["passed"] for e in report["equivalence"].values()):
        print("STOPPED: TF SVM does not reproduce sklearn within tolerance. No Grad-CAM results produced.")
        return 1

    # 2. legacy limitation
    print("[2/4] Legacy CNN head vs SVM agreement", flush=True)
    cnn = tf.keras.models.load_model(args.baseline_run / "models" / "cnn_head.keras")
    b = models["baseline"]
    cnn_pred = np.concatenate([cnn.predict(tf.keras.applications.resnet.preprocess_input(
        load_images(va_rows[i:i + 64], "resize")), verbose=0).argmax(1) for i in range(0, len(va_rows), 64)])
    report["legacy_limitation"] = dict(
        cnn_head_val_accuracy=float((cnn_pred == yva).mean()),
        svm_val_accuracy=float((b["decided"] == yva).mean()),
        cnn_vs_svm_disagreement=float((cnn_pred != b["decided"]).mean()),
        n_disagree=int((cnn_pred != b["decided"]).sum()), n=len(yva))
    print(f"  CNN head disagrees with SVM on {report['legacy_limitation']['n_disagree']}/{len(yva)} val images", flush=True)
    write_json(res_dir / "phase_d.json", report)

    # 3. deletion test
    print("[3/4] Deletion test", flush=True)
    idx = stratified_sample(va_rows, args.deletion_per_class, SEED)
    report["deletion"] = {}
    examples = {}
    for key, m in models.items():
        res, maps, imgs = run_deletion(va_rows, idx, m["mode"], m["flip"], m["pipe"], m["decided"],
                                       SVMGradCAM(backbone, m["pipe"], m["flip"]), extractor,
                                       cnn if key == "baseline" else None, rng)
        report["deletion"][key] = res
        examples[key] = (maps, imgs)
        print("  " + key + ": " + ", ".join(f"{k} AUC {v['auc_margin']:.3f}" for k, v in res.items()), flush=True)
        ex = np.concatenate([np.where(yva[idx] == c)[0][:2] for c in range(len(CLASS_NAMES))])
        save_examples(res_dir / f"gradcam_examples_{key}.png", imgs[ex], {k: v[ex] for k, v in maps.items() if k != "random map"},
                      yva[idx][ex], m["decided"][idx][ex], f"{key} model ({m['mode']}{'+flipavg' if m['flip'] else ''}) - validation images")
    plot_deletion(res_dir / "deletion_curves.png", {f"{k} model": {mm: {kk: vv for kk, vv in r.items() if kk != 'per_image_auc_margin'}
                                                                    for mm, r in v.items()} for k, v in report["deletion"].items()})
    for v in report["deletion"].values():
        for r in v.values():
            r.pop("per_image_auc_margin", None)
    write_json(res_dir / "phase_d.json", report)

    # 4. randomisation sanity checks
    print("[4/4] Randomisation sanity checks", flush=True)
    sidx = stratified_sample(va_rows, args.sanity_per_class, SEED + 1)
    report["sanity"] = {}
    for key, m in models.items():
        imgs = load_images([va_rows[i] for i in sidx], m["mode"])
        base_maps = SVMGradCAM(backbone, m["pipe"], m["flip"])(imgs, m["decided"][sidx])[0]
        report["sanity"][key] = sanity_checks(va_rows, sidx, m["mode"], m["flip"], m["pipe"], m["Xtr"], ytr, backbone,
                                              m["decided"], base_maps, rng)
        for k, v in report["sanity"][key].items():
            print(f"  {key} | {k}: Spearman {v['mean']:.3f} +- {v['sd']:.3f}", flush=True)
    write_json(res_dir / "phase_d.json", report)
    print(f"Done. Outputs in {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
