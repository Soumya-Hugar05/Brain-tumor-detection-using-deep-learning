"""Phase D correction - investigation of the two unsatisfactory Phase D checks.

Does NOT modify run_phase_d.py or any earlier run; writes to runs/<ts>_phaseD_correction/.
Only the TRAIN and VAL splits are read (the test split is locked by neuroinsight.manifest).

Tasks
  1. Diagnose why the backbone re-initialisation maps were constant (RBF kernel saturation) and re-run the
     check validly: the SVM is re-fitted on features of the re-initialised backbone (train subset), and the
     original-SVM variant is reported as INVALID when its maps are constant.
  2. Spearman correlation reports NaN/invalid explicitly (never converted to 0). Verdicts: PASS / FAIL /
     INCONCLUSIVE using rules fixed below, before the numbers were seen.
  3-4. Controlled deletion experiments on the SAME 120 VAL images as Phase D, identical protocol for every map:
       granularity {pixel, 7x7 cell} x fill {black, image-mean, blur}; maps {SVM-faithful (ReLU),
       SVM-faithful signed (no ReLU), legacy CNN head, random x2}. Paired bootstrap CIs vs random.
  5. Relationship between the explained one-vs-rest score and the Platt-based prediction rule.

Verdict rules (fixed in advance)
  randomisation: PASS if all maps valid and mean Spearman <= 0.5; FAIL if valid and > 0.5; INCONCLUSIVE otherwise.
  deletion (per variant, margin AUC, lower = better): PASS if the 95% paired-bootstrap CI of
     (AUC_map - AUC_random) is entirely < 0; FAIL if entirely > 0; else INCONCLUSIVE.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.base import clone

import run_phase_d as D
from neuroinsight import bundle, metrics
from neuroinsight.features import FeatureExtractor, build_backbone, cached_features, combine
from neuroinsight.manifest import CLASS_NAMES, load_split
from neuroinsight.runinfo import SEED, new_run, set_seeds, write_json
from neuroinsight.svm_explain import SVMGradCAM, TFSVM, legacy_cnn_gradcam, upsample

STEPS = D.STEPS
VARIANTS = [("pixel", "black"), ("pixel", "mean"), ("cell", "black"), ("cell", "mean"), ("cell", "blur")]
N_BOOT = 2000


# ----------------------------------------------------------------------------- correlation (task 2)
def rank_corr(a, b):
    """Spearman per image. NaN (constant input) is INVALID, never 0."""
    r = np.array([spearmanr(x.ravel(), y.ravel()).statistic if np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
                  for x, y in zip(a, b)], dtype=float)
    valid = ~np.isnan(r)
    out = dict(n=int(len(r)), n_valid=int(valid.sum()), n_invalid=int((~valid).sum()),
               n_constant_perturbed_maps=int(sum(np.ptp(y) == 0 for y in b)))
    if valid.any():
        out.update(mean=float(r[valid].mean()), sd=float(r[valid].std()), share_above_0_5=float((r[valid] > 0.5).mean()))
    else:
        out.update(mean=None, sd=None, share_above_0_5=None)
    if out["n_invalid"] > 0 or out["mean"] is None:
        out["verdict"] = "INCONCLUSIVE (invalid: constant/undefined maps)"
    else:
        out["verdict"] = "PASS" if out["mean"] <= 0.5 else "FAIL"
    return out


# ----------------------------------------------------------------------------- task 1
def saturation_diagnosis(prefixes_list, backbone, pipe, imgs):
    """Why the constant maps: distance of re-initialised features from the SVM support vectors."""
    sv, gamma = pipe["svm"].support_vectors_, float(pipe["svm"]._gamma)
    rows = []
    for prefixes in prefixes_list:
        model, _ = D.randomised_backbone(backbone, prefixes, SEED) if prefixes else (backbone, 0)
        f = tf.reduce_mean(model(tf.keras.applications.resnet.preprocess_input(tf.constant(imgs)), training=False), [1, 2]).numpy()
        z = (f - pipe["scaler"].mean_) / pipe["scaler"].scale_
        d2 = ((z[:, None, :] - sv[None]) ** 2).sum(-1)
        cam = SVMGradCAM(model, pipe, False)(imgs, np.zeros(len(imgs), int))[0]
        rows.append(dict(reinit=prefixes, feature_mean=float(f.mean()), feature_sd=float(f.std()),
                         min_gamma_d2=float((gamma * d2).min()), max_kernel_value=float(np.exp(-gamma * d2).max()),
                         n_images=len(imgs), n_constant_maps=int((np.ptp(cam.reshape(len(cam), -1), axis=1) == 0).sum())))
        del model
    return rows


def cam_pair(model, pipe, flip, imgs, cls):
    """(ReLU map, pre-ReLU signed map) - the signed map is rarely constant, so it keeps the check informative."""
    gc = SVMGradCAM(model, pipe, flip)
    raw, _ = gc._cam(tf.convert_to_tensor(imgs, tf.float32), tf.constant(np.asarray(cls), tf.int32))
    return gc(imgs, cls)[0], raw.numpy()


def both_corr(base, other):
    r = rank_corr(base[0], other[0])
    r["signed_no_relu"] = rank_corr(base[1], other[1])
    return r


def randomisation(rows_tr, ytr, rows_va, idx_va, mode, flip, pipe, backbone, decided, base_maps, imgs, n_train_per_class, rng):
    out = {}
    cls = decided[idx_va]
    # reference: maps of different images
    out["reference: different images' maps (chance level)"] = both_corr(base_maps, tuple(np.roll(m, 1, axis=0) for m in base_maps))
    # (a) permuted labels, SVM refit on the full training features
    perm = ytr["all"].copy()
    np.random.default_rng(SEED).shuffle(perm)
    out["SVM retrained on permuted labels (full train)"] = both_corr(
        base_maps, cam_pair(backbone, clone(pipe).fit(ytr["X"], perm), flip, imgs, cls))
    # train subset for the refit variants
    sub = D.stratified_sample(rows_tr, n_train_per_class, SEED + 7)
    sub_imgs = D.load_images([rows_tr[i] for i in sub], mode)
    ysub = np.array([rows_tr[i]["y"] for i in sub])

    def feats(model):
        a, b = FeatureExtractor(model)(sub_imgs)
        return (a + b) / 2.0 if flip else a

    # control: same (trained) backbone, SVM refit on the subset -> how much do maps move from refitting alone?
    out[f"CONTROL: trained backbone, SVM refit on {len(sub)}-image train subset"] = both_corr(
        base_maps, cam_pair(backbone, clone(pipe).fit(feats(backbone), ysub), flip, imgs, cls))
    for prefixes in D.CASCADE:
        model, n = D.randomised_backbone(backbone, prefixes, SEED)
        tag = f"{', '.join(prefixes)} ({n} conv layers)"
        # (b) ORIGINAL SVM on re-initialised features: kernel saturates -> constant maps (invalid check)
        out[f"re-init {tag} | original SVM (kernel-saturated)"] = both_corr(base_maps, cam_pair(model, pipe, flip, imgs, cls))
        # (c) valid check: SVM refit on features of the re-initialised backbone
        refit = clone(pipe).fit(feats(model), ysub)
        refit_acc = float((refit.predict(feats(model)) == ysub).mean())
        r = both_corr(base_maps, cam_pair(model, refit, flip, imgs, cls))
        r["refit_train_subset_accuracy"] = refit_acc
        out[f"re-init {tag} | SVM refit on re-initialised features"] = r
        del model
    return out


# ----------------------------------------------------------------------------- tasks 3-4
def make_fill(img, fill):
    if fill == "black":
        return np.zeros_like(img)
    if fill == "mean":
        return np.broadcast_to(img.reshape(-1, 3).mean(0), img.shape).astype(img.dtype).copy()
    if fill == "blur":
        return cv2.GaussianBlur(img.astype(np.float32), (0, 0), 20).astype(img.dtype)
    raise ValueError(fill)


def deletion_batch(img, heat7, gran, fill, rng):
    base = make_fill(img, fill)
    batch = []
    if gran == "pixel":
        heat = upsample(heat7[None])[0] + 1e-7 * rng.random((224, 224))
        order = np.argsort(-heat.ravel())
        for frac in STEPS:
            im = img.copy().reshape(-1, 3)
            im[order[:int(round(frac * order.size))]] = base.reshape(-1, 3)[order[:int(round(frac * order.size))]]
            batch.append(im.reshape(224, 224, 3))
    else:
        h = heat7 + 1e-7 * rng.random(heat7.shape)
        order = np.argsort(-h.ravel())
        for frac in STEPS:
            mask = np.zeros(49, bool)
            mask[order[:int(round(frac * 49))]] = True
            m = np.kron(mask.reshape(7, 7), np.ones((32, 32), bool))
            im = img.copy()
            im[m] = base[m]
            batch.append(im)
    return np.stack(batch)


def margins(batch, cls, extractor, tfsvm, flip):
    a, b = extractor(batch)
    f = ((a + b) / 2.0 if flip else a).astype(np.float64)
    S, _ = tfsvm.conf_sums(f)
    return S.numpy()[:, cls]


def boot_ci(d, rng):
    d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def deletion_experiments(imgs, cls, maps, extractor, tfsvm, flip, rng):
    res = {}
    for gran, fill in VARIANTS:
        auc = {m: [] for m in maps}
        for i in range(len(imgs)):
            for m, H in maps.items():
                s = margins(deletion_batch(imgs[i], H[i], gran, fill, rng), int(cls[i]), extractor, tfsvm, flip)
                auc[m].append(float(np.trapezoid(s, STEPS)))      # raw S_c units, lower = better
            if (i + 1) % 30 == 0:
                print(f"    {gran}/{fill}: {i + 1}/{len(imgs)}", flush=True)
        auc = {m: np.array(v) for m, v in auc.items()}
        rnd = (auc["random map A"] + auc["random map B"]) / 2.0
        entry = {}
        for m, v in auc.items():
            e = dict(mean_auc=float(v.mean()))
            if not m.startswith("random"):
                mean, lo, hi = boot_ci(v - rnd, rng)
                e.update(paired_diff_vs_random=mean, ci95=[lo, hi], share_images_better=float((v < rnd).mean()),
                         verdict="PASS" if hi < 0 else "FAIL" if lo > 0 else "INCONCLUSIVE")
            entry[m] = e
        entry["random (mean of A,B)"] = dict(mean_auc=float(rnd.mean()))
        res[f"{gran}/{fill}"] = entry
        print("  " + f"{gran}/{fill}: " + ", ".join(f"{m} {e['mean_auc']:.2f}" + (f" [{e['verdict']}]" if 'verdict' in e else "")
                                                   for m, e in entry.items()), flush=True)
    return res


# ----------------------------------------------------------------------------- task 5
def rule_relationship(est, meta, pipe, X, y):
    decided, proba = bundle.predict(est, meta, X)
    ovr = pipe.decision_function(X)
    top2 = np.sort(ovr, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    pred_svc = pipe.predict(X)
    dis = decided != ovr.argmax(1)
    out = dict(n=len(y), disagreement_rate=float(dis.mean()), n_disagree=int(dis.sum()),
               accuracy_platt_rule=float((decided == y).mean()), accuracy_ovr_argmax=float((ovr.argmax(1) == y).mean()),
               accuracy_svc_predict_vote=float((pred_svc == y).mean()),
               ovr_argmax_equals_svc_predict=float((ovr.argmax(1) == pred_svc).mean()),
               median_top2_ovr_gap_agree=float(np.median(gap[~dis])), median_top2_ovr_gap_disagree=float(np.median(gap[dis])) if dis.any() else None,
               disagree_correct_by_platt=int(((decided == y) & dis).sum()), disagree_correct_by_ovr=int(((ovr.argmax(1) == y) & dis).sum()),
               disagree_by_true_class={CLASS_NAMES[c]: int((dis & (y == c)).sum()) for c in range(len(CLASS_NAMES))})
    if proba is not None:
        out["median_platt_top_prob_agree"] = float(np.median(proba.max(1)[~dis]))
        out["median_platt_top_prob_disagree"] = float(np.median(proba.max(1)[dis])) if dis.any() else None
    return out, decided, ovr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run", type=Path, required=True)
    ap.add_argument("--improved-run", type=Path, required=True)
    ap.add_argument("--deletion-per-class", type=int, default=30)
    ap.add_argument("--sanity-per-class", type=int, default=10)
    ap.add_argument("--refit-train-per-class", type=int, default=120)
    args = ap.parse_args()
    set_seeds()
    run_dir = new_run("phaseD_correction", dict(baseline_run=args.baseline_run.name, improved_run=args.improved_run.name,
                                                variants=VARIANTS, deletion_per_class=args.deletion_per_class,
                                                sanity_per_class=args.sanity_per_class,
                                                refit_train_per_class=args.refit_train_per_class, n_boot=N_BOOT))
    res_dir = run_dir / "results"
    print(f"Run directory: {run_dir}", flush=True)
    tr_rows, va_rows = load_split("train"), load_split("val")
    yva = np.array([r["y"] for r in va_rows])
    backbone = build_backbone()
    extractor = FeatureExtractor(backbone)
    rng = np.random.default_rng(SEED)

    est, meta = bundle.load_bundle(args.improved_run / "models" / "improved_svm")
    mode, flip = meta["preprocessing"], meta["flip_average"]
    pipe = D.underlying_pipe(est)
    Xtr = combine(cached_features(tr_rows, mode, "train", extractor), flip)
    Xva = combine(cached_features(va_rows, mode, "val", extractor), flip)
    ytr_all = np.array([r["y"] for r in tr_rows])
    report = {}

    # baseline vs improved estimators identical?
    b_est, b_meta = bundle.load_bundle(args.baseline_run / "models" / "baseline_svm")
    report["baseline_vs_improved_max_abs_decision_diff_val"] = float(np.abs(
        D.underlying_pipe(b_est).decision_function(combine(cached_features(va_rows, b_meta["preprocessing"], "val", extractor), b_meta["flip_average"])) -
        pipe.decision_function(Xva)).max())
    print("baseline vs improved decision diff:", report["baseline_vs_improved_max_abs_decision_diff_val"], flush=True)

    # task 5
    print("[5] Score vs prediction-rule relationship", flush=True)
    report["rule_relationship"], decided, ovr = rule_relationship(est, meta, pipe, Xva, yva)
    print("  ", {k: v for k, v in report["rule_relationship"].items() if not isinstance(v, dict)}, flush=True)
    write_json(res_dir / "phase_d_correction.json", report)

    # task 1 diagnosis
    print("[1] Diagnosis of constant re-initialised maps", flush=True)
    sidx = D.stratified_sample(va_rows, args.sanity_per_class, SEED + 1)
    simgs = D.load_images([va_rows[i] for i in sidx], mode)
    report["saturation_diagnosis"] = saturation_diagnosis([[]] + D.CASCADE, backbone, pipe, simgs[:8])
    for r in report["saturation_diagnosis"]:
        print(f"   reinit={len(r['reinit'])} blocks: min gamma*d2 {r['min_gamma_d2']:.3g}, max kernel {r['max_kernel_value']:.3g}, "
              f"constant maps {r['n_constant_maps']}/{r['n_images']}", flush=True)
    write_json(res_dir / "phase_d_correction.json", report)

    # task 1-2 randomisation
    print("[1-2] Randomisation checks (valid)", flush=True)
    base_maps = cam_pair(backbone, pipe, flip, simgs, decided[sidx])
    report["randomisation"] = randomisation(tr_rows, dict(all=ytr_all, X=Xtr), va_rows, sidx, mode, flip, pipe, backbone,
                                            decided, base_maps, simgs, args.refit_train_per_class, rng)
    for k, v in report["randomisation"].items():
        m = "n/a" if v["mean"] is None else f"{v['mean']:.3f} +- {v['sd']:.3f}"
        print(f"   {k}: Spearman {m}  valid {v['n_valid']}/{v['n']}  -> {v['verdict']} | signed: {v['signed_no_relu']['mean']} "
              f"valid {v['signed_no_relu']['n_valid']}/{v['signed_no_relu']['n']} -> {v['signed_no_relu']['verdict']}", flush=True)
    write_json(res_dir / "phase_d_correction.json", report)

    # tasks 3-4 deletion
    print("[3-4] Controlled deletion experiments", flush=True)
    cnn = tf.keras.models.load_model(args.baseline_run / "models" / "cnn_head.keras")
    didx = D.stratified_sample(va_rows, args.deletion_per_class, D.SEED)
    dimgs = D.load_images([va_rows[i] for i in didx], mode)
    dcls = decided[didx]
    gc = SVMGradCAM(backbone, pipe, flip)
    raw, _ = gc._cam(tf.convert_to_tensor(dimgs, tf.float32), tf.constant(dcls, tf.int32))
    raw = raw.numpy()
    relu = gc(dimgs, dcls)[0]
    maps = {"SVM-faithful Grad-CAM (ReLU)": relu,
            "SVM-faithful signed (no ReLU)": (raw - raw.min((1, 2), keepdims=True)) / (np.ptp(raw, axis=(1, 2), keepdims=True) + 1e-8),
            "legacy CNN-head Grad-CAM": legacy_cnn_gradcam(cnn, dimgs, dcls),
            "random map A": rng.random((len(dimgs), 7, 7)), "random map B": rng.random((len(dimgs), 7, 7))}
    agree = decided[didx] == ovr[didx].argmax(1)
    report["deletion_subset"] = dict(n=len(didx), n_rule_disagree=int((~agree).sum()))
    report["deletion"] = deletion_experiments(dimgs, dcls, maps, extractor, TFSVM(pipe, dtype=tf.float64), flip, rng)
    write_json(res_dir / "phase_d_correction.json", report)
    print(f"Done. Outputs in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
