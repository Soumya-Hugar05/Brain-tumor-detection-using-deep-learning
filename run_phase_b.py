"""Phase B - frozen-feature improvements, decided on the VALIDATION split only.

Stages
  1. features for both preprocessing modes (resize = legacy, croppad = brain crop + pad)
  2. shortcut probes (TRAIN -> VAL): do size / encoding / intensity alone predict the class?
  3. default SVM (legacy hyper-parameters) on 4 feature configs: {resize, croppad} x {plain, flip-avg}
  4. perturbation test (causal shortcut check) on VAL images for those models
  5. decisions with explicit rules (paired, duplicate-group bootstrap on VAL):
       croppad  kept if NOT significantly worse on VAL macro-F1 AND it lowers the shortcut flip rate
       flip-avg kept only if it gives a significant VAL macro-F1 gain
  6. SVM hyper-parameter grid (RBF and linear), scored on VAL; the best is kept only if its gain
     over the default is significant on VAL AND it also beats the default in grouped 5-fold CV on TRAIN
  7. probability calibration: legacy SVC Platt vs grouped-CV sigmoid / isotonic (CalibratedClassifierCV),
     compared on VAL NLL / Brier / ECE; kept only if VAL NLL improves
  8. save the selected model bundle and phase_b_report.md

The test split is never read.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neuroinsight import bundle, metrics, shortcut
from neuroinsight.features import FeatureExtractor, cached_features, combine
from neuroinsight.manifest import CLASS_NAMES, NOTUMOR_INDEX, load_split
from neuroinsight.preprocessing import read_rgb
from neuroinsight.runinfo import SEED, new_run, set_seeds, write_json

DEFAULT_SVC = dict(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
RBF_C = [0.3, 1.0, 3.0, 10.0, 30.0]
RBF_GAMMA_MULT = [0.25, 0.5, 1.0, 2.0]      # x the 'scale' value 1 / (n_features * X.var())
LINEAR_C = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
FEATURE_CONFIGS = [("resize", False), ("resize", True), ("croppad", False), ("croppad", True)]


def cfg_name(mode, flip):
    return f"{mode}{'+flipavg' if flip else ''}"


def make_pipe(params: dict, probability: bool = False) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(**params, probability=probability, random_state=SEED))])


def fit_score(params, Xtr, ytr, Xva, yva):
    t = time.time()
    pipe = make_pipe(params).fit(Xtr, ytr)
    pred = pipe.predict(Xva)
    return dict(params=params, val_macro_f1=float(f1_score(yva, pred, average="macro")),
                val_accuracy=float((pred == yva).mean()), n_support=int(pipe["svm"].n_support_.sum()),
                fit_seconds=round(time.time() - t, 1)), pred


def gamma_scale(X):
    Xs = StandardScaler().fit_transform(X)
    return 1.0 / (X.shape[1] * Xs.var())


def perturbation_features(rows, mode, size_fn, quality, extractor):
    """Features of perturbed images (plain + flipped)."""
    feats_p, feats_f = [], []
    with ThreadPoolExecutor(4) as pool:
        for i in range(0, len(rows), 32):
            batch = rows[i:i + 32]
            imgs = list(pool.map(lambda r: shortcut.preprocess_array(
                shortcut.perturb(read_rgb(r["path"]), size_fn(r), quality), mode), batch))
            a, b = extractor(np.stack(imgs))
            feats_p.append(a)
            feats_f.append(b)
    return dict(plain=np.concatenate(feats_p), flipped=np.concatenate(feats_f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-jobs", type=int, default=3)
    args = ap.parse_args()
    set_seeds()
    run_dir = new_run("phaseB_frozen_improvements", dict(
        feature_configs=FEATURE_CONFIGS, default_svc=DEFAULT_SVC, rbf_C=RBF_C, rbf_gamma_mult=RBF_GAMMA_MULT,
        linear_C=LINEAR_C, selection_metric="val macro-F1", bootstrap="paired, duplicate-group, 2000x",
        calibration_cv="StratifiedGroupKFold(5) on train duplicate groups"))
    res_dir = run_dir / "results"
    print(f"Run directory: {run_dir}", flush=True)
    report: dict = {}

    tr_rows, va_rows = load_split("train"), load_split("val")
    ytr = np.array([r["y"] for r in tr_rows])
    yva = np.array([r["y"] for r in va_rows])
    gva = np.array([r["group_id"] for r in va_rows])
    gtr = np.array([r["group_id"] for r in tr_rows])

    # ---- 1. features
    print("[1/8] Features", flush=True)
    extractor = FeatureExtractor()
    feats = {m: (cached_features(tr_rows, m, "train", extractor), cached_features(va_rows, m, "val", extractor))
             for m in ("resize", "croppad")}
    fb = {c: sum(1 for r, inf in zip(tr_rows + va_rows, feats["croppad"][0]["infos"] + feats["croppad"][1]["infos"])
                 if r["label"] == c and inf.get("fallback")) for c in CLASS_NAMES}
    report["croppad_fallbacks_train_val"] = fb

    # ---- 2. shortcut probes
    print("[2/8] Shortcut probes", flush=True)
    with ThreadPoolExecutor(4) as pool:
        props_tr = list(pool.map(shortcut.image_properties, tr_rows))
        props_va = list(pool.map(shortcut.image_properties, va_rows))
    report["image_properties_by_class_train"] = shortcut.describe_by_class(props_tr, ytr)
    report["shortcut_probes"] = shortcut.run_probes(props_tr, ytr, props_va, yva, SEED)
    for k, v in report["shortcut_probes"].items():
        print(f"  probe {k}: val acc {v['val_accuracy']:.3f}  notumor AUC {v['val_notumor_auc']:.3f}", flush=True)
    write_json(res_dir / "phase_b.json", report)

    # ---- 3. default SVM per feature config
    print("[3/8] Default SVM on 4 feature configs", flush=True)
    models, preds, table = {}, {}, []
    for mode, flip in FEATURE_CONFIGS:
        name = cfg_name(mode, flip)
        Xtr, Xva = combine(feats[mode][0], flip), combine(feats[mode][1], flip)
        pipe = make_pipe(DEFAULT_SVC).fit(Xtr, ytr)
        models[name], preds[name] = pipe, pipe.predict(Xva)
        rep = metrics.classification_report(yva, preds[name])
        table.append(dict(config=name, val_accuracy=rep["accuracy"], val_macro_f1=rep["macro_f1"],
                          tumor_sens=rep["tumor_vs_notumor"]["sensitivity"],
                          tumor_spec=rep["tumor_vs_notumor"]["specificity"],
                          per_class_f1={c: rep["per_class"][c]["f1"] for c in CLASS_NAMES}, report=rep))
        print(f"  {name}: val acc {rep['accuracy']:.4f} macro-F1 {rep['macro_f1']:.4f}", flush=True)
    report["default_svm_feature_configs"] = table
    write_json(res_dir / "phase_b.json", report)

    # ---- 4. perturbation test
    print("[4/8] Perturbation (causal shortcut) test", flush=True)
    q_nt = float(np.median([p["jpeg_quality"] for p, y in zip(props_tr, ytr) if y == NOTUMOR_INDEX and p["jpeg_quality"] > 0]))
    q_tu = float(np.median([p["jpeg_quality"] for p, y in zip(props_tr, ytr) if y != NOTUMOR_INDEX and p["jpeg_quality"] > 0]))
    tumor_rows = [r for r in va_rows if r["y"] != NOTUMOR_INDEX]
    nt_rows = [r for r in va_rows if r["y"] == NOTUMOR_INDEX]
    conditions = {
        "tumor -> no-tumor-like (225x225, notumor JPEG quality)": (tumor_rows, lambda r: (225, 225), q_nt),
        "tumor -> JPEG only (original size, notumor JPEG quality)": (tumor_rows, lambda r: (r["width"], r["height"]), q_nt),
        "no-tumor -> tumor-like (512x512, tumor JPEG quality)": (nt_rows, lambda r: (512, 512), q_tu),
        "no-tumor -> JPEG only (original size, tumor JPEG quality)": (nt_rows, lambda r: (r["width"], r["height"]), q_tu),
    }
    pert_feats = {}
    pert = dict(jpeg_quality_notumor_median=q_nt, jpeg_quality_tumor_median=q_tu, conditions={})
    va_index = {r["rel"]: i for i, r in enumerate(va_rows)}
    for cname, (rows, size_fn, q) in conditions.items():
        idx = np.array([va_index[r["rel"]] for r in rows])
        to_nt = cname.startswith("tumor")
        pert["conditions"][cname] = {"n": len(rows)}
        for mode in ("resize", "croppad"):
            pf = perturbation_features(rows, mode, size_fn, q, extractor)
            pert_feats[(cname, mode)] = pf
            for flip in (False, True):
                name = cfg_name(mode, flip)
                p_orig = preds[name][idx]
                p_new = models[name].predict(combine(pf, flip))
                y = yva[idx]
                # flip rate: tumor predicted as tumor before -> notumor after (or reverse)
                if to_nt:
                    before = p_orig != NOTUMOR_INDEX
                    flipped = before & (p_new == NOTUMOR_INDEX)
                else:
                    before = p_orig == NOTUMOR_INDEX
                    flipped = before & (p_new != NOTUMOR_INDEX)
                pert["conditions"][cname][name] = dict(
                    accuracy_before=float((p_orig == y).mean()), accuracy_after=float((p_new == y).mean()),
                    shortcut_flip_rate=float(flipped.sum() / max(before.sum(), 1)),
                    any_prediction_changed=float((p_orig != p_new).mean()))
        print(f"  {cname}: " + ", ".join(f"{n} flip {pert['conditions'][cname][n]['shortcut_flip_rate']:.3f}"
                                         for n in map(lambda c: cfg_name(*c), FEATURE_CONFIGS)), flush=True)
    report["perturbation_test"] = pert
    write_json(res_dir / "phase_b.json", report)

    # ---- 5. decisions: preprocessing and flip-averaging
    print("[5/8] Decisions (paired bootstrap on VAL)", flush=True)
    decisions = {}
    d_pre = metrics.paired_bootstrap_diff(yva, preds["resize"], preds["croppad"], gva)
    sc_cond = [c for c in conditions if "-like" in c]
    flip_resize = np.mean([pert["conditions"][c]["resize"]["shortcut_flip_rate"] for c in sc_cond])
    flip_crop = np.mean([pert["conditions"][c]["croppad"]["shortcut_flip_rate"] for c in sc_cond])
    keep_crop = (not d_pre["macro_f1"]["significant_loss"]) and flip_crop < flip_resize
    decisions["preprocessing"] = dict(
        comparison="croppad vs resize (plain features, default SVM)", bootstrap=d_pre,
        mean_shortcut_flip_rate=dict(resize=float(flip_resize), croppad=float(flip_crop)),
        rule="keep croppad if not significantly worse on VAL macro-F1 AND lower shortcut flip rate",
        selected="croppad" if keep_crop else "resize")
    mode = decisions["preprocessing"]["selected"]
    d_flip = metrics.paired_bootstrap_diff(yva, preds[mode], preds[cfg_name(mode, True)], gva)
    keep_flip = d_flip["macro_f1"]["significant_gain"]
    decisions["flip_average"] = dict(comparison=f"{mode}+flipavg vs {mode}", bootstrap=d_flip,
                                     rule="keep only with a significant VAL macro-F1 gain", selected=bool(keep_flip))
    for k, v in decisions.items():
        print(f"  {k}: selected {v['selected']}  ({v['comparison']}: dF1 {v['bootstrap']['macro_f1']['diff']:+.4f} "
              f"CI {v['bootstrap']['macro_f1']['ci95'][0]:+.4f}..{v['bootstrap']['macro_f1']['ci95'][1]:+.4f})", flush=True)
    report["decisions"] = decisions
    write_json(res_dir / "phase_b.json", report)

    # ---- 6. SVM grid on the selected feature config
    print("[6/8] SVM hyper-parameter grid", flush=True)
    sel = cfg_name(mode, keep_flip)
    Xtr, Xva = combine(feats[mode][0], keep_flip), combine(feats[mode][1], keep_flip)
    gs = gamma_scale(Xtr)
    grid = [dict(kernel="rbf", C=c, gamma=float(gs * g), class_weight="balanced") for c in RBF_C for g in RBF_GAMMA_MULT]
    grid += [dict(kernel="linear", C=c, class_weight="balanced") for c in LINEAR_C]
    t = time.time()
    out = Parallel(n_jobs=args.n_jobs)(delayed(fit_score)(p, Xtr, ytr, Xva, yva) for p in grid)
    print(f"  {len(grid)} fits in {time.time() - t:.0f}s", flush=True)
    grid_res = [o[0] for o in out]
    for g in grid_res:
        if g["params"]["kernel"] == "rbf":
            g["params"]["gamma_mult_of_scale"] = round(g["params"]["gamma"] / gs, 3)
    best_i = int(np.argmax([(g["val_macro_f1"], g["val_accuracy"]) for g in grid_res]))
    best = grid_res[best_i]
    best_params = {k: v for k, v in best["params"].items() if k != "gamma_mult_of_scale"}
    d_tune = metrics.paired_bootstrap_diff(yva, preds[sel], out[best_i][1], gva)
    # grouped 5-fold CV on TRAIN: default vs best
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr))

    def cv_f1(params):
        return [float(f1_score(ytr[te], make_pipe(params).fit(Xtr[trn], ytr[trn]).predict(Xtr[te]), average="macro"))
                for trn, te in cv]
    cv_scores = Parallel(n_jobs=2)(delayed(cv_f1)(p) for p in (DEFAULT_SVC, best_params))
    cv_default, cv_best = cv_scores
    keep_tune = (best_params != DEFAULT_SVC and d_tune["macro_f1"]["significant_gain"]
                 and np.mean(cv_best) > np.mean(cv_default))
    decisions["svm_hyperparameters"] = dict(
        comparison=f"best grid point vs default SVM on {sel}", best=best, bootstrap=d_tune,
        train_grouped_cv_macro_f1=dict(default=cv_default, best=cv_best,
                                        default_mean=float(np.mean(cv_default)), best_mean=float(np.mean(cv_best))),
        rule="keep only with a significant VAL gain AND a higher grouped-CV macro-F1 on TRAIN",
        selected=best_params if keep_tune else DEFAULT_SVC)
    report["svm_grid"] = dict(feature_config=sel, gamma_scale=gs, results=grid_res)
    print(f"  best {best['params']} val F1 {best['val_macro_f1']:.4f} (default {f1_score(yva, preds[sel], average='macro'):.4f}); "
          f"CV default {np.mean(cv_default):.4f} best {np.mean(cv_best):.4f} -> keep {keep_tune}", flush=True)
    write_json(res_dir / "phase_b.json", report)

    # ---- 7. calibration
    print("[7/8] Probability calibration", flush=True)
    svc_params = decisions["svm_hyperparameters"]["selected"]
    cands = {"SVC built-in Platt (legacy)": make_pipe(svc_params, probability=True)}
    for method in ("sigmoid", "isotonic"):
        cands[f"grouped-CV {method}"] = CalibratedClassifierCV(make_pipe(svc_params), method=method, cv=cv,
                                                               ensemble=False)
    cal = {}
    fitted = {}
    for name, est in cands.items():
        est.fit(Xtr, ytr)
        proba = est.predict_proba(Xva)
        svm_pred = underlying_pipe(est).predict(Xva)
        c = metrics.calibration_metrics(yva, proba)
        c.update(accuracy_argmax_proba=float((proba.argmax(1) == yva).mean()),
                 macro_f1_argmax_proba=float(f1_score(yva, proba.argmax(1), average="macro")),
                 agreement_with_svm_predict=float((proba.argmax(1) == svm_pred).mean()))
        cal[name], fitted[name] = c, est
        print(f"  {name}: NLL {c['nll']:.4f} Brier {c['brier']:.4f} ECE {c['ece']:.4f} "
              f"acc {c['accuracy_argmax_proba']:.4f} agree {c['agreement_with_svm_predict']:.4f}", flush=True)
    legacy = "SVC built-in Platt (legacy)"
    best_cal = min(cal, key=lambda k: cal[k]["nll"])
    chosen_cal = best_cal if cal[best_cal]["nll"] < cal[legacy]["nll"] else legacy
    decisions["calibration"] = dict(results=cal, rule="keep the method with the lowest VAL NLL (legacy unless improved)",
                                    selected=chosen_cal)
    metrics.plot_reliability(cal, res_dir / "reliability_val.png", f"Calibration on validation ({sel})")
    write_json(res_dir / "phase_b.json", report)

    # ---- 8. save selected model + report
    print("[8/8] Saving selected model and report", flush=True)
    final = fitted[chosen_cal]
    meta = bundle.save_bundle(
        run_dir / "models" / "improved_svm", final, preprocessing=mode, flip_average=bool(keep_flip),
        decision_rule="argmax_proba",
        description="Phase B selected frozen-feature model (decided on validation only)",
        hyperparameters=dict(svc=svc_params, calibration=chosen_cal),
        extra=dict(decisions={k: v["selected"] for k, v in decisions.items()}))
    pred_final, proba_final = bundle.predict(final, meta, Xva)
    rep_final = metrics.classification_report(yva, pred_final)
    rep_final["calibration"] = metrics.calibration_metrics(yva, proba_final)
    pred_tr, _ = bundle.predict(final, meta, Xtr)
    rep_final_train = metrics.classification_report(ytr, pred_tr)
    report["selected_model"] = dict(bundle=meta, val=rep_final, train=rep_final_train)
    metrics.plot_confusion(rep_final["confusion_matrix"], res_dir / "confusion_val_improved.png",
                           f"Improved model ({sel}) - validation (n={len(yva)})")
    write_json(res_dir / "phase_b.json", report)
    (res_dir / "phase_b_report.md").write_text(render_report(report, sel), encoding="utf-8")
    print(f"Done. Outputs in {run_dir}", flush=True)


def underlying_pipe(est):
    """The fitted scaler+SVC pipeline inside a (possibly calibrated) estimator."""
    if isinstance(est, CalibratedClassifierCV):
        return est.calibrated_classifiers_[0].estimator
    return est


def render_report(r: dict, sel: str) -> str:
    L = ["# Phase B - frozen-feature improvements (validation only)", ""]
    L += ["## Shortcut probes (train -> val, no anatomy used)", "",
          metrics.fmt_table([[k, v["val_accuracy"], v["val_macro_f1"], v["val_notumor_auc"]]
                             for k, v in r["shortcut_probes"].items()],
                            ["Probe", "Val accuracy", "Val macro-F1", "No-tumor AUC"]),
          "", f"Chance accuracy (majority class): {next(iter(r['shortcut_probes'].values()))['chance_accuracy']:.3f}", ""]
    L += ["## Default SVM per feature configuration (val)", "",
          metrics.fmt_table([[t["config"], t["val_accuracy"], t["val_macro_f1"], t["tumor_sens"], t["tumor_spec"]]
                             for t in r["default_svm_feature_configs"]],
                            ["Config", "Accuracy", "Macro-F1", "Tumor sens", "Tumor spec"]), ""]
    L += ["## Perturbation test (val)", ""]
    rows = []
    for c, v in r["perturbation_test"]["conditions"].items():
        for n in [cfg_name(*x) for x in FEATURE_CONFIGS]:
            rows.append([c, n, v["n"], v[n]["shortcut_flip_rate"], v[n]["accuracy_before"], v[n]["accuracy_after"]])
    L += [metrics.fmt_table(rows, ["Condition", "Model", "n", "Shortcut flip rate", "Acc before", "Acc after"]), ""]
    L += ["## Decisions", ""]
    for k, v in r["decisions"].items():
        b = v.get("bootstrap", {}).get("macro_f1")
        extra = f" dF1 {b['diff']:+.4f} (95% CI {b['ci95'][0]:+.4f}..{b['ci95'][1]:+.4f})" if b else ""
        L.append(f"- **{k}**: selected `{v['selected']}`. Rule: {v['rule']}.{extra}")
    s = r["selected_model"]
    L += ["", f"## Selected model ({sel})", "",
          f"Val accuracy {s['val']['accuracy']:.4f}, macro-F1 {s['val']['macro_f1']:.4f}, "
          f"tumor-vs-no-tumor sensitivity {s['val']['tumor_vs_notumor']['sensitivity']:.4f}, "
          f"specificity {s['val']['tumor_vs_notumor']['specificity']:.4f}; train accuracy {s['train']['accuracy']:.4f}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
