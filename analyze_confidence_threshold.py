"""Confidence-threshold analysis for the low-confidence warning (VALIDATION split only, analysis only).

For the Phase B bundle, evaluates "accept a prediction only if its top Platt probability >= t" over a range of
candidate thresholds t on the validation split, and reports for each t:
  accepted / rejected counts, coverage, accuracy among accepted, accuracy among rejected,
  per-class performance on accepted images, and the confusion matrix of accepted images.

This script changes nothing in the app. It loads the validation split only (allow_test=False) and reuses
the cached validation features that evaluate.py uses.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from inference import MODEL_BUNDLE_DIR
from neuroinsight import bundle, metrics
from neuroinsight.features import cached_features, combine
from neuroinsight.manifest import CLASS_NAMES, load_split
from neuroinsight.runinfo import new_run, set_seeds, write_json

CANDIDATES = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]


def per_class_on_accepted(y, pred, accept):
    out = {}
    for k, name in enumerate(CLASS_NAMES):
        true_k, pred_k = (y == k), (pred == k)
        n_true, n_true_acc = int(true_k.sum()), int((true_k & accept).sum())
        n_pred_acc = int((pred_k & accept).sum())
        tp = int((true_k & pred_k & accept).sum())
        out[name] = dict(
            n_true=n_true, n_true_accepted=n_true_acc,
            coverage=n_true_acc / n_true if n_true else None,
            recall_on_accepted=tp / n_true_acc if n_true_acc else None,        # of accepted true-k images, found as k
            precision_on_accepted=tp / n_pred_acc if n_pred_acc else None,     # of accepted predicted-k images, truly k
        )
    return out


def main() -> None:
    set_seeds()
    est, meta = bundle.load_bundle(MODEL_BUNDLE_DIR)
    rows = load_split("val", allow_test=False)
    y = np.array([r["y"] for r in rows])
    feats = cached_features(rows, meta["preprocessing"], "val")
    pred, proba = bundle.predict(est, meta, combine(feats, meta["flip_average"]))
    conf = proba.max(1)
    correct = pred == y
    run_dir = new_run("confidence_threshold_analysis", dict(split="val", n=len(y), bundle=str(MODEL_BUNDLE_DIR),
                                                            candidates=CANDIDATES))
    print(f"val n={len(y)}  overall accuracy {correct.mean():.4f}  (all images, no threshold)")

    results = []
    for t in CANDIDATES:
        acc_mask = conf >= t
        n_acc, n_rej = int(acc_mask.sum()), int((~acc_mask).sum())
        res = dict(
            threshold=t, n_accepted=n_acc, n_rejected=n_rej, coverage=n_acc / len(y),
            accuracy_accepted=float(correct[acc_mask].mean()) if n_acc else None,
            accuracy_rejected=float(correct[~acc_mask].mean()) if n_rej else None,
            errors_accepted=int((~correct[acc_mask]).sum()), errors_rejected=int((~correct[~acc_mask]).sum()),
            share_of_all_errors_flagged=float((~correct[~acc_mask]).sum() / max((~correct).sum(), 1)),
            per_class=per_class_on_accepted(y, pred, acc_mask),
            confusion_matrix_accepted=metrics.confusion(y[acc_mask], pred[acc_mask]).tolist() if n_acc else None,
        )
        if n_acc:
            rep = metrics.classification_report(y[acc_mask], pred[acc_mask])
            res["macro_f1_accepted"] = rep["macro_f1"]
            tv = rep["tumor_vs_notumor"]
            res["tumor_vs_notumor_accepted"] = dict(sensitivity=tv["sensitivity"], specificity=tv["specificity"],
                                                    fn=tv["fn"], fp=tv["fp"])
        results.append(res)

    write_json(run_dir / "results" / "threshold_analysis.json",
               dict(split="val", n=len(y), classes=CLASS_NAMES, overall_accuracy=float(correct.mean()),
                    note="Validation split only. Val was also used to select the model, so these numbers are "
                         "optimistic for new data. No patient IDs exist, so there is no patient-level guarantee.",
                    results=results))

    # human-readable summary
    print(f"\n{'thr':>5} {'accepted':>8} {'rejected':>8} {'coverage':>8} {'acc(acc)':>8} {'acc(rej)':>8} "
          f"{'err acc':>7} {'err rej':>7} {'errs flagged':>12} {'F1(acc)':>8}")
    for r in results:
        aa = "-" if r["accuracy_accepted"] is None else f"{r['accuracy_accepted']:.4f}"
        ar = "-" if r["accuracy_rejected"] is None else f"{r['accuracy_rejected']:.4f}"
        f1 = "-" if r.get("macro_f1_accepted") is None else f"{r['macro_f1_accepted']:.4f}"
        print(f"{r['threshold']:>5.2f} {r['n_accepted']:>8d} {r['n_rejected']:>8d} {r['coverage']:>8.3f} {aa:>8} {ar:>8} "
              f"{r['errors_accepted']:>7d} {r['errors_rejected']:>7d} {r['share_of_all_errors_flagged']:>12.3f} {f1:>8}")
    print(f"\nwritten to {run_dir / 'results' / 'threshold_analysis.json'}")


if __name__ == "__main__":
    main()
