"""Verify that the web app's inference path reproduces the evaluation pipeline exactly (VALIDATION split only).

Three paths are compared on the same validation images:
  A. evaluate.py path   : cached_features(rows, "resize", "val") -> combine -> bundle.predict
  B. fresh batch path   : neuroinsight.features.extract(rows, "resize") (same code, no cache) -> bundle.predict   [--fresh-n images]
  C. web app path       : inference.Predictor.predict_bytes(raw file bytes), one image at a time

Reports max |feature difference|, max |probability difference| and the number of differing labels.
The test split is never loaded (load_split is called with allow_test=False).

    python verify_app_pipeline.py            # all validation images
    python verify_app_pipeline.py --n 40     # quick check
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import time

import numpy as np

from inference import MODEL_BUNDLE_DIR, Predictor
from neuroinsight import bundle, metrics
from neuroinsight.features import cached_features, combine, extract
from neuroinsight.manifest import load_split
from neuroinsight.runinfo import new_run, set_seeds, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=0, help="only the first N validation images (0 = all)")
    ap.add_argument("--fresh-n", type=int, default=64, help="images for the uncached batch path B")
    args = ap.parse_args()

    set_seeds()
    rows_all = load_split("val", allow_test=False)       # val only
    assert all(r["split"] == "val" for r in rows_all) if "split" in rows_all[0] else True
    rows = rows_all[: args.n] if args.n else rows_all
    y = np.array([r["y"] for r in rows])
    run_dir = new_run("verify_app_pipeline", dict(split="val", n=len(rows), bundle=str(MODEL_BUNDLE_DIR)))

    predictor = Predictor()
    est, meta = predictor.estimator, predictor.meta
    print(f"bundle: {MODEL_BUNDLE_DIR}\n  preprocessing={meta['preprocessing']} flip_average={meta['flip_average']} "
          f"decision_rule={meta['decision_rule']}", flush=True)

    # A. evaluate.py path (cached val features), restricted to the chosen rows by relative path
    cache_all = cached_features(rows_all, "resize", "val", predictor.extractor)
    pos = {rel: i for i, rel in enumerate(cache_all["rels"])}
    idx = np.array([pos[r["rel"]] for r in rows])
    XA = combine(dict(plain=cache_all["plain"][idx], flipped=cache_all["flipped"][idx]), meta["flip_average"])
    predA, probaA = bundle.predict(est, meta, XA)

    # B. fresh (uncached) batch extraction on a subset
    nb = min(args.fresh_n, len(rows))
    fb = extract(rows[:nb], "resize", predictor.extractor)
    XB = combine(fb, meta["flip_average"])
    predB, probaB = bundle.predict(est, meta, XB)

    # C. web app path from the raw file bytes
    t0 = time.time()
    XC, probaC, predC = [], [], []
    for i, r in enumerate(rows):
        with open(r["path"], "rb") as fh:
            res = predictor.predict_bytes(fh.read())
        XC.append(res.features)
        probaC.append([res.probabilities[c] for c in predictor.class_names])
        predC.append(res.index)
        if (i + 1) % 100 == 0:
            print(f"  app path {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    XC, probaC, predC = np.array(XC), np.array(probaC), np.array(predC)

    def cmp(name, X1, P1, L1, X2, P2, L2):
        return dict(compare=name, n=int(len(L1)),
                    features_bitwise_identical=bool(np.array_equal(X1, X2)),
                    max_abs_feature_diff=float(np.abs(X1 - X2).max()),
                    max_abs_feature_diff_relative=float(np.abs(X1 - X2).max() / max(np.abs(X1).max(), 1e-12)),
                    probabilities_bitwise_identical=bool(np.array_equal(P1, P2)),
                    max_abs_probability_diff=float(np.abs(P1 - P2).max()),
                    n_labels_differ=int((L1 != L2).sum()))

    out = [cmp("A evaluate-path(cached) vs C app-path", XA, probaA, predA, XC, probaC, predC),
           cmp("B fresh-batch vs C app-path", XB, probaB, predB, XC[:nb], probaC[:nb], predC[:nb]),
           cmp("A evaluate-path(cached) vs B fresh-batch", XA[:nb], probaA[:nb], predA[:nb], XB, probaB, predB)]
    rep_A, rep_C = metrics.classification_report(y, predA), metrics.classification_report(y, predC)
    summary = dict(split="val", n=len(rows), comparisons=out,
                   accuracy_evaluate_path=rep_A["accuracy"], accuracy_app_path=rep_C["accuracy"],
                   macro_f1_evaluate_path=rep_A["macro_f1"], macro_f1_app_path=rep_C["macro_f1"],
                   bundle=predictor.describe())
    write_json(run_dir / "results" / "verification.json", summary)

    print("\n=== VERIFICATION (validation split only) ===")
    for c in out:
        print(f"{c['compare']}: n={c['n']}  features bitwise identical={c['features_bitwise_identical']}  "
              f"max|dFeature|={c['max_abs_feature_diff']:.3g}  max|dProb|={c['max_abs_probability_diff']:.3g}  "
              f"labels differing={c['n_labels_differ']}")
    print(f"accuracy  evaluate path {rep_A['accuracy']:.4f}   app path {rep_C['accuracy']:.4f}")
    print(f"macro-F1  evaluate path {rep_A['macro_f1']:.4f}   app path {rep_C['macro_f1']:.4f}")
    print(f"written to {run_dir / 'results' / 'verification.json'}")


if __name__ == "__main__":
    main()
