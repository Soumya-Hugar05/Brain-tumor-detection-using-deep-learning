"""Phase E - evaluation of frozen model bundles.

Validation (allowed any time):
    python evaluate.py --split val --model runs/<A>/models/baseline_svm --model runs/<B>/models/improved_svm

Freeze the selection (after the user approves the chosen models):
    python evaluate.py --freeze --model ... --model ...
    -> runs/FROZEN_SELECTION.json with the SHA-256 of every estimator and bundle.json

Final test evaluation (only after freezing and the user's approval):
    python evaluate.py --split test --final --model ... --model ...
    * refuses unless every model is in FROZEN_SELECTION.json with identical hashes
    * refuses a second test evaluation unless --allow-repeat (every access is logged in
      runs/test_access_log.jsonl)
    * test features are extracted fresh and never cached

Reports: confusion matrix, accuracy, macro and per-class precision / recall / F1,
class-wise sensitivity and specificity, tumor-vs-no-tumor sensitivity and specificity,
95% bootstrap CIs (resampling duplicate groups), calibration, and a paired bootstrap
comparison of the first two models.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from neuroinsight import bundle, metrics
from neuroinsight.features import FeatureExtractor, cached_features, combine, extract
from neuroinsight.manifest import CLASS_NAMES, load_split, sha256_file, verify_manifest
from neuroinsight.runinfo import RUNS_DIR, new_run, set_seeds, write_json

FROZEN = RUNS_DIR / "FROZEN_SELECTION.json"
ACCESS_LOG = RUNS_DIR / "test_access_log.jsonl"


def bundle_hashes(path: Path) -> dict:
    return dict(path=path.resolve().relative_to(RUNS_DIR.parent).as_posix(),
                estimator_sha256=sha256_file(path / "estimator.joblib"),
                bundle_json_sha256=sha256_file(path / "bundle.json"))


def check_frozen(models: list[Path]) -> None:
    if not FROZEN.is_file():
        sys.exit("REFUSED: no frozen selection (runs/FROZEN_SELECTION.json). Run --freeze after approval.")
    frozen = {m["path"]: m for m in json.loads(FROZEN.read_text(encoding="utf-8"))["models"]}
    for p in models:
        h = bundle_hashes(p)
        f = frozen.get(h["path"])
        if f is None or f["estimator_sha256"] != h["estimator_sha256"] or f["bundle_json_sha256"] != h["bundle_json_sha256"]:
            sys.exit(f"REFUSED: {h['path']} is not in the frozen selection or was modified after freezing.")


def log_access(entry: dict) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def previous_test_runs() -> int:
    if not ACCESS_LOG.is_file():
        return 0
    return sum(1 for line in ACCESS_LOG.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line).get("event") == "test_evaluation_started")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, action="append", required=True, help="bundle directory (repeatable)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required for --split test")
    ap.add_argument("--allow-repeat", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="write runs/FROZEN_SELECTION.json and exit")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    for p in args.model:
        bundle.load_bundle(p)  # validates hashes
    if args.freeze:
        if FROZEN.exists():
            sys.exit(f"REFUSED: {FROZEN} already exists; selection is frozen.")
        write_json(FROZEN, dict(frozen_at=datetime.now().isoformat(timespec="seconds"),
                                manifest_sha256=verify_manifest(), models=[bundle_hashes(p) for p in args.model]))
        print(f"Frozen selection written to {FROZEN}")
        return

    if args.split == "test":
        if not args.final:
            sys.exit("REFUSED: the test split requires --final (after the selection is frozen and approved).")
        check_frozen(args.model)
        n_prev = previous_test_runs()
        if n_prev and not args.allow_repeat:
            sys.exit(f"REFUSED: the test set was already evaluated {n_prev} time(s); use --allow-repeat deliberately.")

    set_seeds()
    run_dir = new_run(f"phaseE_eval_{args.split}", dict(split=args.split, models=[bundle_hashes(p) for p in args.model],
                                                       n_boot=args.n_boot))
    if args.split == "test":
        log_access(dict(event="test_evaluation_started", time=datetime.now().isoformat(timespec="seconds"),
                        run=run_dir.name, models=[bundle_hashes(p)["path"] for p in args.model]))
    rows = load_split(args.split, allow_test=args.split == "test")
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    extractor = FeatureExtractor()
    feat_cache = {}
    results, preds = {}, {}
    for p in args.model:
        est, meta = bundle.load_bundle(p)
        mode = meta["preprocessing"]
        if mode not in feat_cache:
            feat_cache[mode] = (extract(rows, mode, extractor) if args.split == "test"
                                else cached_features(rows, mode, args.split, extractor))
        X = combine(feat_cache[mode], meta["flip_average"])
        pred, proba = bundle.predict(est, meta, X)
        rep = metrics.classification_report(y, pred)
        if proba is not None:
            rep["calibration"] = metrics.calibration_metrics(y, proba)
        rep["bootstrap"] = metrics.bootstrap_ci(y, pred, groups, n_boot=args.n_boot)
        name = p.name if p.name not in results else f"{p.parent.parent.name}/{p.name}"
        results[name], preds[name] = dict(bundle=str(p), meta=meta, report=rep), pred
        metrics.plot_confusion(rep["confusion_matrix"], run_dir / "results" / f"confusion_{name.replace('/', '_')}.png",
                               f"{name} - {args.split} (n={len(y)})")
        print(f"{name}: accuracy {rep['accuracy']:.4f} macro-F1 {rep['macro_f1']:.4f}", flush=True)
    names = list(results)
    comparison = None
    if len(names) >= 2:
        comparison = dict(a=names[0], b=names[1],
                          **metrics.paired_bootstrap_diff(y, preds[names[0]], preds[names[1]], groups, args.n_boot))
    write_json(run_dir / "results" / "evaluation.json", dict(split=args.split, n=len(y), models=results,
                                                             comparison=comparison))
    (run_dir / "results" / "evaluation.md").write_text(render(results, comparison, args.split, len(y)), encoding="utf-8")
    if args.split == "test":
        log_access(dict(event="test_evaluation_finished", time=datetime.now().isoformat(timespec="seconds"),
                        run=run_dir.name))
    print(f"Done. Outputs in {run_dir}")


def render(results, comparison, split, n) -> str:
    L = [f"# Evaluation on the {split} split (n={n})", ""]
    for name, r in results.items():
        rep, ci = r["report"], r["report"]["bootstrap"]["ci95"]
        f = lambda v, k: f"{v:.4f} ({ci[k][0]:.4f}-{ci[k][1]:.4f})"
        L += [f"## {name}", "", f"Preprocessing `{r['meta']['preprocessing']}`, flip-average {r['meta']['flip_average']}, "
              f"decision rule `{r['meta']['decision_rule']}`.", "",
              "| Metric | Value (95% CI) |", "|---|---|",
              f"| Accuracy | {f(rep['accuracy'], 'accuracy')} |",
              f"| Macro precision | {f(rep['macro_precision'], 'macro_precision')} |",
              f"| Macro recall | {f(rep['macro_recall'], 'macro_recall')} |",
              f"| Macro F1 | {f(rep['macro_f1'], 'macro_f1')} |",
              f"| Tumor-vs-no-tumor sensitivity | {f(rep['tumor_vs_notumor']['sensitivity'], 'tumor_vs_notumor.sensitivity')} |",
              f"| Tumor-vs-no-tumor specificity | {f(rep['tumor_vs_notumor']['specificity'], 'tumor_vs_notumor.specificity')} |",
              "", "| Class | n | Precision | Recall / sensitivity | F1 | Specificity |", "|---|---|---|---|---|---|"]
        for c in CLASS_NAMES:
            pc = rep["per_class"][c]
            L.append(f"| {c} | {pc['support']} | {f(pc['precision'], c + '.precision')} | {f(pc['recall'], c + '.recall')} | "
                     f"{f(pc['f1'], c + '.f1')} | {f(pc['specificity'], c + '.specificity')} |")
        L += ["", "Confusion matrix (rows = true, columns = predicted, order " + ", ".join(CLASS_NAMES) + "):", "",
              "```", *["  ".join(f"{v:5d}" for v in row) for row in rep["confusion_matrix"]], "```", ""]
        if "calibration" in rep:
            c = rep["calibration"]
            L += [f"Calibration: NLL {c['nll']:.4f}, Brier {c['brier']:.4f}, ECE {c['ece']:.4f}.", ""]
    if comparison:
        d = comparison["macro_f1"]
        L += ["## Paired comparison", "", f"{comparison['b']} - {comparison['a']}: macro-F1 {d['diff']:+.4f} "
              f"(95% CI {d['ci95'][0]:+.4f}..{d['ci95'][1]:+.4f}); accuracy {comparison['accuracy']['diff']:+.4f} "
              f"(95% CI {comparison['accuracy']['ci95'][0]:+.4f}..{comparison['accuracy']['ci95'][1]:+.4f}); "
              f"{comparison['n_changed_predictions']} predictions differ.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
