"""Classification metrics, calibration metrics, grouped bootstrap CIs and plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import CLASS_NAMES, NOTUMOR_INDEX

K = len(CLASS_NAMES)


def confusion(y_true, y_pred, k: int = K) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def _safe(a, b):
    return float(a / b) if b else float("nan")


def metrics_from_cm(cm: np.ndarray) -> dict:
    n = cm.sum()
    per_class = {}
    for i, c in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n - tp - fn - fp
        prec, rec = _safe(tp, tp + fp), _safe(tp, tp + fn)
        per_class[c] = dict(
            support=int(cm[i].sum()), precision=prec, recall=rec,
            f1=_safe(2 * prec * rec, prec + rec) if not np.isnan(prec + rec) else float("nan"),
            sensitivity=rec, specificity=_safe(tn, tn + fp),
        )
    # tumor (any of glioma/meningioma/pituitary) vs no tumor; positive = tumor
    t = [i for i in range(K) if i != NOTUMOR_INDEX]
    tp = cm[np.ix_(t, t)].sum()
    fn = cm[t, NOTUMOR_INDEX].sum()
    fp = cm[NOTUMOR_INDEX, t].sum()
    tn = cm[NOTUMOR_INDEX, NOTUMOR_INDEX]
    macro = {m: float(np.mean([per_class[c][m] for c in CLASS_NAMES]))
             for m in ("precision", "recall", "f1", "specificity")}
    return dict(
        n=int(n), accuracy=_safe(np.trace(cm), n),
        macro_precision=macro["precision"], macro_recall=macro["recall"], macro_f1=macro["f1"],
        macro_specificity=macro["specificity"],
        per_class=per_class,
        tumor_vs_notumor=dict(sensitivity=_safe(tp, tp + fn), specificity=_safe(tn, tn + fp),
                              tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn)),
        confusion_matrix=cm.tolist(),
    )


def classification_report(y_true, y_pred) -> dict:
    return metrics_from_cm(confusion(y_true, y_pred))


def calibration_metrics(y_true, proba, n_bins: int = 15) -> dict:
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    onehot = np.eye(proba.shape[1])[y_true]
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            bins.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                             accuracy=float(correct[m].mean()), confidence=float(conf[m].mean())))
    return dict(nll=float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true]))),
                brier=float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
                ece=float(ece), mean_confidence=float(conf.mean()), bins=bins)


def bootstrap_ci(y_true, y_pred, groups, n_boot: int = 2000, seed: int = 1337,
                 alpha: float = 0.05) -> dict:
    """95% percentile CIs, resampling duplicate GROUPS (not images) with replacement."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    keys += [f"{c}.{m}" for c in CLASS_NAMES for m in ("precision", "recall", "f1", "specificity")]
    keys += ["tumor_vs_notumor.sensitivity", "tumor_vs_notumor.specificity"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        m = metrics_from_cm(confusion(y_true[idx], y_pred[idx]))
        for k in keys:
            if "." in k:
                a, b = k.split(".")
                v = m["tumor_vs_notumor"][b] if a == "tumor_vs_notumor" else m["per_class"][a][b]
            else:
                v = m[k]
            samples[k].append(v)
    out = {}
    for k, v in samples.items():
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
    return dict(n_boot=n_boot, resampling_unit="duplicate group", n_groups=int(len(uniq)), ci95=out)


def paired_bootstrap_diff(y_true, pred_a, pred_b, groups, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Difference B - A in macro-F1 and accuracy, with a 95% CI from a paired, duplicate-group bootstrap."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = {}
    base = {k: (metrics_from_cm(confusion(y_true, pred_b))[k] - metrics_from_cm(confusion(y_true, pred_a))[k])
            for k in ("macro_f1", "accuracy")}
    samples = {k: [] for k in base}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        ma = metrics_from_cm(confusion(y_true[idx], pred_a[idx]))
        mb = metrics_from_cm(confusion(y_true[idx], pred_b[idx]))
        for k in samples:
            samples[k].append(mb[k] - ma[k])
    for k, v in samples.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = dict(diff=float(base[k]), ci95=[float(lo), float(hi)],
                      significant_gain=bool(lo > 0), significant_loss=bool(hi < 0))
    out["n_changed_predictions"] = int((pred_a != pred_b).sum())
    return out


# ----------------------------------------------------------------------------- plots

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    return plt


def plot_confusion(cm, path: Path, title: str) -> None:
    plt = _plt()
    cm = np.asarray(cm)
    rows = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{rows[i, j]:.1%}", ha="center", va="center", fontsize=8.5,
                    color="white" if rows[i, j] > 0.6 else "#1a1a1a")
    ax.set_xticks(range(K), CLASS_NAMES, rotation=20)
    ax.set_yticks(range(K), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (row-normalised %)")
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(curves: dict, path: Path, title: str) -> None:
    """curves: {name: calibration_metrics(...)}."""
    plt = _plt()
    colors = ["#2a6fdb", "#e07a1f", "#1f9e74", "#8a5cd1"]
    fig, ax = plt.subplots(figsize=(5, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1, ls="--", label="perfect calibration")
    for (name, cal), col in zip(curves.items(), colors):
        b = [x for x in cal["bins"] if x["n"] >= 5]
        ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], marker="o", ms=4, lw=2,
                color=col, label=f"{name} (ECE {cal['ece']:.3f})")
    ax.set_xlabel("Mean predicted confidence (top class)")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0.2, 1.01)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(title, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt_table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)
