"""Shortcut (dataset-bias) analysis: can non-anatomical properties predict the class?

1. Probes - a gradient-boosting classifier trained on TRAIN and scored on VAL using only
   a) geometry      : width, height, aspect ratio, pixel count of the original file
   b) encoding      : bytes per pixel, JPEG quantisation (quality estimate), stored colour mode
   c) intensity     : grey-level statistics of the crop-padded 224x224 image (no geometry)
   d) crop geometry : what is left of geometry after crop+pad (head bbox aspect, area fraction)
   High probe accuracy = the class is predictable from that property alone.

2. Perturbation test - a causal check on the trained classifier: VAL tumor images are given
   no-tumor-like acquisition properties (downscaled to 225x225 = the most common no-tumor size,
   JPEG re-encoded at the median no-tumor quality); VAL no-tumor images are given tumor-like
   ones (upscaled to 512x512 = the tumor-class size, JPEG at the median tumor quality). If
   predictions flip, the classifier relies on size/compression rather than anatomy.
"""
from __future__ import annotations

import io
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from .manifest import CLASS_NAMES, NOTUMOR_INDEX
from .preprocessing import preprocess_croppad, read_rgb, resize_baseline

# Standard IJG luminance quantisation table (quality 50), zig-zag order irrelevant for the mean
_STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
                      69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
                      81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
                     dtype=float)


def jpeg_quality(pil: Image.Image) -> float:
    q = getattr(pil, "quantization", None)
    if not q or 0 not in q:
        return -1.0
    s = 100.0 * np.mean(np.array(q[0], dtype=float)) / _STD_LUMA.mean()
    return float((200 - s) / 2 if s <= 100 else 5000 / s)


def image_properties(row: dict) -> dict:
    path = row["path"]
    with Image.open(path) as pil:
        mode, fmt, quality = pil.mode, pil.format, jpeg_quality(pil)
        chroma = len(getattr(pil, "quantization", {}) or {})
    img = read_rgb(path)
    h, w = img.shape[:2]
    pre, info = preprocess_croppad(img)
    g = pre.mean(axis=2)
    hist = np.histogram(g, bins=32, range=(0, 255))[0] / g.size
    ch, cw = info["crop_hw"]
    return dict(
        width=w, height=h, aspect=w / h, log_pixels=float(np.log(w * h)),
        bytes_per_pixel=path.stat().st_size / (w * h), jpeg_quality=quality, n_quant_tables=chroma,
        stored_gray=int(mode == "L"), is_jpeg=int(fmt == "JPEG"),
        colour_diff=float(np.abs(img[..., 0].astype(int) - img[..., 2].astype(int)).mean()),
        g_mean=float(g.mean()), g_std=float(g.std()),
        **{f"g_p{p}": float(np.percentile(g, p)) for p in (5, 25, 50, 75, 95, 99)},
        g_black=float((g < 10).mean()), g_entropy=float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum()),
        g_laplacian_var=float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var()),
        crop_aspect=cw / ch, crop_area_frac=info["bbox_area_frac"],
    )


PROBE_GROUPS = {
    "geometry (original size/aspect)": ["width", "height", "aspect", "log_pixels"],
    "encoding (bytes/px, JPEG quality, colour mode)": ["bytes_per_pixel", "jpeg_quality", "n_quant_tables",
                                                       "stored_gray", "is_jpeg", "colour_diff"],
    "geometry + encoding": ["width", "height", "aspect", "log_pixels", "bytes_per_pixel", "jpeg_quality",
                            "n_quant_tables", "stored_gray", "is_jpeg", "colour_diff"],
    "crop geometry after crop+pad": ["crop_aspect", "crop_area_frac"],
    "intensity statistics of crop+pad image": ["g_mean", "g_std", "g_p5", "g_p25", "g_p50", "g_p75", "g_p95",
                                               "g_p99", "g_black", "g_entropy", "g_laplacian_var"],
}


def run_probes(props_tr: list[dict], y_tr, props_va: list[dict], y_va, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    out = {}
    for name, cols in PROBE_GROUPS.items():
        Xtr = np.array([[p[c] for c in cols] for p in props_tr], dtype=float)
        Xva = np.array([[p[c] for c in cols] for p in props_va], dtype=float)
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05)
        clf.fit(Xtr, y_tr)
        proba = clf.predict_proba(Xva)
        pred = proba.argmax(1)
        y_nt = (np.asarray(y_va) == NOTUMOR_INDEX).astype(int)
        out[name] = dict(features=cols, val_accuracy=float((pred == y_va).mean()),
                         val_macro_f1=float(f1_score(y_va, pred, average="macro")),
                         val_notumor_auc=float(roc_auc_score(y_nt, proba[:, NOTUMOR_INDEX])),
                         chance_accuracy=float(max(Counter(y_va).values()) / len(y_va)))
    return out


def describe_by_class(props: list[dict], y) -> dict:
    y = np.asarray(y)
    out = {}
    for i, c in enumerate(CLASS_NAMES):
        ps = [p for p, yy in zip(props, y) if yy == i]
        out[c] = {k: float(np.median([p[k] for p in ps])) for k in
                  ("width", "height", "aspect", "bytes_per_pixel", "jpeg_quality", "stored_gray", "g_mean",
                   "g_p95", "g_laplacian_var")}
        out[c]["share_512x512"] = float(np.mean([p["width"] == 512 and p["height"] == 512 for p in ps]))
        out[c]["share_square"] = float(np.mean([p["width"] == p["height"] for p in ps]))
    return out


def perturb(img: np.ndarray, size: tuple[int, int], quality: int) -> np.ndarray:
    """Resample to size=(w, h) (area / cubic) and JPEG re-encode at `quality`."""
    w, h = size
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    small = cv2.resize(img, (w, h), interpolation=interp)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format="JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def preprocess_array(img: np.ndarray, mode: str) -> np.ndarray:
    return resize_baseline(img) if mode == "resize" else preprocess_croppad(img)[0]
