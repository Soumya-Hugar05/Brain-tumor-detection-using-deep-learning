"""Image loading and preprocessing.

Modes
-----
resize   : the legacy pipeline exactly (keras image_dataset_from_directory): decode to RGB and
           bilinear-resize straight to 224x224 (no antialiasing). Non-square images get stretched.
croppad  : brain-region crop + pad to square, then antialiased resize to 224x224. The aspect
           ratio is preserved (no anatomical distortion), the outer background/frame is
           removed, and every image ends up with the same geometry regardless of its original
           size - this removes image size / aspect ratio / border as class cues.

Both modes return float32 RGB in [0, 255]; ResNet50 `preprocess_input` is applied later.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
MODES = ("resize", "croppad")

# croppad parameters (chosen by visual inspection of TRAIN images only)
BLUR_KSIZE = 5
MIN_THRESHOLD = 12          # grey level; lower bound on the foreground threshold
OTSU_FACTOR = 0.5           # foreground threshold = max(MIN_THRESHOLD, OTSU_FACTOR * otsu)
OPEN_FRAC = 0.01            # opening kernel = 1% of the shorter side (removes text / specks)
KEEP_COMPONENT_FRAC = 0.10  # keep components with area >= 10% of the largest one
MARGIN_FRAC = 0.03          # margin around the bounding box (fraction of its longer side)
MIN_AREA_FRAC = 0.03        # fallback to full image if the foreground is smaller than this


def read_rgb(path: str | Path) -> np.ndarray:
    data = tf.io.read_file(str(path))
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    return img.numpy()  # uint8 HxWx3


def resize_baseline(img: np.ndarray) -> np.ndarray:
    return tf.image.resize(img, IMAGE_SIZE, method="bilinear").numpy().astype(np.float32)


def brain_bbox(img: np.ndarray) -> tuple[tuple[int, int, int, int], dict]:
    """Bounding box (x0, y0, x1, y1) of the head region. Returns (bbox, info)."""
    h, w = img.shape[:2]
    gray = img.mean(axis=2).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(MIN_THRESHOLD, OTSU_FACTOR * otsu)
    mask = (blur > thr).astype(np.uint8)
    k = max(3, int(round(OPEN_FRAC * min(h, w))) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    info = dict(threshold=float(thr), otsu=float(otsu), fallback=False)
    if n <= 1:
        info["fallback"] = True
        return (0, 0, w, h), info
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= KEEP_COMPONENT_FRAC * areas.max())[0] + 1
    x0 = stats[keep, cv2.CC_STAT_LEFT].min()
    y0 = stats[keep, cv2.CC_STAT_TOP].min()
    x1 = (stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max()
    y1 = (stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max()
    if areas[keep - 1].sum() < MIN_AREA_FRAC * h * w:
        info["fallback"] = True
        return (0, 0, w, h), info
    m = int(round(MARGIN_FRAC * max(x1 - x0, y1 - y0)))
    bbox = (max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m))
    info["bg_value"] = [float(v) for v in np.median(img[mask == 0], axis=0)] if (mask == 0).any() else [0.0] * 3
    return bbox, info


def crop_pad(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Crop to the head bounding box and pad to a square (no resizing). Returns uint8 image."""
    (x0, y0, x1, y1), info = brain_bbox(img)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    bg = np.array(info.get("bg_value", [0, 0, 0]), dtype=np.float32)
    out = np.empty((side, side, 3), dtype=np.uint8)
    out[:] = np.clip(np.round(bg), 0, 255).astype(np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = crop
    info.update(bbox=[int(x0), int(y0), int(x1), int(y1)], crop_hw=[int(ch), int(cw)],
                bbox_area_frac=float(ch * cw / (img.shape[0] * img.shape[1])))
    return out, info


def preprocess_croppad(img: np.ndarray) -> tuple[np.ndarray, dict]:
    sq, info = crop_pad(img)
    out = tf.image.resize(sq, IMAGE_SIZE, method="bilinear", antialias=True).numpy()
    return np.clip(out, 0, 255).astype(np.float32), info


def load_preprocessed(path: str | Path, mode: str) -> tuple[np.ndarray, dict]:
    img = read_rgb(path)
    if mode == "resize":
        return resize_baseline(img), {}
    if mode == "croppad":
        return preprocess_croppad(img)
    raise ValueError(f"mode must be one of {MODES}")
