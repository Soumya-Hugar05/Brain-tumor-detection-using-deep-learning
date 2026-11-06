"""Frozen ImageNet ResNet50 features (global-average-pooled conv5 output, 2048-d).

GAP(conv5_block3_out) is exactly ResNet50(include_top=False, pooling="avg"), the feature used by
the legacy train.py/app.py. Each extraction also computes the features of the horizontally
flipped image, so 'flip-averaged' features ((f(x) + f(flip x)) / 2) need no extra pass.

Features are cached per (preprocessing mode, split) under runs/cache/ and re-used only if the
manifest hash, the image list and the library versions match.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

from .manifest import verify_manifest
from .preprocessing import IMAGE_SIZE, load_preprocessed
from .runinfo import RUNS_DIR, versions

CACHE_DIR = RUNS_DIR / "cache"
BATCH = 32


def build_backbone() -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet",
                                          input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    return base


class FeatureExtractor:
    def __init__(self, backbone: tf.keras.Model | None = None):
        self.backbone = backbone or build_backbone()

        @tf.function(reduce_retracing=True)
        def _feats(x):
            x_all = tf.concat([x, tf.image.flip_left_right(x)], axis=0)
            maps = self.backbone(preprocess_input(x_all), training=False)
            pooled = tf.reduce_mean(maps, axis=[1, 2])
            n = tf.shape(x)[0]
            return pooled[:n], pooled[n:]

        self._feats = _feats

    def __call__(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """images: float32 (N,224,224,3) in [0,255] -> (plain, flipped) features."""
        a, b = self._feats(tf.convert_to_tensor(images, tf.float32))
        return a.numpy(), b.numpy()


def _load_batch(pool, rows, mode):
    return list(pool.map(lambda r: load_preprocessed(r["path"], mode), rows))


def extract(rows: list[dict], mode: str, extractor: FeatureExtractor | None = None,
            log_every: int = 20) -> dict:
    extractor = extractor or FeatureExtractor()
    plain, flipped, infos = [], [], []
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        nxt = pool.submit(_load_batch, pool, batches[0], mode) if batches else None
        for bi in range(len(batches)):
            loaded = nxt.result()
            if bi + 1 < len(batches):
                nxt = pool.submit(_load_batch, pool, batches[bi + 1], mode)
            imgs = np.stack([im for im, _ in loaded])
            infos.extend(info for _, info in loaded)
            a, b = extractor(imgs)
            plain.append(a)
            flipped.append(b)
            if (bi + 1) % log_every == 0:
                print(f"    {mode}: {min((bi + 1) * BATCH, len(rows))}/{len(rows)} images", flush=True)
    return dict(plain=np.concatenate(plain), flipped=np.concatenate(flipped), infos=infos)


def cached_features(rows: list[dict], mode: str, split: str,
                    extractor: FeatureExtractor | None = None) -> dict:
    """Features for `rows`, from cache if valid. Keys: plain, flipped, infos, rels, y."""
    if split == "test":
        raise RuntimeError("Test features are never cached; extract them in evaluate.py --final.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz, meta_path = CACHE_DIR / f"features_{mode}_{split}.npz", CACHE_DIR / f"features_{mode}_{split}.json"
    rels = [r["rel"] for r in rows]
    meta = dict(mode=mode, split=split, manifest_sha256=verify_manifest(), n=len(rows),
                tensorflow=versions()["tensorflow"], keras=versions()["keras"],
                feature="ResNet50 imagenet, GAP of conv5_block3_out, + horizontally flipped copy")
    if npz.is_file() and meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz, allow_pickle=False)
        if {k: old.get(k) for k in meta} == meta and list(data["rels"]) == rels:
            print(f"  using cached features {npz.name}", flush=True)
            return dict(plain=data["plain"], flipped=data["flipped"], rels=rels,
                        y=np.array([r["y"] for r in rows]), infos=old.get("infos", []))
    print(f"  extracting {mode} features for {split} ({len(rows)} images)...", flush=True)
    out = extract(rows, mode, extractor)
    np.savez(npz, plain=out["plain"], flipped=out["flipped"], rels=np.array(rels))
    meta_path.write_text(json.dumps({**meta, "infos": out["infos"]}), encoding="utf-8")
    return dict(plain=out["plain"], flipped=out["flipped"], rels=rels,
                y=np.array([r["y"] for r in rows]), infos=out["infos"])


def combine(feats: dict, flip_average: bool) -> np.ndarray:
    return (feats["plain"] + feats["flipped"]) / 2.0 if flip_average else feats["plain"]
