from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_class_names(train_dir: str | Path) -> list[str]:
    train_dir = Path(train_dir)
    class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    class_names = sorted([p.name for p in class_dirs])
    if not class_names:
        raise RuntimeError(f"No class folders found under: {train_dir}")
    return class_names


def save_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_image_rgb(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(image_size)
    arr = np.asarray(img, dtype=np.float32)
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / np.sum(e)).astype(np.float32)


def batch_iterable(it: Iterable, batch_size: int):
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
