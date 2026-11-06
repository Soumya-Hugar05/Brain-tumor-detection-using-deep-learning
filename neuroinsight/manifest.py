"""Read the approved split manifest (data_splits/manifest.csv) - the single source of truth.

* The manifest SHA-256 is checked against the approved value before anything is read.
* Excluded images are never returned.
* The test split is locked: it is only returned with allow_test=True, which only
  evaluate.py --final passes.
* Every returned file is MD5-checked against the manifest (dataset unchanged).
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data_splits" / "manifest.csv"
DATASET_DIR = ROOT / "dataset"
APPROVED_MANIFEST_SHA256 = "4d09b01d782fa076890f32bf30bcbbe1a2319c135ead2e58cbd5e1f9ec0d930f"

# Same order as artifacts/class_names.json (alphabetical folder names).
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NOTUMOR_INDEX = CLASS_NAMES.index("notumor")
SPLITS = ("train", "val", "test")


class ManifestError(RuntimeError):
    pass


class TestSetLocked(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(path: Path = MANIFEST_PATH) -> str:
    if not path.is_file():
        raise ManifestError(f"Approved manifest not found: {path}")
    digest = sha256_file(path)
    if digest != APPROVED_MANIFEST_SHA256:
        raise ManifestError(
            f"Manifest SHA-256 mismatch: {digest} != approved {APPROVED_MANIFEST_SHA256}. Stopping."
        )
    return digest


def load_split(split: str, *, allow_test: bool = False, verify_md5: bool = True) -> list[dict]:
    """Rows of one split, sorted by path. Each row: path (absolute Path), rel, label, y, group_id, ..."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "test" and not allow_test:
        raise TestSetLocked("The test split is locked until the final evaluation (evaluate.py --final).")
    verify_manifest()
    rows = []
    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            if r["exclusion_reason"]:
                raise ManifestError(f"Excluded image in split {split}: {r['path']}")
            if r["label"] not in CLASS_NAMES:
                raise ManifestError(f"Unknown label {r['label']} for {r['path']}")
            rows.append(dict(
                rel=r["path"], path=DATASET_DIR / r["path"], label=r["label"],
                y=CLASS_NAMES.index(r["label"]), group_id=r["group_id"], md5=r["md5"],
                width=int(r["width"]), height=int(r["height"]), source_split=r["source_split"],
            ))
    rows.sort(key=lambda r: r["rel"])
    if verify_md5:
        bad = [r["rel"] for r in rows
               if not r["path"].is_file() or hashlib.md5(r["path"].read_bytes()).hexdigest() != r["md5"]]
        if bad:
            raise ManifestError(f"{len(bad)} dataset files missing or changed, e.g. {bad[:3]}")
    return rows
