"""Duplicate-aware dataset audit and group-level train/val/test split.

What it does
------------
1. Scans every image under dataset/Training and dataset/Testing (read-only; nothing is
   moved, renamed or deleted).
2. Computes an MD5 of the file bytes and two 64-bit perceptual hashes per image:
   - pHash: 32x32 grayscale -> 2-D DCT -> top-left 8x8 low frequencies > median.
   - dHash: 9x8 grayscale -> sign of horizontal gradient.
3. Marks approved exclusions (see "Exclusions" below). Excluded images stay in the
   manifest with split = "excluded" and a reason, but are never assigned to train/val/test.
4. Links two images when ANY of these holds:
   a. byte-identical (same MD5);
   b. BOTH hash distances are <= --threshold bits (out of 64);
   c. mirror-aware pixel copy: 64x64 thumbnails correlate >= CANDIDATE_CORR in the original
      OR horizontally flipped orientation (candidate), and at 256x256 in the best
      orientation the pixel correlation is >= CONFIRM_CORR AND the 10th-percentile
      correlation of non-flat 32x32 blocks is >= CONFIRM_BLOCK_P10 (confirmed).
   Linking uses ALL images, including excluded ones, so an excluded image can still hold
   its look-alikes together. Unconfirmed candidates are written to review_pairs.csv and,
   only with --link-related, linked as "related".
5. Builds duplicate groups as connected components of those links (union-find), so
   A~B and B~C put A, B, C in one group.
6. Pools Training and Testing, then assigns whole groups (their eligible images) to
   train / val / test, stratified by each group's majority class, with a fixed seed.
7. Verifies the result and writes the manifest plus an audit report to data_splits/.
   If a hard check fails, no manifest is written and the script exits with code 1.

Exclusions
----------
- pre_augmented: files whose name contains "-aug-" (203 meningioma images). They are
  zoomed / shifted / rotated / flipped copies of other scans that the dataset authors added
  for class balance. They cannot be grouped with their sources reliably and could leak.
- label_conflict_copy: the images listed in LABEL_CONFLICT_COPIES, plus any passed with
  --extra-exclusions. Each is a copy of a scan that is also filed under another class,
  so the true label is unknown.
- A cross-label link between two eligible images fails verification unless the pair is
  listed in LABEL_CONFLICT_KEEP (documented as different scans after visual review).

Thresholds and limitations
--------------------------
- Hash threshold 4: pixel checks found 73-90% of pairs at distance <= 2 are copies; at
  3-4 most are similar slices; at 5-6 none of the sampled pairs were copies.
- Pixel confirmation (corr >= 0.95 and block p10 >= 0.85) was calibrated by visual review:
  it accepts copies that differ by mirroring, resizing, re-encoding or brightness/contrast,
  and rejects adjacent slices of the same patient (block p10 typically 0.3-0.7).
- Not every link is proven to be the same scan. Correlation measures similarity, and
  heavily reprocessed copies (sharpening, blurring, strong gamma) or adjacent slices can
  fall either side of the thresholds. review_pairs.csv lists unconfirmed candidates.
- The dataset has no patient IDs, so this is NOT a patient-level split. Different slices
  of the same patient can still end up in different splits.
- Rotated, cropped or zoomed copies are not detected (only horizontal mirroring is).
- Union-find chaining can grow large groups. The script fails if any group exceeds
  --max-group-frac of the dataset, rather than silently producing a skewed split.
- Hashes and thumbnails depend on Pillow's resampling and BLAS arithmetic. Versions are
  recorded in the report; --verify-only recomputes links and compares them.

Usage
-----
    python audit_dataset.py                 # audit, split, verify, write data_splits/
    python audit_dataset.py --verify-only   # re-verify an existing data_splits/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy.fft import dct

from ml_utils import ensure_dir


DATASET_DIR = Path("dataset")
SOURCE_SPLITS = ("Training", "Testing")
OUT_DIR = Path("data_splits")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 1337
SPLITS = ("train", "val", "test")
EXCLUDED = "excluded"
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_THRESHOLD = 4
MIN_CLASS_BALANCE = 0.75
SENSITIVITY_THRESHOLDS = tuple(range(0, 11))
ROW_CHUNK = 512

# Mirror-aware pixel comparison (calibrated on this dataset; see module docstring).
THUMB_SIZE = 64
CONFIRM_SIZE = 256
CANDIDATE_CORR = 0.93
CONFIRM_CORR = 0.95
CONFIRM_BLOCK_P10 = 0.85
BLOCK = 32
BLOCK_MIN_STD = 4.0
PAIR_CHUNK = 256
RESIDUAL_BANDS = ((0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.0, 0.85))

AUG_PATTERN = re.compile(r"-aug-", re.IGNORECASE)
# Same scan stored under glioma AND meningioma (verified by pixel correlation >= 0.998).
# Approved for exclusion; originally the complete hash groups g00069 and g00349.
LABEL_CONFLICT_COPIES = (
    "Testing/glioma/Te-gl_160.jpg",
    "Testing/meningioma/Te-me_159.jpg",
    "Training/meningioma/Tr-me_1122.jpg",
    "Testing/glioma/Te-gl_68.jpg",
    "Testing/meningioma/Te-me_50.jpg",
    "Testing/meningioma/Te-me_61.jpg",
    "Testing/meningioma/Te-me_96.jpg",
    "Training/meningioma/Tr-me_458.jpg",
    "Training/meningioma/Tr-me_468.jpg",
    "Training/meningioma/Tr-me_714.jpg",
)
# Cross-label links reviewed visually and found to be DIFFERENT scans (kept, documented).
LABEL_CONFLICT_KEEP = (
    # Hash link only (pHash/dHash 4/4, pixel corr 0.73): glioma with a small posterior
    # lesion vs meningioma with a large occipital mass. Different patients.
    ("Testing/glioma/Te-gl_75.jpg", "Testing/meningioma/Te-me_283.jpg"),
)
EXCLUSION_REASONS = {
    "pre_augmented": "Pre-augmented copy shipped with the dataset (filename contains '-aug-'); "
                     "geometric transforms evade perceptual hashing, so it could leak across splits.",
    "label_conflict_copy": "Copy of a scan that is also filed under another class folder; "
                           "the true label is unknown.",
}

MANIFEST_FIELDS = [
    "path", "label", "source_split", "split", "exclusion_reason", "group_id", "group_size",
    "group_eligible_size", "group_type", "group_label_conflict", "md5", "phash", "dhash",
    "width", "height",
]
PAIR_FIELDS = [
    "group_id", "path_a", "path_b", "label_a", "label_b", "split_a", "split_b", "same_label",
    "link_type", "orientation", "phash_dist", "dhash_dist", "corr64", "corr256",
    "block_corr_p10", "grad_corr256",
]


# --------------------------------------------------------------------------- hashing

def _bits_to_uint64(bits: np.ndarray) -> int:
    return int(np.packbits(bits.astype(np.uint8)).view(">u8")[0])


def phash(img: Image.Image) -> int:
    a = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return _bits_to_uint64(low.flatten() > np.median(low))


def dhash(img: Image.Image) -> int:
    a = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.float64)
    return _bits_to_uint64((a[:, 1:] > a[:, :-1]).flatten())


def list_dataset_images(dataset_dir: Path) -> set[str]:
    return {
        p.relative_to(dataset_dir).as_posix()
        for s in SOURCE_SPLITS for p in (dataset_dir / s).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def scan_dataset(dataset_dir: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    records, unreadable = [], []
    for source in SOURCE_SPLITS:
        src_dir = dataset_dir / source
        if not src_dir.is_dir():
            raise RuntimeError(f"Missing folder: {src_dir}")
        for class_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel = path.relative_to(dataset_dir).as_posix()
                try:
                    data = path.read_bytes()
                    with Image.open(path) as img:
                        img.load()
                        records.append(
                            dict(path=rel, label=class_dir.name, source_split=source,
                                 md5=hashlib.md5(data).hexdigest(),
                                 phash=phash(img), dhash=dhash(img),
                                 width=img.width, height=img.height)
                        )
                except Exception as exc:
                    unreadable.append((rel, str(exc)))
        print(f"  scanned {source}: {sum(r['source_split'] == source for r in records)} images", flush=True)
    records.sort(key=lambda r: r["path"])
    return records, unreadable


# ------------------------------------------------------------------------ exclusions

def load_extra_exclusions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    extra = {}
    for r in rows:
        if r["reason"] not in EXCLUSION_REASONS:
            raise RuntimeError(f"Unknown exclusion reason '{r['reason']}' for {r['path']}")
        extra[r["path"]] = r["reason"]
    return extra


def expected_exclusion(path: str, extra: dict[str, str]) -> str:
    reasons = []
    if AUG_PATTERN.search(Path(path).name):
        reasons.append("pre_augmented")
    if path in LABEL_CONFLICT_COPIES:
        reasons.append("label_conflict_copy")
    if path in extra and extra[path] not in reasons:
        reasons.append(extra[path])
    return ";".join(reasons)


def apply_exclusions(records: list[dict], extra: dict[str, str]) -> list[str]:
    known = {r["path"] for r in records}
    missing = [p for p in (*LABEL_CONFLICT_COPIES, *extra) if p not in known]
    for r in records:
        r["exclusion_reason"] = expected_exclusion(r["path"], extra)
    return [f"Listed exclusion not found in dataset: {p}" for p in missing]


# ------------------------------------------------------------------- hash distances

def _hash_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ph = np.array([r["phash"] for r in records], dtype=np.uint64)
    dh = np.array([r["dhash"] for r in records], dtype=np.uint64)
    return ph, dh


def _pair_distance(ph_a, dh_a, ph_b, dh_b) -> tuple[np.ndarray, np.ndarray]:
    dp = np.bitwise_count(ph_a[:, None] ^ ph_b[None, :]).astype(np.int16)
    dd = np.bitwise_count(dh_a[:, None] ^ dh_b[None, :]).astype(np.int16)
    return dp, dd


def find_hash_links(records: list[dict], threshold: int) -> set[tuple[int, int]]:
    """Pairs (i < j) that are byte-identical or within `threshold` on both hashes."""
    ph, dh = _hash_arrays(records)
    md5 = np.array([r["md5"] for r in records])
    pairs = set()
    n = len(records)
    for start in range(0, n, ROW_CHUNK):
        stop = min(start + ROW_CHUNK, n)
        dp, dd = _pair_distance(ph[start:stop], dh[start:stop], ph, dh)
        linked = (np.maximum(dp, dd) <= threshold) | (md5[start:stop, None] == md5[None, :])
        for a, j in zip(*np.nonzero(linked)):
            if j > start + a:
                pairs.add((int(start + a), int(j)))
    return pairs


# ------------------------------------------------------------ mirror-aware pixel links

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def load_thumbnails(records: list[dict], dataset_dir: Path):
    """Return normalized 64x64 thumbnails (original, mirrored) and 256x256 uint8 images."""
    n = len(records)
    small = np.empty((n, THUMB_SIZE * THUMB_SIZE), np.float32)
    small_mirror = np.empty_like(small)
    big = np.empty((n, CONFIRM_SIZE, CONFIRM_SIZE), np.uint8)
    for k, r in enumerate(records):
        with Image.open(dataset_dir / r["path"]) as img:
            gray = img.convert("L")
            s = np.asarray(gray.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS), np.float32)
            small[k] = s.ravel()
            small_mirror[k] = s[:, ::-1].ravel()
            big[k] = np.asarray(gray.resize((CONFIRM_SIZE, CONFIRM_SIZE), Image.LANCZOS), np.uint8)
    return _normalize_rows(small), _normalize_rows(small_mirror), big


def find_pixel_candidates(small: np.ndarray, small_mirror: np.ndarray) -> set[tuple[int, int]]:
    """Pairs (i < j) whose 64x64 thumbnails correlate >= CANDIDATE_CORR in either orientation."""
    pairs = set()
    n = len(small)
    for start in range(0, n, ROW_CHUNK):
        block = small[start:start + ROW_CHUNK]
        for ref in (small, small_mirror):
            for a, j in zip(*np.nonzero(block @ ref.T >= CANDIDATE_CORR)):
                if j > start + a:
                    pairs.add((int(start + a), int(j)))
    return pairs


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


def _block_p10(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """10th percentile of 32x32 block correlations, ignoring flat (background) blocks."""
    k, n = a.shape[0], CONFIRM_SIZE // BLOCK
    A = a.reshape(k, n, BLOCK, n, BLOCK).transpose(0, 1, 3, 2, 4).reshape(k, n * n, -1)
    B = b.reshape(k, n, BLOCK, n, BLOCK).transpose(0, 1, 3, 2, 4).reshape(k, n * n, -1)
    valid = (A.std(-1) >= BLOCK_MIN_STD) & (B.std(-1) >= BLOCK_MIN_STD)
    r = np.where(valid, _row_corr(A, B), np.nan)
    out = np.zeros(k, np.float32)
    has = valid.any(axis=1)
    if has.any():
        out[has] = np.nanpercentile(r[has], 10, axis=1)
    return out


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]
    gy[:, 1:-1, :] = x[:, 2:, :] - x[:, :-2, :]
    return np.hypot(gx, gy)


def pair_metrics(pairs: list[tuple[int, int]], small, small_mirror, big) -> dict:
    """Similarity of each pair in its better orientation (by 256x256 pixel correlation)."""
    I = np.array([p[0] for p in pairs], dtype=np.int64)
    J = np.array([p[1] for p in pairs], dtype=np.int64)
    out = {}
    for start in range(0, len(pairs), PAIR_CHUNK):
        sl = slice(start, start + PAIR_CHUNK)
        c64 = {"same": np.einsum("ij,ij->i", small[I[sl]], small[J[sl]]),
               "mirror": np.einsum("ij,ij->i", small_mirror[I[sl]], small[J[sl]])}
        a = big[I[sl]].astype(np.float32)
        b = big[J[sl]].astype(np.float32)
        k = len(a)
        c_same = _row_corr(a.reshape(k, -1), b.reshape(k, -1))
        c_mirror = _row_corr(a[:, :, ::-1].reshape(k, -1), b.reshape(k, -1))
        use_mirror = c_mirror > c_same
        a[use_mirror] = a[use_mirror][:, :, ::-1]
        blk = _block_p10(a, b)
        grad = _row_corr(_gradient_magnitude(a).reshape(k, -1), _gradient_magnitude(b).reshape(k, -1))
        for n in range(k):
            orient = "mirror" if use_mirror[n] else "same"
            out[(int(I[start + n]), int(J[start + n]))] = dict(
                orientation=orient,
                corr64=float(c64[orient][n]),
                corr256=float(c_mirror[n] if use_mirror[n] else c_same[n]),
                block_corr_p10=float(blk[n]),
                grad_corr256=float(grad[n]),
            )
    return out


def is_pixel_copy(m: dict) -> bool:
    return m["corr256"] >= CONFIRM_CORR and m["block_corr_p10"] >= CONFIRM_BLOCK_P10


def find_links(records: list[dict], dataset_dir: Path, threshold: int, link_related: bool):
    """Return (links, review). Each is a list of dicts with i, j, link_type and metrics."""
    t0 = time.perf_counter()
    hash_pairs = find_hash_links(records, threshold)
    print(f"  hash pairs: {len(hash_pairs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    small, small_mirror, big = load_thumbnails(records, dataset_dir)
    print(f"  thumbnails loaded ({time.perf_counter() - t0:.0f}s)", flush=True)
    candidates = find_pixel_candidates(small, small_mirror)
    pairs = sorted(hash_pairs | candidates)
    print(f"  pixel candidates: {len(candidates)} ({time.perf_counter() - t0:.0f}s); confirming "
          f"{len(pairs)} pairs at {CONFIRM_SIZE}x{CONFIRM_SIZE}...", flush=True)
    metrics = pair_metrics(pairs, small, small_mirror, big)
    print(f"  pair metrics done ({time.perf_counter() - t0:.0f}s)", flush=True)
    ph, dh = _hash_arrays(records)

    links, review = [], []
    for i, j in pairs:
        m = metrics[(i, j)]
        row = dict(i=i, j=j, phash_dist=int(np.bitwise_count(ph[i] ^ ph[j])),
                   dhash_dist=int(np.bitwise_count(dh[i] ^ dh[j])), **m)
        if records[i]["md5"] == records[j]["md5"]:
            row["link_type"] = "exact_md5"
        elif (i, j) in hash_pairs:
            row["link_type"] = "near_hash"
        elif is_pixel_copy(m):
            row["link_type"] = f"pixel_{m['orientation']}"
        elif link_related:
            row["link_type"] = "related"
        else:
            row["link_type"] = "unconfirmed"
            review.append(row)
            continue
        links.append(row)
    return links, review


# ------------------------------------------------------------------------- groups

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_groups(records: list[dict], links: list[dict]) -> list[dict]:
    uf = UnionFind(len(records))
    for link in links:
        uf.union(link["i"], link["j"])
    members = defaultdict(list)
    for idx in range(len(records)):
        members[uf.find(idx)].append(idx)

    # Records are path-sorted, so ordering by first member gives stable group IDs.
    groups = []
    for gnum, idxs in enumerate(sorted(members.values(), key=lambda m: m[0]), start=1):
        eligible = [i for i in idxs if not records[i]["exclusion_reason"]]
        labels = Counter(records[i]["label"] for i in (eligible or idxs))
        top = max(labels.values())
        majority = sorted(lbl for lbl, c in labels.items() if c == top)[0]
        if len(idxs) == 1:
            gtype = "singleton"
        elif len({records[i]["md5"] for i in idxs}) == 1:
            gtype = "exact"
        else:
            gtype = "near"
        groups.append(dict(
            group_id=f"g{gnum:05d}", members=idxs, eligible=eligible, majority_label=majority,
            label_conflict=len({records[i]["label"] for i in idxs}) > 1,
            group_type=gtype, sources=sorted({records[i]["source_split"] for i in idxs}),
        ))
    return groups


# ---------------------------------------------------------------------------- split

def assign_splits(records: list[dict], groups: list[dict], ratios: tuple[float, ...], seed: int) -> None:
    """Greedy group-level stratified split over eligible images. Sets group/record 'split'."""
    by_class = defaultdict(list)
    for g in groups:
        g["split"] = EXCLUDED
        if g["eligible"]:
            by_class[g["majority_label"]].append(g)

    for class_idx, label in enumerate(sorted(by_class)):
        class_groups = by_class[label]
        rng = np.random.default_rng([seed, class_idx])
        order = rng.permutation(len(class_groups))
        # Place large groups first (they are hardest to balance); random order breaks ties.
        ordered = sorted((class_groups[k] for k in order), key=lambda g: -len(g["eligible"]))
        total = sum(len(g["eligible"]) for g in class_groups)
        targets = [r * total for r in ratios]
        counts = [0, 0, 0]
        for g in ordered:
            deficits = [(targets[s] - counts[s]) / targets[s] for s in range(len(SPLITS))]
            s = int(np.argmax(deficits))
            counts[s] += len(g["eligible"])
            g["split"] = SPLITS[s]

    for g in groups:
        for i in g["members"]:
            r = records[i]
            r["split"] = EXCLUDED if r["exclusion_reason"] else g["split"]
            r["group_id"] = g["group_id"]
            r["group_size"] = len(g["members"])
            r["group_eligible_size"] = len(g["eligible"])
            r["group_type"] = g["group_type"]
            r["group_label_conflict"] = g["label_conflict"]


# --------------------------------------------------------------------------- checks

def nearest_cross_split(records: list[dict], query: str, reference: str) -> np.ndarray:
    """For each `query`-split image, the min max(pHash, dHash) distance to any `reference` image."""
    q = [r for r in records if r["split"] == query]
    ref = [r for r in records if r["split"] == reference]
    qp, qd = _hash_arrays(q)
    rp, rd = _hash_arrays(ref)
    best = np.full(len(q), 64, dtype=np.int16)
    for start in range(0, len(q), ROW_CHUNK):
        dp, dd = _pair_distance(qp[start:start + ROW_CHUNK], qd[start:start + ROW_CHUNK], rp, rd)
        best[start:start + ROW_CHUNK] = np.maximum(dp, dd).min(axis=1)
    return best


def residual_candidates(records: list[dict], review: list[dict]) -> dict:
    """Held-out images that have an UNCONFIRMED pixel candidate in another split, by band."""
    out = {}
    for query, ref in (("test", "train"), ("val", "train"), ("test", "val")):
        best = {}
        for row in review:
            for q, r in ((row["i"], row["j"]), (row["j"], row["i"])):
                if records[q]["split"] == query and records[r]["split"] == ref:
                    best[q] = max(best.get(q, 0.0), row["corr256"])
        out[f"{query}->{ref}"] = {
            f"{lo:.2f}-{min(hi, 1.0):.2f}": sum(lo <= v < hi for v in best.values())
            for lo, hi in RESIDUAL_BANDS
        }
    return out


def run_checks(records, groups, links, review, ratios, threshold, max_group_frac, dataset_dir,
               ratio_tol, extra):
    """Return (checks, warnings, sensitivity, residual). checks is a list of {name, passed, detail}."""
    checks, warnings = [], []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    labels = sorted({r["label"] for r in records})
    eligible = [r for r in records if r["split"] != EXCLUDED]

    # 1. Every image on disk is in the manifest exactly once.
    paths = [r["path"] for r in records]
    dupes = [p for p, c in Counter(paths).items() if c > 1]
    on_disk = list_dataset_images(dataset_dir)
    missing = sorted(set(paths) - on_disk)
    extra_files = sorted(on_disk - set(paths))
    check("Every original image accounted for exactly once",
          not dupes and not missing and not extra_files,
          f"{len(paths)} rows, {len(on_disk)} files on disk; repeated {len(dupes)}, "
          f"missing on disk {len(missing)}, not in manifest {len(extra_files)}"
          + (f"; e.g. {(dupes + missing + extra_files)[:3]}" if dupes or missing or extra_files else ""))

    # 2. Exclusions are exactly the approved/listed ones and never sit in a train/val/test split.
    wrong = [r["path"] for r in records
             if r["exclusion_reason"] != expected_exclusion(r["path"], extra)
             or (r["split"] == EXCLUDED) != bool(r["exclusion_reason"])
             or r["split"] not in (*SPLITS, EXCLUDED)]
    reasons = Counter(r["exclusion_reason"] for r in records if r["exclusion_reason"])
    check("Excluded images are exactly the listed ones and in no train/val/test split",
          not wrong and set(LABEL_CONFLICT_COPIES) <= set(paths),
          f"excluded {sum(reasons.values())}: {dict(reasons)}; inconsistent rows {len(wrong)}"
          + (f", e.g. {wrong[:3]}" if wrong else ""))

    # 3. Label-conflict exclusions cover complete groups (every copy of a conflicting scan).
    by_group = defaultdict(list)
    for r in records:
        by_group[r["group_id"]].append(r)
    conflict_gids = sorted({r["group_id"] for r in records if "label_conflict_copy" in r["exclusion_reason"]})
    partial = [g for g in conflict_gids if any(m["split"] != EXCLUDED for m in by_group[g])]
    check("Label-conflict exclusions cover complete duplicate groups",
          not partial, f"{len(conflict_gids)} groups; with eligible images left: {partial}")

    # 4. No undocumented cross-label link between eligible images.
    keep = {frozenset(p) for p in LABEL_CONFLICT_KEEP}
    cross_label = [l for l in links
                   if records[l["i"]]["split"] != EXCLUDED and records[l["j"]]["split"] != EXCLUDED
                   and records[l["i"]]["label"] != records[l["j"]]["label"]]
    undocumented = [l for l in cross_label
                    if frozenset((records[l["i"]]["path"], records[l["j"]]["path"])) not in keep]
    check("No undocumented cross-label duplicate link among eligible images",
          not undocumented,
          f"{len(cross_label)} cross-label links, {len(cross_label) - len(undocumented)} documented as "
          f"different scans; undocumented {len(undocumented)}"
          + (": " + "; ".join(f"{records[l['i']]['path']} ~ {records[l['j']]['path']} ({l['link_type']})"
                              for l in undocumented[:5]) if undocumented else ""))

    # 5. A group's eligible images live in exactly one split.
    group_splits = defaultdict(set)
    for r in eligible:
        group_splits[r["group_id"]].add(r["split"])
    crossing = sorted(g for g, s in group_splits.items() if len(s) > 1)
    check("No duplicate group spans more than one of train/val/test",
          not crossing, f"{len(group_splits)} groups with eligible images; crossing {len(crossing)}"
          + (f": {crossing[:10]}" if crossing else ""))

    # 6. No link of any type (hash or confirmed pixel copy) connects two different splits.
    split_links = [l for l in links
                   if EXCLUDED not in (records[l["i"]]["split"], records[l["j"]]["split"])
                   and records[l["i"]]["split"] != records[l["j"]]["split"]]
    by_type = Counter(l["link_type"] for l in links)
    check("No duplicate link (hash or mirror-aware pixel copy) connects two different splits",
          not split_links, f"{len(links)} links checked {dict(by_type)}; crossing {len(split_links)}")

    # 7. Runaway chaining.
    largest = max(len(g["members"]) for g in groups)
    check("No oversized duplicate group (chaining guard)",
          largest <= max_group_frac * len(records),
          f"largest group {largest} images; limit {max_group_frac:.0%} of {len(records)} = "
          f"{int(max_group_frac * len(records))}")

    # 8-10. Class coverage, per-class ratios and balance.
    counts = {s: Counter(r["label"] for r in eligible if r["split"] == s) for s in SPLITS}
    absent = [f"{lbl}/{s}" for s in SPLITS for lbl in labels if counts[s][lbl] == 0]
    check("All classes present in train, val and test", not absent,
          "missing: " + (", ".join(absent) if absent else "none"))

    off = []
    for lbl in labels:
        total = sum(counts[s][lbl] for s in SPLITS)
        for s, target in zip(SPLITS, ratios):
            share = counts[s][lbl] / total if total else 0.0
            if abs(share - target) > ratio_tol:
                off.append(f"{lbl}/{s} {share:.1%} vs {target:.0%}")
    check(f"Per-class split ratios within {ratio_tol:.0%} of target", not off,
          "off target: " + (", ".join(off) if off else "none"))

    balance = {s: (min(counts[s][lbl] for lbl in labels) / max(counts[s][lbl] for lbl in labels)
                   if counts[s] else 0.0) for s in SPLITS}
    check(f"Class balance in each split (smallest/largest class >= {MIN_CLASS_BALANCE})",
          all(b >= MIN_CLASS_BALANCE for b in balance.values()),
          ", ".join(f"{s} {b:.2f}" for s, b in balance.items()))

    sensitivity = {}
    for query, ref in (("test", "train"), ("val", "train"), ("test", "val")):
        best = nearest_cross_split(records, query, ref)
        sensitivity[f"{query}->{ref}"] = {str(t): int((best <= t).sum()) for t in SENSITIVITY_THRESHOLDS}

    residual = residual_candidates(records, review)
    for key, bands in residual.items():
        n = sum(bands.values())
        if n:
            warnings.append(
                f"{n} {key.split('->')[0]} image(s) have an UNCONFIRMED pixel candidate in "
                f"{key.split('->')[1]} (64px corr >= {CANDIDATE_CORR}, failed 256px confirmation); "
                f"by 256px corr band {bands}. Mostly reprocessed no-tumor copies or adjacent slices "
                "of the same patient; see review_pairs.csv."
            )
    return checks, warnings, sensitivity, residual


# --------------------------------------------------------------------------- output

def split_summary(records, groups):
    labels = sorted({r["label"] for r in records})
    summary = {}
    for s in (*SPLITS, EXCLUDED):
        rows = [r for r in records if r["split"] == s]
        summary[s] = dict(
            images=len(rows),
            groups=len({r["group_id"] for r in rows}),
            classes={lbl: sum(r["label"] == lbl for r in rows) for lbl in labels},
            from_original=dict(Counter(r["source_split"] for r in rows)),
        )
    return summary


def exclusion_summary(records):
    by_reason = defaultdict(list)
    for r in records:
        if r["exclusion_reason"]:
            by_reason[r["exclusion_reason"]].append(r)
    out = {}
    for reason, rows in sorted(by_reason.items()):
        out[reason] = dict(
            count=len(rows),
            description=" + ".join(EXCLUSION_REASONS[x] for x in reason.split(";")),
            by_class=dict(Counter(r["label"] for r in rows)),
            by_original_folder=dict(Counter(r["source_split"] for r in rows)),
        )
        if "label_conflict_copy" in reason:
            groups = defaultdict(list)
            for r in rows:
                groups[r["group_id"]].append(r["path"])
            out[reason]["groups"] = dict(sorted(groups.items()))
    return out


def mixed_label_groups(records, links) -> list[dict]:
    """Every group whose members (eligible or excluded) carry more than one class label."""
    by_group = defaultdict(list)
    for idx, r in enumerate(records):
        by_group[r["group_id"]].append(idx)
    keep = {frozenset(p) for p in LABEL_CONFLICT_KEEP}
    out = []
    for gid, idxs in sorted(by_group.items()):
        if len({records[i]["label"] for i in idxs}) < 2:
            continue
        member_set = set(idxs)
        cross = [l for l in links if l["i"] in member_set and l["j"] in member_set
                 and records[l["i"]]["label"] != records[l["j"]]["label"]]
        out.append(dict(
            group_id=gid,
            members=[dict(path=records[i]["path"], label=records[i]["label"],
                          source=records[i]["source_split"], split=records[i]["split"],
                          exclusion=records[i]["exclusion_reason"]) for i in idxs],
            cross_label_links=[dict(a=records[l["i"]]["path"], b=records[l["j"]]["path"],
                                    link_type=l["link_type"], orientation=l["orientation"],
                                    corr256=round(l["corr256"], 4),
                                    block_corr_p10=round(l["block_corr_p10"], 4),
                                    documented_keep=frozenset((records[l["i"]]["path"], records[l["j"]]["path"])) in keep)
                               for l in cross],
            all_excluded=all(records[i]["split"] == EXCLUDED for i in idxs),
        ))
    return out


def write_manifest(path: Path, records: list[dict]) -> str:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in MANIFEST_FIELDS}
            row["phash"] = f"{r['phash']:016x}"
            row["dhash"] = f"{r['dhash']:016x}"
            writer.writerow(row)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pairs(path: Path, records, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for l in rows:
            a, b = records[l["i"]], records[l["j"]]
            writer.writerow(dict(
                group_id=a.get("group_id", ""), path_a=a["path"], path_b=b["path"],
                label_a=a["label"], label_b=b["label"], split_a=a.get("split", ""),
                split_b=b.get("split", ""), same_label=a["label"] == b["label"],
                link_type=l["link_type"], orientation=l["orientation"],
                phash_dist=l["phash_dist"], dhash_dist=l["dhash_dist"],
                corr64=f"{l['corr64']:.4f}", corr256=f"{l['corr256']:.4f}",
                block_corr_p10=f"{l['block_corr_p10']:.4f}", grad_corr256=f"{l['grad_corr256']:.4f}",
            ))


def dataset_fingerprint(records: list[dict]) -> str:
    lines = "\n".join(f"{r['path']} {r['md5']}" for r in sorted(records, key=lambda r: r["path"]))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def write_markdown(path: Path, report: dict) -> None:
    cfg, stats, labels = report["config"], report["dataset"], report["labels"]
    lines = [
        "# Dataset audit and split report",
        "",
        f"Status: **{report['status']}**  ",
        f"Generated: {report['generated_utc']}  ",
        f"Hash rule: identical MD5, or both pHash and dHash distance <= {cfg['threshold']} of 64 bits  ",
        f"Pixel rule: {cfg['pixel_rule']}  ",
        f"Related candidates linked: {cfg['link_related']}  ",
        f"Ratios train/val/test (of eligible images): {cfg['ratios']}  |  Seed: {cfg['seed']}  ",
        f"Dataset fingerprint (SHA-256 of path + MD5 list): `{stats['fingerprint']}`  ",
        f"Manifest SHA-256: `{report.get('manifest_sha256', 'not written')}`",
        "",
        "## Totals",
        "",
        f"- Original images: **{stats['images']}**",
        f"- Eligible images (assigned to train/val/test): **{stats['eligible_images']}**",
        f"- Excluded images: **{stats['excluded_images']}**",
        "",
        "## Excluded images",
        "",
        "| Reason | Count | By class | By original folder | Why |",
        "|---|---|---|---|---|",
    ]
    for reason, d in report["exclusions"].items():
        lines.append(f"| `{reason}` | {d['count']} | {d['by_class']} | {d['by_original_folder']} | {d['description']} |")
    lines += [
        "",
        "## Splits",
        "",
        "| Split | Images | Groups | " + " | ".join(labels) + " | From Training | From Testing |",
        "|---|---|---|" + "---|" * len(labels) + "---|---|",
    ]
    for s, d in report["splits"].items():
        lines.append(
            f"| {s} | {d['images']} | {d['groups']} | "
            + " | ".join(str(d["classes"][lbl]) for lbl in labels)
            + f" | {d['from_original'].get('Training', 0)} | {d['from_original'].get('Testing', 0)} |"
        )
    lines += [
        "",
        "## Duplicates found (all original images)",
        "",
        f"- Groups: {stats['groups']}  |  multi-image groups: {stats['multi_image_groups']} "
        f"(all members byte-identical {stats['exact_groups']}, other {stats['near_groups']})",
        f"- Images sharing a group with at least one other image: {stats['images_in_multi_groups']}",
        f"- Largest group: {stats['largest_group']} images",
        f"- Links by type: {stats['links_by_type']}",
        f"- Pixel-confirmed links not found by hashing: {stats['pixel_only_links']} "
        f"(mirrored {stats['pixel_only_mirror']}, same orientation {stats['pixel_only_same']})",
        f"- Unconfirmed pixel candidates (not linked, see review_pairs.csv): {stats['unconfirmed_candidates']}",
        "",
        "## Mixed-label groups",
        "",
    ]
    if not report["mixed_label_groups"]:
        lines.append("None.")
    for g in report["mixed_label_groups"]:
        status = "all excluded" if g["all_excluded"] else "has eligible images"
        lines.append(f"### {g['group_id']} ({len(g['members'])} images, {status})")
        lines.append("")
        lines.append("| Path | Label | Original folder | Split | Exclusion |")
        lines.append("|---|---|---|---|---|")
        lines += [f"| `{m['path']}` | {m['label']} | {m['source']} | {m['split']} | {m['exclusion'] or '-'} |"
                  for m in g["members"]]
        lines.append("")
        lines.append("Cross-label links: " + ("; ".join(
            f"`{l['a']}` ~ `{l['b']}` ({l['link_type']}, {l['orientation']}, corr {l['corr256']}, "
            f"block p10 {l['block_corr_p10']}{', documented different scans' if l['documented_keep'] else ''})"
            for l in g["cross_label_links"]) or "none (mixed only through excluded or same-label links)"))
        lines.append("")
    lines += ["## Verification", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in report["checks"]]
    lines += ["", "### Warnings", ""]
    lines += [f"- {w}" for w in report["warnings"]] or ["- None."]
    lines += [
        "",
        "## Remaining possible leakage",
        "",
        "Held-out images with an unconfirmed pixel candidate (64px corr >= "
        f"{CANDIDATE_CORR}, either orientation) in another split, by best 256px correlation:",
        "",
        "| Direction | " + " | ".join(next(iter(report["residual"].values())).keys()) + " |",
        "|---|" + "---|" * len(next(iter(report["residual"].values()))),
    ]
    for key, bands in report["residual"].items():
        lines.append(f"| {key} | " + " | ".join(str(v) for v in bands.values()) + " |")
    lines += [
        "",
        "Hash-distance sensitivity (held-out images with a neighbour in the other split within "
        "distance d; must be 0 for d <= threshold):",
        "",
        "| d | " + " | ".join(report["sensitivity"]) + " |",
        "|---|" + "---|" * len(report["sensitivity"]),
    ]
    for t in SENSITIVITY_THRESHOLDS:
        lines.append(f"| {t} | " + " | ".join(str(v[str(t)]) for v in report["sensitivity"].values()) + " |")
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


LIMITATIONS = [
    "No patient IDs are available, so this is NOT a patient-level split. Adjacent slices of the "
    "same patient (visible as unconfirmed candidates with low block correlation) can sit in "
    "different splits, so test scores may still be somewhat optimistic.",
    "A link means the images are near-identical after optional horizontal mirroring and "
    "brightness/contrast normalization, not proof of the same scan. Thresholds were calibrated "
    "by visual review of sampled pairs, not every pair.",
    "Heavily reprocessed copies (sharpening, blurring, strong gamma), mostly in the no-tumor "
    "class, can fail pixel confirmation and remain in different splits. They are counted in "
    "'Remaining possible leakage' and listed in review_pairs.csv.",
    "Rotated, cropped or zoomed copies are not detected; only horizontal mirroring is handled. "
    "The '-aug-' exclusion removes the known pre-augmented files.",
    "Exclusions are rule-based ('-aug-' filenames) or explicit lists of label-conflict copies "
    "verified by pixel correlation. Label errors that are not duplicates cannot be detected.",
    "Hash and thumbnail values depend on Pillow resampling and floating-point arithmetic; "
    "versions are recorded in audit_report.json and --verify-only recomputes all links.",
]


# --------------------------------------------------------------------------- verify

def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["phash"] = int(r["phash"], 16)
        r["dhash"] = int(r["dhash"], 16)
        r["group_size"] = int(r["group_size"])
        r["group_eligible_size"] = int(r["group_eligible_size"])
        r["group_label_conflict"] = r["group_label_conflict"] == "True"
    return rows


def verify_only(args) -> int:
    manifest = args.out_dir / "manifest.csv"
    report_path = args.out_dir / "audit_report.json"
    if not manifest.is_file() or not report_path.is_file():
        print(f"ERROR: {manifest} or {report_path} not found. Run without --verify-only first.")
        return 1
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = saved["config"]
    extra = {e["path"]: e["reason"] for e in cfg.get("extra_exclusions", [])}
    records = load_manifest(manifest)
    errors = []

    if hashlib.sha256(manifest.read_bytes()).hexdigest() != saved.get("manifest_sha256"):
        errors.append("manifest.csv SHA-256 does not match audit_report.json (file was edited).")

    print("Re-hashing files (MD5) to confirm the dataset is unchanged...", flush=True)
    changed = [r["path"] for r in records
               if not (args.dataset_dir / r["path"]).is_file()
               or hashlib.md5((args.dataset_dir / r["path"]).read_bytes()).hexdigest() != r["md5"]]
    if changed:
        errors.append(f"{len(changed)} files missing or changed since the audit, e.g. {changed[:3]}")
    if dataset_fingerprint(records) != saved["dataset"]["fingerprint"]:
        errors.append("Dataset fingerprint differs from the saved report.")

    print("Recomputing hash and mirror-aware pixel links...", flush=True)
    links, review = find_links(records, args.dataset_dir, cfg["threshold"], cfg["link_related"])
    by_group = defaultdict(list)
    for idx, r in enumerate(records):
        by_group[r["group_id"]].append(idx)
    groups = [dict(group_id=g, members=m) for g, m in by_group.items()]
    checks, warnings, sensitivity, residual = run_checks(
        records, groups, links, review, tuple(cfg["ratios"]), cfg["threshold"],
        cfg["max_group_frac"], args.dataset_dir, cfg["ratio_tolerance"], extra,
    )
    errors += [f"{c['name']}: {c['detail']}" for c in checks if not c["passed"]]
    if dict(Counter(l["link_type"] for l in links)) != saved["dataset"]["links_by_type"]:
        errors.append(f"Recomputed links {dict(Counter(l['link_type'] for l in links))} differ from "
                      f"the report {saved['dataset']['links_by_type']}.")
    if sensitivity != saved["sensitivity"] or residual != saved["residual"]:
        errors.append("Recomputed leakage tables differ from the saved report.")

    print(f"\nVerified {len(records)} rows, {len(groups)} groups.")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"  [{'PASS' if not changed else 'FAIL'}] Dataset files unchanged since audit (MD5 of every file)")
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("VERIFY PASSED" if not errors else "VERIFY FAILED")
    return 0 if not errors else 1


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="Max bits (of 64) for BOTH pHash and dHash to link two images.")
    parser.add_argument("--ratios", type=float, nargs=3, default=DEFAULT_RATIOS,
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-group-frac", type=float, default=0.02,
                        help="Fail if any group holds more than this fraction of all images.")
    parser.add_argument("--ratio-tolerance", type=float, default=0.02,
                        help="Fail if a class's share in a split misses its target by more than this.")
    parser.add_argument("--extra-exclusions", type=Path, default=None,
                        help="CSV with columns path,reason[,note] of additional exclusions.")
    parser.add_argument("--link-related", action="store_true",
                        help="Also link unconfirmed pixel candidates (stricter, larger groups).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-verify an existing manifest instead of rebuilding it.")
    args = parser.parse_args()

    if args.verify_only:
        return verify_only(args)

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) <= 0:
        parser.error("--ratios must be positive and sum to 1.")
    extra = load_extra_exclusions(args.extra_exclusions)

    print(f"[1/5] Scanning {args.dataset_dir} (read-only)...", flush=True)
    records, unreadable = scan_dataset(args.dataset_dir)
    if unreadable:
        for path, err in unreadable:
            print(f"  ERROR unreadable: {path}: {err}")
        print(f"STOPPED: {len(unreadable)} unreadable image(s). Fix or remove them, then re-run.")
        return 1

    print("[2/5] Applying exclusions...", flush=True)
    exclusion_errors = apply_exclusions(records, extra)
    if exclusion_errors:
        for e in exclusion_errors:
            print(f"  ERROR: {e}")
        print("STOPPED: exclusion list does not match the dataset.")
        return 1

    print(f"[3/5] Linking duplicates (hash threshold {args.threshold}, mirror-aware pixel check)...", flush=True)
    links, review = find_links(records, args.dataset_dir, args.threshold, args.link_related)
    groups = build_groups(records, links)

    print(f"[4/5] Assigning groups to splits (seed {args.seed})...", flush=True)
    assign_splits(records, groups, ratios, args.seed)

    print("[5/5] Verifying...", flush=True)
    checks, warnings, sensitivity, residual = run_checks(
        records, groups, links, review, ratios, args.threshold, args.max_group_frac,
        args.dataset_dir, args.ratio_tolerance, extra,
    )
    errors = [f"{c['name']}: {c['detail']}" for c in checks if not c["passed"]]

    multi = [g for g in groups if len(g["members"]) > 1]
    pixel_only = [l for l in links if l["link_type"].startswith("pixel_")]
    n_excluded = sum(r["split"] == EXCLUDED for r in records)
    report = dict(
        status="PASSED" if not errors else "FAILED",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        config=dict(
            dataset_dir=args.dataset_dir.as_posix(), threshold=args.threshold, ratios=list(ratios),
            seed=args.seed, max_group_frac=args.max_group_frac, ratio_tolerance=args.ratio_tolerance,
            hashes="pHash (32x32 DCT, 8x8 low freq) + dHash (9x8), both <= threshold; or equal MD5",
            pixel_rule=(f"{THUMB_SIZE}px thumbnail corr >= {CANDIDATE_CORR} (original or mirrored) -> "
                        f"{CONFIRM_SIZE}px corr >= {CONFIRM_CORR} and {BLOCK}px-block corr p10 >= "
                        f"{CONFIRM_BLOCK_P10} in the better orientation"),
            link_related=args.link_related,
            extra_exclusions=[dict(path=p, reason=r) for p, r in sorted(extra.items())],
            exclusion_rules=dict(pre_augmented=f"filename matches /{AUG_PATTERN.pattern}/i",
                                 label_conflict_copy=sorted({*LABEL_CONFLICT_COPIES, *extra})),
            documented_keep=[list(p) for p in LABEL_CONFLICT_KEEP],
            versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                          pillow=PIL.__version__, scipy=scipy.__version__),
        ),
        labels=sorted({r["label"] for r in records}),
        dataset=dict(
            images=len(records), eligible_images=len(records) - n_excluded, excluded_images=n_excluded,
            fingerprint=dataset_fingerprint(records),
            groups=len(groups), multi_image_groups=len(multi),
            exact_groups=sum(g["group_type"] == "exact" for g in groups),
            near_groups=sum(g["group_type"] == "near" for g in groups),
            images_in_multi_groups=sum(len(g["members"]) for g in multi),
            largest_group=max(len(g["members"]) for g in groups),
            group_size_histogram={str(k): v for k, v in sorted(Counter(len(g["members"]) for g in groups).items())},
            links_by_type=dict(Counter(l["link_type"] for l in links)),
            pixel_only_links=len(pixel_only),
            pixel_only_mirror=sum(l["orientation"] == "mirror" for l in pixel_only),
            pixel_only_same=sum(l["orientation"] == "same" for l in pixel_only),
            mirror_links_total=sum(l["orientation"] == "mirror" for l in links),
            unconfirmed_candidates=len(review),
            label_conflict_groups=sum(g["label_conflict"] for g in groups),
        ),
        exclusions=exclusion_summary(records),
        splits=split_summary(records, groups),
        mixed_label_groups=mixed_label_groups(records, links),
        checks=checks, errors=errors, warnings=warnings, sensitivity=sensitivity,
        residual=residual, limitations=LIMITATIONS,
    )

    ensure_dir(args.out_dir)
    if not errors:
        report["manifest_sha256"] = write_manifest(args.out_dir / "manifest.csv", records)
        write_pairs(args.out_dir / "duplicate_pairs.csv", records, links)
    write_pairs(args.out_dir / "review_pairs.csv", records, review)
    (args.out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.out_dir / "audit_report.md", report)

    d = report["dataset"]
    print(f"\nOriginal {d['images']} | eligible {d['eligible_images']} | excluded {d['excluded_images']} "
          f"{ {k: v['count'] for k, v in report['exclusions'].items()} }")
    print(f"Groups {d['groups']} | multi-image {d['multi_image_groups']} | largest {d['largest_group']} | "
          f"links {d['links_by_type']} | unconfirmed candidates {d['unconfirmed_candidates']}")
    for s, info in report["splits"].items():
        print(f"  {s:8s}: {info['images']:5d} images, {info['groups']:5d} groups, classes {info['classes']}")
    print(f"Mixed-label groups: {len(report['mixed_label_groups'])}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"\nSTOPPED: split is not trustworthy; no manifest written. See {args.out_dir / 'audit_report.md'}")
        return 1
    print(f"\nPASSED. Wrote {args.out_dir}/manifest.csv, duplicate_pairs.csv, review_pairs.csv, "
          "audit_report.json, audit_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
