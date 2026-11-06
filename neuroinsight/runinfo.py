"""Run directories, seeds and environment records."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from .manifest import ROOT, verify_manifest

SEED = 1337
RUNS_DIR = ROOT / "runs"
PACKAGES = ["tensorflow", "keras", "scikit-learn", "numpy", "scipy", "opencv-python",
            "pillow", "matplotlib", "joblib"]


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - older TF
        pass


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": (run("status", "--porcelain") or "").count("\n") + 1}


def new_run(name: str, config: dict) -> Path:
    """Create runs/<timestamp>_<name>/ with models/ and results/ and write config.json."""
    manifest_sha = verify_manifest()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{name}"
    (run_dir / "models").mkdir(parents=True, exist_ok=False)
    (run_dir / "results").mkdir()
    record = dict(run=run_dir.name, created=datetime.now().isoformat(timespec="seconds"),
                  argv=sys.argv, seed=SEED, manifest_sha256=manifest_sha,
                  versions=versions(), git=git_state(), config=config)
    write_json(run_dir / "config.json", record)
    return run_dir


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(type(o))
