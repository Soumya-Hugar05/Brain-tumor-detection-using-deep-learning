"""Copy a small, reproducible set of demo MRI images for the web app (VALIDATION split only).

2 images per class, drawn at random (seed 1337) from the approved validation split of data_splits/manifest.csv.
They are NOT chosen by whether the model gets them right. The test split is never loaded.
Writes static/samples/<name>.<ext> and static/samples/samples.json.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from neuroinsight.manifest import CLASS_NAMES, load_split, verify_manifest

OUT = Path(__file__).resolve().parent / "static" / "samples"
PER_CLASS, SEED = 2, 1337


def main() -> None:
    rows = load_split("val", allow_test=False)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    samples = []
    for cls in CLASS_NAMES:
        pool = [r for r in rows if r["label"] == cls]
        for i, r in enumerate(sorted(rng.sample(pool, PER_CLASS), key=lambda r: r["rel"]), 1):
            name = f"sample_{cls}_{i}{Path(r['rel']).suffix.lower()}"
            shutil.copyfile(r["path"], OUT / name)
            samples.append(dict(file=name, dataset_label=cls, source=r["rel"]))
    (OUT / "samples.json").write_text(json.dumps(dict(
        note="Demo samples drawn at random from the approved validation split (seed 1337). Not test data.",
        manifest_sha256=verify_manifest(), split="val", samples=samples), indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
