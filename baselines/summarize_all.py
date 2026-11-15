from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SEEDS = tuple(range(3020, 3025))
BASELINES = (
    "yolov8l", "unetpp_watershed", "mask_rcnn", "cascade_mask_rcnn",
    "htc", "solov2", "condinst", "queryinst", "mask2former", "rtmdet_ins",
    "detr_crop_unet", "edgeattnet", "fullres_unet", "maskrcnn_consensus",
)


def result_path(root: Path, method: str, seed: int) -> Path:
    if method == "m2_lite":
        return root / f"magfilo_v1_m2lite_s{seed}_test.json"
    if method == "yolov8l":
        return root / f"magfilo_v1_yolov8l_s{seed}" / "test_best.json"
    return root / f"baseline_magfilo_v1_{method}_s{seed}" / "test_best.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(); root = args.artifact_root / "results"; rows = []
    missing = []
    for method in ("m2_lite", *BASELINES):
        for seed in SEEDS:
            path = result_path(root, method, seed)
            if path.is_file():
                rows.append({"method": method, "seed": seed, **json.loads(path.read_text())})
            else:
                missing.append((method, seed, str(path)))
    frame = pd.DataFrame(rows)
    if frame.empty: raise RuntimeError(f"No test result JSON files below {root}")
    metrics = [name for name in ("global_pq", "mean_pq", "sq", "rq", "tp", "fp", "fn") if name in frame]
    summary = frame.groupby("method")[metrics].agg(["mean", "std", "count"])
    output = root / "five_seed_summary.csv"; summary.to_csv(output)
    frame.to_csv(root / "five_seed_runs.csv", index=False)
    print(summary.to_string()); print(f"Saved: {output}")
    if missing:
        print(f"WARNING: {len(missing)} of 75 expected test results are missing")
        for method, seed, path in missing: print(f"  {method} seed={seed}: {path}")


if __name__ == "__main__": main()
