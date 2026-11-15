from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(); root = args.artifact_root / "results"; rows = []
    for seed in range(3020, 3025):
        own = root / f"m2lite_s{seed}_test.json"
        if own.is_file(): rows.append({"method": "m2_lite", "seed": seed, **json.loads(own.read_text())})
        yolo = root / f"yolov8l_s{seed}" / "test_best.json"
        if yolo.is_file(): rows.append({"method": "yolov8l", "seed": seed, **json.loads(yolo.read_text())})
        unetpp = root / f"baseline_unetpp_watershed_s{seed}" / "test_best.json"
        if unetpp.is_file(): rows.append({"method": "unetpp_watershed", "seed": seed, **json.loads(unetpp.read_text())})
        for method in ("mask_rcnn", "cascade_mask_rcnn", "htc", "solov2", "condinst", "queryinst", "mask2former", "rtmdet_ins"):
            path = root / f"baseline_{method}_s{seed}" / "test_best.json"
            if path.is_file(): rows.append({"method": method, "seed": seed, **json.loads(path.read_text())})
    frame = pd.DataFrame(rows)
    if frame.empty: raise RuntimeError(f"No test result JSON files below {root}")
    metrics = [name for name in ("global_pq", "mean_pq", "sq", "rq", "tp", "fp", "fn") if name in frame]
    summary = frame.groupby("method")[metrics].agg(["mean", "std", "count"])
    output = root / "five_seed_summary.csv"; summary.to_csv(output)
    frame.to_csv(root / "five_seed_runs.csv", index=False)
    print(summary.to_string()); print(f"Saved: {output}")


if __name__ == "__main__": main()
