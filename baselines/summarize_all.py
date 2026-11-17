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
METHODS = ("m2_lite", *BASELINES)


def result_path(root: Path, method: str, seed: int) -> Path:
    if method == "m2_lite":
        return root / f"magfilo_v1_m2lite_s{seed}_test.json"
    if method == "yolov8l":
        return root / f"magfilo_v1_yolov8l_s{seed}" / "test_best.json"
    return root / f"baseline_magfilo_v1_{method}_s{seed}" / "test_best.json"


def checkpoint_path(artifact_root: Path, method: str, seed: int) -> Path:
    if method == "m2_lite":
        return artifact_root / "runs" / f"magfilo_v1_m2lite_s{seed}_sira" / "best.pt"
    if method == "yolov8l":
        return artifact_root / "runs" / f"magfilo_v1_yolov8l_s{seed}" / "weights" / "best.pt"
    if method == "unetpp_watershed":
        return artifact_root / "baseline_work/magfilo_v1/unetpp_watershed" / f"seed{seed}" / "best.pt"
    if method in {"fullres_unet", "edgeattnet", "maskrcnn_consensus"}:
        return artifact_root / "baseline_work/magfilo_v1" / method / f"seed{seed}" / "best.pt"
    if method == "detr_crop_unet":
        return artifact_root / "baseline_work/magfilo_v1/detr_crop_unet" / f"seed{seed}" / "crop_unet_best.pt"
    return artifact_root / "baseline_work/magfilo_v1" / method / f"seed{seed}" / "best.pth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(); artifact_root = args.artifact_root.resolve(); root = artifact_root / "results"; root.mkdir(parents=True, exist_ok=True); rows = []
    skipped = []
    for method in METHODS:
        for seed in SEEDS:
            path = result_path(root, method, seed)
            checkpoint = checkpoint_path(artifact_root, method, seed)
            if not path.is_file() or path.stat().st_size == 0:
                skipped.append({
                    "method": method, "seed": seed, "reason": "missing_result",
                    "checkpoint_exists": checkpoint.is_file(),
                    "checkpoint_path": str(checkpoint), "result_path": str(path),
                })
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                skipped.append({
                    "method": method, "seed": seed, "reason": f"invalid_result:{type(error).__name__}",
                    "checkpoint_exists": checkpoint.is_file(),
                    "checkpoint_path": str(checkpoint), "result_path": str(path),
                })
                continue
            if "global_pq" not in payload:
                skipped.append({
                    "method": method, "seed": seed, "reason": "missing_global_pq",
                    "checkpoint_exists": checkpoint.is_file(),
                    "checkpoint_path": str(checkpoint), "result_path": str(path),
                })
                continue
            rows.append({"method": method, "seed": seed, **payload})
    frame = pd.DataFrame(rows)
    metrics = [name for name in (
        "global_pq", "mean_pq", "sq", "rq",
        "semantic_dice", "semantic_iou", "mean_semantic_dice", "mean_semantic_iou",
        "tp", "fp", "fn",
    ) if name in frame]
    summary_rows = []
    for method in METHODS:
        group = frame[frame.method == method].sort_values("seed") if not frame.empty else frame
        if group.empty:
            continue
        row = {
            "method": method, "n_seeds": len(group),
            "seeds": "|".join(f"s{int(seed)}" for seed in group.seed.tolist()),
            "complete_5_seeds": len(group) == len(SEEDS),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else float("nan")
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else float("nan")
        summary_rows.append(row)
    summary_columns = ["method", "n_seeds", "seeds", "complete_5_seeds"] + [
        f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")
    ]
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    output = root / "five_seed_summary.csv"
    skipped_frame = pd.DataFrame(skipped, columns=(
        "method", "seed", "reason", "checkpoint_exists", "checkpoint_path", "result_path"
    ))
    summary.to_csv(output, index=False)
    if frame.empty:
        pd.DataFrame(columns=("method", "seed")).to_csv(root / "five_seed_runs.csv", index=False)
    else:
        frame.sort_values(["method", "seed"]).to_csv(root / "five_seed_runs.csv", index=False)
    skipped_frame.to_csv(root / "five_seed_skipped.csv", index=False)
    print(summary.to_string(index=False)); print(f"Saved: {output}")
    print(f"Included {len(frame)}/75 runs; skipped {len(skipped_frame)}/75")
    if len(skipped_frame):
        print(skipped_frame[["method", "seed", "reason", "checkpoint_exists"]].to_string(index=False))


if __name__ == "__main__": main()
