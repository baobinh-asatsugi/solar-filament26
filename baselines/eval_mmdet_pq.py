from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

from filament.config import Paths, discover_data
from filament.postprocess import (
    accumulate_metric_record, finalize_metric_totals, mask_to_rle_object,
    metric_totals, polygon_to_rle, pq_stats_rle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official pooled-PQ evaluation for an MMDetection baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, default=None)
    args = parser.parse_args()
    annotation_path, image_root, _ = discover_data(Paths(args.data_root.resolve(), args.artifact_root.resolve()))
    manifest = pd.read_csv(args.manifest)
    records = manifest[manifest["split"] == args.split]
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)
    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    cache = {}
    for file_name in tqdm(sorted(records.file_name.unique()), desc=f"infer {args.split}"):
        sample = inference_detector(model, str(image_root / file_name)).pred_instances.cpu()
        masks = sample.masks.numpy().astype(bool) if len(sample) else np.zeros((0, 1, 1), bool)
        cache[file_name] = (masks, sample.scores.numpy() if len(sample) else np.empty(0))
    configurations = [(float(t), int(a)) for t in (0.05, .10, .15, .20, .25, .30, .40, .50) for a in (100, 200, 300)]
    if args.frozen_config:
        frozen = json.loads(args.frozen_config.read_text())
        configurations = [(float(frozen["score_threshold"]), int(frozen["min_area"]))]
    rows = []
    for threshold, min_area in configurations:
        stats = metric_totals()
        for file_name, group in records.groupby("file_name"):
            masks, scores = cache[file_name]
            predictions = [mask_to_rle_object(mask) for mask, score in zip(masks, scores)
                           if score >= threshold and int(mask.sum()) >= min_area]
            for row in group.itertuples(index=False):
                info = images[str(row.image_id)]
                gt = [polygon_to_rle(item["segmentation"], info["height"], info["width"])
                      for item in annotations[str(row.image_id)]]
                values = pq_stats_rle(gt, predictions)
                accumulate_metric_record(stats, values)
        rows.append({"score_threshold": threshold, "min_area": min_area,
                     **finalize_metric_totals(stats)})
    ranking = pd.DataFrame(rows).sort_values("global_pq", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output_dir / f"{args.split}_pq.csv", index=False)
    best = ranking.iloc[0].to_dict()
    (args.output_dir / f"{args.split}_best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
