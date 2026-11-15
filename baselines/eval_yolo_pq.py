from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from filament.config import Paths, discover_data
from filament.postprocess import mask_to_rle_object, polygon_to_rle, pq_stats_rle


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled-PQ evaluation for the plain YOLOv8l-seg baseline")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--frozen-config", type=Path, default=None)
    args = parser.parse_args()
    annotation_path, image_root, _ = discover_data(Paths(args.data_root.resolve(), args.artifact_root.resolve()))
    manifest = pd.read_csv(args.manifest); records = manifest[manifest.split == args.split]
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(x["id"]): x for x in coco["images"]}; annotations = defaultdict(list)
    for item in coco["annotations"]: annotations[str(item["image_id"])].append(item)
    model = YOLO(str(args.checkpoint)); cache = {}
    for name in tqdm(sorted(records.file_name.unique()), desc=f"YOLO {args.split}"):
        result = model.predict(str(image_root / name), imgsz=2048, conf=.03, iou=.75, max_det=200,
                               retina_masks=True, device=args.device, verbose=False)[0]
        if result.masks is None: cache[name] = (np.zeros((0, 1, 1), bool), np.empty(0))
        else: cache[name] = (result.masks.data.cpu().numpy().astype(bool), result.boxes.conf.cpu().numpy())
    configs = [(t, a) for t in (.05, .10, .15, .20, .25, .30, .40, .50) for a in (100, 200, 300)]
    if args.frozen_config:
        frozen = json.loads(args.frozen_config.read_text()); configs = [(float(frozen["score_threshold"]), int(frozen["min_area"]))]
    rows = []
    for threshold, area in configs:
        stat = dict(iou_sum=0., pq_sum=0., records=0, tp=0, fp=0, fn=0)
        for name, group in records.groupby("file_name"):
            masks, scores = cache[name]
            pred = [mask_to_rle_object(mask) for mask, score in zip(masks, scores) if score >= threshold and mask.sum() >= area]
            for row in group.itertuples(index=False):
                info = images[str(row.image_id)]
                gt = [polygon_to_rle(x["segmentation"], info["height"], info["width"]) for x in annotations[str(row.image_id)]]
                value = pq_stats_rle(gt, pred)
                for key in ("iou_sum", "tp", "fp", "fn"): stat[key] += value[key]
                stat["pq_sum"] += value["pq"]; stat["records"] += 1
        denominator = stat["tp"] + .5 * stat["fp"] + .5 * stat["fn"]
        rows.append({"score_threshold": threshold, "min_area": area,
                     "global_pq": stat["iou_sum"] / denominator if denominator else 0.,
                     "mean_pq": stat["pq_sum"] / max(stat["records"], 1),
                     "sq": stat["iou_sum"] / max(stat["tp"], 1), "rq": stat["tp"] / denominator if denominator else 0.,
                     "tp": stat["tp"], "fp": stat["fp"], "fn": stat["fn"]})
    ranking = pd.DataFrame(rows).sort_values("global_pq", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True); ranking.to_csv(args.output_dir / f"{args.split}_pq.csv", index=False)
    (args.output_dir / f"{args.split}_best.json").write_text(json.dumps(ranking.iloc[0].to_dict(), indent=2) + "\n")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__": main()
