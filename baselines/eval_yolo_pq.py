from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from filament.config import Paths, discover_data
from filament.postprocess import (
    accumulate_metric_record, finalize_metric_totals, mask_to_rle_object,
    metric_totals, polygon_to_rle, pq_stats_rle,
)


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
        if result.masks is None:
            cache[name] = []
        else:
            masks = result.masks.data.cpu().numpy().astype(bool)
            scores = result.boxes.conf.cpu().numpy()
            cache[name] = [
                (mask_to_rle_object(mask), float(score), int(mask.sum()))
                for mask, score in zip(masks, scores)
            ]
        del result

    # PQ tuning below is CPU-only. Release the detector and its CUDA cache so a
    # waiting training phase may use this GPU while the grid is evaluated.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"YOLO {args.split}: GPU inference complete; detector memory released before PQ sweep", flush=True)
    configs = [(t, a) for t in (.05, .10, .15, .20, .25, .30, .40, .50) for a in (100, 200, 300)]
    if args.frozen_config:
        frozen = json.loads(args.frozen_config.read_text()); configs = [(float(frozen["score_threshold"]), int(frozen["min_area"]))]
    rows = []
    for threshold, area in tqdm(configs, desc=f"YOLO {args.split} PQ sweep"):
        stat = metric_totals()
        for name, group in records.groupby("file_name"):
            pred = [
                encoded for encoded, score, mask_area in cache[name]
                if score >= threshold and mask_area >= area
            ]
            for row in group.itertuples(index=False):
                info = images[str(row.image_id)]
                gt = [polygon_to_rle(x["segmentation"], info["height"], info["width"]) for x in annotations[str(row.image_id)]]
                value = pq_stats_rle(gt, pred)
                accumulate_metric_record(stat, value)
        rows.append({"score_threshold": threshold, "min_area": area,
                     **finalize_metric_totals(stat)})
    ranking = pd.DataFrame(rows).sort_values("global_pq", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True); ranking.to_csv(args.output_dir / f"{args.split}_pq.csv", index=False)
    (args.output_dir / f"{args.split}_best.json").write_text(json.dumps(ranking.iloc[0].to_dict(), indent=2) + "\n")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__": main()
