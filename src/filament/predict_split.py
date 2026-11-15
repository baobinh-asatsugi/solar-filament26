from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_path_args, discover_data, paths_from_args
from .oof_bank import ProposalBank, proposal_record
from .semantic_inference import SemanticPredictionCache, infer_semantic_image, load_semantic_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache detector and semantic predictions for one fixed split")
    add_path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--base-conf", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--max-det", type=int, default=200)
    parser.add_argument("--semantic-batch", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = paths_from_args(args)
    _, image_dir, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    files = sorted(manifest.loc[manifest["split"] == args.split, "file_name"].unique())
    if not files:
        raise ValueError(f"No {args.split} files in {args.manifest}")
    raw = ProposalBank(paths.artifact_root / "predictions" / args.run_name / "raw" / args.split)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    semantic_model, semantic_config = load_semantic_checkpoint(args.semantic.resolve(), device)
    tile_size = int(semantic_config["tile_size"])
    feature_mode = semantic_config["feature_mode"]
    semantic = SemanticPredictionCache(
        paths.artifact_root / "predictions" / args.run_name / "semantic" / args.split,
        args.semantic.resolve(), {"tile_size": tile_size, "overlap": 0.5, "feature_mode": feature_mode, "tta": False},
    )
    detector = YOLO(str(args.detector.resolve()))
    for file_name in tqdm(files, desc=f"Cache {args.split}", unit="image"):
        image_path = image_dir / file_name
        if args.overwrite or not raw.path_for(file_name).is_file():
            result = detector.predict(str(image_path), imgsz=args.imgsz, conf=args.base_conf, iou=args.nms_iou,
                                      max_det=args.max_det, retina_masks=True, device=args.device, verbose=False)[0]
            instances = []
            if result.masks is not None:
                instances = [proposal_record(mask, score, box, args.run_name)
                             for mask, score, box in zip(result.masks.data.cpu().numpy().astype(bool),
                                                         result.boxes.conf.cpu().numpy(),
                                                         result.boxes.xyxy.cpu().numpy())]
            raw.save(file_name, instances, metadata={"split": args.split, "detector": str(args.detector.resolve())})
        if args.overwrite or not semantic.has(file_name):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            probability = infer_semantic_image(semantic_model, image, device, tile_size, 0.5,
                                               feature_mode, args.semantic_batch, False)
            semantic.save(file_name, probability)
    print(f"Cached {len(files)} {args.split} images under {paths.artifact_root / 'predictions' / args.run_name}")


if __name__ == "__main__":
    main()
