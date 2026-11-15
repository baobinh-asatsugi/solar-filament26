from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .oof_bank import ProposalBank, proposal_record
from .postprocess import mask_to_rle_object
from .sira_inference import load_sira, refine_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply SIRA to one fixed split")
    add_path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = paths_from_args(args)
    _, image_dir, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    files = sorted(manifest.loc[manifest["split"] == args.split, "file_name"].unique())
    root = paths.artifact_root / "predictions" / args.run_name
    raw_bank = ProposalBank(root / "raw" / args.split)
    refined_bank = ProposalBank(root / "refined" / args.split)
    semantic_root = root / "semantic" / args.split
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model, config = load_sira(args.checkpoint.resolve(), device)
    for file_name in tqdm(files, desc=f"Refine {args.split}", unit="image"):
        if refined_bank.path_for(file_name).is_file() and not args.overwrite:
            continue
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        raw = raw_bank.load(file_name, decode_masks=False)["instances"]
        with np.load(semantic_root / f"{Path(file_name).stem}.npz") as values:
            semantic = values["probability"].astype(np.float32)[0]
        outputs = refine_candidates(model, config, image, raw, semantic, device, args.batch)
        saved = []
        for item in outputs:
            x0, y0, x1, y1 = item["bounds"]
            mask = np.zeros(image.shape, bool)
            mask[y0:y1, x0:x1] = item["probability"] >= 0.5
            box = np.asarray(mask_utils.toBbox(mask_to_rle_object(mask)), dtype=float)
            bx, by, bw, bh = box
            uncertainty = float((1.0 - 2.0 * np.abs(item["probability"] - 0.5)).clip(0, 1).mean())
            features = {
                "detector_score": float(item["score"]), "mean_probability": item["mean_probability"],
                "q90_probability": item["q90_probability"], "seed_agreement": item["seed_agreement"],
                "semantic_support": float(semantic[mask].mean()) if mask.any() else 0.0,
                "boundary_uncertainty": uncertainty, "elongation": max(bw, bh) / max(min(bw, bh), 1.0),
                "radial_position": float(np.hypot(bx + bw / 2 - image.shape[1] / 2,
                                                   by + bh / 2 - image.shape[0] / 2) / (min(image.shape) / 2)),
                "area": int(mask.sum()), "components": int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1),
            }
            record = proposal_record(mask, item["score"], np.asarray(item["box"]), item["source"])
            record["features"] = features
            saved.append(record)
        refined_bank.save(file_name, saved, metadata={"split": args.split, "checkpoint": str(args.checkpoint.resolve())})
    print(f"Refined {len(files)} {args.split} images: {refined_bank.root}")


if __name__ == "__main__":
    main()
