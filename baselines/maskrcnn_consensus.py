"""Full-resolution port of the public Mask R-CNN consensus/alignment solution."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from baselines.common import (
    INPUT_SIZE,
    annotation_maps,
    consensus_instances,
    load_problem,
    pooled_pq,
    resolve_scored_masks,
    write_result,
)
from filament.postprocess import mask_to_rle_object


def build_model(pretrained: bool):
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None, min_size=INPUT_SIZE, max_size=INPUT_SIZE,
        box_detections_per_img=300
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 2)
    return model


class ConsensusDataset(Dataset):
    def __init__(self, data_root: Path, manifest, coco: dict, split: str, cache_root: Path, training: bool):
        self.image_root = data_root / "images"
        self.names = sorted(manifest.loc[manifest["split"] == split, "file_name"].unique())
        self.images, self.annotations, self.records = annotation_maps(coco)
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.training = training

    def __len__(self):
        return len(self.names)

    def _masks(self, name: str) -> np.ndarray:
        path = self.cache_root / f"{Path(name).stem}.npz"
        if not path.is_file():
            masks = consensus_instances(name, self.images, self.annotations, self.records)
            stacked = np.stack(masks).astype(np.uint8) if masks else np.zeros((0, INPUT_SIZE, INPUT_SIZE), np.uint8)
            np.savez_compressed(path, masks=stacked)
        return np.load(path)["masks"]

    def __getitem__(self, index: int):
        name = self.names[index]
        gray = cv2.imread(str(self.image_root / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(self.image_root / name)
        masks = self._masks(name)
        if self.training and random.random() < 0.5:
            gray, masks = gray[:, ::-1].copy(), masks[:, :, ::-1].copy()
        if self.training and random.random() < 0.5:
            gray, masks = gray[::-1].copy(), masks[:, ::-1, :].copy()
        if self.training and random.random() < 0.8:
            angle = random.uniform(-180.0, 180.0)
            matrix = cv2.getRotationMatrix2D((INPUT_SIZE / 2, INPUT_SIZE / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, matrix, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            masks = np.stack([
                cv2.warpAffine(mask, matrix, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
                for mask in masks
            ]) if len(masks) else masks
        if self.training and random.random() < 0.5:
            contrast, brightness = random.uniform(0.8, 1.2), random.uniform(-51, 51)
            gray = np.clip(gray.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        if self.training and random.random() < 0.15:
            gray = cv2.GaussianBlur(gray, (random.choice((3, 5)),) * 2, 0)
        boxes, kept = [], []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if len(xs) < 15:
                continue
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            kept.append(mask)
        target = {
            "boxes": torch.as_tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(kept), dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(kept).astype(np.uint8)) if kept else torch.zeros((0, INPUT_SIZE, INPUT_SIZE), dtype=torch.uint8),
            "image_id": torch.tensor([index]),
        }
        image = torch.from_numpy(gray).float()[None].repeat(3, 1, 1) / 255.0
        return image, target, name


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def inference(model, dataset: ConsensusDataset, device: torch.device):
    model.eval(); cache = {}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    for images, _, names in tqdm(loader, desc="Mask R-CNN inference"):
        output = model([images[0].to(device)])[0]
        scores = output["scores"].cpu().numpy()
        masks = output["masks"][:, 0].cpu().numpy()
        cache[names[0]] = [
            (float(score), mask_to_rle_object(mask >= 0.5))
            for score, mask in zip(scores, masks) if score >= 0.05
        ]
    return cache


def aligned(mask: np.ndarray, shift: int, erode: bool) -> np.ndarray:
    if erode:
        mask = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    if shift:
        mask = np.roll(mask, (shift, shift), axis=(0, 1))
        mask[:shift] = False; mask[:, :shift] = False
    return mask


def tune(coco, records, cache, frozen=None):
    configs = [
        (score, area, shift, erode)
        for score in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
        for area in (100, 200, 300)
        for shift, erode in ((0, False), (0, True), (1, False), (1, True))
    ]
    if frozen:
        configs = [(float(frozen["score_threshold"]), int(frozen["min_area"]),
                    int(frozen["shift"]), bool(frozen["erode"]))]
    rows = []
    for score, area, shift, erode in tqdm(configs, desc="Mask R-CNN PQ grid"):
        predictions = {
            name: [
                aligned(mask, shift, erode)
                for mask in resolve_scored_masks(
                    [(value, mask_utils.decode(rle).astype(bool)) for value, rle in items], score, area
                )
            ]
            for name, items in cache.items()
        }
        rows.append({"score_threshold": score, "min_area": area, "shift": shift, "erode": erode,
                     **pooled_pq(coco, records, predictions)})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("global_pq", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-config", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    coco, manifest, _ = load_problem(args.data_root, args.manifest)
    cache_root = args.artifact_root / "baseline_cache" / "magfilo_v1" / "consensus_instances"
    run_dir = args.artifact_root / "baseline_work" / "magfilo_v1" / "maskrcnn_consensus" / f"seed{args.seed}"
    result_dir = args.artifact_root / "results" / f"baseline_magfilo_v1_maskrcnn_consensus_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    model = build_model(pretrained=args.command == "train").to(device)

    if args.command == "train":
        training = ConsensusDataset(args.data_root, manifest, coco, "train", cache_root, True)
        validation = ConsensusDataset(args.data_root, manifest, coco, "val", cache_root, False)
        loader = DataLoader(training, batch_size=1, shuffle=True, num_workers=args.workers,
                            collate_fn=collate, pin_memory=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
        accumulation = 8  # upstream effective batch: batch 2 x four accumulated steps
        total_steps = args.epochs * max(math.ceil(len(loader) / accumulation), 1)
        step = 0; best = -1.0
        for epoch in range(1, args.epochs + 1):
            model.train(); running = 0.0; optimizer.zero_grad(set_to_none=True)
            for batch_index, (images, targets, _) in enumerate(
                tqdm(loader, desc=f"maskrcnn consensus {epoch}/{args.epochs}"), start=1
            ):
                if not len(targets[0]["boxes"]):
                    continue
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                losses = model(images, targets); loss = sum(losses.values())
                (loss / accumulation).backward(); running += float(loss.item())
                if batch_index % accumulation == 0 or batch_index == len(loader):
                    factor = min(1.0, (step + 1) / 100) * 0.5 * (
                        1 + math.cos(math.pi * min(step / max(total_steps, 1), 1.0))
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = 1e-4 * factor
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if epoch >= 8 and (epoch % 2 == 0 or epoch == args.epochs):
                val_cache = inference(model, validation, device)
                ranking = tune(coco, manifest[manifest["split"] == "val"], val_cache)
                score = float(ranking.iloc[0]["global_pq"])
                print(f"epoch={epoch} loss={running/max(len(loader),1):.5f} val_pq={score:.6f}")
                if score > best:
                    best = score
                    torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed,
                                "input_size": INPUT_SIZE}, checkpoint)
        return

    payload = torch.load(checkpoint, map_location=device)
    if int(payload.get("input_size", 0)) != INPUT_SIZE:
        raise RuntimeError("Checkpoint input size mismatch")
    model.load_state_dict(payload["model"])
    dataset = ConsensusDataset(args.data_root, manifest, coco, args.split, cache_root, False)
    cache = inference(model, dataset, device)
    frozen = json.loads(args.frozen_config.read_text()) if args.frozen_config else None
    ranking = tune(coco, manifest[manifest["split"] == args.split], cache, frozen)
    ranking.to_csv(result_dir / f"{args.split}_pq.csv", index=False)
    best = ranking.iloc[0].to_dict(); best.update({"seed": args.seed, "input_size": INPUT_SIZE})
    write_result(result_dir / f"{args.split}_best.json", best)
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
