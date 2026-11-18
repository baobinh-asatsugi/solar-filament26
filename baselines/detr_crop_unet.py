"""RT-DETR-family detector -> filament crop -> plain U-Net paper reconstruction."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import RTDETR
from pycocotools import mask as mask_utils

from baselines.common import INPUT_SIZE, load_problem, pooled_pq, resolve_scored_masks, write_result
from filament.data.annotators import select_annotation_records
from filament.postprocess import mask_to_rle_object


CROP_SIZE = 256


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.layers(value)


class PlainUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128), ConvBlock(128, 256)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up = nn.ModuleList([nn.ConvTranspose2d(512, 256, 2, 2), nn.ConvTranspose2d(256, 128, 2, 2),
                                 nn.ConvTranspose2d(128, 64, 2, 2), nn.ConvTranspose2d(64, 32, 2, 2)])
        self.decoders = nn.ModuleList([ConvBlock(512, 256), ConvBlock(256, 128), ConvBlock(128, 64), ConvBlock(64, 32)])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        skips = []; value = image
        for encoder in self.encoders:
            value = encoder(value); skips.append(value); value = self.pool(value)
        value = self.bottleneck(value)
        for up, decoder, skip in zip(self.up, self.decoders, reversed(skips)):
            value = decoder(torch.cat([up(value), skip], dim=1))
        return self.head(value)


def expanded_box(box, width=INPUT_SIZE, height=INPUT_SIZE, expansion=0.10):
    x, y, w, h = map(float, box); side_x, side_y = w * expansion, h * expansion
    x0, y0 = max(0, int(np.floor(x - side_x))), max(0, int(np.floor(y - side_y)))
    x1, y1 = min(width, int(np.ceil(x + w + side_x))), min(height, int(np.ceil(y + h + side_y)))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def crop_mask(segmentation, box):
    x0, y0, x1, y1 = box
    mask = np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
    sx, sy = CROP_SIZE / (x1 - x0), CROP_SIZE / (y1 - y0)
    for polygon in segmentation:
        points = np.asarray(polygon, np.float32).reshape(-1, 2)
        points[:, 0] = (points[:, 0] - x0) * sx; points[:, 1] = (points[:, 1] - y0) * sy
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


class CropDataset(Dataset):
    def __init__(self, data_root: Path, manifest: pd.DataFrame, coco: dict, split: str, training: bool):
        selected = select_annotation_records(manifest[manifest["split"] == split], coco, "medoid", 3020)
        selected_ids = set(selected["image_id"].astype(str))
        images = {str(item["id"]): item for item in coco["images"] if str(item["id"]) in selected_ids}
        self.items = [(images[str(item["image_id"])], item) for item in coco["annotations"] if str(item["image_id"]) in selected_ids]
        self.image_root, self.training = data_root / "images", training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        info, annotation = self.items[index]
        image = cv2.imread(str(self.image_root / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        box = expanded_box(annotation["bbox"], int(info["width"]), int(info["height"]))
        x0, y0, x1, y1 = box
        crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = crop_mask(annotation["segmentation"], box)
        if self.training and random.random() < 0.5:
            crop, mask = crop[:, ::-1].copy(), mask[:, ::-1].copy()
        if self.training and random.random() < 0.5:
            crop, mask = crop[::-1].copy(), mask[::-1].copy()
        return torch.from_numpy(crop).float()[None] / 255.0, torch.from_numpy(mask).float()[None]


def iou_loss(logits, target):
    # Keep the reduction in FP32. A 256x256 crop has 65,536 pixels, which is
    # already beyond float16's largest finite value (65,504). Summing mask
    # probabilities under autocast can therefore overflow and poison the
    # refiner even when the input and parameters are otherwise valid.
    probability = logits.float().sigmoid(); target = target.float()
    intersection = (probability * target).sum((1, 2, 3))
    union = probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) - intersection
    return (1 - (intersection + 1) / (union + 1)).mean()


@torch.no_grad()
def infer(detector, refiner, image_root: Path, names: list[str], device: str):
    refiner.eval(); torch_device = next(refiner.parameters()).device; cache = {}
    for name in tqdm(names, desc="DETR -> crop U-Net inference"):
        image = cv2.imread(str(image_root / name), cv2.IMREAD_GRAYSCALE)
        result = detector.predict(str(image_root / name), imgsz=INPUT_SIZE, conf=0.03, iou=0.75,
                                  max_det=200, device=device, verbose=False)[0]
        items = []
        for box_tensor, score_tensor in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = expanded_box([box_tensor[0], box_tensor[1], box_tensor[2]-box_tensor[0], box_tensor[3]-box_tensor[1]])
            crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(crop).float()[None, None].to(torch_device) / 255.0
            # This small refiner is intentionally evaluated in FP32. Its
            # activations can overflow FP16 on high-contrast solar crops.
            probability = refiner(tensor).float().sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError(f"Non-finite crop probability for {name}")
            probability = np.ascontiguousarray(probability, dtype=np.float32)
            local = cv2.resize(probability, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR) >= 0.5
            mask = np.zeros((INPUT_SIZE, INPUT_SIZE), bool); mask[y0:y1, x0:x1] = local
            items.append((float(score_tensor), mask_to_rle_object(mask)))
        cache[name] = items
    return cache


def tune(coco, records, cache, frozen=None):
    configs = [(score, area) for score in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40) for area in (100, 200, 300)]
    if frozen:
        configs = [(float(frozen["score_threshold"]), int(frozen["min_area"]))]
    rows = []
    for score, area in tqdm(configs, desc="DETR crop PQ grid"):
        predictions = {
            name: resolve_scored_masks(
                [(value, mask_utils.decode(rle).astype(bool)) for value, rle in items], score, area
            )
            for name, items in cache.items()
        }
        rows.append({"score_threshold": score, "min_area": area, **pooled_pq(coco, records, predictions)})
    return pd.DataFrame(rows).sort_values("global_pq", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-refiner", "check-refiner", "eval"))
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--detector", type=Path, required=True); parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("val", "test"), default="val"); parser.add_argument("--frozen-config", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    coco, manifest, image_root = load_problem(args.data_root, args.manifest)
    run_dir = args.artifact_root / "baseline_work" / "magfilo_v1" / "detr_crop_unet" / f"seed{args.seed}"
    result_dir = args.artifact_root / "results" / f"baseline_magfilo_v1_detr_crop_unet_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "crop_unet_best.pt"; refiner = PlainUNet().to(device)
    if args.command == "train-refiner":
        training = CropDataset(args.data_root, manifest, coco, "train", True)
        validation = CropDataset(args.data_root, manifest, coco, "val", False)
        train_loader = DataLoader(training, batch_size=16, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(validation, batch_size=16, shuffle=False, num_workers=args.workers)
        optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-3); best = float("inf")
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        for epoch in range(1, args.epochs + 1):
            refiner.train(); total = 0.0; finite_batches = 0; skipped_batches = 0
            for image, target in tqdm(train_loader, desc=f"crop U-Net {epoch}/{args.epochs}"):
                image, target = image.to(device), target.to(device); optimizer.zero_grad(set_to_none=True)
                # Keep the convolutional forward under AMP for memory, but
                # iou_loss explicitly promotes logits and reductions to FP32.
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = refiner(image)
                loss = iou_loss(logits, target)
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True); skipped_batches += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients_finite = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in refiner.parameters()
                )
                if not gradients_finite:
                    optimizer.zero_grad(set_to_none=True); scaler.update(); skipped_batches += 1
                    continue
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=5.0)
                scaler.step(optimizer); scaler.update(); total += float(loss.item())
                finite_batches += 1
            refiner.eval(); values = []
            with torch.no_grad():
                for image, target in val_loader:
                    candidate = float(iou_loss(refiner(image.to(device)), target.to(device)).item())
                    if np.isfinite(candidate): values.append(candidate)
            if not values:
                raise FloatingPointError(f"No finite validation losses at epoch {epoch}")
            value = float(np.mean(values))
            print(f"epoch={epoch} train={total/max(finite_batches,1):.6f} val_iou_loss={value:.6f} skipped_nonfinite={skipped_batches}")
            if np.isfinite(value) and value < best:
                best = value; torch.save({"model": refiner.state_dict(), "seed": args.seed,
                                          "crop_size": CROP_SIZE, "input_size": INPUT_SIZE}, checkpoint)
        return
    payload = torch.load(checkpoint, map_location=device); refiner.load_state_dict(payload["model"])
    if args.command == "check-refiner":
        if not all(torch.isfinite(parameter).all() for parameter in refiner.parameters()):
            raise FloatingPointError(f"Non-finite parameter in {checkpoint}")
        validation = CropDataset(args.data_root, manifest, coco, "val", False)
        loader = DataLoader(validation, batch_size=16, shuffle=False, num_workers=args.workers)
        refiner.eval()
        with torch.no_grad():
            for batch_index, (image, _) in enumerate(loader):
                output = refiner(image.to(device))
                if not torch.isfinite(output).all():
                    raise FloatingPointError(
                        f"Non-finite FP32 output in {checkpoint} at validation batch {batch_index}"
                    )
        print(f"Valid FP32 crop U-Net checkpoint: {checkpoint}")
        return
    detector = RTDETR(str(args.detector)); names = sorted(manifest.loc[manifest["split"] == args.split, "file_name"].unique())
    cache = infer(detector, refiner, image_root, names, args.device)
    frozen = json.loads(args.frozen_config.read_text()) if args.frozen_config else None
    ranking = tune(coco, manifest[manifest["split"] == args.split], cache, frozen)
    ranking.to_csv(result_dir / f"{args.split}_pq.csv", index=False)
    best = ranking.iloc[0].to_dict(); best.update({"seed": args.seed, "input_size": INPUT_SIZE, "crop_size": CROP_SIZE})
    write_result(result_dir / f"{args.split}_best.json", best); print(ranking.head(10).to_string(index=False))


if __name__ == "__main__": main()
