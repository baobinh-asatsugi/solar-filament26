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
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch.utils.data import DataLoader
from tqdm import tqdm

from filament.config import Paths, discover_data
from filament.postprocess import mask_to_rle_object, polygon_to_rle, pq_stats_rle
from filament.semantic_data import SemanticTileDataset, normalize_intensity


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, value): return self.layers(value)


class OfficialUNetPlusPlus(nn.Module):
    """Original five-level nested U-Net++ with four deep-supervision heads."""
    def __init__(self):
        super().__init__(); channels = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2); self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.x00 = ConvBlock(1, channels[0]); self.x10 = ConvBlock(channels[0], channels[1])
        self.x20 = ConvBlock(channels[1], channels[2]); self.x30 = ConvBlock(channels[2], channels[3]); self.x40 = ConvBlock(channels[3], channels[4])
        self.x01 = ConvBlock(channels[0] + channels[1], channels[0]); self.x11 = ConvBlock(channels[1] + channels[2], channels[1])
        self.x21 = ConvBlock(channels[2] + channels[3], channels[2]); self.x31 = ConvBlock(channels[3] + channels[4], channels[3])
        self.x02 = ConvBlock(channels[0] * 2 + channels[1], channels[0]); self.x12 = ConvBlock(channels[1] * 2 + channels[2], channels[1])
        self.x22 = ConvBlock(channels[2] * 2 + channels[3], channels[2])
        self.x03 = ConvBlock(channels[0] * 3 + channels[1], channels[0]); self.x13 = ConvBlock(channels[1] * 3 + channels[2], channels[1])
        self.x04 = ConvBlock(channels[0] * 4 + channels[1], channels[0])
        self.heads = nn.ModuleList([nn.Conv2d(channels[0], 1, 1) for _ in range(4)])
    def forward(self, x):
        x00 = self.x00(x); x10 = self.x10(self.pool(x00)); x20 = self.x20(self.pool(x10)); x30 = self.x30(self.pool(x20)); x40 = self.x40(self.pool(x30))
        x01 = self.x01(torch.cat([x00, self.up(x10)], 1)); x11 = self.x11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], 1)); x31 = self.x31(torch.cat([x30, self.up(x40)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], 1)); x12 = self.x12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], 1)); x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))
        return [head(value) for head, value in zip(self.heads, (x01, x02, x03, x04))]


def bce_dice(outputs, target):
    losses = []
    for logits in outputs:
        probability = logits.sigmoid(); intersection = (probability * target).sum((1, 2, 3))
        dice = 1 - ((2 * intersection + 1e-6) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)).mean()
        losses.append(F.binary_cross_entropy_with_logits(logits, target) + dice)
    return torch.stack(losses).mean()


def positions(length: int, tile: int, stride: int):
    if length <= tile: return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile: values.append(length - tile)
    return values


@torch.no_grad()
def predict_full(model, image, device, tile=512, overlap=.5, batch_size=4):
    stride = int(tile * (1 - overlap)); ys, xs = positions(image.shape[0], tile, stride), positions(image.shape[1], tile, stride)
    weight = np.clip(np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32), .05, None)
    total = np.zeros(image.shape, np.float32); weights = np.zeros(image.shape, np.float32); tensors, locations = [], []
    def flush():
        if not tensors: return
        logits = model(torch.from_numpy(np.stack(tensors)).to(device))
        prediction = torch.stack([value.sigmoid() for value in logits]).mean(0)[:, 0].cpu().numpy()
        for patch, (y, x, h, w) in zip(prediction, locations):
            total[y:y+h, x:x+w] += patch[:h, :w] * weight[:h, :w]; weights[y:y+h, x:x+w] += weight[:h, :w]
        tensors.clear(); locations.clear()
    for y in ys:
        for x in xs:
            patch = image[y:min(y+tile, image.shape[0]), x:min(x+tile, image.shape[1])]; h, w = patch.shape
            if (h, w) != (tile, tile): patch = cv2.copyMakeBorder(patch, 0, tile-h, 0, tile-w, cv2.BORDER_REFLECT_101)
            tensors.append(normalize_intensity(patch)[None]); locations.append((y, x, h, w))
            if len(tensors) == batch_size: flush()
    flush(); return total / np.maximum(weights, 1e-6)


def instances_from_probability(probability, threshold, min_distance, min_area):
    foreground = probability >= threshold
    distance = ndi.distance_transform_edt(foreground)
    coordinates = peak_local_max(distance, labels=foreground, min_distance=min_distance, exclude_border=False)
    markers = np.zeros(foreground.shape, np.int32)
    for index, (y, x) in enumerate(coordinates, 1): markers[y, x] = index
    if not len(coordinates): markers, _ = ndi.label(foreground)
    labels = watershed(-distance, markers, mask=foreground)
    return [mask_to_rle_object(labels == index) for index in range(1, int(labels.max()) + 1) if int((labels == index).sum()) >= min_area]


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    annotation, image_root, _ = discover_data(Paths(args.zip_path.resolve(), args.artifact_root.resolve()))
    common = dict(annotation_path=annotation, image_dir=image_root, manifest_path=args.manifest.resolve(), tile_size=512,
                  feature_mode="intensity", positive_fraction=.75, seed=args.seed)
    training = SemanticTileDataset(split="train", samples=6000, annotation_strategy="medoid", training=True, **common)
    validation = SemanticTileDataset(split="val", samples=1200, annotation_strategy="duplicate", training=False, **common)
    loaders = [DataLoader(training, batch_size=args.batch, shuffle=False, num_workers=args.workers, drop_last=True),
               DataLoader(validation, batch_size=args.batch, shuffle=False, num_workers=args.workers)]
    device = torch.device(f"cuda:{args.device}"); model = OfficialUNetPlusPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4); scaler = torch.cuda.amp.GradScaler()
    args.output_dir.mkdir(parents=True, exist_ok=True); best, bad_epochs = -1., 0
    for epoch in range(1, 101):
        model.train(); total = 0.
        for batch in tqdm(loaders[0], desc=f"UNet++ {epoch}/100"):
            image = batch["image"][:, :1].to(device); target = batch["target"][:, :1].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): loss = bce_dice(model(image), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += float(loss)
        model.eval(); intersection = denominator = 0.
        with torch.no_grad():
            for batch in loaders[1]:
                image = batch["image"][:, :1].to(device); target = batch["target"][:, :1].to(device)
                pred = torch.stack([x.sigmoid() for x in model(image)]).mean(0) >= .5
                intersection += float((pred & target.bool()).sum()); denominator += float(pred.sum() + target.sum())
        dice = 2 * intersection / max(denominator, 1); print(json.dumps({"epoch": epoch, "loss": total/len(loaders[0]), "val_dice": dice}))
        payload = {"model": model.state_dict(), "epoch": epoch, "seed": args.seed}
        torch.save(payload, args.output_dir / "last.pt")
        if dice > best: best, bad_epochs = dice, 0; torch.save(payload, args.output_dir / "best.pt")
        else: bad_epochs += 1
        if bad_epochs >= 20: break


def evaluate(args):
    annotation_path, image_root, _ = discover_data(Paths(args.zip_path.resolve(), args.artifact_root.resolve()))
    manifest = pd.read_csv(args.manifest); records = manifest[manifest.split == args.split]
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(x["id"]): x for x in coco["images"]}; annotations = defaultdict(list)
    for x in coco["annotations"]: annotations[str(x["image_id"])].append(x)
    device = torch.device(f"cuda:{args.device}"); model = OfficialUNetPlusPlus().to(device)
    model.load_state_dict(torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)["model"]); model.eval()
    cache_dir = args.output_dir / "probabilities" / args.split; cache_dir.mkdir(parents=True, exist_ok=True); cache = {}
    for name in tqdm(sorted(records.file_name.unique()), desc=f"UNet++ {args.split}"):
        path = cache_dir / f"{Path(name).stem}.npz"
        if path.is_file(): probability = np.load(path)["probability"].astype(np.float32)
        else:
            image = cv2.imread(str(image_root / name), cv2.IMREAD_GRAYSCALE); probability = predict_full(model, image, device, batch_size=args.batch)
            np.savez_compressed(path, probability=probability.astype(np.float16))
        cache[name] = probability
    configs = [(t, d, a) for t in (.35, .45, .50, .55, .65) for d in (8, 16, 24, 32) for a in (100, 200, 300)]
    frozen_path = args.output_dir / "val_best.json"
    if args.split == "test":
        frozen = json.loads(frozen_path.read_text()); configs = [(frozen["threshold"], int(frozen["min_distance"]), int(frozen["min_area"]))]
    rows = []
    for threshold, distance, area in tqdm(configs, desc="watershed grid"):
        stat = dict(iou_sum=0., pq_sum=0., records=0, tp=0, fp=0, fn=0)
        for name, group in records.groupby("file_name"):
            prediction = instances_from_probability(cache[name], threshold, distance, area)
            for row in group.itertuples(index=False):
                info = images[str(row.image_id)]; gt = [polygon_to_rle(x["segmentation"], info["height"], info["width"]) for x in annotations[str(row.image_id)]]
                value = pq_stats_rle(gt, prediction)
                for key in ("iou_sum", "tp", "fp", "fn"): stat[key] += value[key]
                stat["pq_sum"] += value["pq"]; stat["records"] += 1
        den = stat["tp"] + .5*stat["fp"] + .5*stat["fn"]
        rows.append({"threshold": threshold, "min_distance": distance, "min_area": area,
                     "global_pq": stat["iou_sum"]/den if den else 0., "mean_pq": stat["pq_sum"]/stat["records"],
                     "sq": stat["iou_sum"]/max(stat["tp"], 1), "rq": stat["tp"]/den if den else 0.,
                     "tp": stat["tp"], "fp": stat["fp"], "fn": stat["fn"]})
    ranking = pd.DataFrame(rows).sort_values("global_pq", ascending=False); ranking.to_csv(args.output_dir / f"{args.split}_pq.csv", index=False)
    (args.output_dir / f"{args.split}_best.json").write_text(json.dumps(ranking.iloc[0].to_dict(), indent=2) + "\n"); print(ranking.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("train", "eval")); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val"); parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2); parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(); train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__": main()
