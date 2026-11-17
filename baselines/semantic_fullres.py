"""Five-seed full-resolution semantic baselines with instance decoding.

The ``fullres_unet`` model ports the public ResNet34 U-Net baseline. The
``edgeattnet`` model is a paper-faithful reconstruction because the public
EdgeAttNet repository does not contain the advertised model class.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baselines.cache_io import atomic_write_bytes
from baselines.common import (
    INPUT_SIZE,
    annotation_maps,
    connected_instances,
    load_problem,
    semantic_union,
    write_result,
)
from filament.postprocess import (
    accumulate_metric_record,
    finalize_metric_totals,
    mask_to_rle_object,
    metric_totals,
    polygon_to_rle,
    pq_stats_rle,
)


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        value = nn.functional.interpolate(value, scale_factor=2, mode="nearest")
        if skip is not None:
            value = torch.cat([value, skip], dim=1)
        return self.conv(value)


class EdgeGuidedWindowAttention(nn.Module):
    """Resolution-safe EG-MHSA reconstruction (windowed only for 2048 inputs)."""

    def __init__(self, channels: int = 512, heads: int = 4, window: int = 8):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels, self.heads, self.window = channels, heads, window
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.1)
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, feature: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        window = self.window
        pad_h, pad_w = (-height) % window, (-width) % window
        feature = nn.functional.pad(feature, (0, pad_w, 0, pad_h))
        edge = nn.functional.pad(edge, (0, pad_w, 0, pad_h))
        hp, wp = feature.shape[-2:]
        tokens = feature.permute(0, 2, 3, 1).reshape(batch, hp // window, window, wp // window, window, channels)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)
        edge_tokens = edge.permute(0, 2, 3, 1).reshape(batch, hp // window, window, wp // window, window, channels)
        edge_tokens = edge_tokens.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)
        normalized = self.norm(tokens)
        qkv = self.qkv(normalized).reshape(tokens.shape[0], tokens.shape[1], 3, self.heads, channels // self.heads)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
        edge_heads = edge_tokens.reshape(edge_tokens.shape[0], edge_tokens.shape[1], self.heads, channels // self.heads).permute(0, 2, 1, 3)
        query, key = query + edge_heads, key + edge_heads
        logits = query @ key.transpose(-2, -1) / math.sqrt(channels // self.heads)
        attended = (logits.softmax(dim=-1) @ value).permute(0, 2, 1, 3).reshape(tokens.shape)
        tokens = self.output_norm(tokens + self.dropout(self.projection(attended)))
        tokens = tokens.reshape(batch, hp // window, wp // window, window, window, channels)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(batch, hp, wp, channels)
        return tokens[:, :height, :width].permute(0, 3, 1, 2)


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = torchvision.models.resnet34(weights=weights)
        grayscale_weight = encoder.conv1.weight.detach().sum(dim=1, keepdim=True)
        encoder.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            encoder.conv1.weight.copy_(grayscale_weight)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layers = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
        self.u4, self.u3 = Up(512, 256, 256), Up(256, 128, 128)
        self.u2, self.u1, self.u0 = Up(128, 64, 64), Up(64, 64, 32), Up(32, 0, 16)
        self.mask_head = nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(image)
        one = self.layers[0](self.pool(stem))
        two = self.layers[1](one)
        three = self.layers[2](two)
        four = self.layers[3](three)
        value = self.u4(four, three)
        value = self.u3(value, two)
        value = self.u2(value, one)
        value = self.u1(value, stem)
        value = self.u0(value, None)
        return {"mask": self.mask_head(value)}


class DoubleConv(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
        )
    def forward(self, value): return self.layers(value)


class DecoderBlock(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int):
        super().__init__(); self.up = nn.ConvTranspose2d(inputs, outputs, 2, 2)
        self.conv = DoubleConv(outputs + skip, outputs)
    def forward(self, value, skip): return self.conv(torch.cat([self.up(value), skip], dim=1))


class PaperEdgeAttNet(nn.Module):
    """Paper architecture: learned edge prior + two 4-head EG-MHSA blocks."""
    def __init__(self):
        super().__init__(); self.pool = nn.MaxPool2d(2)
        self.e1, self.e2 = DoubleConv(1, 64), DoubleConv(64, 128)
        self.e3, self.e4 = DoubleConv(128, 256), DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 512)
        self.edge_map = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.Sigmoid())
        self.edge_projection = nn.Conv2d(1, 512, 1)
        self.attention = nn.ModuleList([EdgeGuidedWindowAttention(512, 4) for _ in range(2)])
        self.d4, self.d3 = DecoderBlock(512, 512, 512), DecoderBlock(512, 256, 256)
        self.d2, self.d1 = DecoderBlock(256, 128, 128), DecoderBlock(128, 64, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, image):
        one = self.e1(image); two = self.e2(self.pool(one)); three = self.e3(self.pool(two)); four = self.e4(self.pool(three))
        value = self.bottleneck(self.pool(four))
        edge = self.edge_projection(nn.functional.interpolate(
            self.edge_map(image), size=value.shape[-2:], mode="bilinear", align_corners=False
        ))
        for attention in self.attention: value = attention(value, edge)
        value = self.d4(value, four); value = self.d3(value, three)
        value = self.d2(value, two); value = self.d1(value, one)
        return {"mask": self.head(value)}


class FullImageDataset(Dataset):
    def __init__(self, data_root: Path, manifest: pd.DataFrame, coco: dict, split: str, cache_root: Path,
                 training: bool, crop_size: int = 1024, augment: bool = True):
        self.image_root = data_root / "images"
        self.names = sorted(manifest.loc[manifest["split"] == split, "file_name"].unique())
        self.images, self.annotations, self.records = annotation_maps(coco)
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.training = training
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def target(self, name: str) -> np.ndarray:
        path = self.cache_root / f"{Path(name).stem}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if mask is None:
            # A missing or legacy truncated cache entry is rebuilt. Encoding to
            # memory plus atomic replace prevents concurrent seeds/workers from
            # observing a partially written PNG.
            mask = semantic_union(name, self.images, self.annotations, self.records)
            success, encoded = cv2.imencode(".png", mask * 255)
            if not success:
                raise OSError(f"Cannot encode {path}")
            atomic_write_bytes(path, encoded.tobytes())
            return mask.astype(np.uint8)
        return (mask > 0).astype(np.uint8)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = cv2.imread(str(self.image_root / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.image_root / name)
        target = self.target(name)
        if self.training:
            if self.crop_size < image.shape[0]:
                y = random.randint(0, image.shape[0] - self.crop_size)
                x = random.randint(0, image.shape[1] - self.crop_size)
                image = image[y:y + self.crop_size, x:x + self.crop_size]
                target = target[y:y + self.crop_size, x:x + self.crop_size]
            if self.augment and random.random() < 0.5:
                image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
            if self.augment and random.random() < 0.5:
                image, target = image[::-1].copy(), target[::-1].copy()
            if self.augment:
                rotation = random.randrange(4)
                image, target = np.rot90(image, rotation).copy(), np.rot90(target, rotation).copy()
        image_tensor = (torch.from_numpy(image).float()[None] / 255.0 - 0.5) / 0.25
        if self.training and self.augment:
            image_tensor = image_tensor * random.uniform(0.85, 1.15) + random.uniform(-0.15, 0.15)
        return image_tensor, torch.from_numpy(target.astype(np.float32))[None], name


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    numerator = 2 * (probability * target).sum(dim=(1, 2, 3)) + 1
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1
    return 1 - (numerator / denominator).mean()


def edge_target(mask: torch.Tensor) -> torch.Tensor:
    dilated = nn.functional.max_pool2d(mask, 3, 1, 1)
    eroded = -nn.functional.max_pool2d(-mask, 3, 1, 1)
    return (dilated - eroded).clamp(0, 1)


@torch.no_grad()
def iter_probabilities(model, dataset: FullImageDataset, device: torch.device):
    model.eval()
    for image, _, name in tqdm(DataLoader(dataset, batch_size=1, shuffle=False), desc="full-res inference"):
        image = image.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            if isinstance(model, PaperEdgeAttNet):
                tile, stride = 512, 384
                positions = list(range(0, INPUT_SIZE - tile + 1, stride))
                if positions[-1] != INPUT_SIZE - tile: positions.append(INPUT_SIZE - tile)
                total = torch.zeros((INPUT_SIZE, INPUT_SIZE), device=device)
                weight = torch.zeros_like(total)
                for y in positions:
                    for x in positions:
                        patch = model(image[..., y:y + tile, x:x + tile])["mask"].sigmoid()[0, 0]
                        total[y:y + tile, x:x + tile] += patch; weight[y:y + tile, x:x + tile] += 1
                probability = total / weight.clamp_min(1)
            else:
                probability = model(image)["mask"].sigmoid()[0, 0]
        yield name[0], probability.cpu().numpy().astype(np.float16)


def tune_stream(coco: dict, records: pd.DataFrame, probabilities, frozen: dict | None = None):
    configs = [(threshold, area) for threshold in (0.35, 0.45, 0.50, 0.55, 0.65) for area in (100, 200, 300)]
    if frozen:
        configs = [(float(frozen["mask_threshold"]), int(frozen["min_area"]))]
    images, annotations, _ = annotation_maps(coco)
    records_by_name = {
        name: list(group.itertuples(index=False)) for name, group in records.groupby("file_name")
    }
    totals = [metric_totals() for _ in configs]
    for name, probability in probabilities:
        record_rows = records_by_name.get(name, [])
        ground_truth_sets = []
        for row in record_rows:
            info = images[str(row.image_id)]
            ground_truth_sets.append([
                polygon_to_rle(item["segmentation"], int(info["height"]), int(info["width"]))
                for item in annotations[str(row.image_id)]
            ])
        for config_index, (threshold, area) in enumerate(configs):
            masks = connected_instances(probability, threshold, area)
            pred_rles = [mask_to_rle_object(mask) for mask in masks]
            del masks
            for ground_truth in ground_truth_sets:
                accumulate_metric_record(totals[config_index], pq_stats_rle(ground_truth, pred_rles))
            del pred_rles
        del ground_truth_sets
        del probability
    rows = [
        {"mask_threshold": threshold, "min_area": area, **finalize_metric_totals(total)}
        for (threshold, area), total in zip(configs, totals)
    ]
    return pd.DataFrame(rows).sort_values("global_pq", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--model", choices=("fullres_unet", "edgeattnet"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-config", type=Path)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    coco, manifest, _ = load_problem(args.data_root, args.manifest)
    run_name = f"baseline_magfilo_v1_{args.model}_s{args.seed}"
    run_dir = args.artifact_root / "baseline_work" / "magfilo_v1" / args.model / f"seed{args.seed}"
    result_dir = args.artifact_root / "results" / run_name
    cache_root = args.artifact_root / "baseline_cache" / "magfilo_v1" / "semantic_union"
    run_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    last_checkpoint = run_dir / "last.pt"
    resume_checkpoint = None
    if args.command == "train" and args.resume:
        if last_checkpoint.is_file():
            resume_checkpoint = last_checkpoint
        elif checkpoint.is_file():
            resume_checkpoint = checkpoint
    model = (PaperEdgeAttNet() if args.model == "edgeattnet" else ResNetUNet(
        pretrained=args.command == "train" and resume_checkpoint is None
    )).to(device)

    if args.command == "train":
        training = FullImageDataset(args.data_root, manifest, coco, "train", cache_root, True,
                                    args.crop_size, augment=args.model != "edgeattnet")
        validation = FullImageDataset(args.data_root, manifest, coco, "val", cache_root, False)
        loader = DataLoader(training, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
        edge_method = args.model == "edgeattnet"
        optimizer = (torch.optim.Adam(model.parameters(), lr=1e-4) if edge_method else
                     torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4))
        steps_per_epoch = max(len(loader), 1)
        total_steps = steps_per_epoch * args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / steps_per_epoch)
            * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))),
        ) if not edge_method else None
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        bce = nn.BCEWithLogitsLoss()
        best = -1.0; start_epoch = 1
        if resume_checkpoint is not None:
            payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            start_epoch = int(payload.get("epoch", 0)) + 1
            stored_best = payload.get("best_pq", payload.get("val_pq"))
            if stored_best is None:
                recovered = tune_stream(
                    coco,
                    manifest[manifest["split"] == "val"],
                    iter_probabilities(model, validation, device),
                    frozen={"mask_threshold": 0.5, "min_area": 300},
                )
                best = float(recovered.iloc[0]["global_pq"])
                print(f"Recovered legacy best checkpoint PQ={best:.6f}")
            else:
                best = float(stored_best)
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None and payload.get("scheduler") is not None:
                scheduler.load_state_dict(payload["scheduler"])
            if "scaler" in payload:
                scaler.load_state_dict(payload["scaler"])
            print(
                f"Resuming {args.model} from {resume_checkpoint} at "
                f"epoch {start_epoch}/{args.epochs}; best_pq={best:.6f}"
            )
        for epoch in range(start_epoch, args.epochs + 1):
            model.train(); total = 0.0
            for image, target, _ in tqdm(loader, desc=f"{args.model} {epoch}/{args.epochs}"):
                image, target = image.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model(image)
                    loss = bce(outputs["mask"], target) + dice_loss(outputs["mask"], target)
                    if "edge" in outputs:
                        loss = loss + 0.2 * bce(outputs["edge"], edge_target(target))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                if scheduler is not None: scheduler.step()
                total += float(loss.item())
            validation_ranking = tune_stream(
                coco,
                manifest[manifest["split"] == "val"],
                iter_probabilities(model, validation, device),
                frozen={"mask_threshold": 0.5, "min_area": 300},
            )
            metrics = validation_ranking.iloc[0].to_dict()
            print(f"epoch={epoch} loss={total/max(len(loader),1):.5f} val_pq={metrics['global_pq']:.6f}")
            if metrics["global_pq"] > best:
                best = metrics["global_pq"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed,
                            "input_size": INPUT_SIZE, "crop_size": args.crop_size,
                            "model_name": args.model, "val_pq": best}, checkpoint)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(), "epoch": epoch, "best_pq": best,
                "seed": args.seed, "input_size": INPUT_SIZE,
                "crop_size": args.crop_size, "model_name": args.model,
            }, last_checkpoint)
        last_checkpoint.unlink(missing_ok=True)
        return

    payload = torch.load(checkpoint, map_location=device)
    if int(payload.get("input_size", 0)) != INPUT_SIZE:
        raise RuntimeError("Checkpoint was not trained with the shared 2048-pixel input protocol")
    model.load_state_dict(payload["model"])
    dataset = FullImageDataset(args.data_root, manifest, coco, args.split, cache_root, False)
    frozen = json.loads(args.frozen_config.read_text()) if args.frozen_config else None
    ranking = tune_stream(
        coco,
        manifest[manifest["split"] == args.split],
        iter_probabilities(model, dataset, device),
        frozen,
    )
    ranking.to_csv(result_dir / f"{args.split}_pq.csv", index=False)
    best = ranking.iloc[0].to_dict()
    best.update({"model": args.model, "seed": args.seed, "input_size": INPUT_SIZE})
    write_result(result_dir / f"{args.split}_best.json", best)
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
