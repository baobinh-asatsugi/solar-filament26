"""Shared full-MAGFiLO data and pooled-PQ utilities for comparison baselines."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from filament.postprocess import (
    accumulate_metric_record, finalize_metric_totals, mask_to_rle_object,
    metric_totals, polygon_to_rle, pq_stats_rle,
)


INPUT_SIZE = 2048


def load_problem(data_root: Path, manifest_path: Path):
    annotation_path = data_root / "magfilo_2024_v1.0.json"
    image_root = data_root / "images"
    with annotation_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    manifest = pd.read_csv(manifest_path)
    if manifest.groupby("file_name")["split"].nunique().max() != 1:
        raise RuntimeError("Physical-image leakage in split manifest")
    if len({item["file_name"] for item in coco["images"]}) != 958:
        raise RuntimeError("Expected full MAGFiLO v1.0 with 958 physical images")
    dimensions = {(int(item["width"]), int(item["height"])) for item in coco["images"]}
    if dimensions != {(INPUT_SIZE, INPUT_SIZE)}:
        raise RuntimeError(f"Expected every source annotation at 2048x2048, found {sorted(dimensions)}")
    missing = sorted({item["file_name"] for item in coco["images"]} - {path.name for path in image_root.glob("*.jpeg")})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} MAGFiLO JPEGs; examples: {missing[:5]}")
    return coco, manifest, image_root


def annotation_maps(coco: dict):
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    records: dict[str, list[str]] = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)
    for image_id, item in images.items():
        records[item["file_name"]].append(image_id)
    return images, annotations, records


def rasterize(annotation: dict, height: int = INPUT_SIZE, width: int = INPUT_SIZE) -> np.ndarray:
    return mask_utils.decode(polygon_to_rle(annotation["segmentation"], height, width)).astype(np.uint8)


def semantic_union(file_name: str, images: dict, annotations: dict, records: dict) -> np.ndarray:
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    for image_id in records[file_name]:
        for annotation in annotations[image_id]:
            mask |= rasterize(annotation)
    return mask


def consensus_instances(
    file_name: str,
    images: dict,
    annotations: dict,
    records: dict,
    min_votes: int = 2,
    cluster_iou: float = 0.30,
) -> list[np.ndarray]:
    record_ids = records[file_name]
    if len(record_ids) == 1:
        return [rasterize(item) for item in annotations[record_ids[0]]]
    clusters: list[dict] = []
    for annotator_index, image_id in enumerate(record_ids):
        for annotation in annotations[image_id]:
            mask = rasterize(annotation)
            encoded = mask_to_rle_object(mask)
            best, best_iou = None, cluster_iou
            for cluster in clusters:
                if annotator_index in cluster["annotators"]:
                    continue
                iou = float(mask_utils.iou([encoded], [cluster["representative"]], [0])[0, 0])
                if iou > best_iou:
                    best, best_iou = cluster, iou
            if best is None:
                clusters.append({"annotators": {annotator_index}, "representative": encoded, "masks": [mask]})
            else:
                best["annotators"].add(annotator_index)
                best["masks"].append(mask)
    result = []
    required = min(min_votes, len(record_ids))
    for cluster in clusters:
        if len(cluster["annotators"]) < required:
            continue
        votes = np.sum(cluster["masks"], axis=0)
        mask = (votes * 2 >= len(cluster["masks"])).astype(np.uint8)
        if mask.any():
            result.append(mask)
    return result


def connected_instances(probability: np.ndarray, threshold: float, min_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= threshold).astype(np.uint8), connectivity=8
    )
    return [labels == index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_area]


def pooled_pq(coco: dict, records_frame: pd.DataFrame, predictions: dict[str, list[np.ndarray]]) -> dict:
    images, annotations, _ = annotation_maps(coco)
    stats = metric_totals()
    encoded_predictions = {
        name: [mask_to_rle_object(mask) for mask in masks] for name, masks in predictions.items()
    }
    for row in records_frame.itertuples(index=False):
        info = images[str(row.image_id)]
        ground_truth = [
            polygon_to_rle(item["segmentation"], int(info["height"]), int(info["width"]))
            for item in annotations[str(row.image_id)]
        ]
        values = pq_stats_rle(ground_truth, encoded_predictions.get(row.file_name, []))
        accumulate_metric_record(stats, values)
    return finalize_metric_totals(stats)


def resolve_scored_masks(
    scored_masks: list[tuple[float, np.ndarray]], score_threshold: float, min_area: int
) -> list[np.ndarray]:
    occupied = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    output = []
    for score, source in sorted(scored_masks, key=lambda pair: -pair[0]):
        if score < score_threshold:
            continue
        mask = source.astype(bool) & ~occupied
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count > 2:
            index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = labels == index
        if int(mask.sum()) < min_area:
            continue
        output.append(mask)
        occupied |= mask
    return output


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
