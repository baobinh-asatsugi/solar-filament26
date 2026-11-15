from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor

from .gated_selection import FEATURES, build_candidate_features, select_set_aware_instances
from .oof_bank import ProposalBank
from .postprocess import (
    accumulate_metric_record, finalize_metric_totals, metric_totals,
    polygon_to_rle, pq_stats_rle, rle_from_json,
)


def quality_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=220, learning_rate=0.04,
                                         max_leaf_nodes=15, min_samples_leaf=30,
                                         l2_regularization=1.0, random_state=seed)


def unique_iou_targets(predictions: list[dict], targets: list[dict]) -> np.ndarray:
    output = np.zeros(len(predictions), np.float32)
    if not predictions or not targets:
        return output
    iou = mask_utils.iou(predictions, targets, [0] * len(targets))
    rows, columns = linear_sum_assignment(-iou)
    output[rows] = iou[rows, columns]
    return output


def load_ground_truth(annotation_path: Path):
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)
    return images, annotations


def gt_sets(records: pd.DataFrame, images: dict, annotations: dict) -> list[list[dict]]:
    output = []
    for record in records.itertuples(index=False):
        image_id = str(record.image_id)
        info = images[image_id]
        output.append([polygon_to_rle(item["segmentation"], info["height"], info["width"])
                       for item in annotations[image_id]])
    return output


def candidate_table(manifest: pd.DataFrame, split: str, prediction_root: Path,
                    images: dict, annotations: dict) -> tuple[pd.DataFrame, dict]:
    rows, payloads = [], {}
    raw_bank = ProposalBank(prediction_root / "raw" / split)
    refined_bank = ProposalBank(prediction_root / "refined" / split)
    selected = manifest[manifest["split"] == split]
    for file_name, records in selected.groupby("file_name", sort=True):
        raw = raw_bank.load(file_name, decode_masks=False)["instances"]
        refined = refined_bank.load(file_name, decode_masks=False)["instances"]
        features = build_candidate_features(raw, refined)
        rles = [rle_from_json(item["rle"]) for item in refined]
        ground_truth = gt_sets(records, images, annotations)
        target = np.mean([unique_iou_targets(rles, gt) for gt in ground_truth], axis=0) if ground_truth else np.zeros(len(rles))
        for index, feature in features.iterrows():
            rows.append({"file_name": file_name, "candidate_index": int(index),
                         **{name: float(feature[name]) for name in FEATURES}, "target": float(target[index])})
        payloads[file_name] = (records, refined, ground_truth)
    return pd.DataFrame(rows), payloads


def score_payloads(payloads: dict, candidates: pd.DataFrame, score_column: str, *, score_threshold: float,
                   min_area: int, cluster_iou: float, duplicate_iou: float) -> dict:
    grouped = {name: part.sort_values("candidate_index") for name, part in candidates.groupby("file_name")}
    stats = {**metric_totals(), "selected": 0}
    for file_name, (_, instances, ground_truth) in payloads.items():
        scores = grouped[file_name][score_column].to_numpy(np.float32)
        rles = [rle_from_json(item["rle"]) for item in instances]
        predictions = [rle for rle, _ in select_set_aware_instances(
            rles, scores, score_threshold, min_area, cluster_iou, duplicate_iou)]
        stats["selected"] += len(predictions)
        for gt in ground_truth:
            values = pq_stats_rle(gt, predictions)
            accumulate_metric_record(stats, values)
    return {**finalize_metric_totals(stats), "selected_instances": int(stats["selected"])}
