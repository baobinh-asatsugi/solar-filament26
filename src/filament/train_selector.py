from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import add_path_args, discover_data, paths_from_args
from .gated_selection import FEATURES
from .lite_pipeline import candidate_table, load_ground_truth, quality_model, score_payloads


def _values(text: str, cast):
    return [cast(value) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 17-feature M2-Lite selector on train; tune decoder on val")
    add_path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--seed", type=int, default=3020)
    parser.add_argument("--score-thresholds", default="0.10,0.125,0.15,0.175,0.20,0.225,0.25")
    parser.add_argument("--min-area-values", default="200,300")
    parser.add_argument("--cluster-ious", default="0.30,0.45")
    parser.add_argument("--duplicate-iou", type=float, default=0.65)
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation, _, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    images, annotations = load_ground_truth(annotation)
    root = paths.artifact_root / "predictions" / args.prediction_run
    train, _ = candidate_table(manifest, "train", root, images, annotations)
    valid, payloads = candidate_table(manifest, "val", root, images, annotations)
    model = quality_model(args.seed)
    model.fit(train[FEATURES], train["target"])
    valid["quality"] = np.clip(model.predict(valid[FEATURES]), 0.0, 1.0)
    rows = []
    for threshold in _values(args.score_thresholds, float):
        for area in _values(args.min_area_values, int):
            for cluster in _values(args.cluster_ious, float):
                metrics = score_payloads(payloads, valid, "quality", score_threshold=threshold,
                                         min_area=area, cluster_iou=cluster, duplicate_iou=args.duplicate_iou)
                rows.append({"score_threshold": threshold, "min_area": area, "cluster_iou": cluster,
                             "duplicate_iou": args.duplicate_iou, **metrics})
    ranking = pd.DataFrame(rows).sort_values("global_pq", ascending=False).reset_index(drop=True)
    output = paths.experiment_results(args.run_name)
    ranking.to_csv(output / "val_tuning.csv", index=False)
    best = ranking.iloc[0].to_dict()
    (output / "best_decoder.json").write_text(json.dumps(best, indent=2) + "\n")
    joblib.dump({"quality_model": model, "features": FEATURES}, output / "quality.joblib")
    train.to_csv(output / "train_quality_targets.csv", index=False)
    valid.to_csv(output / "val_quality_predictions.csv", index=False)
    print(ranking.head(10).to_string(index=False))
    print(f"Selector: {output / 'quality.joblib'}")


if __name__ == "__main__":
    main()
