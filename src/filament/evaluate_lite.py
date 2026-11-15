from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import add_path_args, discover_data, paths_from_args
from .gated_selection import FEATURES
from .lite_pipeline import candidate_table, load_ground_truth, score_payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen M2-Lite model on val or annotated test")
    add_path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise RuntimeError("Final test is intentionally locked. Re-run with --confirm-final-test after freezing the method.")
    paths = paths_from_args(args)
    annotation, _, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    images, annotations = load_ground_truth(annotation)
    frame, payloads = candidate_table(manifest, args.split,
                                      paths.artifact_root / "predictions" / args.prediction_run,
                                      images, annotations)
    bundle = joblib.load(args.selector_dir / "quality.joblib")
    config = json.loads((args.selector_dir / "best_decoder.json").read_text())
    frame["quality"] = np.clip(bundle["quality_model"].predict(frame[FEATURES]), 0.0, 1.0)
    metrics = score_payloads(payloads, frame, "quality",
                             score_threshold=float(config["score_threshold"]),
                             min_area=int(config["min_area"]), cluster_iou=float(config["cluster_iou"]),
                             duplicate_iou=float(config["duplicate_iou"]))
    result = {"split": args.split, "prediction_run": args.prediction_run,
              "decoder": {key: config[key] for key in ("score_threshold", "min_area", "cluster_iou", "duplicate_iou")},
              **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
