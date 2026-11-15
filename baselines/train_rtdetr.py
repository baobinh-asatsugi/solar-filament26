"""Train the detector stage of the DETR -> crop U-Net baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR

from baselines.common import INPUT_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run_name = f"magfilo_v1_detr_crop_unet_detector_s{args.seed}"
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=str(args.data.resolve()), project=str((args.artifact_root / "runs").resolve()), name=run_name,
        imgsz=INPUT_SIZE, epochs=args.epochs, batch=1, workers=6, device=args.device,
        seed=args.seed, deterministic=True, cache=False, amp=True, patience=10,
        mosaic=0.0, mixup=0.0, translate=0.0, scale=0.1, fliplr=0.5, flipud=0.5,
    )
    print(args.artifact_root / "runs" / run_name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
