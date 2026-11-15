from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .data.annotators import select_annotation_records
from .experiment import record_event, sha256_file


def materialize(src, dst) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def safe_stem(image_id: str) -> str:
    return image_id.replace("/", "_").replace("\\", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert full MAGFiLO v1.0 to YOLO segmentation format")
    add_path_args(parser)
    parser.add_argument("--seed", type=int, default=3020)
    parser.add_argument(
        "--annotation-strategy",
        choices=("duplicate", "medoid", "max-completeness", "first", "random-single"),
        default="duplicate",
    )
    parser.add_argument("--use-categories", action="store_true", help="Train four chirality classes like the public 0.69 notebook")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--input-manifest", type=str, required=True, help="Frozen train/val/test manifest")
    parser.add_argument("--val-annotation-strategy", choices=("all", "same"), default="all")
    parser.add_argument(
        "--export-manifest",
        type=Path,
        default=None,
        help="Write the full pre-selection split manifest for reuse by another annotation strategy.",
    )
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation_path, train_image_dir, _ = discover_data(paths)

    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = pd.DataFrame(coco["images"])
    anns_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    category_to_index = {item["id"]: index for index, item in enumerate(categories)}

    manifest_path = Path(args.input_manifest).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"image_id", "file_name", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Input manifest lacks {required - set(manifest.columns)}: {manifest_path}")
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise ValueError("Fixed manifest must contain exactly train, val, and test")
    if args.export_manifest is not None:
        export_path = args.export_manifest.expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(export_path, index=False)
    if manifest.groupby("file_name")["split"].nunique().max() != 1:
        raise RuntimeError(f"Physical-image leakage in {manifest_path}")
    training_records = select_annotation_records(
        manifest[manifest["split"] == "train"], coco, args.annotation_strategy, args.seed
    )
    validation_records = manifest[manifest["split"] == "val"].copy()
    if args.val_annotation_strategy == "same":
        validation_records = select_annotation_records(validation_records, coco, args.annotation_strategy, args.seed)
    manifest = pd.concat([training_records, validation_records], ignore_index=True)
    overlap = set(manifest.loc[manifest["split"] == "train", "file_name"]) & set(
        manifest.loc[manifest["split"] == "val", "file_name"]
    )
    if overlap:
        raise RuntimeError(f"Physical-image leakage detected: {len(overlap)} files")

    dataset_name = args.dataset_name or f"fixed_{args.annotation_strategy}_s{args.seed}"
    dataset_root = paths.dataset_root(dataset_name)

    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    image_lookup = {str(row.id): row for row in images.itertuples(index=False)}
    for selected in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="COCO -> YOLO"):
        row = image_lookup[str(selected.image_id)]
        stem = safe_stem(str(row.id))
        source_image = train_image_dir / row.file_name
        target_image = dataset_root / "images" / selected.split / f"{stem}.jpeg"
        target_label = dataset_root / "labels" / selected.split / f"{stem}.txt"
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        materialize(source_image, target_image)

        lines = []
        for annotation in anns_by_image[row.id]:
            for polygon in annotation["segmentation"]:
                points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
                if len(points) < 3:
                    continue
                points[:, 0] = np.clip(points[:, 0] / row.width, 0, 1)
                points[:, 1] = np.clip(points[:, 1] / row.height, 0, 1)
                coords = " ".join(f"{x:.7f} {y:.7f}" for x, y in points)
                class_id = category_to_index[annotation["category_id"]] if args.use_categories else 0
                lines.append(f"{class_id} {coords}")
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest_rows.append(
            {
                "sample_stem": stem,
                "image_id": row.id,
                "file_name": row.file_name,
                "split": selected.split,
                "annotation_strategy": args.annotation_strategy,
            }
        )

    selected_manifest = pd.DataFrame(manifest_rows)
    selected_manifest_path = dataset_root / "manifest.csv"
    selected_manifest.to_csv(selected_manifest_path, index=False)
    names = {index: item["name"] for index, item in enumerate(categories)} if args.use_categories else {0: "filament"}
    dataset = {"path": str(dataset_root), "train": "images/train", "val": "images/val", "names": names}
    data_yaml = dataset_root / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False))
    record_event(
        paths.artifact_root,
        dataset_name,
        "prepare",
        {
            "dataset_name": dataset_name,
            "annotation_strategy": args.annotation_strategy,
            "val_annotation_strategy": args.val_annotation_strategy,
            "use_categories": args.use_categories,
            "seed": args.seed,
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": sha256_file(manifest_path),
            "selected_manifest": str(selected_manifest_path),
            "selected_manifest_sha256": sha256_file(selected_manifest_path),
        },
    )
    print(selected_manifest.groupby("split").agg(records=("image_id", "size"), physical_images=("file_name", "nunique")))
    print(f"Source images: {len(list(train_image_dir.glob('*.jpeg')))}")
    print(f"Dataset YAML: {data_yaml}")


if __name__ == "__main__":
    main()

