"""Export the frozen full-MAGFiLO split as a one-class YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from filament.data.annotators import select_annotation_records


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.data_root / "magfilo_2024_v1.0.json").open(encoding="utf-8") as handle:
        coco = json.load(handle)
    manifest = pd.read_csv(args.manifest)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)
    output = args.output_dir.resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        selected = select_annotation_records(manifest[manifest["split"] == split], coco, "medoid", 3020)
        for row in selected.itertuples(index=False):
            info = images[str(row.image_id)]
            stem = str(row.image_id).replace("/", "_").replace("\\", "_")
            link_or_copy(args.data_root / "images" / info["file_name"], output / "images" / split / f"{stem}.jpeg")
            lines = []
            for item in annotations[str(row.image_id)]:
                x, y, width, height = map(float, item["bbox"])
                cx, cy = (x + width / 2) / info["width"], (y + height / 2) / info["height"]
                lines.append(f"0 {cx:.8f} {cy:.8f} {width/info['width']:.8f} {height/info['height']:.8f}")
            (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(output), "train": "images/train", "val": "images/val", "names": {0: "filament"}
    }, sort_keys=False), encoding="utf-8")
    print(output / "dataset.yaml")


if __name__ == "__main__":
    main()
