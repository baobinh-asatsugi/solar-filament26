from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from filament.config import Paths, discover_data
from filament.data.annotators import select_annotation_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the frozen split as one-class COCO JSON")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    annotation_path, _, _ = discover_data(Paths(args.zip_path.resolve(), args.artifact_root.resolve()))
    with annotation_path.open() as handle:
        coco = json.load(handle)
    manifest = pd.read_csv(args.manifest)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    image_by_id = {str(item["id"]): item for item in coco["images"]}
    anns_by_id: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        anns_by_id[str(item["image_id"])].append(item)
    for split in ("train", "val", "test"):
        selected = manifest[manifest["split"] == split].copy()
        if split == "train":
            selected = select_annotation_records(selected, coco, "medoid", 3020)
        images, annotations, next_ann = [], [], 1
        for next_image, row in enumerate(selected.itertuples(index=False), start=1):
            source = image_by_id[str(row.image_id)]
            images.append({"id": next_image, "file_name": source["file_name"],
                           "width": source["width"], "height": source["height"]})
            for item in anns_by_id[str(row.image_id)]:
                copied = {key: value for key, value in item.items() if key not in {"id", "image_id", "category_id"}}
                copied.update({"id": next_ann, "image_id": next_image, "category_id": 1, "iscrowd": 0})
                annotations.append(copied); next_ann += 1
        payload = {"images": images, "annotations": annotations,
                   "categories": [{"id": 1, "name": "filament", "supercategory": "filament"}]}
        (output / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
        print(split, "records=", len(images), "instances=", len(annotations))


if __name__ == "__main__":
    main()
