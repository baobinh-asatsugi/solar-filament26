import json
from pathlib import Path

import pandas as pd
import pytest

from filament.data.fixed_split import build_fixed_annotated_split


def _write_coco(path: Path, physical_images: int = 40) -> None:
    images, annotations, annotation_id = [], [], 0
    for index in range(physical_images):
        file_name = f"2020{index:010d}{'Ch' if index % 2 else 'Mh'}.jpeg"
        for annotator in range(2):
            image_id = f"{annotator:02d}-{index}"
            images.append({"id": image_id, "file_name": file_name, "width": 32, "height": 32})
            for _ in range(1 + index % 4):
                annotations.append({"id": annotation_id, "image_id": image_id, "area": 10 + index,
                                    "segmentation": [[0, 0, 1, 0, 1, 1]], "category_id": 1})
                annotation_id += 1
    path.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")


def test_fixed_split_is_disjoint_deterministic_grouped_and_has_no_folds(tmp_path: Path):
    annotation, output = tmp_path / "annotations.json", tmp_path / "split"
    _write_coco(annotation)
    first = build_fixed_annotated_split(annotation, output, expected_physical_images=40)
    second = build_fixed_annotated_split(annotation, output, expected_physical_images=40)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    manifest = pd.read_csv(output / "manifest.csv")
    assert "fold" not in manifest
    assert set(manifest.split) == {"train", "val", "test"}
    assert manifest.groupby("file_name").split.nunique().max() == 1
    counts = manifest.groupby("split").file_name.nunique().to_dict()
    assert counts == {"test": 8, "train": 26, "val": 6}


def test_fixed_split_refuses_changed_existing_manifest(tmp_path: Path):
    annotation, output = tmp_path / "annotations.json", tmp_path / "split"
    _write_coco(annotation)
    build_fixed_annotated_split(annotation, output, expected_physical_images=40)
    (output / "manifest.csv").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        build_fixed_annotated_split(annotation, output, expected_physical_images=40)
