import hashlib
from pathlib import Path

import pandas as pd


EXPECTED_MANIFEST_SHA256 = "224a710c143af577caef185d423c42dc3d92783e6c5a71a78ab13a72b52971cb"


def test_committed_full_magfilo_split_is_frozen_grouped_and_complete():
    path = Path("splits/magfilo_v1/manifest.csv")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    manifest = pd.read_csv(path)
    assert len(manifest) == 1593
    assert manifest["file_name"].nunique() == 958
    assert int(manifest["record_instance_count"].sum()) == 10244
    assert manifest.groupby("file_name")["split"].nunique().max() == 1
    assert manifest.groupby("split")["file_name"].nunique().to_dict() == {
        "test": 192, "train": 622, "val": 144,
    }


def test_every_baseline_uses_the_shared_2048_source_protocol():
    common = Path("baselines/common.py").read_text(encoding="utf-8")
    adapter = Path("baselines/mmdet_adapter.py").read_text(encoding="utf-8")
    yolo = Path("scripts/run_yolov8_seed.sh").read_text(encoding="utf-8")
    detector = Path("baselines/train_rtdetr.py").read_text(encoding="utf-8")
    assert "INPUT_SIZE = 2048" in common
    assert 'default=2048' in adapter
    assert "--imgsz 2048" in yolo
    assert "imgsz=INPUT_SIZE" in detector
