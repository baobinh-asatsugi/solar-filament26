import json
import sys
from pathlib import Path

import pandas as pd

from baselines.summarize_all import main


def test_partial_summary_reports_seed_count_and_skips_missing_runs(tmp_path: Path, monkeypatch):
    result = tmp_path / "results/magfilo_v1_yolov8l_s3020/test_best.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"global_pq": 0.4, "sq": 0.7, "rq": 0.57}))
    checkpoint = tmp_path / "runs/magfilo_v1_yolov8l_s3020/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(sys, "argv", ["summarize_all", "--artifact-root", str(tmp_path)])
    main()

    summary = pd.read_csv(tmp_path / "results/five_seed_summary.csv")
    assert summary.method.tolist() == ["yolov8l"]
    yolo = summary[summary.method == "yolov8l"].iloc[0]
    assert yolo.n_seeds == 1
    assert yolo.seeds == "s3020"
    assert not bool(yolo.complete_5_seeds)
    assert yolo.global_pq_mean == 0.4

    skipped = pd.read_csv(tmp_path / "results/five_seed_skipped.csv")
    assert len(skipped) == 74


def test_summary_with_no_completed_runs_writes_empty_readable_summary(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["summarize_all", "--artifact-root", str(tmp_path)])
    main()

    summary = pd.read_csv(tmp_path / "results/five_seed_summary.csv")
    skipped = pd.read_csv(tmp_path / "results/five_seed_skipped.csv")
    assert summary.empty
    assert summary.columns.tolist() == ["method", "n_seeds", "seeds", "complete_5_seeds"]
    assert len(skipped) == 75
    missing_yolo = skipped[(skipped.method == "yolov8l") & (skipped.seed == 3021)].iloc[0]
    assert missing_yolo.reason == "missing_result"
