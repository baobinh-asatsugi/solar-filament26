from pathlib import Path

import pytest

from filament.checkpoint_policy import retain_only_best_checkpoint


def test_retain_only_best_checkpoint_removes_other_weights(tmp_path: Path):
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    (weights / "epoch10.pt").write_bytes(b"periodic")
    (run / "history.json").write_text("{}")

    removed = retain_only_best_checkpoint(run, best)

    assert {path.name for path in removed} == {"last.pt", "epoch10.pt"}
    assert best.read_bytes() == b"best"
    assert (run / "history.json").is_file()


def test_retain_only_best_checkpoint_rejects_external_best(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "best.pt"
    outside.write_bytes(b"best")
    with pytest.raises(ValueError, match="outside run directory"):
        retain_only_best_checkpoint(run, outside)


def test_training_sources_follow_best_only_policy():
    root = Path(__file__).resolve().parents[1]
    direct_savers = [
        root / "src/filament/train_semantic.py",
        root / "src/filament/train_sira.py",
        root / "baselines/unetpp_watershed.py",
    ]
    for path in direct_savers:
        assert 'torch.save(checkpoint, run_dir / "last.pt")' not in path.read_text()
        assert 'args.output_dir / "last.pt"' not in path.read_text()

    yolo = (root / "src/filament/train.py").read_text()
    rtdetr = (root / "baselines/train_rtdetr.py").read_text()
    assert "save_period=-1" in yolo
    assert "save_period=-1" in rtdetr

    mmdet = (root / "baselines/mmdet_adapter.py").read_text()
    launcher = (root / "scripts/run_mmdet_seed.sh").read_text()
    assert 'save_best = "coco/segm_mAP"' in mmdet
    assert "save_optimizer = False" in mmdet
    assert '${WORK}/best_checkpoint' in launcher
    assert '${WORK}/last_checkpoint' not in launcher


def test_gpu_gate_excludes_libreoffice_but_fails_closed():
    root = Path(__file__).resolve().parents[1]
    gate = (root / "scripts/run_with_gpu_limit.sh").read_text()
    assert 'GPU_IGNORED_PROCESS_REGEX="${GPU_IGNORED_PROCESS_REGEX:-soffice\\\\.bin$}"' in gate
    assert "--query-compute-apps=pid,process_name" in gate
    assert 'echo "${MAX_JOBS_PER_GPU}"' in gate
