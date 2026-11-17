from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_prefers_project_local_yolo_weights_with_download_fallback():
    trainer = (ROOT / "src/filament/train.py").read_text()
    assert '"weights" / "yolov8l-seg.pt"' in trainer
    assert "local_pretrained.is_file()" in trainer
    assert "allowing Ultralytics to download" in trainer
    assert "YOLO(model_source)" in trainer


def test_m2_and_yolo_runners_pass_local_weight_path():
    for relative in ("scripts/train_one_seed.sh", "scripts/run_yolov8_seed.sh"):
        text = (ROOT / relative).read_text()
        assert '${PROJECT_ROOT}/weights/yolov8l-seg.pt' in text
        assert 'YOLO_MODEL="yolov8l-seg.pt"' in text
        assert '--model "${YOLO_MODEL}"' in text
        assert 'weights/last.pt' in text
        assert '--resume auto' in text
