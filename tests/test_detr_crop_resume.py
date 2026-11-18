from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_detr_crop_casts_amp_probability_before_opencv_resize():
    source = (ROOT / "baselines/detr_crop_unet.py").read_text()
    assert ".float().cpu().numpy()" in source
    assert "np.ascontiguousarray(probability, dtype=np.float32)" in source
    assert "Non-finite crop probability" in source


def test_detr_crop_skips_completed_training_and_evaluation_stages():
    runner = (ROOT / "scripts/run_detr_crop_unet_seed.sh").read_text()
    trainer = (ROOT / "baselines/train_rtdetr.py").read_text()
    assert "detector checkpoint exists; skipping RT-DETR training" in runner
    assert "crop U-Net checkpoint exists; skipping refiner training" in runner
    assert "validation result exists; skipping val" in runner
    assert "test result exists; skipping test" in runner
    assert "Existing RT-DETR checkpoint; skipping training" in trainer


def test_detr_crop_ignores_nonfinite_training_losses():
    source = (ROOT / "baselines/detr_crop_unet.py").read_text()
    assert "if not torch.isfinite(loss)" in source
    assert "skipped_nonfinite" in source
    assert "No finite validation losses" in source
