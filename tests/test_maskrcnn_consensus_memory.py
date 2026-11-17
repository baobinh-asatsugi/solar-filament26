from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_maskrcnn_grid_streams_one_image_without_dense_prediction_dictionary():
    source = (ROOT / "baselines/maskrcnn_consensus.py").read_text()
    assert "def resolve_aligned_rles" in source
    assert "prediction_rles = resolve_aligned_rles" in source
    assert "accumulate_metric_record" in source
    assert "predictions = {" not in source
    assert "mask_utils.decode(encoded)" in source


def test_maskrcnn_training_uses_fixed_decoder_and_resumable_last_checkpoint():
    source = (ROOT / "baselines/maskrcnn_consensus.py").read_text()
    runner = (ROOT / "scripts/run_maskrcnn_consensus_seed.sh").read_text()
    assert 'frozen={"score_threshold": 0.5' in source
    assert 'last_checkpoint = run_dir / "last.pt"' in source
    assert 'optimizer.load_state_dict(payload["optimizer"])' in source
    assert "last_checkpoint.unlink(missing_ok=True)" in source
    assert "--resume" in runner
