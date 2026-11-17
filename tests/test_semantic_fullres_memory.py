from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fullres_validation_streams_probabilities_and_metrics():
    source = (ROOT / "baselines/semantic_fullres.py").read_text()
    assert "def iter_probabilities" in source
    assert "yield name[0], probability" in source
    assert "def tune_stream" in source
    assert "accumulate_metric_record" in source
    assert "predictions = {name:" not in source
    assert "output[name[0]]" not in source


def test_fullres_training_and_eval_both_use_streaming_path():
    source = (ROOT / "baselines/semantic_fullres.py").read_text()
    assert source.count("iter_probabilities(model,") >= 3
    assert source.count("tune_stream(") >= 3
