from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_domain_launcher_skips_completed_test_results():
    text = (ROOT / "scripts/launch_domain_baselines.sh").read_text()
    assert "baseline_magfilo_v1_${METHOD}_s${SEED}/test_best.json" in text
    assert '[[ -s "${TEST_RESULT}" ]]' in text
    assert "No pending domain baselines" in text


def test_domain_runners_are_independently_idempotent():
    scripts = (
        "run_semantic_baseline_seed.sh",
        "run_maskrcnn_consensus_seed.sh",
        "run_detr_crop_unet_seed.sh",
    )
    for name in scripts:
        text = (ROOT / "scripts" / name).read_text()
        assert '[[ -s "${RESULT}/test_best.json" ]]' in text
        assert "skipping completed run" in text
