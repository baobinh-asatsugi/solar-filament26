from pathlib import Path


def test_official_baseline_groups_are_disjoint_and_named():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/launch_official_baselines.sh").read_text()
    assert "MODELS=(mask_rcnn cascade_mask_rcnn htc solov2)" in launcher
    assert "MODELS=(condinst queryinst mask2former rtmdet_ins)" in launcher
    assert 'SESSION_PREFIX="mmdet_group1"' in launcher
    assert 'SESSION_PREFIX="mmdet_group2"' in launcher
    assert 'BASELINE_CONDA_ENV:-finetune-sd' in launcher
    assert (root / "scripts/launch_official_baselines_group1.sh").is_file()
    assert (root / "scripts/launch_official_baselines_group2.sh").is_file()


def test_mmdet_setup_uses_finetune_torch_without_build_isolation():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "scripts/setup_official_baselines.sh").read_text()
    assert 'import torch' in setup
    assert '"finetune-sd" not in sys.executable' in setup
    assert 'pip install "${UPSTREAM}" --no-deps --no-build-isolation' in setup
    assert 'pip install -e "${UPSTREAM}"' not in setup
    assert 'shapely terminaltables' in setup
