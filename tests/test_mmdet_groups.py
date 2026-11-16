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


def test_mmdet_keeps_seed_without_forcing_unsupported_deterministic_kernels():
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "baselines/mmdet_adapter.py").read_text()
    assert "dict(seed=args.seed, deterministic=False)" in adapter
    assert "dict(seed=args.seed, deterministic=True)" not in adapter


def test_htc_builds_and_loads_semantic_targets():
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "baselines/mmdet_adapter.py").read_text()
    assert "def _prepare_semantic_targets" in adapter
    assert 'target[mask] = 0' in adapter
    assert 'transform["with_seg"] = True' in adapter
    assert 'data_prefix["seg"]' in adapter


def test_mmdet_runs_skip_completed_test_results():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/launch_official_baselines.sh").read_text()
    runner = (root / "scripts/run_mmdet_seed.sh").read_text()
    assert '[[ -s "${TEST_RESULT}" ]]' in launcher
    assert "No pending MMDetection baselines" in launcher
    assert '[[ -s "${OUT}/test_best.json" ]]' in runner
