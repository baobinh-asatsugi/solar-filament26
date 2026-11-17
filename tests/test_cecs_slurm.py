from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cecs_job_runs_two_seeds_per_gpu_with_two_gpu_quota():
    job = (ROOT / "jobs/cecs_method.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in job
    assert "#SBATCH --gres=gpu:1" in job
    assert "seeds=(3020 3021)" in job
    assert "seeds=(3022 3023)" in job
    assert "tail_seed=3024" in job
    assert 'run_seed "${seed}" &' in job
    assert 'YOLO_WORKERS="${YOLO_WORKERS:-3}"' in job
    assert "#SBATCH --cpus-per-task=8" in job
    assert "#SBATCH --mem=32G" in job
    assert "#SBATCH --time=04:00:00" in job


def test_cecs_submission_covers_ours_and_fourteen_baselines_sequentially():
    submit = (ROOT / "scripts/submit_cecs_all.sh").read_text()
    methods = (
        "m2_lite", "yolov8l", "unetpp_watershed", "fullres_unet",
        "edgeattnet", "maskrcnn_consensus", "detr_crop_unet", "mask_rcnn",
        "cascade_mask_rcnn", "htc", "solov2", "condinst", "queryinst",
        "mask2former", "rtmdet_ins",
    )
    assert all(method in submit for method in methods)
    assert 'afterany:${previous_job}' in submit
    assert 'METHODS=("$@")' in submit


def test_cecs_defaults_resolve_requested_home_layout_without_identity_literals():
    for relative in ("jobs/cecs_method.sbatch", "scripts/prepare_cecs.sh", "scripts/submit_cecs_all.sh"):
        text = (ROOT / relative).read_text()
        assert "${HOME}/ml26" in text
        assert "${HOME}/datasets/magfilom" in text
        assert "${HOME}/.venv" in text
        assert "/mnt/users/24mai.tp" not in text
        assert "ml26_artifacts" not in text


def test_slurm_bypasses_node_wide_gpu_gate():
    gate = (ROOT / "scripts/run_with_gpu_limit.sh").read_text()
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in gate
    assert "executing directly" in gate


def test_prepare_accepts_gpu_free_login_node_but_rejects_cpu_only_torch():
    prepare = (ROOT / "scripts/prepare_cecs.sh").read_text()
    assert "torch.version.cuda is None" in prepare
    assert "if not torch.cuda.is_available()" not in prepare


def test_cecs_pins_numpy_opencv_after_optional_baseline_install_and_preflights_jobs():
    prepare = (ROOT / "scripts/prepare_cecs.sh").read_text()
    submit = (ROOT / "scripts/submit_cecs_all.sh").read_text()
    job = (ROOT / "jobs/cecs_method.sbatch").read_text()
    requirements = (ROOT / "requirements.txt").read_text()
    checker = (ROOT / "scripts/check_cecs_env.py").read_text()
    assert prepare.rfind("numpy==1.26.4") > prepare.rfind("setup_official_baselines.sh")
    assert "opencv-python-headless==4.11.0.86" in prepare
    assert "opencv-python-headless==4.11.0.86" in requirements
    assert "check_cecs_env.py" in submit
    assert "check_cecs_env.py" in job
    assert 'np.__version__ != "1.26.4"' in checker


def test_cecs_m2_lite_runs_frozen_final_test_and_skips_existing_result():
    dispatcher = (ROOT / "scripts/run_cecs_method_seed.sh").read_text()
    evaluator = (ROOT / "scripts/evaluate_final_test.sh").read_text()
    assert "evaluate_final_test.sh" in dispatcher
    assert "final test result exists; skipping" in evaluator
