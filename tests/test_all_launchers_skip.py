from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_work_launcher_checks_completion_before_enqueueing():
    launchers = (
        "launch_domain_baselines.sh",
        "launch_five_seeds_tmux.sh",
        "launch_official_baselines.sh",
        "launch_unetpp_watershed_five_seeds.sh",
        "launch_yolov8_five_seeds.sh",
        "launch_yolov8_test_metrics.sh",
    )
    for name in launchers:
        text = (ROOT / "scripts" / name).read_text()
        assert "Skip complete:" in text, name
        assert "PENDING" in text, name


def test_group_launchers_delegate_to_checked_official_launcher():
    for name in (
        "launch_official_baselines_group1.sh",
        "launch_official_baselines_group2.sh",
    ):
        text = (ROOT / "scripts" / name).read_text()
        assert "launch_official_baselines.sh" in text
