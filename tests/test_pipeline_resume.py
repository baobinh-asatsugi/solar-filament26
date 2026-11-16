from pathlib import Path

from filament.pipeline_state import complete


ROOT = Path(__file__).resolve().parents[1]


def test_cache_completion_requires_every_expected_artifact(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file_name,split\na.jpeg,train\nb.jpeg,train\nc.jpeg,val\n")
    root = tmp_path / "predictions"
    for folder in (root / "raw/train", root / "semantic/train"):
        folder.mkdir(parents=True)
    for stem in ("a", "b"):
        (root / f"raw/train/{stem}.json.gz").write_bytes(b"x")
        (root / f"semantic/train/{stem}.npz").write_bytes(b"x")
    assert complete(root, manifest, "train", "prediction")[0]
    (root / "semantic/train/b.npz").write_bytes(b"")
    assert not complete(root, manifest, "train", "prediction")[0]


def test_one_seed_runner_skips_completed_stages_and_resumes_sira():
    text = (ROOT / "scripts/train_one_seed.sh").read_text()
    assert "detector checkpoint exists; skipping YOLO training" in text
    assert "semantic checkpoint exists; skipping semantic training" in text
    assert "filament.pipeline_state" in text
    assert '"${SIRA_COMPLETE}"' in text
    assert "--resume" in text
    assert 'SIRA_BATCH="${SIRA_BATCH:-2}"' in text
