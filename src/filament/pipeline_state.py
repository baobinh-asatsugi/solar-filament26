from __future__ import annotations

import argparse
import csv
from pathlib import Path


def expected_files(manifest: Path, split: str) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return sorted({row["file_name"] for row in csv.DictReader(handle) if row["split"] == split})


def complete(root: Path, manifest: Path, split: str, stage: str) -> tuple[bool, dict[str, int]]:
    def ready(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    names = expected_files(manifest, split)
    missing_raw = missing_semantic = missing_refined = 0
    for name in names:
        stem = Path(name).stem
        if stage == "prediction":
            missing_raw += not ready(root / "raw" / split / f"{stem}.json.gz")
            missing_semantic += not ready(root / "semantic" / split / f"{stem}.npz")
        elif stage == "refined":
            missing_refined += not ready(root / "refined" / split / f"{stem}.json.gz")
        else:
            raise ValueError(stage)
    counts = {
        "expected": len(names), "missing_raw": int(missing_raw),
        "missing_semantic": int(missing_semantic), "missing_refined": int(missing_refined),
    }
    return not any((missing_raw, missing_semantic, missing_refined)), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a resumable M2-Lite cache stage is complete")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--stage", choices=("prediction", "refined"), required=True)
    args = parser.parse_args()
    is_complete, counts = complete(args.prediction_root, args.manifest, args.split, args.stage)
    state = "complete" if is_complete else "incomplete"
    print(f"{args.stage} {args.split}: {state} {counts}")
    raise SystemExit(0 if is_complete else 1)


if __name__ == "__main__":
    main()
