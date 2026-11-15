#!/usr/bin/env python3
"""Remove historical non-best checkpoints from managed artifact directories.

The command is a dry run unless ``--apply`` is supplied.  It only scans ``runs``
and ``baseline_work`` below ARTIFACT_ROOT and never touches pretrained/upstream data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def is_best(path: Path) -> bool:
    name = path.name.lower()
    return name in {"best.pt", "best.pth", "crop_unet_best.pt"} or name.startswith("best_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path(os.environ["ARTIFACT_ROOT"]) if "ARTIFACT_ROOT" in os.environ else None,
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete; default is dry-run")
    args = parser.parse_args()
    if args.artifact_root is None:
        parser.error("set ARTIFACT_ROOT or pass --artifact-root")

    candidates: list[Path] = []
    for relative in ("runs", "baseline_work"):
        root = (args.artifact_root / relative).resolve()
        if not root.is_dir():
            continue
        for pattern in ("*.pt", "*.pth"):
            candidates.extend(path for path in root.rglob(pattern) if not is_best(path))

    candidates = sorted(set(candidates))
    total = sum(path.stat().st_size for path in candidates)
    action = "DELETE" if args.apply else "WOULD DELETE"
    for path in candidates:
        print(f"{action} {path} ({path.stat().st_size / 2**20:.1f} MiB)")
        if args.apply:
            path.unlink()
    print(
        f"{action}: {len(candidates)} non-best checkpoints, "
        f"{total / 2**30:.2f} GiB. Best-named checkpoints were retained."
    )


if __name__ == "__main__":
    main()
