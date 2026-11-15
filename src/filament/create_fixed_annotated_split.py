from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import add_path_args, discover_data, paths_from_args
from .data.fixed_split import build_fixed_annotated_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one grouped train/val/test split (no folds)")
    add_path_args(parser)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=3020)
    parser.add_argument("--physical-images", type=int, default=958)
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation, _, _ = discover_data(paths)
    output = args.output_dir or paths.results_root / "fixed_split"
    result = build_fixed_annotated_split(annotation, output, val_fraction=args.val_fraction,
                                         test_fraction=args.test_fraction, seed=args.seed,
                                         expected_physical_images=args.physical_images)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
