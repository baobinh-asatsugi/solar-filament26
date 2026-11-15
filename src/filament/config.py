from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Paths:
    data_root: Path
    artifact_root: Path

    @property
    def yolo_root(self) -> Path:
        return self.artifact_root / "yolo_dataset"

    def dataset_root(self, dataset_name: str) -> Path:
        return self.artifact_root / "datasets" / dataset_name

    @property
    def runs_root(self) -> Path:
        return self.artifact_root / "runs"

    @property
    def results_root(self) -> Path:
        return self.artifact_root / "results"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_root / run_name

    def experiment_results(self, run_name: str) -> Path:
        path = self.results_root / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prediction_cache(self, run_name: str) -> Path:
        path = self.artifact_root / "predictions" / "raw" / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path


def add_path_args(parser: argparse.ArgumentParser) -> None:
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", project_root / "magfilo")),
        help="MAGFiLO v1.0 root containing magfilo_2024_v1.0.json and images/",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("ARTIFACT_ROOT", project_root / "artifacts")),
    )


def add_run_args(parser: argparse.ArgumentParser, *, checkpoint: bool = True) -> None:
    parser.add_argument("--run-name", default="yolov8l_seg_grouped_baseline")
    if checkpoint:
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=None,
            help="Explicit checkpoint. Defaults to <artifact-root>/runs/<run-name>/weights/best.pt.",
        )


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML defaults; explicit CLI flags take precedence")


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=None)
    known, _ = preliminary.parse_known_args()
    if known.config is not None:
        config_path = known.config.expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")
        actions = {action.dest: action for action in parser._actions}
        converted = {}
        for raw_key, value in payload.items():
            key = str(raw_key).replace("-", "_")
            if key not in actions:
                raise ValueError(f"Unknown config key '{raw_key}' in {config_path}")
            action = actions[key]
            if value is not None and action.type is not None:
                value = action.type(value)
            converted[key] = value
        parser.set_defaults(**converted)
    return parser.parse_args()


def resolve_checkpoint(paths: Paths, run_name: str, checkpoint: Path | None) -> Path:
    resolved = checkpoint.expanduser().resolve() if checkpoint else paths.run_dir(run_name) / "weights" / "best.pt"
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def paths_from_args(args: argparse.Namespace) -> Paths:
    paths = Paths(args.data_root.expanduser().resolve(), args.artifact_root.expanduser().resolve())
    for directory in (
        paths.artifact_root,
        paths.yolo_root,
        paths.artifact_root / "datasets",
        paths.runs_root,
        paths.results_root,
        paths.artifact_root / "predictions" / "raw",
        paths.artifact_root / "registry",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def discover_data(paths: Paths) -> tuple[Path, Path, Path]:
    annotation_path = paths.data_root / "magfilo_2024_v1.0.json"
    image_root = paths.data_root / "images"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing full MAGFiLO annotation: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing full MAGFiLO image directory: {image_root}")
    return annotation_path, image_root, image_root

