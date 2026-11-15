from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def _diff(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff(left[key], right[key], path))
        return paths
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        paths = []
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right): paths.append(path)
            else: paths.extend(_diff(left[index], right[index], path))
        return paths
    return [] if left == right else [prefix]


def _allowed(path: str) -> bool:
    model_class = path.startswith("model.") and path.rsplit(".", 1)[-1] in {
        "num_classes", "num_things_classes", "num_stuff_classes"
    }
    dataset = path.startswith(("train_dataloader.dataset", "val_dataloader.dataset", "test_dataloader.dataset"))
    evaluator = path.startswith(("val_evaluator.", "test_evaluator.")) and path.rsplit(".", 1)[-1] in {"ann_file", "metric"}
    runtime = path in {"work_dir", "resume", "randomness", "auto_scale_lr.enable"}
    return model_class or dataset or evaluator or runtime


def _set_num_classes(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"num_classes", "num_things_classes"}:
                value[key] = 1
            elif key == "num_stuff_classes":
                value[key] = 0
            else:
                _set_num_classes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _set_num_classes(child)


def _set_dataset(value, *, ann_file: str, image_root: str) -> None:
    if not isinstance(value, dict):
        return
    if "dataset" in value:
        _set_dataset(value["dataset"], ann_file=ann_file, image_root=image_root)
    if value.get("type") in {"CocoDataset", "ClassBalancedDataset", "RepeatDataset"} or "ann_file" in value:
        if value.get("type") == "CocoDataset" or "ann_file" in value:
            value["type"] = "CocoDataset"
            value["data_root"] = ""
            value["ann_file"] = ann_file
            value["data_prefix"] = {"img": image_root}
            value["metainfo"] = {"classes": ("filament",), "palette": [(255, 0, 0)]}
            value["filter_cfg"] = {"filter_empty_gt": False, "min_size": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-only adapter for official MMDetection configs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config.resolve())
    original = cfg.to_dict()
    _set_num_classes(cfg.model)
    _set_dataset(cfg.train_dataloader.dataset, ann_file=str(args.train_json.resolve()), image_root=str(args.image_root.resolve()))
    _set_dataset(cfg.val_dataloader.dataset, ann_file=str(args.val_json.resolve()), image_root=str(args.image_root.resolve()))
    _set_dataset(cfg.test_dataloader.dataset, ann_file=str(args.test_json.resolve()), image_root=str(args.image_root.resolve()))
    for evaluator, annotation in ((cfg.val_evaluator, args.val_json), (cfg.test_evaluator, args.test_json)):
        if isinstance(evaluator, dict):
            evaluator["ann_file"] = str(annotation.resolve())
            evaluator["metric"] = ["bbox", "segm"]
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.randomness = dict(seed=args.seed, deterministic=True)
    cfg.resume = args.resume
    if "auto_scale_lr" in cfg:
        cfg.auto_scale_lr.enable = True
    args.work_dir.mkdir(parents=True, exist_ok=True)
    changed = _diff(original, cfg.to_dict())
    unexpected = [path for path in changed if not _allowed(path)]
    if unexpected:
        raise RuntimeError(f"Adapter changed forbidden config fields: {unexpected}")
    (args.work_dir / "adaptation_audit.json").write_text(json.dumps({
        "upstream_config": str(args.config.resolve()), "changed_paths": changed,
        "invariant": "Only dataset/evaluator paths, one-class dimensions, seed/runtime, and auto LR scaling may change.",
    }, indent=2) + "\n")
    cfg.dump(args.work_dir / "adapted_config.py")
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
