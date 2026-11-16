from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from PIL import Image
from pycocotools import mask as mask_utils

from filament.checkpoint_policy import retain_only_best_checkpoint


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
    checkpoint_policy = path.startswith("default_hooks.checkpoint.") and path.rsplit(".", 1)[-1] in {
        "interval", "max_keep_ckpts", "rule", "save_best", "save_last",
        "save_optimizer", "save_param_scheduler",
    }
    shared_input = path.startswith(("train_dataloader.dataset", "val_dataloader.dataset", "test_dataloader.dataset")) and any(
        token in path for token in (".scale", ".scales", ".img_scale", ".size")
    )
    return model_class or dataset or evaluator or runtime or shared_input or checkpoint_policy


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


def _set_input_size(value, input_size: int) -> None:
    if isinstance(value, dict):
        transform_type = str(value.get("type", ""))
        if "Resize" in transform_type:
            if "scale" in value:
                value["scale"] = (input_size, input_size)
            if "scales" in value:
                value["scales"] = [(input_size, input_size)]
        if transform_type in {"BatchFixedSizePad", "FixedSizeCrop"} and "size" in value:
            value["size"] = (input_size, input_size)
        for child in value.values():
            _set_input_size(child, input_size)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _set_input_size(child, input_size)


def _has_semantic_head(model) -> bool:
    roi_head = model.get("roi_head", {}) if isinstance(model, dict) else {}
    return isinstance(roi_head, dict) and roi_head.get("semantic_head") is not None


def _decode_annotation_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def _prepare_semantic_targets(annotation_path: Path, output_dir: Path) -> int:
    """Build HTC thing maps: filament=0, unlabeled background=255."""
    with annotation_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images_by_id = {str(item["id"]): item for item in coco["images"]}
    image_ids_by_file: dict[str, list[str]] = defaultdict(list)
    annotations_by_id: dict[str, list[dict]] = defaultdict(list)
    for item in coco["images"]:
        image_ids_by_file[item["file_name"]].append(str(item["id"]))
    for annotation in coco["annotations"]:
        annotations_by_id[str(annotation["image_id"])].append(annotation)

    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, image_ids in image_ids_by_file.items():
        info = images_by_id[image_ids[0]]
        height, width = int(info["height"]), int(info["width"])
        target = np.full((height, width), 255, dtype=np.uint8)
        for image_id in image_ids:
            for annotation in annotations_by_id[image_id]:
                mask = _decode_annotation_mask(annotation["segmentation"], height, width)
                target[mask] = 0
        target_path = output_dir / Path(file_name).with_suffix(".png")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(target, mode="L").save(target_path)
    return len(image_ids_by_file)


def _enable_semantic_targets(value, semantic_root: Path) -> None:
    if not isinstance(value, dict):
        return
    if "dataset" in value:
        _enable_semantic_targets(value["dataset"], semantic_root)
    if value.get("type") == "CocoDataset" or "ann_file" in value:
        data_prefix = dict(value.get("data_prefix", {}))
        data_prefix["seg"] = str(semantic_root.resolve())
        value["data_prefix"] = data_prefix
        value["seg_map_suffix"] = ".png"
        for transform in value.get("pipeline", []):
            if isinstance(transform, dict) and transform.get("type") == "LoadAnnotations":
                transform["with_seg"] = True


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
    parser.add_argument("--input-size", type=int, default=2048)
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config.resolve())
    original = cfg.to_dict()
    _set_num_classes(cfg.model)
    _set_dataset(cfg.train_dataloader.dataset, ann_file=str(args.train_json.resolve()), image_root=str(args.image_root.resolve()))
    _set_dataset(cfg.val_dataloader.dataset, ann_file=str(args.val_json.resolve()), image_root=str(args.image_root.resolve()))
    _set_dataset(cfg.test_dataloader.dataset, ann_file=str(args.test_json.resolve()), image_root=str(args.image_root.resolve()))
    _set_input_size(cfg.train_dataloader.dataset, args.input_size)
    _set_input_size(cfg.val_dataloader.dataset, args.input_size)
    _set_input_size(cfg.test_dataloader.dataset, args.input_size)
    semantic_target_count = 0
    if _has_semantic_head(cfg.model):
        semantic_root = args.work_dir / "semantic_targets"
        semantic_target_count = _prepare_semantic_targets(args.train_json, semantic_root)
        _enable_semantic_targets(cfg.train_dataloader.dataset, semantic_root)
    for evaluator, annotation in ((cfg.val_evaluator, args.val_json), (cfg.test_evaluator, args.test_json)):
        if isinstance(evaluator, dict):
            evaluator["ann_file"] = str(annotation.resolve())
            evaluator["metric"] = ["bbox", "segm"]
    cfg.work_dir = str(args.work_dir.resolve())
    # Several official instance heads (notably CondInst) use CUDA replication
    # padding whose backward pass has no deterministic implementation in the
    # pinned Torch 2.0 build. Keep the controlled seed, but do not request
    # strict deterministic kernels that would abort training.
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.resume = args.resume
    checkpoint_hook = cfg.default_hooks.checkpoint
    # Disable periodic epoch snapshots. Validation still triggers save_best.
    checkpoint_hook.interval = -1
    checkpoint_hook.save_best = "coco/segm_mAP"
    checkpoint_hook.rule = "greater"
    checkpoint_hook.max_keep_ckpts = 1
    checkpoint_hook.save_last = False
    checkpoint_hook.save_optimizer = False
    checkpoint_hook.save_param_scheduler = False
    if "auto_scale_lr" in cfg:
        cfg.auto_scale_lr.enable = True
    args.work_dir.mkdir(parents=True, exist_ok=True)
    changed = _diff(original, cfg.to_dict())
    unexpected = [path for path in changed if not _allowed(path)]
    if unexpected:
        raise RuntimeError(f"Adapter changed forbidden config fields: {unexpected}")
    (args.work_dir / "adaptation_audit.json").write_text(json.dumps({
        "upstream_config": str(args.config.resolve()), "changed_paths": changed,
        "shared_input_size": args.input_size,
        "semantic_target_count": semantic_target_count,
        "invariant": "Only dataset/evaluator paths, one-class dimensions, the shared 2048 input size, seed/runtime, best-only checkpoint retention, and auto LR scaling may change.",
    }, indent=2) + "\n")
    cfg.dump(args.work_dir / "adapted_config.py")
    Runner.from_cfg(cfg).train()

    best_candidates = sorted(
        args.work_dir.glob("best_*.pth"), key=lambda path: path.stat().st_mtime
    )
    if not best_candidates:
        raise RuntimeError(
            f"MMDetection did not produce a best validation checkpoint in {args.work_dir}"
        )
    selected = best_candidates[-1]
    best_checkpoint = args.work_dir / "best.pth"
    if selected.resolve() != best_checkpoint.resolve():
        selected.replace(best_checkpoint)
    retain_only_best_checkpoint(args.work_dir, best_checkpoint)
    (args.work_dir / "best_checkpoint").write_text(str(best_checkpoint.resolve()) + "\n")
    legacy_pointer = args.work_dir / "last_checkpoint"
    if legacy_pointer.exists():
        legacy_pointer.unlink()


if __name__ == "__main__":
    main()
