from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strata(frame: pd.DataFrame) -> pd.Series:
    ranked = frame["instance_count"].rank(method="first")
    bins = pd.qcut(ranked, q=min(4, len(frame)), labels=False, duplicates="drop").fillna(0).astype(int)
    detailed = frame["source"].astype(str) + "_c" + bins.astype(str)
    counts = detailed.value_counts()
    coarse = detailed.where(detailed.map(counts) >= 2, frame["source"].astype(str))
    counts = coarse.value_counts()
    return coarse.where(coarse.map(counts) >= 2, "other")


def build_fixed_annotated_split(annotation_path: Path, output_dir: Path, *, val_fraction: float = 0.15,
                                test_fraction: float = 0.20, seed: int = 3020,
                                expected_physical_images: int | None = 707) -> dict:
    """Create one immutable grouped train/val/test split with no folds."""
    annotation_path, output_dir = annotation_path.resolve(), output_dir.resolve()
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 0.5:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than 0.5")
    with annotation_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    anns: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        anns[str(item["image_id"])].append(item)
    records = pd.DataFrame([{
        "image_id": str(image["id"]), "file_name": image["file_name"],
        "record_instance_count": len(anns[str(image["id"])]),
        "record_foreground_area": sum(float(a.get("area", 0.0)) for a in anns[str(image["id"])]),
    } for image in coco["images"]])
    physical = records.groupby("file_name", as_index=False).agg(
        instance_count=("record_instance_count", "mean"), foreground_area=("record_foreground_area", "mean"),
        annotator_count=("image_id", "size"),
    ).sort_values("file_name").reset_index(drop=True)
    if expected_physical_images is not None and len(physical) != expected_physical_images:
        raise ValueError(f"Expected {expected_physical_images} physical images, found {len(physical)}")
    physical["source"] = physical["file_name"].str.replace(r"\.[^.]+$", "", regex=True).str[-2:]
    physical["stratum"] = _strata(physical)
    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    development_idx, test_idx = next(outer.split(physical, physical["stratum"]))
    physical["split"] = "train"
    physical.loc[test_idx, "split"] = "test"
    development = physical.loc[development_idx].copy()
    development["stratum"] = _strata(development)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction / (1.0 - test_fraction), random_state=seed + 1)
    _, val_local_idx = next(inner.split(development, development["stratum"]))
    physical.loc[development.index[val_local_idx], "split"] = "val"

    images = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id"})
    images["image_id"] = images["image_id"].astype(str)
    manifest = images.merge(physical[["file_name", "split", "source", "instance_count", "foreground_area",
                                      "annotator_count"]], on="file_name", validate="many_to_one")
    manifest["sample_stem"] = manifest["image_id"].str.replace(r"[/\\]", "_", regex=True)
    manifest["annotator"] = manifest["image_id"].str.split("-", n=1).str[0]
    manifest = manifest[["sample_stem", "image_id", "file_name", "split", "annotator", "source",
                         "instance_count", "foreground_area", "annotator_count"]]
    manifest = manifest.sort_values(["file_name", "image_id"]).reset_index(drop=True)
    if manifest.groupby("file_name")["split"].nunique().max() != 1:
        raise RuntimeError("Physical-image leakage across splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    csv_bytes = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != csv_bytes:
        raise RuntimeError(f"Refusing to overwrite a different frozen split: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(csv_bytes)
    summary = {split: {"physical_images": int(part["file_name"].nunique()),
                       "annotation_records": int(len(part)),
                       "instances": int(round(part.groupby("file_name")["instance_count"].first().sum()))}
               for split, part in manifest.groupby("split", sort=True)}
    metadata = {
        "protocol": "fixed-grouped-train-val-test-v1", "status": "FROZEN", "seed": seed,
        "val_fraction": val_fraction, "test_fraction": test_fraction, "group_key": "file_name",
        "stratification": "source + instance-count quartile", "annotation_sha256": _sha256(annotation_path),
        "manifest_sha256": _sha256(manifest_path), "summary": summary,
        "rules": ["Train networks and quality regressor only with train rows.",
                  "Use val for checkpoint selection and decoder thresholds.",
                  "Use test once after freezing the method; never tune on it.",
                  "Use this same manifest for every training seed."],
    }
    metadata_path = output_dir / "metadata.json"
    text = json.dumps(metadata, indent=2) + "\n"
    if metadata_path.exists() and metadata_path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Frozen split metadata mismatch: {metadata_path}")
    if not metadata_path.exists():
        metadata_path.write_text(text, encoding="utf-8")
    return metadata
