from __future__ import annotations

from pathlib import Path


def retain_only_best_checkpoint(run_dir: Path, best_checkpoint: Path) -> list[Path]:
    """Delete model checkpoints in ``run_dir`` except the selected best one.

    Training histories, metrics, plots, prediction caches, and configuration files are
    deliberately untouched.  This is called only after training has completed, so an
    interrupted run can still retain its framework-specific recovery checkpoint.
    """
    run_dir = run_dir.resolve()
    best_checkpoint = best_checkpoint.resolve()
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint was not produced: {best_checkpoint}")
    if run_dir != best_checkpoint.parent and run_dir not in best_checkpoint.parents:
        raise ValueError(f"Best checkpoint {best_checkpoint} is outside run directory {run_dir}")

    removed: list[Path] = []
    for pattern in ("*.pt", "*.pth"):
        for candidate in run_dir.rglob(pattern):
            if candidate.resolve() == best_checkpoint:
                continue
            candidate.unlink()
            removed.append(candidate)
    return removed
