"""Race-safe cache writers shared by concurrently trained baseline seeds."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes atomically so readers never observe a partial cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    """Atomically publish a compressed NumPy archive without suffix rewriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
