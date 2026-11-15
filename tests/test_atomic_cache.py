from pathlib import Path

import numpy as np

from baselines.cache_io import atomic_savez_compressed, atomic_write_bytes


def test_atomic_write_bytes_replaces_existing_file(tmp_path: Path):
    target = tmp_path / "cache.png"
    target.write_bytes(b"truncated")
    atomic_write_bytes(target, b"complete")
    assert target.read_bytes() == b"complete"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_savez_compressed_is_readable(tmp_path: Path):
    target = tmp_path / "cache.npz"
    expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
    atomic_savez_compressed(target, masks=expected)
    with np.load(target, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["masks"], expected)
    assert list(tmp_path.glob(".*.tmp")) == []
