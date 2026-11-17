from __future__ import annotations

import sys


def main() -> None:
    try:
        import cv2
        import numpy as np
        import torch
    except Exception as error:
        raise SystemExit(f"CECS environment import failed: {error}") from error

    if torch.version.cuda is None:
        raise SystemExit("CECS virtual environment contains a CPU-only PyTorch build")

    # Check the real ABI boundary instead of rejecting a NumPy version number.
    # This catches an actually incompatible PyTorch build while allowing NumPy 2.
    try:
        array = np.zeros((2, 2), dtype=np.float32)
        tensor = torch.from_numpy(array)
        if tensor.shape != (2, 2) or tensor.numpy().shape != (2, 2):
            raise RuntimeError("unexpected NumPy/Torch bridge result")
        if cv2.resize(array, (1, 1)).shape != (1, 1):
            raise RuntimeError("unexpected OpenCV/NumPy bridge result")
    except Exception as error:
        raise SystemExit(f"CECS NumPy/Torch/OpenCV ABI check failed: {error}") from error

    print(
        f"CECS environment OK: python={sys.executable} torch={torch.__version__} "
        f"cuda_build={torch.version.cuda} numpy={np.__version__} opencv={cv2.__version__}"
    )


if __name__ == "__main__":
    main()
