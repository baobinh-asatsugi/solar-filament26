from __future__ import annotations

import sys


REPAIR = (
    "python -m pip install --force-reinstall --no-cache-dir "
    "numpy==1.26.4 opencv-python-headless==4.11.0.86"
)


def main() -> None:
    try:
        import cv2
        import numpy as np
        import torch
    except Exception as error:
        raise SystemExit(f"CECS environment import failed: {error}\nRepair with:\n  {REPAIR}") from error

    if np.__version__ != "1.26.4":
        raise SystemExit(
            f"CECS requires NumPy 1.26.4, found {np.__version__}.\nRepair with:\n  {REPAIR}"
        )
    if cv2.__version__ != "4.11.0":
        raise SystemExit(
            f"CECS requires OpenCV 4.11.0, found {cv2.__version__}.\nRepair with:\n  {REPAIR}"
        )
    if torch.version.cuda is None:
        raise SystemExit("CECS virtual environment contains a CPU-only PyTorch build")

    print(
        f"CECS environment OK: python={sys.executable} torch={torch.__version__} "
        f"cuda_build={torch.version.cuda} numpy={np.__version__} opencv={cv2.__version__}"
    )


if __name__ == "__main__":
    main()
