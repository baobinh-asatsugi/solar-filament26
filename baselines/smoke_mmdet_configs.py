from __future__ import annotations

import argparse
import gc
from pathlib import Path

from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

from baselines.mmdet_adapter import _set_num_classes


CONFIGS = {
    "mask_rcnn": "configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
    "cascade_mask_rcnn": "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
    "htc": "configs/htc/htc_r50_fpn_1x_coco.py",
    "solov2": "configs/solov2/solov2_r50_fpn_1x_coco.py",
    "condinst": "configs/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py",
    "queryinst": "configs/queryinst/queryinst_r50_fpn_1x_coco.py",
    "mask2former": "configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
    "rtmdet_ins": "configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args(); register_all_modules()
    for name, relative in CONFIGS.items():
        path = args.upstream / relative
        if not path.is_file(): raise FileNotFoundError(path)
        cfg = Config.fromfile(path); _set_num_classes(cfg.model)
        model = MODELS.build(cfg.model)
        parameters = sum(value.numel() for value in model.parameters())
        print(f"OK {name}: {parameters:,} parameters")
        del model; gc.collect()


if __name__ == "__main__": main()
