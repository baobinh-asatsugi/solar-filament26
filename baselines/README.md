# Official instance-segmentation baselines

All eight baselines are loaded from unmodified official MMDetection model-zoo
configs. The adapter changes only: dataset paths, one-class output dimensions,
random seed, work directory, and checkpoint initialization. Backbone, neck,
head, losses, optimizer, augmentation, and training schedule remain those of
the named upstream config.

| key | paper/model | official config |
|---|---|---|
| `mask_rcnn` | Mask R-CNN | `mask-rcnn_r50_fpn_1x_coco` |
| `cascade_mask_rcnn` | Cascade Mask R-CNN | `cascade-mask-rcnn_r50_fpn_1x_coco` |
| `htc` | Hybrid Task Cascade | `htc_r50_fpn_1x_coco` |
| `solov2` | SOLOv2 | `solov2_r50_fpn_1x_coco` |
| `condinst` | CondInst | `condinst_r50_fpn_ms-poly-90k_coco_instance` |
| `queryinst` | QueryInst | `queryinst_r50_fpn_1x_coco` |
| `mask2former` | Mask2Former | `mask2former_r50_8xb2-lsj-50e_coco` |
| `rtmdet_ins` | RTMDet-Ins | `rtmdet-ins_l_8xb32-300e_coco` |

The upstream source is downloaded at run time under the ignored artifact root;
it is not vendored or modified in this repository.

The domain-specific `unetpp_watershed` baseline follows the authors' original
five-level channel layout (32–512), nested dense skip paths, four deep-
supervision heads, BCE+Dice objective, Adam at 3e-4, and early stopping.
Watershed is only the required semantic-to-instance adapter; its parameters are
selected on validation and frozen for test.
