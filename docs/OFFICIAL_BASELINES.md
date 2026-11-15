# Five-seed comparison protocol

All methods use the same immutable grouped `train/val/test` manifest. Neural
training seeds are 3020–3024. Decoder thresholds are selected on validation
and then frozen for annotated test. The final table reports test mean ± sample
standard deviation across the five seeds.

## Compared systems

1. **M2-Lite (ours)** — YOLOv8l proposal branch, U-Net++ semantic branch,
   semantic-assisted SIRA, and the minimal 17-feature quality decoder.
2. **Plain YOLOv8l-seg** — the detector alone, trained at 2048 px.
3. **Mask R-CNN** — two-stage RoI instance segmentation
   ([paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf),
   [code](https://github.com/open-mmlab/mmdetection)).
4. **Cascade Mask R-CNN** — cascaded box/mask refinement
   ([paper](https://arxiv.org/abs/1712.00726),
   [code](https://github.com/open-mmlab/mmdetection)).
5. **Hybrid Task Cascade (HTC)** — interleaved detection and segmentation
   stages with semantic context ([paper](https://arxiv.org/abs/1901.07518),
   [code](https://github.com/open-mmlab/mmdetection)).
6. **SOLOv2** — box-free, location-based dynamic mask kernels
   ([paper](https://papers.neurips.cc/paper/2020/file/cd3afef9b8b89558cd56638c3631868a-Paper.pdf),
   [official repo](https://github.com/WXinlong/SOLO)).
7. **CondInst** — conditional convolution instance masks without RoIAlign
   ([paper](https://arxiv.org/abs/2003.05664),
   [official repo](https://github.com/aim-uofa/AdelaiDet)).
8. **QueryInst** — query-driven parallel box/mask supervision
   ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Instances_As_Queries_ICCV_2021_paper.pdf),
   [official repo](https://github.com/hustvl/QueryInst)).
9. **Mask2Former** — masked-attention universal segmentation
   ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf),
   [official repo](https://github.com/facebookresearch/Mask2Former)).
10. **RTMDet-Ins-L** — real-time CSPNeXt instance segmentation
    ([paper](https://arxiv.org/abs/2212.07784),
    [code](https://github.com/open-mmlab/mmdetection)).
11. **UNet++ + watershed** — domain-specific nested medical semantic
    segmentation followed by validation-tuned instance separation
    ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/),
    [official repo](https://github.com/MrGiovanni/UNetPlusPlus)).

The eight non-YOLO baselines run their MMDetection v3.3.0 official model-zoo
configs. The adapter changes only dataset path, class count (80→1), seed,
work directory, and automatic learning-rate scaling for the actual GPU count.
It does not replace backbones, heads, losses, augmentation pipelines,
optimizers, or schedules.

## Commands

Prepare the shared split and COCO files:

```bash
bash scripts/prepare_fixed_split.sh
bash scripts/prepare_baselines.sh
```

Ours (four GPUs, seed 3024 follows seed 3020 on GPU 0):

```bash
bash scripts/launch_five_seeds_tmux.sh
# after validation is accepted/frozen:
for seed in 3020 3021 3022 3023 3024; do
  bash scripts/evaluate_final_test.sh "$seed"
done
```

Plain YOLOv8l-seg:

```bash
bash scripts/launch_yolov8_five_seeds.sh
```

Domain-specific U-Net++ + watershed:

```bash
conda activate finetune-sd
bash scripts/launch_unetpp_watershed_five_seeds.sh
```

Official baselines should use a separate environment because MMDetection pins
MMCV independently from Ultralytics:

```bash
conda create -n mmdet-baselines python=3.11 -y
conda activate mmdet-baselines
# install a CUDA PyTorch build compatible with the server first
export PYTHON=/path/to/mmdet-baselines/bin/python
bash scripts/setup_official_baselines.sh
bash scripts/prepare_baselines.sh
bash scripts/launch_official_baselines.sh
```

The setup command includes a CPU smoke test that loads the eight official
configs, applies only the one-class adaptation, and constructs every model.
Do not launch the 40 training runs unless all eight lines print `OK`.

One baseline/seed can be debugged directly:

```bash
bash scripts/run_mmdet_seed.sh mask_rcnn 3020 0
```

Summarize every completed run:

```bash
conda activate finetune-sd
bash scripts/summarize_five_seed_results.sh
```

Outputs:

```text
$ARTIFACT_ROOT/results/five_seed_runs.csv
$ARTIFACT_ROOT/results/five_seed_summary.csv
```

## Upstream source handling

Official repositories were cloned temporarily during adapter development for
config/path verification. They are not part of Git (`external/` is ignored).
At runtime, `setup_official_baselines.sh` checks out the pinned MMDetection
v3.3.0 source into `$ARTIFACT_ROOT/upstream`, keeping third-party code and
weights outside this repository.
