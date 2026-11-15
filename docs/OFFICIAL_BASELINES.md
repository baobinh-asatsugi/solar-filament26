# Fourteen-baseline protocol

All baselines use the committed grouped MAGFiLO split, source resolution
2048×2048, seeds 3020–3024, validation-only checkpoint/decoder selection, and
the shared pooled-PQ implementation. Test is never used for tuning.

## Methods and provenance

| key | method | implementation source |
|---|---|---|
| `yolov8l` | YOLOv8l-seg | Ultralytics release |
| `unetpp_watershed` | original five-level U-Net++ + watershed | official U-Net++ architecture; local instance adapter |
| `mask_rcnn` | Mask R-CNN R50-FPN | MMDetection v3.3.0 config |
| `cascade_mask_rcnn` | Cascade Mask R-CNN R50-FPN | MMDetection v3.3.0 config |
| `htc` | Hybrid Task Cascade R50-FPN | MMDetection v3.3.0 config |
| `solov2` | SOLOv2 R50-FPN | MMDetection v3.3.0 config |
| `condinst` | CondInst R50-FPN | MMDetection v3.3.0 config |
| `queryinst` | QueryInst R50-FPN | MMDetection v3.3.0 config |
| `mask2former` | Mask2Former R50 | MMDetection v3.3.0 config |
| `rtmdet_ins` | RTMDet-Ins-L | MMDetection v3.3.0 config |
| `fullres_unet` | full-resolution ResNet34 U-Net | port of `rodchemist/solar-filament-segmentation-2026` |
| `edgeattnet` | edge-guided attention U-Net | paper-faithful reconstruction |
| `detr_crop_unet` | RT-DETR → filament crop → U-Net | paper-faithful reconstruction of the published framework |
| `maskrcnn_consensus` | Mask R-CNN consensus/alignment | port of `gitfirsthub/magfilo-filament-segmentation` |

The eight MMDetection adapters alter only paths, one-class dimensions, random
seed, working directory, and 2048 input transforms. Their backbones, heads,
losses, optimizer, and schedules stay in the named upstream configs.

The DETR–crop U-Net authors released the SWEFil data companion repository but
not their model implementation. EdgeAttNet's public `edgeattnet_model.py` is a
self-importing placeholder and omits the advertised model class. Therefore
those two baselines are explicitly labelled reconstructions, not official code
ports. Full-resolution U-Net and Mask R-CNN consensus/alignment are direct
ports of publicly released implementations.

## Prepare

```bash
export PROJECT_ROOT=/path/to/solar-filament26
export DATA_ROOT=/private/path/MAGFiLO
export ARTIFACT_ROOT=/private/path/artifacts
export PYTHON=/path/to/environment/bin/python
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
cd "$PROJECT_ROOT"
mkdir -p "$ARTIFACT_ROOT/logs"

bash scripts/prepare_fixed_split.sh
bash scripts/prepare_baselines.sh
```

## Direct one-seed commands

```bash
# YOLOv8l
bash scripts/run_yolov8_seed.sh 3020 0

# U-Net++ + watershed
bash scripts/run_unetpp_watershed_seed.sh 3020 0

# Any MMDetection key listed above
bash scripts/run_mmdet_seed.sh mask_rcnn 3020 0

# Full-resolution U-Net or EdgeAttNet
bash scripts/run_semantic_baseline_seed.sh fullres_unet 3020 0
bash scripts/run_semantic_baseline_seed.sh edgeattnet 3020 0

# DETR → crop U-Net
bash scripts/run_detr_crop_unet_seed.sh 3020 0

# Consensus/alignment Mask R-CNN
bash scripts/run_maskrcnn_consensus_seed.sh 3020 0
```

Each runner performs train → validation tuning → test with the frozen
validation configuration. For formal experiments, launch all five seeds and
do not inspect test until every design choice is frozen.

## Five-seed launchers

```bash
bash scripts/launch_yolov8_five_seeds.sh
bash scripts/launch_unetpp_watershed_five_seeds.sh
bash scripts/launch_domain_baselines.sh
```

MMDetection has binary dependency constraints and uses a separate environment:

```bash
conda create -n mmdet-baselines python=3.11 -y
conda activate mmdet-baselines
# Install the CUDA PyTorch build appropriate for the machine first.
export PYTHON=/path/to/mmdet-baselines/bin/python
bash scripts/setup_official_baselines.sh
bash scripts/launch_official_baselines.sh
```

`setup_official_baselines.sh` pins MMDetection v3.3.0 and performs a config/model
construction smoke test. Do not launch the 40 runs unless all eight configs
print `OK`.

## Summary

```bash
bash scripts/summarize_five_seed_results.sh
```

The expected experiment contains 75 final results: M2-Lite plus 14 baselines,
five seeds each. Missing method/seed pairs are printed explicitly. Report
pooled PQ mean ± sample standard deviation; SQ/RQ and TP/FP/FN diagnose shape
quality versus recognition errors.
