# Solar Filament Instance Segmentation

Reproducible code for **M2-Lite** and 14 comparison baselines on the full
MAGFiLO v1.0 dataset. Every method uses the same frozen physical-image split,
the same five seeds, the same 2048×2048 source images, validation-only model
selection, and the same pooled panoptic-quality evaluator.

All reported results must use the committed full-MAGFiLO protocol below and be
reported as five-seed mean ± standard deviation.

## M2-Lite architecture

```text
2048×2048 grayscale image
  ├── YOLOv8l-seg proposals (2048 input)
  └── U-Net++ / EfficientNet-B3 semantic map (overlapping 768 crops)
                 │
                 ▼
       SIRA U-Net / ResNet34 mask refinement
       [intensity, CLAHE, seed, affinity, semantic]
                 │
                 ▼
       17-feature HGBR quality scoring
                 │
                 ▼
       threshold + mask-IoU NMS + overlap resolution
                 │
                 ▼
             instance masks
```

The model code intentionally keeps only this best-performing chain. Historical
graph selectors, Q2/Q3 heads, ensembles, and experimental set models are not
part of the runnable protocol.

## Dataset

Download the complete MAGFiLO v1.0 release and arrange it at any private path:

```text
/private/path/MAGFiLO/
├── magfilo_2024_v1.0.json
└── images/
    ├── ... 958 image files ...
```

The annotation file contains 1,593 annotation records, 958 unique physical
images, and 10,244 filament instances. Data, weights, caches, results, external
repositories, and credentials are excluded by `.gitignore`.

## Environment

Python 3.10–3.11 and an NVIDIA GPU are recommended. Install a CUDA-enabled
PyTorch/torchvision pair compatible with the host driver first, then:

```bash
conda activate <project-environment>
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

export PROJECT_ROOT=/path/to/solar-filament26
export DATA_ROOT=/private/path/MAGFiLO
export ARTIFACT_ROOT=/private/path/solar-filament-artifacts
export PYTHON=/path/to/environment/bin/python
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
cd "$PROJECT_ROOT"
mkdir -p "$ARTIFACT_ROOT/logs"
```

No personal username, mount point, or server-specific path is required.

## Frozen split

The committed manifest `splits/magfilo_v1/manifest.csv` is grouped by physical
`file_name` and stratified by source plus instance-count quartile:

| split | physical images | annotation records | instances |
|---|---:|---:|---:|
| train | 622 | 1,016 | 6,622 |
| validation | 144 | 247 | 1,550 |
| test | 192 | 330 | 2,072 |

Manifest SHA-256:
`224a710c143af577caef185d423c42dc3d92783e6c5a71a78ab13a72b52971cb`.

Verify the local dataset and reproduce the prepared training views:

```bash
bash scripts/prepare_fixed_split.sh \
  2>&1 | tee "$ARTIFACT_ROOT/logs/prepare_fixed_split.log"
bash scripts/prepare_baselines.sh \
  2>&1 | tee "$ARTIFACT_ROOT/logs/prepare_baselines.log"
```

The split builder refuses to replace a frozen manifest with a different one.
Train only on `train`, tune checkpoints/thresholds only on `val`, and evaluate
`test` once after the method is frozen.

## Run M2-Lite

One seed:

```bash
bash scripts/train_one_seed.sh 3020 0 \
  2>&1 | tee "$ARTIFACT_ROOT/logs/m2lite_s3020.log"
# only after validation choices are frozen
bash scripts/evaluate_final_test.sh 3020
```

Five seeds (3020–3024) on four GPUs:

```bash
bash scripts/launch_five_seeds_tmux.sh
tmux ls
```

## Run 14 baselines

All methods consume the same 2048×2048 source canvas. Architecture-native
operations—M2-Lite semantic crops, U-Net++ tiles, RoI heads, and the DETR crop
refiner—do not resize the source dataset or change the evaluation resolution.

| group | methods | five-seed launcher |
|---|---|---|
| detector | YOLOv8l-seg | `scripts/launch_yolov8_five_seeds.sh` |
| semantic instance adapter | U-Net++ + watershed | `scripts/launch_unetpp_watershed_five_seeds.sh` |
| official MMDetection | Mask R-CNN, Cascade Mask R-CNN, HTC, SOLOv2, CondInst, QueryInst, Mask2Former, RTMDet-Ins | `scripts/launch_official_baselines.sh` |
| solar/domain methods | full-resolution U-Net, EdgeAttNet, DETR→crop U-Net, Mask R-CNN consensus/alignment | `scripts/launch_domain_baselines.sh` |

Commands:

```bash
bash scripts/launch_yolov8_five_seeds.sh
bash scripts/launch_unetpp_watershed_five_seeds.sh
bash scripts/launch_domain_baselines.sh
```

The eight MMDetection methods need their isolated environment:

```bash
conda create -n mmdet-baselines python=3.11 -y
conda activate mmdet-baselines
# Install a compatible CUDA PyTorch build first.
export PYTHON=/path/to/mmdet-baselines/bin/python
bash scripts/setup_official_baselines.sh
bash scripts/launch_official_baselines.sh
```

The setup script checks out pinned MMDetection v3.3.0 under the ignored
artifact directory and builds all eight adapted configs before training.
Detailed provenance and direct one-seed commands are in
[`docs/OFFICIAL_BASELINES.md`](docs/OFFICIAL_BASELINES.md).

## Summarize all 75 test runs

After five seeds of M2-Lite and 14 baselines have finished:

```bash
bash scripts/summarize_five_seed_results.sh
```

Outputs:

```text
$ARTIFACT_ROOT/results/five_seed_runs.csv
$ARTIFACT_ROOT/results/five_seed_summary.csv
```

The summary script explicitly reports missing method/seed pairs. The primary
metric is pooled/global PQ with one-to-one matching at IoU `> 0.5`; it also
reports mean PQ, SQ, RQ, TP, FP, and FN.

## Verification

```bash
python -m compileall -q src baselines
pytest -q
```

Third-party source used while creating the adapters lives under ignored
`external/` and is never committed. Two domain baselines require transparent
reconstructions: the DETR–U-Net authors did not release model code, while the
public EdgeAttNet repository contains a placeholder model file. These are
marked as reconstructions in the baseline documentation and results.
