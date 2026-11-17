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

Every launcher uses `scripts/run_with_gpu_limit.sh`. The default global gate is
at most two compute jobs per physical GPU, including jobs already visible to
`nvidia-smi` from another launcher. A queued job waits and starts automatically
when its assigned GPU drops below the limit. Override only when intentional:

```bash
export MAX_JOBS_PER_GPU=2
export GPU_POLL_SECONDS=15
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

Install and run the eight MMDetection methods in the same `finetune-sd`
environment used by the other experiments:

```bash
conda activate finetune-sd
export BASELINE_CONDA_ENV=finetune-sd
export PYTHON=/path/to/finetune-sd/bin/python
bash scripts/setup_official_baselines.sh
bash scripts/launch_official_baselines_group1.sh  # Mask/Cascade/HTC/SOLOv2
bash scripts/launch_official_baselines_group2.sh  # CondInst/QueryInst/Mask2Former/RTMDet-Ins
```

The setup script checks out pinned MMDetection v3.3.0 under the ignored
artifact directory and builds all eight adapted configs before training.
Detailed provenance and direct one-seed commands are in
[`docs/OFFICIAL_BASELINES.md`](docs/OFFICIAL_BASELINES.md).

## Checkpoint storage policy

Every trainer retains only its validation-selected best model checkpoint after
successful training. Periodic epoch snapshots and `last.pt`/`last.pth` are not
kept; small histories, metrics, configs, and plots remain available for audit.
To inspect historical artifacts before pruning them:

```bash
python scripts/prune_nonbest_checkpoints.py --artifact-root "$ARTIFACT_ROOT"
```

After reviewing the dry-run list, apply the cleanup with:

```bash
python scripts/prune_nonbest_checkpoints.py --artifact-root "$ARTIFACT_ROOT" --apply
```

## Summarize all 75 test runs

After five seeds of M2-Lite and 14 baselines have finished:

```bash
bash scripts/summarize_five_seed_results.sh
```

Outputs:

```text
$ARTIFACT_ROOT/results/five_seed_runs.csv
$ARTIFACT_ROOT/results/five_seed_summary.csv
$ARTIFACT_ROOT/results/five_seed_skipped.csv
```

The summary includes explicit `n_seeds`, the included seed IDs (for example,
`s3020|s3021`), and a
`complete_5_seeds` flag for every included method. Models with no valid result
are omitted. Missing/empty/invalid results are not
averaged; they are written to `five_seed_skipped.csv` together with whether a
checkpoint exists. The primary
metric is pooled/global PQ with one-to-one matching at IoU `> 0.5`; it also
reports mean PQ, SQ, RQ, TP, FP, and FN. Every evaluator also reports pooled
foreground Dice/IoU and mean-per-annotation-record Dice/IoU. Model selection
and decoder tuning remain based only on validation global PQ.

## CECS Slurm execution

The CECS launch path follows the cluster's `student` QoS and two-GPU cap. Paths
default relative to `$HOME`, so on the project account they resolve to:

```text
$HOME/ml26                 code
$HOME/datasets/magfilo     MAGFiLO data
$HOME/.venv                Python virtual environment
$HOME/ml26/outputs         checkpoints, caches, logs, and results
```

Prepare once from the login node (dependency installation requires network
access). The MMDetection checkout is pinned under the artifact root:

```bash
cd "$HOME/ml26"
source "$HOME/.venv/bin/activate"
export PROJECT_ROOT="$HOME/ml26"
export DATA_ROOT="$HOME/datasets/magfilo"
export ARTIFACT_ROOT="$HOME/ml26/outputs"
export VENV_PATH="$HOME/.venv"
bash scripts/prepare_cecs.sh
```

If the environment is already installed but `outputs/` is new or empty, prepare
only the fixed split and model input views (no package installation):

```bash
bash scripts/prepare_cecs_data.sh
```

The submitter validates the actual NumPy–PyTorch–OpenCV ABI bridge rather than
rejecting NumPy 2 by version number. It also snapshots the exact active virtual
environment on every submission:

```text
outputs/environment/requirements-cecs.lock.txt
```

Submit the proposed method plus all 14 baselines (75 runs):

```bash
bash scripts/submit_cecs_all.sh
```

Submit exactly one model (all five seeds) with any one of these commands:

```bash
# 1. Proposed M2-Lite
bash scripts/submit_cecs_all.sh m2_lite

# 2. YOLOv8l-seg
bash scripts/submit_cecs_all.sh yolov8l

# 3. U-Net++ + watershed
bash scripts/submit_cecs_all.sh unetpp_watershed

# 4-7. Solar/domain baselines
bash scripts/submit_cecs_all.sh fullres_unet
bash scripts/submit_cecs_all.sh edgeattnet
bash scripts/submit_cecs_all.sh maskrcnn_consensus
bash scripts/submit_cecs_all.sh detr_crop_unet

# 8-15. Official MMDetection baselines
bash scripts/submit_cecs_all.sh mask_rcnn
bash scripts/submit_cecs_all.sh cascade_mask_rcnn
bash scripts/submit_cecs_all.sh htc
bash scripts/submit_cecs_all.sh solov2
bash scripts/submit_cecs_all.sh condinst
bash scripts/submit_cecs_all.sh queryinst
bash scripts/submit_cecs_all.sh mask2former
bash scripts/submit_cecs_all.sh rtmdet_ins
```

Standard methods use Slurm array `0-1%2`: each task requests one GPU, four CPUs,
and 16 GB host RAM, and runs two seeds concurrently. Thus `3020+3021` run on
the first GPU and `3022+3023` on the second (four simultaneous seeds); seed
`3024` follows on the first. GPU-heavy methods (`cascade_mask_rcnn`, `htc`,
`queryinst`, and `mask2former`) instead use array `0-4%2`, one seed per GPU and
at most two simultaneous seeds. Method jobs are dependency-chained, so only one
method occupies the two-GPU quota at a time. To test selected methods:

```bash
bash scripts/submit_cecs_all.sh m2_lite yolov8l unetpp_watershed
bash scripts/submit_cecs_all.sh htc condinst
```

Monitor and inspect logs:

```bash
squeue --me -o '%.18i %.28j %.2t %.10M %.6D %R %b'
tail -f slurm_logs/ml26-m2_lite-*.out
tail -f "$PROJECT_ROOT/outputs/logs/cecs_m2_lite_s3020.log"
```

All seed runners remain idempotent: completed final results are skipped, while
M2-Lite resumes its detector, semantic cache, SIRA, refinement, selector, and
evaluation stages independently. CECS uses batch size one for memory-sensitive
trainers so that two seeds can share a GPU; source/evaluation resolution remains
2048 and model architectures, objectives, and schedules are unchanged.

After any subset finishes, manually regenerate the partial-safe aggregate:

```bash
export ARTIFACT_ROOT="$PROJECT_ROOT/outputs"
bash scripts/summarize_five_seed_results.sh
```

The aggregate never treats a missing checkpoint/result as score zero. It skips
that run, reports the exact number and IDs of included seeds, and records the
reason in `outputs/results/five_seed_skipped.csv`.

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
