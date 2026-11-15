# Solar Filament Instance Segmentation — M2-Lite

Minimal training and evaluation code for the best architecture developed in
this project. The historical reference is `submission_m2_lite_single_fold2.csv`
with public Kaggle PQ `0.39`. Kaggle is no longer used by the current protocol;
new results are measured on one frozen annotated test split.

## Architecture

```text
grayscale solar image
  ├─ YOLOv8l-seg, 2048 px ─────────────── low-confidence instance seeds
  └─ U-Net++ / EfficientNet-B3, 768 tiles ─ dense semantic/topology map
                         │
                         ▼
              SIRA U-Net / ResNet34
       [intensity, CLAHE, seed, affinity, semantic]
                         │
                         ▼
       17-feature HGBR candidate quality score
                         │
                         ▼
       score filter + mask-IoU NMS + overlap resolver
                         │
                         ▼
                  instance masks
```

Only this chain is retained. There is no OOF/fold ensemble, graph selector,
fusion branch, Q2/Q3, P5, or Mask2Former experiment in the runnable code.

## Data protocol

One immutable, physical-image-grouped split is used:

- 65% train: fit YOLO, semantic model, SIRA, and quality regressor;
- 15% validation: choose checkpoints and decoder thresholds;
- 20% annotated test: final evaluation only.

All annotation records for the same `file_name` remain in one split. Five
seeds mean five independent training runs on this same split, not five folds.
Do not pick the best seed from test; report mean and standard deviation.

## Installation

Python 3.11 and an NVIDIA GPU are recommended. Install a CUDA-enabled PyTorch
build compatible with the server driver first, then:

```bash
conda activate finetune-sd
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Check CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
assert torch.cuda.is_available()
PY
```

## Data placement

Put the competition archive at any private path. Generated data and weights
also belong outside Git:

```bash
export PROJECT_ROOT=/path/to/solar-filament26
export ZIP_PATH=/path/to/filament-segmentation-2026.zip
export ARTIFACT_ROOT=/path/to/artifacts
export PYTHON=/path/to/conda/envs/finetune-sd/bin/python
export PYTHONPATH="$PROJECT_ROOT/src"
cd "$PROJECT_ROOT"
```

The archive must contain the MAGFiLO annotation JSON and train images. The
repository `.gitignore` excludes archives, extracted data, checkpoints,
prediction caches, and results.

## Prepare the one fixed split

Run exactly once and keep the resulting manifest:

```bash
bash scripts/prepare_fixed_split.sh 2>&1 | tee "$ARTIFACT_ROOT/logs/prepare.log"
```

Important outputs:

```text
$ARTIFACT_ROOT/results/fixed_split/manifest.csv
$ARTIFACT_ROOT/results/fixed_split/metadata.json
$ARTIFACT_ROOT/datasets/m2_lite_fixed/dataset.yaml
```

The manifest contains only `train`, `val`, and `test`; it contains no fold ID.
The split command refuses to overwrite a different frozen manifest.

## Train one seed

```bash
bash scripts/train_one_seed.sh 3020 0 \
  2>&1 | tee "$ARTIFACT_ROOT/logs/m2lite_s3020.log"
```

This trains and runs, in order: detector, semantic model, cached predictions,
SIRA, refined predictions, quality model, and validation evaluation. It does
not evaluate the final test split.

## Train five seeds

Prepare first, then launch seeds 3020–3024 across four GPUs:

```bash
bash scripts/launch_five_seeds_tmux.sh
tmux ls
tail -f "$ARTIFACT_ROOT/logs/m2lite_s3020.log"
```

GPU 0 runs seed 3024 sequentially after seed 3020; the other three GPUs each
run one seed. At most four training jobs are active at once.

## Final annotated test

Inspect validation results first:

```bash
cat "$ARTIFACT_ROOT/results/m2lite_s3020_val.json"
```

After the method and thresholds are frozen, evaluate each seed exactly once:

```bash
for seed in 3020 3021 3022 3023 3024; do
  bash scripts/evaluate_final_test.sh "$seed"
done
```

Results are written to `$ARTIFACT_ROOT/results/m2lite_s<seed>_test.json` and
contain pooled/global PQ, mean PQ, SQ, RQ, TP, FP, and FN. Global PQ uses
one-to-one instance matching at strict IoU `> 0.5`.

## Tests

```bash
export PYTHONPATH="$PROJECT_ROOT/src"
python -m compileall -q src
pytest -q
```

## Official five-seed baselines

The repository also contains dataset-only adapters for plain YOLOv8l-seg and
eight official instance-segmentation baselines. Their papers, exact upstream
configs, environment setup, launch commands, and mean±std summarizer are in
[docs/OFFICIAL_BASELINES.md](docs/OFFICIAL_BASELINES.md).

## Repository layout

```text
src/filament/   the single M2-Lite pipeline
scripts/        prepare, one-seed, five-seed, and final-test commands
tests/          split, metric, semantic-feature, and decoder tests
docs/           report/proposal material and historical results
```
