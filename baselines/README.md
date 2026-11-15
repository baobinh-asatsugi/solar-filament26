# Baseline adapters

This directory contains the dataset/evaluation adapters for 14 baselines. The
shared invariants are:

- full MAGFiLO v1.0 source images at 2048×2048;
- committed grouped train/validation/test manifest;
- seeds 3020–3024 on the identical split;
- checkpoint and threshold selection on validation only;
- pooled PQ at strict IoU `> 0.5` on the frozen test set.

The eight MMDetection methods retain upstream v3.3.0 model-zoo architecture,
loss, optimizer, augmentation, and schedule. `mmdet_adapter.py` changes only
the dataset, class count, seed/work directory, and image transforms needed for
the common 2048 canvas.

`fullres_unet` and `maskrcnn_consensus` are ports of released solar-filament
code. `edgeattnet` and `detr_crop_unet` are marked reconstructions because the
referenced public repositories do not contain runnable model implementations.
See `models.yaml` and `docs/OFFICIAL_BASELINES.md` for exact provenance.
