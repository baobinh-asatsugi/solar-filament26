# M2-Lite Solar Filament Instance Segmentation

> **COMP3020 Deliverable 3 working document**  
> Status: 15 November 2026  
> This file contains (A) a compact proposal-ready draft intended to fit the
> three-page limit after conversion to the course LaTeX template, and (B)
> technical notes for answering questions and writing later reports. Replace
> every bracketed placeholder before submission.

> **Frozen champion definition.** Throughout this document, “M2-Lite” means
> the exact pipeline that generated `submission_m2_lite_single_fold2.csv`
> (public PQ `0.39`): fold-2 `m1_d3` detector, fold-2 `m1_s2` semantic model,
> fold-2 `m2_sira` refiner, the final `m2_lite_refined` HGBR, and the fixed
> M2-Lite decoder. It does not refer to the later no-semantic retraining or to
> an M2/M3 ensemble.
>
> **Post-competition evaluation rule.** Kaggle is no longer used for model
> selection. We now freeze 20% of the 707 annotated physical images as a final
> test set and use only the remaining 80% for four-fold development CV. The
> previous five-fold OOF PQ (`0.432964`) and public score (`0.39`) are
> preliminary historical results; the new fixed-test result is pending a clean
> rerun from scratch.

---

# Part A — Proposal-ready draft

## Title and project information

**M2-Lite: Proposal-Guided Refinement and Quality-Aware Selection for Solar
Filament Instance Segmentation**

**Category:** Physical Sciences / Computer Vision  
**Authors:** [FULL NAME — VINUNI EMAIL], [FULL NAME — VINUNI EMAIL]  
**Mentor:** [MENTOR NAME AND AFFILIATION, OR “No external mentor”]

## Abstract

We study instance segmentation of solar filaments in full-disk H-alpha images,
where low contrast, thin curved morphology, overlapping predictions, and
annotation ambiguity make both localization and instance separation difficult.
We propose **M2-Lite**, a two-branch pipeline that combines a high-recall
YOLOv8l-seg proposal detector with semantic U-Net++ context, followed by a
seed-conditioned U-Net refiner and a lightweight learned quality selector. On
our previous five-fold grouped evaluation, M2-Lite obtained a pooled Panoptic
Quality (PQ) of **0.433**. Before the competition submission endpoint closed,
its fold-2 inference pipeline produced `submission_m2_lite_single_fold2.csv`,
which recorded **0.39 public PQ**,
improving over our detector baseline (`0.33`) and early semantic fusion system
(`0.35`). Our experiments further identify recognition and proposal selection,
rather than matched-mask geometry, as the main remaining bottleneck.

## 1. Introduction

Solar filaments are elongated clouds of relatively cool, dense plasma suspended
above the solar surface. Their morphology is relevant to solar activity
analysis, but manually tracing individual filaments in H-alpha observations is
slow and subjective. Automatic segmentation is challenging because filaments
may be faint, fragmented, branched, or close to sunspots and other dark solar
structures. Moreover, the task requires **instances**, not merely a binary
foreground map: two nearby filaments must remain separate, while duplicate
predictions for one filament must be suppressed.

Existing semantic segmentation approaches can delineate filament foreground,
but do not directly solve instance counting and separation. Conversely,
detector-based instance segmentation generates distinct proposals but often
misses thin extremities and produces duplicate or imperfect masks. End-to-end
set-prediction models are conceptually attractive, yet require substantial data
and optimization; our small custom set model achieved only `0.166` validation
PQ. We therefore investigate whether a modular proposal-and-refinement design
can exploit the strengths of both detection and dense segmentation while
remaining interpretable and feasible on the available data.

Our contributions are threefold. First, we construct a leakage-safe evaluation
protocol by grouping every annotation of the same physical image into one
fold. Second, we develop M2-Lite, which uses dense semantic evidence to refine
high-recall detector seeds and then learns a continuous instance-quality score
for duplicate-aware selection. Third, we perform stage-wise and oracle
diagnostics showing that the principal error lies in recognition quality—false
positives, false negatives, and incorrect candidate ranking—rather than the IoU
of already matched masks. This analysis provides a concrete direction for the
next stage: a visual crop-based quality model that can inspect image appearance
instead of relying only on scalar candidate features.

## 2. Related work

The MAGFiLO dataset provides manually annotated solar filaments in H-alpha
observations and includes instance segmentation information, filament spines,
and bounding boxes [1]. Solar-filament research has explored U-Net with
multi-scale and adversarial extensions [7], the channel-efficient Flat U-Net
[8], and EdgeAttNet, which injects learned edge evidence into bottleneck
attention to recover fine barbs [9]. These methods motivate structure-aware
processing for thin solar objects, but their semantic masks do not by
themselves identify separate instances.

U-Net combines a contracting path for contextual features with a symmetric
expanding path for precise localization [2]. UNet++ replaces direct skip
connections with nested dense pathways to reduce the semantic gap between its
encoder and decoder [3]. Because filaments are thin and curved, we also examine
centerline-aware supervision inspired by clDice, a topology-preserving loss for
tubular structures [4]. However, our ablations show that a simpler
foreground/centerline objective transfers better than adding all structural
losses. At the instance level, Mask2Former formulates segmentation as set
prediction using masked attention [5]. We tested a smaller custom
Mask2Former-style model, but it was not competitive on this dataset; hence the
current work focuses on a data-efficient detector–semantic–refiner pipeline.
Our main metric is Panoptic Quality, which factorizes mask Segmentation Quality
(SQ) and instance Recognition Quality (RQ) [6].

## 3. Method

### 3.1 Problem statement

Let an input image be \(x\in\mathbb{R}^{H\times W}\), and let its ground-truth
filament instances be \(G=\{g_j\}_{j=1}^{M}\), where each \(g_j\) is a binary
mask. The model predicts a variable-size set
\(P=\{(p_i,s_i)\}_{i=1}^{N}\) of masks and confidence scores. A predicted mask
and a ground-truth mask form a one-to-one true-positive match only when their
IoU exceeds 0.5. We seek parameters that maximize pooled PQ:

\[
\mathrm{PQ}=\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}
{|TP|+\tfrac12|FP|+\tfrac12|FN|}=\mathrm{SQ}\times\mathrm{RQ}.
\]

This objective penalizes missed filaments and duplicate/background masks in
addition to inaccurate mask boundaries.

### 3.2 M2-Lite architecture

This is the exact architecture used for the best submission, not an averaged
description of the larger M2/M3 research pipeline. Its frozen artifacts are:

```text
detector:  m1_d3_fold2/weights/best.pt
semantic:  m1_s2_fold2/best.pt
refiner:   m2_sira_fold2/best.pt
selector:  m2_lite_refined/m2_lite_quality.joblib
decoder:   m2_lite_refined/best_m2_lite.json
output:    submission_m2_lite_single_fold2.csv
public PQ: 0.39
```

```text
Input H-alpha image
   ├── YOLOv8l-seg at 2048 px ──────────► raw instance proposals
   └── U-Net++/EfficientNet-B3 ─────────► semantic probability map
                                            │
raw proposal + image + semantic map ────────┘
                       ▼
          seed-conditioned SIRA U-Net/ResNet34
                       ▼
             one refined mask per proposal
                       ▼
        17-feature continuous-IoU quality regressor
                       ▼
 threshold + minimum area + mask-IoU duplicate suppression
                       ▼
                  instance RLE masks
```

**High-recall proposals.** The fold-2 `m1_d3` YOLOv8l-seg detector processes
images at 2048 px.
At inference, a low confidence threshold (`0.03`), NMS IoU `0.75`, and maximum
of 200 detections preserve weak candidates for downstream evaluation. The
detector is trained with the standard Ultralytics box, classification,
distribution-focal, and segmentation objectives.

**Semantic context.** The fold-2 `m1_s2` tiled U-Net++ with an
ImageNet-pretrained EfficientNet-B3 encoder predicts filament foreground and
centerline evidence.
It uses grayscale, CLAHE-enhanced, and radial-position image information. Its
loss is a foreground BCE–Dice objective augmented by centerline BCE–Dice
(`0.1`) and skeleton-recall (`0.2`) terms. This branch provides dense context
but does not directly generate the final instance set.

**Seed-conditioned refinement.** For every detector proposal, the fold-2
`m2_sira` model extracts a 512-pixel local crop and applies a ResNet34 U-Net.
Its five inputs are grayscale
intensity, CLAHE intensity, the detector seed, seed-affinity evidence, and the
semantic foreground probability. A weighted BCE–Dice loss trains the network
to output one refined mask for the seeded instance. This design preserves
instance identity while recovering faint filament pixels.

**Quality-aware selection.** The final scorer stored by `m2_lite_refined` is
fitted after leakage-safe cross-fitted development on all OOF candidates. Each
refined candidate is represented by 17
features describing detector confidence, predicted probability, semantic
support, seed agreement, shape, area, and raw–refined consistency. A
HistGradientBoostingRegressor predicts the continuous IoU assigned by
one-to-one Hungarian matching, averaged across available annotation records.
Cross-fitting ensures that every validation prediction comes from a quality
model that did not train on that physical image. The final decoder uses score
threshold `0.15`, minimum area `300`, cluster IoU `0.30`, and duplicate IoU
`0.65`.

## 4. Experiments

### Dataset and protocol

We use the Solar Filament Segmentation 2026 competition data derived from
MAGFiLO. The training set contains **707 unique physical images** in our
manifest; some physical images have multiple annotation records. Across the
five-fold audit, these records contain **8,199 pooled ground-truth instance
occurrences**. Detector and refiner inference produces **20,417 OOF candidate
instances**. The hidden-label test set contains **180 images**, so test masks
can only be evaluated through the competition server.

We generate one immutable grouped split with seed 3020. Approximately 80% of
the 707 physical images form the development set and 20% form the final
annotated test set. All records from the same filename remain together, and
the outer split is stratified by image source and physical-image instance-count
quartile. The development portion is further divided into four grouped folds
for training, hyperparameter selection, and ablation. Final-test annotations
are not used for training, early stopping, threshold calibration, qualitative
selection, or model choice. The test set is evaluated once after the complete
method is frozen. The historical public submission used matching fold-2
detector, semantic, and SIRA checkpoints followed by the final all-OOF HGBR;
it is reported as preliminary evidence only.

### Metrics

The primary metric is pooled PQ. We additionally report SQ, RQ, TP, FP, FN,
per-fold PQ, semantic Dice/IoU, detector proposal recall, and quality-ranking
diagnostics. Mean image PQ is reported only as a secondary measure because it
weights images differently from the official pooled objective.

### Baselines and ablations

We compare: (1) YOLOv8l-seg alone; (2) early detector–semantic fusion; (3)
M2-Lite without semantic context; (4) the complete M2-Lite system; and (5) a
custom set-prediction model. We also test progressively more complicated
quality scorers and structural semantic losses to determine whether their
additional complexity improves end-to-end PQ.

### Implementation details

The detector is fine-tuned for 30 epochs at 2048 px with conservative
augmentation and batch size 1. The semantic U-Net++ is trained for 20 epochs on
768-pixel tiles with batch size 2. Five SIRA models are trained for 25 epochs
with 512-pixel crops, batch size 4, cosine learning-rate decay, and one held-out
physical-image fold each. Neural models use PyTorch and CUDA; the final HGBR is
trained with 220 boosting iterations, learning rate `0.04`, 15 leaf nodes,
minimum 30 samples per leaf, and L2 regularization `1.0`.

### Preliminary results

| Method | Previous grouped OOF PQ | Historical public PQ |
| --- | ---: | ---: |
| YOLOv8l-seg baseline | 0.354 | 0.33 |
| Best early semantic fusion | 0.418 | 0.35 |
| M2-Lite without semantic | 0.420 | 0.37 |
| **M2-Lite full** | **0.433** | **0.39** |
| Custom set-prediction model, fold 0 | 0.166 | not submitted |

Before the new fixed split was introduced, the complete M2-Lite model achieved
`PQ=0.432964` under five-fold grouped OOF evaluation. Its fold PQ values were
`0.4296`, `0.4354`, `0.4601`, `0.4215`, and `0.4180`. Removing semantic context
reduces OOF PQ by `0.0127` and public PQ by `0.02`, confirming that the dense
branch contributes complementary information.

These are preliminary results and will be rerun from scratch on the fixed
development/final-test protocol. Stage-wise oracle analysis on the previous OOF
predictions gives `PQ=0.4329`, `SQ=0.6833`, and `RQ=0.6336` for
the current model. Replacing only its learned quality scores with oracle scores
raises PQ to `0.4975`; oracle selection among refined masks reaches `0.5365`,
and oracle raw/refined selection reaches `0.5666`. The final oracle SQ remains
`0.6838`, almost unchanged from the current `0.6833`, whereas RQ rises to
`0.8287`. Thus, the next planned experiment is a compact visual ROI quality
model that directly observes image crops, raw masks, refined masks, and
semantic probabilities. We expect this model to reduce duplicate/background
FPs and choose between raw and refined variants more reliably than the current
17 scalar features.

## Acknowledgement and contributions

We thank **[MENTOR NAME]** for **[specific technical or scientific guidance]**.
**[MEMBER 1]** contributed **[data inspection, detector training, evaluation]**;
**[MEMBER 2]** contributed **[semantic/refinement models, diagnostics, writing]**;
and **[MEMBER 3, if applicable]** contributed **[specific work]**. All members
participated in experiment design, result analysis, and revision of the final
proposal. **[If no mentor: “This project currently has no external mentor.”]**

## References

[1] Ahmadzadeh et al., “A dataset of manually annotated filaments from H-alpha
observations,” *Scientific Data*, 2024.
https://doi.org/10.1038/s41597-024-03876-y

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for
Biomedical Image Segmentation,” *MICCAI*, 2015.
https://arxiv.org/abs/1505.04597

[3] Z. Zhou et al., “UNet++: A Nested U-Net Architecture for Medical Image
Segmentation,” 2018. https://arxiv.org/abs/1807.10165

[4] S. Shit et al., “clDice — A Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation,” *CVPR*, 2021.
https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html

[5] B. Cheng et al., “Masked-attention Mask Transformer for Universal Image
Segmentation,” *CVPR*, 2022. https://arxiv.org/abs/2112.01527

[6] A. Kirillov et al., “Panoptic Segmentation,” *CVPR*, 2019.
https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html

[7] D. Liu, W. Song, G. Lin, and H. Wang, “Solar Filament Segmentation
Based on Improved U-Nets,” *Solar Physics*, vol. 296, 2021.
https://doi.org/10.1007/s11207-021-01920-3

[8] G. Zhu, G. Lin, X. Yang, and C. Zeng, “Flat U-Net: An Efficient
Ultralightweight Model for Solar Filament Segmentation in Full-disk H-alpha
Images,” 2025. https://arxiv.org/abs/2502.07259

[9] V. Solomon, P. Martens, J. Liu, and R. Angryk, “EdgeAttNet: Towards
Barb-Aware Filament Segmentation,” 2025.
https://arxiv.org/abs/2509.02964

---

# Part B — Technical notes (do not paste all of this into the 3-page proposal)

## B.1 Exact meaning of “best model”

The best currently submitted minimal model is:

```text
submission_m2_lite_single_fold2.csv
Kaggle public PQ = 0.39
```

“Single fold 2” means:

- detector checkpoint: `runs/m1_d3_fold2/weights/best.pt`;
- semantic checkpoint: `runs/m1_s2_fold2/best.pt`;
- semantic-assisted refiner: `runs/m2_sira_fold2/best.pt`;
- raw test bank: `predictions/test_proposals/m1_d3_5fold/fold2`;
- refined test bank: `predictions/sira_test/m2/fold2`;
- quality model: `results/m2_lite_refined/m2_lite_quality.joblib`;
- decoder config: `results/m2_lite_refined/best_m2_lite.json`;
- no five-network test-time ensemble;
- the quality HGBR is still trained using leakage-safe OOF supervision before
  fitting the final model on all OOF candidates.

It must not be confused with:

```text
submission_m2_lite_no_semantic_single_fold2.csv
Kaggle public PQ = 0.37
```

The latter uses the retrained `m2_lite_sira_oof` branch created with
`--no-semantic-channel`. The champion uses the earlier `m2_sira_oof` bank with
semantic evidence.

The champion is also **not** any of the following: the M4-M2 ensemble, the
M4-M3 ensemble, P5.3 cluster-first, Q2, Q3, or Q3-R. Those runs are ablations
and diagnostics only.

## B.2 Complete M2-Lite data flow

### Step 1: detector proposal bank

For image \(x\), YOLO predicts raw candidates
\(R=\{(r_i,c_i)\}_{i=1}^{K}\), where \(r_i\) is a raw mask and \(c_i\) is its
detector confidence. Low-confidence inference intentionally makes \(K\) larger
than the expected true count. This improves recall but creates many duplicate
and background candidates.

Measured detector proposal recall:

| Mask-IoU threshold | Recall |
| ---: | ---: |
| 0.1 | 0.952433 |
| 0.3 | 0.925967 |
| 0.5 | 0.809245 |

### Step 2: semantic map

The U-Net++ branch produces a dense probability map \(S(x)\). It helps the
refiner determine whether the seed lies on plausible filament foreground and
how the structure continues beyond an imperfect detector mask.

Semantic audit:

| Dice | IoU | Skeleton recall |
| ---: | ---: | ---: |
| 0.638245 | 0.482117 | 0.839138 |

### Step 3: SIRA refinement

For every raw candidate \(r_i\), SIRA predicts:

\[
\hat r_i=f_\theta(x_{\mathrm{gray}},x_{\mathrm{CLAHE}},r_i,
a(r_i),S(x)),
\]

where \(a(r_i)\) is seed-affinity evidence. The refiner is local and
seed-conditioned, so it cannot recover a filament for which no detector seed
exists.

| SIRA diagnostic | Value |
| --- | ---: |
| Mean IoU change | +0.057351 |
| Fraction crossing from below to above IoU 0.5 | 0.167703 |
| Fraction damaged from above to below IoU 0.5 | 0.039281 |
| Identity-switch rate | 0.023902 |

### Step 4: quality target and features

For every annotation record, Hungarian assignment gives each GT to at most one
candidate. The assigned IoUs are averaged across records to produce a
continuous unique-IoU target \(q_i\). This prevents all duplicate proposals
from receiving the same positive label.

The HGBR input comprises:

```text
detector_score, mean_probability, q90_probability,
seed_agreement, semantic_support, boundary_uncertainty,
elongation, radial_position, area, components,
raw_area, refined_area, log_area_ratio, raw_refined_iou,
raw_box_fill, refined_box_fill, raw_refined_box_iou
```

The HGBR predicts \(\hat q_i\in[0,1]\). This model is preferred over Q2, Q3,
and Q3-R because it has the highest cross-fitted pooled PQ, even though Q3-R
has better candidate-level AUC and MAE.

### Step 5: deterministic set selection

Candidates are sorted by \(\hat q_i\). A candidate is retained if it passes
the score and area thresholds and is not suppressed as a duplicate by a
higher-scored overlapping candidate. The output set is encoded as competition
RLE strings.

## B.3 Training, validation, and test separation

### Frozen annotated split protocol v1

```text
master: results/fixed_annotated_split_v1/master_manifest.csv
development: results/fixed_annotated_split_v1/development_fold_manifest.csv
final test: results/fixed_annotated_split_v1/final_test_manifest.csv
seed: 3020
development/final-test ratio: 80/20
development CV folds: 4
group key: physical file_name
stratification: source + physical-image instance-count band
primary final metric: pooled/global PQ on the fixed annotated final test
development metric: pooled/global PQ over four development OOF folds
secondary metrics: SQ, RQ, TP, FP, FN, and per-fold development PQ
```

This new manifest is frozen before the clean rerun. It must not be regenerated
with a more favorable seed. All annotation records of one physical image remain
in one role. During development, methods are compared on the same four-fold OOF
predictions. The final-test CSV contains annotations for scoring but must not be
opened for model design, qualitative error selection, threshold tuning, or
checkpoint selection. The team acknowledges that the images originate from a
dataset previously explored during earlier experiments; the split nevertheless
provides a fixed, predeclared evaluation for the clean rerun rather than a
repeatedly changing validation subset.

```text
707 annotated physical images
        │
        ├── development: approximately 565 images
        │      ├── grouped development fold 0
        │      ├── grouped development fold 1
        │      ├── grouped development fold 2
        │      └── grouped development fold 3
        │
        └── fixed annotated final test: approximately 142 images

For development fold k:
  train/tune only with development data
  generate OOF predictions on development fold k

Concatenate development OOF predictions
  → fit/cross-fit selector
  → choose architecture and decoder on development only

Freeze the complete method
  → train final components using development data only
  → infer the fixed annotated final test once
  → compute pooled PQ/SQ/RQ/TP/FP/FN
```

The old 180-image competition test set had no accessible GT and is no longer
part of model selection. In the clean rerun, “final testing” means evaluation
on the fixed annotated 20% split. After that result is recorded, a separate
deployment model may be trained on all 707 images, but its score remains the
previously recorded fixed-test result rather than an in-sample measurement.

## B.4 Detailed results

### Main progression

| Run | OOF/validation PQ | Public PQ | Interpretation |
| --- | ---: | ---: | --- |
| YOLOv8l grouped baseline | 0.353937 | 0.33 | detector-only starting point |
| P2 D3 conservative, 2048 | 0.401513 | 0.33 | better proposals, no public gain alone |
| P2 D4 medoid, 2048 | 0.402090 | 0.33 | useful diversity, no public gain alone |
| P4 Fusion S0 | 0.411010 | 0.34 | semantic fusion helps |
| P4 Fusion S1 | 0.419020 | 0.34 | stronger semantic backbone |
| P4 Fusion S2 | 0.418111 | 0.35 | best early public fusion |
| P4 Fusion S3 | 0.376460 | 0.31 | structural complexity hurts |
| M4-M2 ensemble | 0.432520 | 0.39 | strong but complex |
| M4-M3 ensemble | 0.435865 | 0.39 | strong but complex |
| M2-Lite no semantic | 0.420264 | 0.37 | semantic ablation |
| **M2-Lite full** | **0.432964** | **0.39** | minimal current champion |

Validation values above were produced under different historical protocols in
some early runs. They show development progression but should not be used as a
strict statistical comparison unless the split is identical. The M2-Lite
semantic/no-semantic comparison uses the same five-fold framework.

### Champion OOF counts

```text
global PQ: 0.432964
mean PQ:   0.417082
SQ:        0.683268
RQ:        0.633666
TP:        4,826
FP:        2,207
FN:        3,373
selected instances: 4,308
```

The selected-instance count is counted once per physical prediction set,
whereas TP/FP/FN are pooled across annotation records; therefore those totals
are not expected to be identical.

### Per-fold OOF results

| Fold | PQ | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.429611 | 1026 | 515 | 683 |
| 1 | 0.435408 | 984 | 467 | 641 |
| 2 | **0.460132** | 1001 | 422 | 568 |
| 3 | 0.421541 | 911 | 398 | 738 |
| 4 | 0.418039 | 904 | 405 | 743 |

Fold 2 was the strongest historical held-out fold, but the spread also warns
that the dataset is small and heterogeneous. These values are preliminary
evidence only. The clean rerun must report four-fold development OOF during
model design and the locked 20% annotated test after the method is frozen; fold
2 must not be presented alone as unbiased final performance.

### Oracle stage decomposition

| Stage | PQ | SQ | RQ | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw masks ordered by detector score | 0.392279 | 0.662939 | 0.591728 | 5143 | 4041 | 3056 |
| Refined masks ordered by detector score | 0.423585 | 0.676967 | 0.625710 | 5286 | 3411 | 2913 |
| Current M2-Lite quality | **0.432901** | **0.683265** | 0.633576 | 4825 | 2207 | 3374 |
| Oracle quality, same decoder | 0.497497 | 0.677264 | 0.734569 | 5986 | 2113 | 2213 |
| Oracle refined set | 0.536475 | 0.678114 | 0.791128 | 5939 | 876 | 2260 |
| Oracle raw/refined set | **0.566641** | **0.683804** | **0.828659** | 6471 | 948 | 1728 |

The negligible SQ difference between current and final oracle masks shows why
the next contribution should target recognition/selection rather than simply a
larger segmentation backbone.

## B.5 Negative experiments worth reporting

| Experiment | Result | Scientific conclusion |
| --- | ---: | --- |
| Semantic removal | −0.0127 OOF PQ, −0.02 public | semantic context is necessary |
| clDice/boundary semantic S3 | 0.31 public | more structural losses can hurt transfer |
| Frangi/iterative semantic variants | no gain | vessel priors are not sufficient |
| Custom Mask2Former-style M6 | 0.1656 fold-0 PQ | small custom set model is undertrained/data-limited |
| Q2 two-head quality model | 0.432954 OOF | complex objective does not beat continuous HGBR |
| Q3 classifier × regressor | 0.411485 OOF | positive weighting produces too many FPs |
| Q3-R 51-feature HGBR | 0.427059 OOF | better AUC/MAE does not imply better set-level PQ |

These failures strengthen the proposal: they demonstrate controlled ablation
and lead to the specific hypothesis that visual evidence is missing from the
quality selector.

## B.6 Planned experiments for the final project

1. **Visual quality/action scorer:** concatenate an ROI image, raw mask,
   refined mask, and semantic probability; predict raw IoU, refined IoU, and
   unique-match probability using a small pretrained CNN.
2. **Strict cross-fitting:** train the visual scorer on four physical-image
   folds and score the fifth; repeat for all folds.
3. **Controlled ablation:** compare image-only, masks-only, image+masks, and
   image+masks+semantic inputs under the identical decoder.
4. **Raw/refined choice:** compare always-refined output with learned choice.
5. **Error analysis:** stratify FP/FN by contrast, area, elongation, radial
   position, and duplicate-cluster size.
6. **Final ensemble only after single-model improvement:** ensemble folds only
   if the same single-model method first improves OOF PQ.

Acceptance criteria:

```text
PQ <= 0.433: reject
0.433 < PQ < 0.445: weak evidence; inspect per-fold stability
PQ >= 0.445: meaningful improvement
PQ >= 0.450: successful quality-module upgrade
```

## B.7 Statements to avoid in the proposal

- Do not call the public test set a validation set; it has no accessible GT.
- Do not claim `0.39` is test accuracy; it is public PQ.
- Do not claim that Mask2Former itself was disproved; only the custom M6
  implementation was unsuccessful.
- Do not claim that M2-Lite is end-to-end; it is a modular pipeline.
- Do not compare OOF `0.433` and public `0.39` as though they were computed on
  the same images.
- Do not promise public PQ `0.7–0.8`; the current raw/refined oracle ceiling is
  approximately `0.567` on OOF candidates.
- Do not describe Q3-R as successful merely because its ROC-AUC/PR-AUC
  improved; its official PQ decreased.

## B.8 Three-page editing guidance

When moving Part A into the course LaTeX template:

1. keep the architecture diagram as one compact figure;
2. keep only the five-row main result table;
3. express the oracle analysis in two sentences, not another full table;
4. use six references unless space permits more;
5. keep acknowledgement outside the technical narrative but include explicit
   member contributions;
6. if the result table overflows, remove the custom set-model row before
   removing the semantic ablation;
7. do not paste Part B into the three-page submission.
