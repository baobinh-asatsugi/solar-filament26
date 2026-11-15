# Solar Filament Segmentation 2026: Architecture, Runs, and Diagnostics

> Status: 2026-11-15  
> Purpose: single source of truth for the current system, every important
> experiment (including negative results), and the evidence-backed bottleneck.

## 1. Executive summary

The task is **instance segmentation of solar filaments**. The output is not a
single foreground mask: each physical filament must be returned as a separate
RLE instance. The official objective is pooled Panoptic Quality (PQ), so the
system must simultaneously:

1. find the correct number of filaments;
2. suppress false and duplicate instances;
3. keep distinct nearby filaments separate; and
4. produce masks with IoU above the strict matching threshold.

The best submitted minimal system is currently:

```text
YOLOv8l-seg proposals
        +
semantic U-Net++ context
        ↓
SIRA U-Net mask refinement
        ↓
17-feature HGBR quality score
        ↓
fixed threshold + mask-IoU duplicate suppression
        ↓
instance RLE submission
```

Its important results are:

| System | Validation / public result |
| --- | ---: |
| M2-Lite semantic, 5-fold OOF | **0.432964 pooled PQ** |
| `submission_m2_lite_single_fold2.csv` | **0.39 public PQ** |
| M2-Lite without semantic, 5-fold OOF | 0.420264 pooled PQ |
| `submission_m2_lite_no_semantic_single_fold2.csv` | 0.37 public PQ |
| Old M4-M2/M3 ensembles | 0.39 public PQ |
| Best early P4 fusion | 0.35 public PQ |

The champion audit shows that the main remaining error is **instance
selection/ranking (RQ), not mask geometry (SQ)**:

```text
Current M2-Lite:                  PQ 0.432901, SQ 0.683265, RQ 0.633576
Oracle quality, same decoder:     PQ 0.497497, SQ 0.677264, RQ 0.734569
Oracle refined set:              PQ 0.536475
Oracle raw+refined set:          PQ 0.566641, SQ 0.683804, RQ 0.828659
```

The current and full-oracle SQ values are almost identical (`0.6833` versus
`0.6838`). The large gap is therefore caused primarily by false positives,
false negatives, duplicate resolution, and choosing the wrong proposal/mask
variant.

Three attempts to replace the baseline quality scorer did not improve PQ:

| Quality scorer | OOF PQ | Result |
| --- | ---: | --- |
| Baseline 17-feature continuous HGBR | **0.432964** | KEEP |
| Q2 two-head MLP + ranking loss | 0.432954 | REJECT |
| Q3 51-feature classifier × regressor | 0.411485 | REJECT |
| Q3-R 51-feature continuous HGBR | 0.427059 | REJECT |

Q3 and Q3-R improved candidate-level ROC/PR metrics, but not official PQ.
This is strong evidence that scalar candidate features and candidate-level
objectives do not contain enough information to solve the set-level decision.

## 2. Task and data

### 2.1 Input and output

- **Input:** solar observation image.
- **Output:** zero or more filament instances, each encoded as an RLE mask.
- **Submission columns:** `filament_id`, `segmentation_rle`.
- A physical image can have multiple annotation records/annotators.
- Every split must group by physical filename. Splitting annotation records
  independently leaks the same physical image between train and validation.

The current five-fold diagnostic manifest contains:

- 707 unique physical images;
- 20,417 detector/refiner candidates;
- 8,199 pooled ground-truth instances in the champion audit;
- five grouped folds with no physical-image leakage.

The early legacy split had 142 validation images and 233 annotation records.
Its numbers are useful historically but must not be mixed with five-fold OOF
results.

### 2.2 Official metric

A prediction and GT instance are a true-positive match only when they are
one-to-one matched and their mask IoU is strictly above the competition
threshold (the implementation uses `IoU > 0.5`).

For all pooled records:

```text
PQ = sum(IoU of matched pairs) / (TP + 0.5 FP + 0.5 FN)
SQ = sum(IoU of matched pairs) / TP
RQ = TP / (TP + 0.5 FP + 0.5 FN)
PQ = SQ × RQ
```

Consequences:

- an extra duplicate costs `0.5 FP` even if another mask matched correctly;
- a missed filament costs `0.5 FN`;
- a visually plausible mask below IoU 0.5 contributes no IoU and becomes an
  FP while leaving an FN;
- optimizing Dice/AUC/MAE alone does not guarantee higher PQ;
- per-image mean PQ and pooled/global PQ are different objectives. Model
  selection uses pooled/global PQ.

## 3. Current champion architecture

### 3.1 End-to-end diagram

```text
Solar image
   │
   ├──────────────────────────────────────────────────┐
   │                                                  │
   ▼                                                  ▼
YOLOv8l-seg detector                            Semantic U-Net++
2048 px, low confidence                        EfficientNet-B3 encoder
raw instance proposals                         dense foreground context
   │                                                  │
   │ raw mask, score, geometry                        │ probability map
   └───────────────────────┬──────────────────────────┘
                           ▼
                  SIRA local mask refiner
                  U-Net, ResNet34 encoder
                  input = image evidence + seed + semantic
                           │
                           ▼
              refined mask for every detector proposal
                           │
                           ▼
               17 handcrafted candidate features
                           │
                           ▼
            HistGradientBoostingRegressor (HGBR)
            target = continuous unique refined IoU
                           │
                           ▼
        score threshold 0.15 + minimum area 300 px
        cluster IoU 0.30 + duplicate IoU 0.65
                           │
                           ▼
                   selected instance masks
                           │
                           ▼
                       RLE CSV
```

### 3.2 Detector branch

The detector is YOLOv8l-seg at 2048 px. The five-fold proposal bank is derived
from the D3-style conservative detector setup. It is deliberately run at low
confidence to prioritize proposal recall; the quality/decoder stages remove
weak candidates later.

Important detector settings used across the successful branch:

```text
architecture: YOLOv8l-seg
image size: 2048
conservative augmentation: no mosaic/mixup/translation; scale 0.1
horizontal and vertical flips: 0.5
test proposal confidence: approximately 0.03 for the low-confidence bank
NMS IoU: approximately 0.75
maximum detections: 200
```

The detector uses the standard Ultralytics segmentation objective: box,
classification, distribution focal loss, and mask segmentation losses.

Detector coverage measured by the champion audit:

| IoU threshold | Proposal recall |
| ---: | ---: |
| 0.1 | 0.952433 |
| 0.3 | 0.925967 |
| 0.5 | 0.809245 |

Interpretation: proposal coverage is strong at loose IoU, but about 19% of GT
instances do not have a detector proposal already above IoU 0.5. The detector
is not the first module to replace, but it becomes the ceiling after selection
is improved.

### 3.3 Semantic branch

The semantic model supplies dense image context; it does not directly produce
the final instances.

```text
architecture: U-Net++
encoder: EfficientNet-B3
tile size: 768
feature mode: gray/CLAHE/radial-position channels
outputs: foreground, centerline, boundary-related dense evidence
```

The successful semantic loss is approximately:

```text
Lsemantic = BCE+Dice(foreground)
          + 0.1 × BCE+Dice(centerline)
          + 0.2 × SkeletonRecall(foreground/centerline structure)
```

The more complicated boundary/clDice variants did not improve the end-to-end
score. For the successful S2 branch, boundary/clDice weights are effectively
zero.

Five-fold audit metrics:

| Dice | IoU | Skeleton recall |
| ---: | ---: | ---: |
| 0.638245 | 0.482117 | 0.839138 |

Semantic ablation:

| Pipeline | OOF PQ | Public PQ |
| --- | ---: | ---: |
| With semantic context | **0.432964** | **0.39** |
| Without semantic context | 0.420264 | 0.37 |
| Gain | **+0.012700** | **+0.02** |

Conclusion: semantic context is a proven useful component and should be kept.

### 3.4 SIRA refiner

Every detector proposal becomes a seed for a local binary refinement network.

```text
architecture: U-Net
encoder: ResNet34
crop size: 512
input channels:
  1. grayscale image
  2. CLAHE image
  3. detector seed mask
  4. seed affinity/distance evidence
  5. semantic foreground probability
output: one refined mask for the proposal
loss: weighted BCE + Dice
```

The refiner is intentionally applied proposal-by-proposal. It improves mask
shape but cannot create a completely missing detector proposal.

Champion audit:

| Diagnostic | Value |
| --- | ---: |
| Mean IoU change | +0.057351 |
| Gain crossing IoU 0.5 | 0.167703 |
| Damage crossing IoU 0.5 | 0.039281 |
| Identity-switch rate | 0.023902 |

The broader pipeline audit produced similar values: mean paired delta
`+0.052950`, gain crossing `0.203115`, damage crossing `0.054905`, and identity
switch `0.023118`.

Conclusion: SIRA works on average and must be kept. However, always choosing
the refined mask is suboptimal because a minority of proposals are damaged or
switch identity.

### 3.5 Baseline quality scorer

The champion quality module is deliberately small:

```text
model: HistGradientBoostingRegressor
loss: squared error
max_iter: 220
learning_rate: 0.04
max_leaf_nodes: 15
min_samples_leaf: 30
l2_regularization: 1.0
```

It uses 17 scalar features:

```text
detector_score
mean_probability
q90_probability
seed_agreement
semantic_support
boundary_uncertainty
elongation
radial_position
area
components
raw_area
refined_area
log_area_ratio
raw_refined_iou
raw_box_fill
refined_box_fill
raw_refined_box_iou
```

The target is not binary. It is the mean continuous IoU assigned by one-to-one
Hungarian matching across the annotation records for that physical image. One
GT cannot assign a high target to multiple duplicate candidates.

Five physical-image folds are used for cross-fitting. Thus every OOF quality
prediction is made by a scorer that was not trained on that physical image.

Baseline quality diagnostics:

| MAE | ROC-AUC at IoU>0.5 | PR-AUC | cluster top-1 | ranking regret |
| ---: | ---: | ---: | ---: | ---: |
| 0.173053 | 0.758254 | 0.448809 | 0.491620 | 0.211746 |

### 3.6 Deterministic decoder

The final M2-Lite decoder is intentionally simple:

1. discard masks below the score threshold;
2. discard masks smaller than the minimum area;
3. order candidates by predicted quality;
4. use mask-IoU clustering/suppression to remove duplicates;
5. output the remaining refined masks.

Champion configuration:

```text
score threshold: 0.15
cluster IoU: 0.30
minimum area: 300 px
duplicate IoU: 0.65
```

No graph neural network, pair head, action chain, or learned decoder is needed
for the submitted M2-Lite champion.

## 4. Run history

### 4.1 Detector baselines and P2

| Run / submission | Main change | Validation global PQ | Public PQ | Decision |
| --- | --- | ---: | ---: | --- |
| `yolov8l_seg_grouped_baseline` | YOLOv8l-seg, 1536 | 0.353937 best grid | 0.33 | baseline |
| `p2_d2_yolov8l_2048_standard` | 2048 standard augmentation | 0.393649 | 0.32 | reject |
| `p2_d3_yolov8l_2048_conservative` | 2048 conservative augmentation | 0.401513 | 0.33 | keep as proposal source |
| `p2_d4_yolov8l_2048_medoid` | 2048 medoid labels | 0.402090 | 0.33 | useful diversity only |
| `p2_ref_yolo11s_1024_4class` | smaller YOLO11, four classes | 0.324090 | not competitive | reject |

The 2048 models improved validation but not public PQ by themselves. This was
the first sign that thresholding/instance selection and validation-to-test
transfer mattered more than raw detector mAP.

### 4.2 Semantic and P4 fusion

| Run | Semantic model / strategy | Validation global PQ | Public PQ | Decision |
| --- | --- | ---: | ---: | --- |
| `p4_fusion_s0` | U-Net ResNet34 | 0.411010 | 0.34 | positive but not best |
| `p4_fusion_s1` | U-Net++ EfficientNet-B3 | 0.419020 | 0.34 | positive |
| `p4_fusion_s2` | skeleton-recall semantic | 0.418111 | **0.35** | best early fusion |
| `p4_fusion_s3` | clDice + boundary | 0.376460 | 0.31 | reject |
| S4 | Frangi vesselness + structural losses | did not beat the simpler branch | not selected | reject |
| S5 | iterative residual U-Net++ | did not produce a leaderboard gain | not selected | reject |

Important negative result: adding vesselness, boundary, clDice, and iterative
refinement complexity did not reliably improve PQ. Solar filaments are
elongated like vessels, but borrowing every medical-vessel prior did not solve
instance counting and duplicate selection.

### 4.3 M1–M4 five-fold pipeline

The next phase built five-fold detector proposals, SIRA refiners, and calibrated
quality models.

Five-fold pooled results before M4 calibration:

| Branch | Best pooled PQ | Parameters |
| --- | ---: | --- |
| M2 | 0.423709 | mask threshold 0.35, conf 0.25, min area 300 |
| M3 | **0.427543** | mask threshold 0.35, conf 0.25, min area 300 |

M4 quality calibration:

| Branch | Best M4 OOF PQ | Threshold / area |
| --- | ---: | --- |
| M2 | 0.432520 | quality 0.5, area 300 |
| M3 | **0.435865** | quality 0.5, area 300 |

Leaderboard:

| Submission | Public PQ |
| --- | ---: |
| `submission_m4_m2_ensemble_s2.csv` | 0.39 |
| `submission_m4_m3_ensemble_s2.csv` | 0.39 |

The displayed public scores tied after rounding. The user-observed ordering
favored M2 slightly, while M3 had the higher OOF PQ. This discrepancy is one
reason later work moved to single-model diagnosis before ensembling.

### 4.4 Proposal-oracle diagnostic

The low-confidence D3/D4 proposal bank was evaluated independently of learned
selection.

| Split | Existing-proposal oracle PQ | Perfect-mask proposal oracle PQ |
| --- | ---: | ---: |
| Five-fold/current data | 0.637044 | 0.977871 |
| Legacy validation | 0.610609 | 0.973252 |

Current-data proposal recall included `0.9567 @ IoU 0.1`, `0.9421 @ 0.3`, and
`0.8724 @ 0.5` for that combined proposal diagnostic. This established that a
large part of the early gap came from choosing/rejecting existing proposals,
not from complete absence of possible masks.

These are GT-assisted upper bounds, not deployable models.

### 4.5 P5 oracle supervision and learned selectors

P5.1 generated corrected oracle supervision for reject/raw/refined decisions.

```text
physical images: 707
candidates: 20,417
annotation-candidate rows: 33,308
oracle action counts:
  reject: 15,753
  raw:     1,855
  refined: 2,809
oracle global PQ: 0.572379
oracle TP/FP/FN: 6609 / 976 / 1590
```

P5.2 learned candidate actions and duplicate relations:

| Metric | Value |
| --- | ---: |
| Candidate selection ROC-AUC | 0.769563 |
| Candidate selection PR-AUC | 0.578269 |
| Candidate action accuracy | 0.718813 |
| Pair ROC-AUC | 0.863063 |
| Pair PR-AUC | 0.921772 |
| End-to-end PQ after duplicate graph | 0.420432 |

The pair head was strong, but the candidate selector was not strong enough.
Adding a good pair classifier to weak node decisions did not improve the final
system.

P5.3 used cluster-first selection and explicit raw/refined choice:

| Metric | Value |
| --- | ---: |
| Global PQ | **0.440819** |
| SQ / RQ | 0.677447 / 0.650706 |
| TP / FP / FN | 5069 / 2312 / 3130 |
| Selected candidates | 4508 |
| Raw / refined fraction | 0.1779 / 0.8221 |
| Match ROC-AUC / PR-AUC | 0.819192 / 0.821587 |
| Raw IoU MAE | 0.144330 |
| Refined IoU MAE | 0.157749 |
| Variant action accuracy | 0.581690 |

This is the best recorded OOF result from the complex P5 diagnostic pipeline,
but it was not established as a better public single-fold submission. It also
introduced substantial training and inference complexity. It is retained as
evidence that cluster-first decisions and raw/refined choice can help, not as
the current minimal champion.

### 4.6 Full pipeline stage audit

The unified audit evaluated 707 physical images and assigned the following
module status:

| Module | Status | Evidence |
| --- | --- | ---: |
| Detector | KEEP | recall@0.1 = 0.952433 |
| Semantic | KEEP after ablation | +0.0127 OOF PQ, +0.02 public |
| SIRA | KEEP, later gate raw/refined | mean IoU delta about +0.053 to +0.057 |
| Pair head | technically strong | PR-AUC = 0.921772 |
| Quality head | FIX | cluster top-1 was only 0.398090 in complex pipeline |
| Cluster decoder | simplify or improve | P5.3 PQ = 0.440819 |

Decoder ablation:

| Stage | Global PQ | SQ | RQ | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P5.2 candidate-only | 0.390372 | 0.687955 | 0.567438 | 4083 | 2109 | 4116 |
| P5.2 + duplicate graph | 0.420432 | 0.681139 | 0.617249 | 4813 | 2583 | 3386 |
| P5.3 paired-IoU only | 0.288348 | 0.695394 | 0.414654 | 3503 | 5194 | 4696 |
| P5.3 cluster-first | **0.440819** | 0.677447 | 0.650706 | 5069 | 2312 | 3130 |
| O1 current M4 | 0.432462 | 0.677860 | 0.637981 | 5271 | 3054 | 2928 |
| O4 raw+refined oracle set | **0.572379** | 0.683494 | 0.837430 | 6609 | 976 | 1590 |

### 4.7 M6 set-prediction experiment

M6 was a custom Mask2Former-style experiment, not the official pretrained
Mask2Former implementation.

```text
encoder: ResNet34
queries: 64
hidden dimension: 192
decoder layers: 4
attention heads: 8
loss/matching: class + mask + Dice set costs
fold tested: fold 0
```

Result:

| PQ | SQ | RQ | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.165625 | 0.611715 | 0.270756 | 618 | 2238 | 1091 |

The main failure was recognition/counting, not only mask quality. The custom
model was severely underdeveloped relative to a mature Mask2Former stack and
was rejected for the deadline. This run does **not** prove that official
Mask2Former can never work; it proves that replacing the mature proposal
pipeline with this small custom implementation was not competitive.

### 4.8 M2-Lite simplification

M2-Lite removed graph, pair, and action machinery and kept only:

```text
detector → semantic-assisted SIRA → one HGBR → deterministic IoU NMS
```

Result:

| Run | OOF PQ | Public PQ | Decision |
| --- | ---: | ---: | --- |
| `m2_lite_refined` | **0.432964** | **0.39** on single fold 2 | champion |
| `m2_lite_no_semantic` | 0.420264 | 0.37 on single fold 2 | semantic is necessary |

Per-fold M2-Lite semantic results:

| Fold | Global PQ |
| ---: | ---: |
| 0 | 0.429611 |
| 1 | 0.435408 |
| 2 | **0.460132** |
| 3 | 0.421541 |
| 4 | 0.418039 |

The best public submission used the fold-2 neural checkpoints and the final
quality model. “Single fold 2” means no test-time five-model ensemble; it does
not mean the quality model was evaluated without grouped cross-fitting.

### 4.9 Context-channel ablation

A simplified additional context experiment was tested on fold 2.

| Evaluation | PQ |
| --- | ---: |
| Same fixed config (`mask=.50`, `conf=.15`, `area=300`) | 0.407671 |
| Best post-hoc grid config | 0.425377 |

It did not beat the corresponding champion fold (`0.460132`) and did not
justify added architecture complexity. Reject.

## 5. Champion oracle audit

The champion audit decomposed the same OOF candidate population stage by
stage:

| Stage | Global PQ | Mean PQ | SQ | RQ | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 raw detector score | 0.392279 | 0.379186 | 0.662939 | 0.591728 | 5143 | 4041 | 3056 |
| S1 refined detector score | 0.423585 | 0.413422 | 0.676967 | 0.625710 | 5286 | 3411 | 2913 |
| S2 champion predicted quality | **0.432901** | 0.417024 | **0.683265** | 0.633576 | 4825 | 2207 | 3374 |
| S3 oracle quality, same decoder | 0.497497 | 0.497613 | 0.677264 | 0.734569 | 5986 | 2113 | 2213 |
| S4 refined oracle set, unresolved | 0.536475 | 0.530432 | 0.678114 | 0.791128 | 5939 | 876 | 2260 |
| S5 refined oracle set, resolved | 0.531630 | 0.526135 | 0.678813 | 0.783176 | 5819 | 842 | 2380 |
| S6 raw+refined oracle set, resolved | **0.566641** | 0.558972 | **0.683804** | **0.828659** | 6471 | 948 | 1728 |

Measured recoverable gap by module:

| Bottleneck | Recoverable pooled PQ |
| --- | ---: |
| Quality ranking | **+0.064596** |
| Raw-versus-refined choice | +0.035010 |
| Decoder/set selection | +0.034133 |
| SIRA improvement over detector-score ordering | +0.031305 already realized |

The categories overlap and therefore must not be summed as independent gains.

Key interpretation:

- SIRA is useful: S0 → S1 adds about `+0.0313` PQ.
- The learned quality model adds only about `+0.0093` beyond S1.
- Oracle quality with the same decoder would add `+0.0646` over the champion.
- Raw/refined choice matters because SIRA occasionally damages a good raw mask.
- Oracle SQ is essentially unchanged, so instance recognition/selection is the
  dominant error.

## 6. Quality-scorer experiments

### 6.1 Q2: two-task neural scorer

Q2 attempted:

```text
head 1: P(unique IoU > 0.5), weighted BCE
head 2: conditional expected IoU
score: P(match) × predicted IoU
additional within-cluster ranking margin
```

Result:

| Metric | Baseline HGBR | Q2 |
| --- | ---: | ---: |
| Global PQ | **0.432964** | 0.432954 |
| Gain | — | −0.000009 |
| MAE | **0.173053** | 0.199731 |
| ROC-AUC | **0.758254** | 0.756052 |
| PR-AUC | **0.448809** | 0.440914 |
| Cluster top-1 | **0.491620** | 0.477654 |
| Ranking regret | **0.211746** | 0.214601 |

Q2 reproduced the final PQ only by changing its operating threshold, while all
quality diagnostics worsened. It was rejected.

### 6.2 Q3: relative features plus classifier/regressor

Q3 expanded the feature set from 17 to 51 with:

- cluster size and overlap statistics;
- image candidate/cluster counts;
- within-cluster ranks;
- gaps to cluster maxima;
- deltas from cluster medians;
- relative area and medoid evidence.

It retained the binary classifier and conditional regressor formulation.

| Metric | Q3 |
| --- | ---: |
| Global PQ | **0.411485** |
| Gain over baseline | **−0.021479** |
| TP / FP / FN | 5204 / **3807** / 2995 |
| SQ / RQ | 0.680404 / 0.604765 |
| MAE | 0.169038 |
| ROC-AUC / PR-AUC | 0.792253 / 0.527578 |
| Cluster top-1 / regret | 0.500645 / 0.199221 |

Candidate-level discrimination improved, but positive weighting and the binary
objective produced too many false positives. Official PQ collapsed. Reject.

### 6.3 Q3-R: relative features with direct continuous regression

Q3-R was the clean ablation:

```text
same 51 features
same HGBR hyperparameters as the champion
one continuous unique-IoU target
no binary classifier
no class weighting
no probability × IoU score product
```

| Metric | Baseline | Q3-R |
| --- | ---: | ---: |
| Global PQ | **0.432964** | 0.427059 |
| Gain | — | **−0.005905** |
| TP | 4826 | 5038 |
| FP | **2207** | 2815 |
| FN | 3373 | **3161** |
| SQ | **0.683268** | 0.680345 |
| RQ | **0.633666** | 0.627710 |
| MAE | 0.173053 | **0.158588** |
| ROC-AUC | 0.758254 | **0.798681** |
| PR-AUC | 0.448809 | **0.538728** |
| Cluster top-1 | 0.491620 | **0.497636** |
| Ranking regret | 0.211746 | **0.197442** |

Q3-R found 212 additional TP but introduced 608 additional FP. The recall gain
could not pay the PQ penalty. Its best threshold (`0.142113`) was already swept,
so this was not merely a calibration-threshold oversight.

Per-fold Q3-R PQ:

| Fold | Q3-R | M2-Lite baseline |
| ---: | ---: | ---: |
| 0 | 0.423943 | **0.429611** |
| 1 | 0.425858 | **0.435408** |
| 2 | 0.449820 | **0.460132** |
| 3 | 0.421161 | **0.421541** |
| 4 | 0.414908 | **0.418039** |

Q3-R was worse on all five folds. Reject.

### 6.4 What Q2/Q3/Q3-R prove

The negative runs are informative:

1. increasing tabular feature count improves global candidate AUC but not the
   image-level instance set;
2. converting noisy continuous IoU into a rare binary label is harmful;
3. candidate-wise MAE/AUC/PR-AUC are insufficient acceptance metrics;
4. within-cluster top-1 remains around 50%, far below the oracle requirement;
5. scalar features cannot directly see whether a mask follows the actual dark
   filament, crosses a nearby structure, is broken, is over-thick, or merges
   two filaments;
6. future quality work must be accepted only by cross-fitted pooled PQ.

## 7. Evidence-backed bottleneck

### 7.1 Primary bottleneck: quality ranking and set recognition

Evidence:

- champion to oracle-quality gap: `+0.064596` PQ;
- quality cluster top-1: only `0.491620`;
- Q3-R improves candidate PR-AUC to `0.538728` but loses PQ;
- current SQ is within `0.00054` of the raw+refined oracle SQ;
- FP and FN, not matched-mask IoU, dominate the remaining denominator.

The problem is not “make every mask a little prettier.” It is:

```text
Which candidates correspond to unique real filaments?
Which candidates are duplicates or background?
Should the raw or refined version be used?
How many final instances should this image contain?
```

### 7.2 Secondary bottleneck: raw-versus-refined choice

The refiner has positive mean gain, but damage crossing and identity switching
are nonzero. Oracle raw/refined choice provides about `+0.035` recoverable PQ.
Always-refined is a good minimal baseline, not the optimal final policy.

### 7.3 Later bottleneck: proposal geometry and recall

Detector recall is high at loose IoU but only `0.8092` at IoU 0.5 in the
champion audit. Once selection approaches its oracle, improvements will require
better proposals, better mask refinement, or a mature set-prediction model.

### 7.4 Why a 0.6–0.8 public target is not supported yet

The deployable system is around `0.39` public. The measured raw+refined oracle
on current proposals is approximately `0.5666` OOF. That oracle already uses
GT to make impossible-at-test decisions. Therefore the current proposal bank
cannot justify a claim of `0.7–0.8` without a major new proposal/model family
and validation evidence. A realistic staged target is first `0.44–0.46` OOF
for a learned selector that transfers, then improve proposal recall/geometry.

## 8. Current keep/reject decisions

### Keep

- grouped physical-image five-fold protocol;
- YOLOv8l low-confidence proposal bank;
- semantic U-Net++ EfficientNet-B3 context;
- SIRA U-Net ResNet34 refinement;
- continuous unique-IoU supervision;
- simple deterministic mask-IoU decoder;
- baseline 17-feature HGBR until a replacement beats pooled PQ;
- SQ/RQ/TP/FP/FN and oracle diagnostics alongside PQ.

### Reject or archive

- YOLO11s 1024 four-class reference run;
- P4 S3 clDice/boundary configuration;
- Frangi/structural-loss complexity without PQ gain;
- iterative residual semantic complexity without PQ gain;
- custom M6 Mask2Former-style model;
- semantic removal;
- context-channel fold-2 experiment;
- Q2 two-head quality scorer;
- Q3 classifier × regressor;
- Q3-R 51-feature regressor;
- graph/pair/action chains as the default champion pipeline.

### Preserve only as research evidence

- P5.1 oracle supervision;
- P5.2 pair-head diagnostics;
- P5.3 cluster-first OOF result;
- champion and full pipeline audit artifacts;
- M6 negative result;
- Q2/Q3/Q3-R OOF predictions and diagnostics.

## 9. Recommended next experiment

The next justified quality experiment is a **minimal visual ROI scorer**, not
another tabular model:

```text
ROI image crop
+ raw mask
+ refined mask
+ semantic probability
        ↓
small pretrained CNN
        ├── predicted raw unique IoU
        ├── predicted refined unique IoU
        └── optional unique-match probability
        ↓
choose raw/refined + quality score
        ↓
the existing deterministic decoder
```

Minimal loss:

```text
L = SmoothL1(predicted raw IoU, raw unique IoU)
  + SmoothL1(predicted refined IoU, refined unique IoU)
  + optional 0.5 × BCE(unique match)
```

To preserve causal interpretability, begin without graph layers, pair heads,
or a learned global decoder. The visual model must be cross-fitted by physical
image and evaluated using the exact existing decoder.

Acceptance gates:

| Outcome | Decision |
| --- | --- |
| `PQ ≤ 0.433` | reject visual scorer |
| `0.433 < PQ < 0.445` | weak signal; inspect folds/calibration |
| `PQ ≥ 0.445` | meaningful improvement |
| `PQ ≥ 0.450` | quality-module objective achieved |

Only after a single visual scorer improves pooled OOF PQ should five-fold test
ensembling be reconsidered.

## 10. Reproducibility and comparison rules

1. Never compare legacy 142-image validation directly with five-fold OOF as if
   they were the same sample.
2. Never compare OOF PQ directly with Kaggle public PQ without labeling both.
3. Do not select a model using candidate AUC/MAE alone.
4. Every learned postprocessor must use physical-image cross-fitting.
5. Do not tune on test or leaderboard feedback beyond recording submissions.
6. Report at least: pooled PQ, SQ, RQ, TP, FP, FN, per-fold PQ, and public PQ if
   submitted.
7. Keep exact decoder parameters with every saved model.
8. Treat all GT-assisted oracle stages as diagnostic ceilings only.

## 11. Artifact map

Use environment-independent roots:

```text
${PROJECT_ROOT}/src/filament/             implementation
${PROJECT_ROOT}/scripts/                  launch/evaluation scripts
${PROJECT_ROOT}/configs/                  model configurations
${ARTIFACT_ROOT}/runs/                    checkpoints
${ARTIFACT_ROOT}/predictions/             proposal/refinement banks
${ARTIFACT_ROOT}/results/                 metrics, models, submissions
${ARTIFACT_ROOT}/logs/                    execution logs
```

Important result directories/files:

```text
results/m2_lite_refined/
results/m2_lite_no_semantic/
results/m2_lite_champion_audit/
results/m2_lite_q2/
results/m2_lite_q3_relative_hgb/
results/m2_lite_q3r_relative_regression/
results/p5_1_oracle_supervision_m2/
results/p5_2_graph_selector_m2/
results/p5_3_cluster_first_m2/
results/pipeline_stage_audit_m2/
```

The current minimal submission reference is:

```text
submission_m2_lite_single_fold2.csv
public PQ: 0.39
```

The early reproducible README may still describe P4 Fusion S2 (`0.35 public`).
That is a historical reproduction recipe, not the current champion method.
