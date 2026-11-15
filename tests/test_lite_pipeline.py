import numpy as np

from filament.lite_pipeline import unique_iou_targets
from filament.postprocess import mask_to_rle_object


def test_unique_targets_enforce_one_prediction_per_ground_truth():
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    rle = mask_to_rle_object(mask)
    target = unique_iou_targets([rle, rle], [rle])
    assert sorted(target.tolist()) == [0.0, 1.0]
