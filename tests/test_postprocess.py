import numpy as np

from filament.postprocess import (
    accumulate_metric_record, finalize_metric_totals, mask_to_rle_object,
    metric_totals, pq_stats_rle, resolve_instances,
)


def test_pq_identity_and_empty_cases():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:8, 3:9] = True
    rle = mask_to_rle_object(mask)
    assert pq_stats_rle([rle], [rle])["pq"] == 1.0
    assert pq_stats_rle([], [rle])["fp"] == 1
    assert pq_stats_rle([rle], [])["fn"] == 1
    assert pq_stats_rle([], [])["pq"] == 0.0


def test_resolved_instances_never_overlap():
    masks = np.zeros((2, 16, 16), dtype=bool)
    masks[0, 2:10, 2:10] = True
    masks[1, 6:14, 6:14] = True
    instances = resolve_instances(masks, np.asarray([0.9, 0.8]), conf=0.1, min_area=1, duplicate_iou=0.95)
    assert len(instances) == 2
    assert not np.logical_and(instances[0], instances[1]).any()


def test_semantic_dice_and_iou_use_instance_unions():
    ground_truth = np.zeros((8, 8), dtype=bool)
    ground_truth[2:4, 2:4] = True
    prediction = np.zeros_like(ground_truth)
    prediction[2:3, 2:4] = True
    values = pq_stats_rle(
        [mask_to_rle_object(ground_truth)], [mask_to_rle_object(prediction)]
    )
    assert values["semantic_iou"] == 0.5
    assert np.isclose(values["semantic_dice"], 2.0 / 3.0)

    total = metric_totals()
    accumulate_metric_record(total, values)
    final = finalize_metric_totals(total)
    assert final["semantic_iou"] == 0.5
    assert np.isclose(final["semantic_dice"], 2.0 / 3.0)
    assert final["mean_semantic_iou"] == 0.5
    assert np.isclose(final["mean_semantic_dice"], 2.0 / 3.0)
