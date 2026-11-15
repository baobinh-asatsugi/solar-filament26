import numpy as np
import torch

from baselines.unetpp_watershed import OfficialUNetPlusPlus, instances_from_probability


def test_unetpp_has_four_full_resolution_deep_supervision_heads():
    model = OfficialUNetPlusPlus().eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 1, 64, 64))
    assert len(outputs) == 4
    assert all(tuple(value.shape) == (1, 1, 64, 64) for value in outputs)


def test_watershed_separates_two_probability_peaks():
    yy, xx = np.mgrid[:64, :64]
    probability = np.maximum(
        np.exp(-((yy - 32) ** 2 + (xx - 20) ** 2) / 100),
        np.exp(-((yy - 32) ** 2 + (xx - 44) ** 2) / 100),
    ).astype(np.float32)
    instances = instances_from_probability(probability, threshold=.2, min_distance=10, min_area=20)
    assert len(instances) == 2
