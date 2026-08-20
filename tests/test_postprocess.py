"""Unit tests for detection post-processing."""

import numpy as np

from ascend_ocr.config import DetConfig
from ascend_ocr.postprocess import postprocess_detection


def test_postprocess_detection_basic():
    # Simulate a 4x4 probability map with one hot spot.
    pred = np.zeros((1, 1, 64, 64), dtype=np.float32)
    pred[0, 0, 20:40, 20:40] = 0.9

    cfg = DetConfig(thresh=0.3, box_thresh=0.3, min_box_area=1.0)
    boxes = postprocess_detection(
        pred,
        src_shape=(128, 128),
        scale=(0.5, 0.5),
        cfg=cfg,
    )
    assert len(boxes) >= 1
    # Each box should have 4 points.
    for box in boxes:
        assert box.shape[0] == 4


def test_postprocess_no_detection():
    pred = np.zeros((1, 1, 32, 32), dtype=np.float32)
    cfg = DetConfig()
    boxes = postprocess_detection(pred, (64, 64), (0.5, 0.5), cfg)
    assert boxes == []
