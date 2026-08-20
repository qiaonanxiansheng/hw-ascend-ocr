"""Unit tests for advanced detection post-processing switches."""

import numpy as np

from ascend_ocr.config import DetConfig
from ascend_ocr.postprocess import postprocess_detection


def _make_hotspot(pred_h=64, pred_w=64):
    pred = np.zeros((1, 1, pred_h, pred_w), dtype=np.float32)
    pred[0, 0, 20:40, 20:40] = 0.9
    return pred


def test_box_type_minarearect():
    pred = _make_hotspot()
    cfg = DetConfig(thresh=0.3, box_thresh=0.3, box_type="minarearect")
    boxes = postprocess_detection(pred, (128, 128), (1.0, 1.0), cfg)
    assert len(boxes) >= 1
    assert boxes[0].shape[0] == 4


def test_box_type_poly():
    pred = _make_hotspot()
    cfg = DetConfig(thresh=0.3, box_thresh=0.3, box_type="poly")
    boxes = postprocess_detection(pred, (128, 128), (1.0, 1.0), cfg)
    assert len(boxes) >= 1


def test_sort_modes():
    pred = _make_hotspot()
    for mode in ("natural", "top2bottom", "left2right"):
        cfg = DetConfig(thresh=0.3, box_thresh=0.3, sort_mode=mode)
        boxes = postprocess_detection(pred, (128, 128), (1.0, 1.0), cfg)
        assert len(boxes) >= 1


def test_area_filter():
    pred = _make_hotspot()
    cfg = DetConfig(thresh=0.3, box_thresh=0.3, min_box_area=10000)
    boxes = postprocess_detection(pred, (128, 128), (1.0, 1.0), cfg)
    assert len(boxes) == 0


def test_nms():
    pred = _make_hotspot()
    cfg = DetConfig(thresh=0.3, box_thresh=0.3, nms_threshold=0.5)
    boxes = postprocess_detection(pred, (128, 128), (1.0, 1.0), cfg)
    # With one hotspot, NMS should not remove it.
    assert len(boxes) >= 1
