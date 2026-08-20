"""Unit tests for preprocessing switches."""

import numpy as np

from yuntu_ascend_ocr.config import ClsConfig, DetConfig, RecConfig
from yuntu_ascend_ocr.preprocess import (
    preprocess_for_classification,
    preprocess_for_detection,
    preprocess_for_recognition,
)


def test_det_preprocess_pad():
    img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    cfg = DetConfig(resize_mode="pad", limit_side_len=320, limit_multiple=32)
    tensor, scale, shape = preprocess_for_detection(img, cfg)
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] % 32 == 0
    assert tensor.shape[3] % 32 == 0
    assert scale[0] > 0 and scale[1] > 0


def test_det_preprocess_stretch():
    img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    cfg = DetConfig(resize_mode="stretch", limit_side_len=320, limit_multiple=32)
    tensor, scale, shape = preprocess_for_detection(img, cfg)
    assert tensor.shape == (1, 3, shape[0], shape[1])


def test_det_normalize_modes():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    for mode in ("imagenet", "ppocr", "none", "custom"):
        cfg = DetConfig(normalize_mode=mode, limit_side_len=128, limit_multiple=32)
        tensor, _, _ = preprocess_for_detection(img, cfg)
        assert tensor.shape == (1, 3, 128, 128)


def test_cls_normalize_modes():
    img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
    for mode in ("imagenet", "ppocr", "none", "custom"):
        cfg = ClsConfig(normalize_mode=mode)
        tensor = preprocess_for_classification(img, cfg)
        assert tensor.shape == (1, 3, 224, 224)


def test_rec_preprocess_pad_modes():
    img = np.random.randint(0, 255, (40, 200, 3), dtype=np.uint8)
    for align in ("center", "left", "right"):
        cfg = RecConfig(resize_mode="fixed_height_pad", pad_align=align)
        tensor, shape = preprocess_for_recognition(img, cfg)
        assert tensor.shape == (1, 3, 48, 320)


def test_rec_preprocess_stretch():
    img = np.random.randint(0, 255, (40, 200, 3), dtype=np.uint8)
    cfg = RecConfig(resize_mode="fixed_size_stretch")
    tensor, shape = preprocess_for_recognition(img, cfg)
    assert tensor.shape == (1, 3, 48, 320)
