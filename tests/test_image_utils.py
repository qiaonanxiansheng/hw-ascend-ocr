"""Unit tests for image utilities (no Ascend hardware required)."""

import numpy as np
import pytest

from yuntu_ascend_ocr.image_utils import (
    load_image,
    order_points,
    perspective_transform,
    rotate_image,
)


def test_load_image_from_array():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    loaded = load_image(img)
    assert loaded.shape == (100, 200, 3)


def test_load_image_from_bytes():
    import cv2

    img = np.zeros((50, 80, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    loaded = load_image(buf.tobytes())
    assert loaded.shape[:2] == (50, 80)


def test_rotate_image():
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    assert rotate_image(img, 0).shape == (100, 50, 3)
    assert rotate_image(img, 90).shape == (50, 100, 3)
    assert rotate_image(img, 180).shape == (100, 50, 3)
    assert rotate_image(img, 270).shape == (50, 100, 3)


def test_order_points():
    pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    ordered = order_points(pts)
    expected = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert np.allclose(ordered, expected)


def test_perspective_transform():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    pts = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float32)
    warped = perspective_transform(img, pts, (100, 30))
    assert warped.shape == (30, 100, 3)
