"""
Image I/O and geometric utilities.

- Load images from local path or HTTP/HTTPS URL.
- Rotate images by large angles (0 / 90 / 180 / 270).
- Perspective transform for text-line rectification.
- Padding and resizing helpers.
"""

import logging
from io import BytesIO
from typing import Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np

from .exceptions import ImageLoadError, PreprocessError

logger = logging.getLogger(__name__)


ImageSource = Union[str, bytes, np.ndarray]


def is_url(path: str) -> bool:
    """Return True if ``path`` looks like a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def load_image(source: ImageSource) -> np.ndarray:
    """
    Load an image as a BGR numpy array.

    Args:
        source: Local file path, HTTP(S) URL, raw bytes, or existing numpy array.

    Returns:
        BGR image with shape (H, W, 3).
    """
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    if isinstance(source, bytes):
        data = source
    elif isinstance(source, str) and is_url(source):
        try:
            import urllib.request

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.0"
                )
            }
            req = urllib.request.Request(source, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
        except Exception as exc:
            raise ImageLoadError(f"Failed to download image from {source}: {exc}") from exc
    elif isinstance(source, str):
        try:
            with open(source, "rb") as f:
                data = f.read()
        except Exception as exc:
            raise ImageLoadError(f"Failed to read image file {source}: {exc}") from exc
    else:
        raise ImageLoadError(f"Unsupported image source type: {type(source)}")

    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ImageLoadError(f"cv2.imdecode failed: {exc}") from exc

    if img is None:
        raise ImageLoadError("cv2.imdecode returned None; image may be corrupt")
    return img


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate an image counter-clockwise by a multiple of 90 degrees.

    Args:
        img: Input image.
        angle: One of 0, 90, 180, 270.

    Returns:
        Rotated image.
    """
    angle = angle % 360
    if angle == 0:
        return img
    if angle == 90:
        # cv2.ROTATE_90_COUNTER_CLOCKWISE == 2
        return cv2.rotate(img, 2)
    if angle == 180:
        # cv2.ROTATE_180 == 1
        return cv2.rotate(img, 1)
    if angle == 270:
        # cv2.ROTATE_90_CLOCKWISE == 0
        return cv2.rotate(img, 0)
    raise PreprocessError(f"Unsupported rotation angle: {angle}")


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2).

    Returns:
        Ordered array of shape (4, 2).
    """
    pts = pts.reshape(4, 2)
    if pts.dtype != np.float32:
        pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect = np.empty((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


# 缓存 dst 数组，避免每次调用 perspective_transform 时重复分配
_dst_cache: dict = {}


def perspective_transform(
    img: np.ndarray, pts: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Crop and rectify a quadrilateral region into a rectangle.

    Args:
        img: Source image.
        pts: Four corner points of the text box, shape (4, 2) or (N, 2).
        target_size: (width, height) of the output rectangle.

    Returns:
        Rectified BGR image.
    """
    if pts.shape[0] < 4:
        raise PreprocessError(" perspective_transform needs at least 4 points")

    src = order_points(pts[:4])
    # 从缓存获取 dst 数组，避免重复分配
    dst = _dst_cache.get(target_size)
    if dst is None:
        tw, th = target_size
        dst = np.array(
            [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
            dtype=np.float32,
        )
        _dst_cache[target_size] = dst
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, target_size)


def resize_with_aspect_ratio(
    img: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int, int, int]]:
    """
    Resize an image to fit inside ``target_size`` while preserving aspect ratio,
    then pad with ``pad_color``.

    Returns:
        (padded_image, (scale_h, scale_w), (pad_top, pad_bottom, pad_left, pad_right))
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return padded, (scale, scale), (pad_top, pad_bottom, pad_left, pad_right)


def normalize_image(
    img: np.ndarray,
    scale: float = 1.0 / 255.0,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Normalize a uint8 BGR image to float32 CHW tensor.

    Args:
        img: BGR image, uint8, shape (H, W, 3).
        scale: Initial scale factor (default 1/255).
        mean: Per-channel mean to subtract.
        std: Per-channel std to divide.

    Returns:
        Normalized float32 array with shape (3, H, W).
    """
    img = img.astype(np.float32)
    if mean and std:
        # 融合运算: (img * scale - mean) / std = img * (scale/std) + (-mean/std)
        # 减少到一次乘法和一次加法，比原始的 3 次数组操作少 2 次
        inv_std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        np.divide(scale, inv_std, out=inv_std)
        bias = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        np.negative(bias, out=bias)
        np.divide(bias, np.array(std, dtype=np.float32).reshape(1, 1, 3), out=bias)
        img *= inv_std
        img += bias
    else:
        img *= scale
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))


def draw_boxes(
    img: np.ndarray,
    boxes: list,
    texts: list = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw detection boxes and optional text on a copy of the image."""
    vis = img.copy()
    for idx, box in enumerate(boxes):
        pts = np.array(box, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(vis, [pts], True, color, thickness)
        if texts and idx < len(texts):
            cv2.putText(
                vis,
                str(texts[idx]),
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA,
            )
    return vis
