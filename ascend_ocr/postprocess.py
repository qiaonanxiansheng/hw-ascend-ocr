"""
Detection post-processing: DBNet-style probability map -> ordered text boxes.

Includes polygon un-clipping (pyclipper-based), rectangle filtering,
reading-order sorting and optional binary-map dilation.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .config import DetConfig

logger = logging.getLogger(__name__)


def _unclip(points: np.ndarray, unclip_ratio: float, use_pyclipper: bool = True) -> np.ndarray:
    """
    Expand a polygon by ``unclip_ratio``.

    Uses ``pyclipper`` (PaddleOCR's approach) when available and requested,
    otherwise falls back to a geometric approximation.
    """
    points = points.reshape(-1, 2).astype(np.float32)
    area = cv2.contourArea(points)
    length = cv2.arcLength(points, True)
    if length < 1e-6:
        return points.reshape(-1, 1, 2)

    # DBNet unclip distance formula.
    distance = area * (unclip_ratio ** 2 - 1) / length

    if use_pyclipper:
        try:
            import pyclipper

            scaling = 1.0
            subject = [(p[0] * scaling, p[1] * scaling) for p in points]
            clipper = pyclipper.PyclipperOffset()
            clipper.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = clipper.Execute(distance * scaling)
            if not expanded:
                return points.reshape(-1, 1, 2)
            expanded = np.array(expanded[0], dtype=np.float32) / scaling
            return expanded.reshape(-1, 1, 2)
        except Exception:
            pass

    # Fallback approximation.
    offset = np.ones(points.shape, dtype=np.float32) * distance
    expanded = points + offset
    expanded_center = expanded.mean(axis=0)
    original_center = points.mean(axis=0)
    expanded = expanded - expanded_center + original_center
    return expanded.reshape(-1, 1, 2)


def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """Mean probability inside the polygon ``box``."""
    h, w = bitmap.shape[:2]
    box = box.reshape(-1, 2).astype(np.int32)
    xmin, ymin = np.clip(box.min(axis=0), 0, [w - 1, h - 1])
    xmax, ymax = np.clip(box.max(axis=0), 0, [w - 1, h - 1])
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted_box = box - np.array([xmin, ymin])
    cv2.fillPoly(mask, [shifted_box.reshape(-1, 1, 2)], 1)
    roi = bitmap[ymin : ymax + 1, xmin : xmax + 1]
    return float(roi[mask.astype(bool)].mean())


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two rotated rectangles approximated by their bounding boxes."""
    p1 = box1.reshape(-1, 2)
    p2 = box2.reshape(-1, 2)
    x1_min, y1_min = p1.min(axis=0)
    x1_max, y1_max = p1.max(axis=0)
    x2_min, y2_min = p2.min(axis=0)
    x2_max, y2_max = p2.max(axis=0)

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    """Simple NMS based on bounding-box IoU."""
    if threshold < 0 or len(boxes) <= 1:
        return boxes
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            if _box_iou(boxes[i], boxes[j]) > threshold:
                keep[j] = False
    return [boxes[i] for i, k in enumerate(keep) if k]


def _order_boxes(
    boxes: List[np.ndarray], mode: str = "natural", line_factor: float = 0.5
) -> List[np.ndarray]:
    """
    Sort boxes in reading order.

    Args:
        boxes: List of 4+ point polygons.
        mode: "natural" | "top2bottom" | "left2right".
        line_factor: Used by "natural" to group boxes on the same text line.
    """
    if not boxes:
        return boxes

    def natural_key(box):
        pts = box.reshape(-1, 2)
        cy = pts[:, 1].mean()
        cx = pts[:, 0].mean()
        h = max(pts[:, 1].max() - pts[:, 1].min(), 1.0)
        row = round(cy / max(h * line_factor, 5.0))
        return (row, cx)

    def top2bottom_key(box):
        pts = box.reshape(-1, 2)
        return (pts[:, 1].min(), pts[:, 0].mean())

    def left2right_key(box):
        pts = box.reshape(-1, 2)
        return (pts[:, 0].min(), pts[:, 1].mean())

    key_map = {
        "natural": natural_key,
        "top2bottom": top2bottom_key,
        "left2right": left2right_key,
    }
    key = key_map.get(mode, natural_key)
    return sorted(boxes, key=key)


def _to_box(points: np.ndarray, box_type: str) -> np.ndarray:
    """Convert a polygon to the requested box representation."""
    pts = points.reshape(-1, 2).astype(np.float32)
    if box_type in ("minarearect", "quad"):
        rect = cv2.minAreaRect(pts)
        return cv2.boxPoints(rect)
    return pts  # "poly"


def postprocess_detection(
    pred: np.ndarray,
    src_shape: Tuple[int, int],
    scale: Tuple[float, float],
    cfg: DetConfig,
) -> List[np.ndarray]:
    """
    Convert a DBNet probability map to a list of text box polygons.

    Args:
        pred: Model output. Expected shape (1, 1, H, W) or (1, H, W).
        src_shape: (src_h, src_w) of the original image.
        scale: (scale_h, scale_w) from original -> model input (kept for API compat).
        cfg: Detection configuration.

    Returns:
        List of polygons in original image coords, ordered by cfg.sort_mode.
    """
    if pred.ndim == 4:
        pred = pred[0, 0]
    elif pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[:, :, 0]

    pred = pred.astype(np.float32)
    bitmap = (pred > cfg.thresh).astype(np.uint8)

    if cfg.use_dilate:
        kernel = np.ones((cfg.dilate_kernel_size, cfg.dilate_kernel_size), np.uint8)
        bitmap = cv2.dilate(bitmap, kernel, iterations=1)

    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    src_h, src_w = src_shape
    pred_h, pred_w = pred.shape[:2]

    for contour in contours[: cfg.max_candidates]:
        if contour.size < 8:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if points.shape[0] < 4:
            continue

        score = _box_score_fast(pred, points)
        if score < cfg.box_thresh:
            continue

        # Unclip polygon.
        unclipped = _unclip(points, cfg.unclip_ratio, cfg.use_pyclipper)
        box = unclipped.reshape(-1, 2)

        # Convert to requested box type.
        box = _to_box(box, cfg.box_type)

        # Scale back to original image coordinates.
        scaled_box = box.copy()
        scaled_box[:, 0] = box[:, 0] * (src_w / pred_w)
        scaled_box[:, 1] = box[:, 1] * (src_h / pred_h)

        # Clip to image bounds.
        scaled_box[:, 0] = np.clip(scaled_box[:, 0], 0, src_w - 1)
        scaled_box[:, 1] = np.clip(scaled_box[:, 1], 0, src_h - 1)

        # Area filter.
        area = cv2.contourArea(scaled_box.astype(np.float32))
        if area < cfg.min_box_area or area > cfg.max_box_area:
            continue

        # Aspect ratio filter.
        x_min, y_min = scaled_box.min(axis=0)
        x_max, y_max = scaled_box.max(axis=0)
        w, h = max(x_max - x_min, 1), max(y_max - y_min, 1)
        aspect = w / h
        if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
            continue

        boxes.append(scaled_box)

    # NMS and sorting.
    boxes = _nms(boxes, cfg.nms_threshold)
    boxes = _order_boxes(boxes, cfg.sort_mode, cfg.line_cluster_factor)
    return boxes


def boxes_to_min_area_rect(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """Convert arbitrary polygons to 4-point minimum-area rectangles."""
    rects = []
    for box in boxes:
        pts = box.reshape(-1, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        rect_pts = cv2.boxPoints(rect)
        rects.append(rect_pts)
    return rects
