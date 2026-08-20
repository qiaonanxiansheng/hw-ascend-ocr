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

            subject = [(float(p[0]), float(p[1])) for p in points]
            clipper = pyclipper.PyclipperOffset()
            clipper.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = clipper.Execute(distance)
            if not expanded:
                return points.reshape(-1, 1, 2)
            expanded = np.array(expanded[0], dtype=np.float32)
            return expanded.reshape(-1, 1, 2)
        except Exception:
            pass

    # Fallback approximation.
    expanded = points + distance
    expanded -= expanded.mean(axis=0)
    expanded += points.mean(axis=0)
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
    shifted_box = box.reshape(-1, 1, 2)
    shifted_box[:, :, 0] -= int(xmin)
    shifted_box[:, :, 1] -= int(ymin)
    cv2.fillPoly(mask, [shifted_box], 1)
    roi = bitmap[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1]
    # mask 本身是 uint8(0/1)，numpy 可直接作为布尔索引，无需 .astype(bool)
    return float(roi[mask].mean())


def _nms(boxes: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    """Vectorized NMS based on bounding-box IoU."""
    if threshold < 0 or len(boxes) <= 1:
        return boxes
    n = len(boxes)
    # 一次性提取所有框的外接矩形
    bboxes = np.empty((n, 4), dtype=np.float32)
    for i, box in enumerate(boxes):
        pts = box.reshape(-1, 2)
        bboxes[i, 0] = pts[:, 0].min()
        bboxes[i, 1] = pts[:, 1].min()
        bboxes[i, 2] = pts[:, 0].max()
        bboxes[i, 3] = pts[:, 1].max()

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        # 计算 box[i] 与所有后续框的 IoU
        j_start = i + 1
        xx1 = np.maximum(bboxes[i, 0], bboxes[j_start:, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[j_start:, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[j_start:, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[j_start:, 3])
        inter_w = np.clip(xx2 - xx1, 0, None)
        inter_h = np.clip(yy2 - yy1, 0, None)
        inter = inter_w * inter_h
        union = areas[i] + areas[j_start:] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        # 抑制 IoU > threshold 的框
        suppressed = np.where(iou > threshold)[0] + j_start
        for j in suppressed:
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
    """Convert a polygon to the requested box representation.

    For minarearect/quad, returns 4 points in clockwise order:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = points.reshape(-1, 2).astype(np.float32)
    if box_type in ("minarearect", "quad"):
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        # Reorder to clockwise: TL, TR, BR, BL.
        s = box.sum(axis=1)
        diff = np.diff(box, axis=1).ravel()
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = box[np.argmin(s)]    # top-left (smallest x+y)
        ordered[2] = box[np.argmax(s)]    # bottom-right (largest x+y)
        ordered[1] = box[np.argmin(diff)] # top-right (smallest x-y)
        ordered[3] = box[np.argmax(diff)] # bottom-left (largest x-y)
        return ordered
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
    # 预计算缩放因子，避免在循环中重复创建
    scale_arr = np.array([src_w / pred_w, src_h / pred_h], dtype=np.float32)

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

        # Scale back to original image coordinates (单次矩阵乘法，无需 copy).
        scaled_box = box * scale_arr
        np.clip(scaled_box[:, 0], 0, src_w - 1, out=scaled_box[:, 0])
        np.clip(scaled_box[:, 1], 0, src_h - 1, out=scaled_box[:, 1])

        # Area filter (已经是 float32，无需重复转换).
        area = cv2.contourArea(scaled_box)
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
