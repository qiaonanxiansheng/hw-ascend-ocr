"""
Table structure recognition using YOLO model (detects both table and cell).

Pipeline:
    1. Preprocess image → 640×640 tensor
    2. YOLO inference → raw detections [1, 6, 8400]
    3. Post-process: parse → filter by class → NMS → scale to original coords
    4. Assign cells to table regions (center containment)
    5. Row/column clustering with adaptive thresholds
    6. Rowspan/colspan computation via edge alignment
    7. HTML table generation with occupancy matrix
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import TableConfig
from .model import AscendModel

logger = logging.getLogger(__name__)

# YOLO class IDs (must match model training)
CLASS_TABLE = 0
CLASS_CELL = 1


@dataclass
class DetectedBox:
    """A raw detected bounding box."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    class_id: int


@dataclass
class TableCell:
    """A detected table cell with structure information."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    rowstart: int = 0
    rowend: int = 0
    colstart: int = 0
    colend: int = 0
    lines: list = field(default_factory=list)

    @property
    def rowspan(self) -> int:
        return self.rowend - self.rowstart

    @property
    def colspan(self) -> int:
        return self.colend - self.colstart

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class TableStructure:
    """Complete table structure with cells and HTML."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    rows: int
    cols: int
    cells: List[TableCell]
    html: str = ""


def _box_contains_cx_cy(
    cx: float, cy: float,
    x1: int, y1: int, x2: int, y2: int,
) -> bool:
    """Check if a point (cx, cy) is inside a bounding box."""
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _compute_overlap_ratio(
    tx1: float, ty1: float, tx2: float, ty2: float,
    cx1: float, cy1: float, cx2: float, cy2: float,
) -> float:
    """Compute intersection area / text box area."""
    text_area = max(tx2 - tx1, 0) * max(ty2 - ty1, 0)
    if text_area <= 0:
        return 0.0
    ix1 = max(tx1, cx1)
    iy1 = max(ty1, cy1)
    ix2 = min(tx2, cx2)
    iy2 = min(ty2, cy2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    return inter / text_area


def _filter_repeat_boxes(boxes: List[DetectedBox], iou_threshold: float = 0.67) -> List[DetectedBox]:
    """
    Remove overlapping detections: if two boxes overlap > iou_threshold
    of the smaller box, keep the one with higher score.

    Mirrors the reference project's filter_repeat_objects logic.
    """
    if len(boxes) <= 1:
        return boxes

    # Sort by score descending
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    removed = set()

    for i, b1 in enumerate(sorted_boxes):
        if i in removed:
            continue
        x1a, y1a, x2a, y2a = b1.bbox
        area_a = max(x2a - x1a, 0) * max(y2a - y1a, 0)
        if area_a <= 0:
            continue

        for j in range(i + 1, len(sorted_boxes)):
            if j in removed:
                continue
            b2 = sorted_boxes[j]
            x1b, y1b, x2b, y2b = b2.bbox
            area_b = max(x2b - x1b, 0) * max(y2b - y1b, 0)
            if area_b <= 0:
                continue

            # Compute intersection
            ix1 = max(x1a, x1b)
            iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b)
            iy2 = min(y2a, y2b)
            inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)

            smaller_area = min(area_a, area_b)
            if smaller_area > 0 and inter > iou_threshold * smaller_area:
                removed.add(j)

    return [b for i, b in enumerate(sorted_boxes) if i not in removed]


class TableRecognizer:
    """YOLO-based table and cell detector with structure analysis."""

    def __init__(
        self,
        model_path: str,
        cfg: Optional[TableConfig] = None,
        device_id: int = 0,
        decrypt_callback: Optional[Callable[[str], bytes]] = None,
    ):
        if cfg is None:
            cfg = TableConfig()
        self.cfg = cfg

        logger.info("Loading table model from %s", model_path)
        self.model = AscendModel(
            model_path, device_id=device_id, decrypt_callback=decrypt_callback
        )

        # Log model info
        for i in range(self.model._input_num):
            shape = self.model.get_input_shape(i)
            logger.info("Table model input[%d] shape: %s", i, shape)
        for i in range(self.model._output_num):
            shape = self.model.get_output_shape(i)
            logger.info("Table model output[%d] shape: %s", i, shape)

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for YOLO table model.

        Returns:
            (tensor, scale_w, scale_h) where tensor is [1, 3, 640, 640].
        """
        src_h, src_w = img.shape[:2]
        target = self.cfg.input_size

        resized = cv2.resize(img, (target, target), interpolation=cv2.INTER_LINEAR)
        img_norm = resized.astype(np.float32) * self.cfg.scale
        tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        scale_w = src_w / target
        scale_h = src_h / target

        return tensor, scale_w, scale_h

    def _postprocess(
        self,
        output: np.ndarray,
        src_h: int,
        src_w: int,
        scale_w: float,
        scale_h: float,
    ) -> List[DetectedBox]:
        """
        Parse YOLO output, apply confidence filter and NMS.

        Output format: [1, 6, 8400] where 6 = [cx, cy, w, h, score, class_id]
        (YOLO11 outputs center-x, center-y, width, height in model input space)

        Returns:
            List of DetectedBox in original image coordinates.
        """
        logger.debug("[postprocess] raw output: shape=%s, dtype=%s", output.shape, output.dtype)

        # [1, 6, 8400] -> [8400, 6]
        if output.ndim == 3:
            output = output[0]
        output = output.T  # [8400, 6]

        boxes_raw = output[:, :4]  # cx, cy, w, h
        # Columns 4-5 are class scores (already probabilities, same as ultralytics ONNX output)
        # Confidence = max(class_scores), class = argmax(class_scores)
        class_scores = output[:, 4:]  # [N, 2] = [table_score, cell_score]

        # Debug: raw score distribution
        logger.debug(
            "[postprocess] raw class_scores(col4/table): min=%.4f, max=%.4f, mean=%.4f",
            float(class_scores[:, 0].min()), float(class_scores[:, 0].max()), float(class_scores[:, 0].mean()),
        )
        logger.debug(
            "[postprocess] raw class_scores(col5/cell): min=%.4f, max=%.4f, mean=%.4f",
            float(class_scores[:, 1].min()), float(class_scores[:, 1].max()), float(class_scores[:, 1].mean()),
        )
        logger.debug(
            "[postprocess] raw boxes(cxcywh): cx=[%.1f,%.1f] cy=[%.1f,%.1f] w=[%.1f,%.1f] h=[%.1f,%.1f]",
            float(boxes_raw[:, 0].min()), float(boxes_raw[:, 0].max()),
            float(boxes_raw[:, 1].min()), float(boxes_raw[:, 1].max()),
            float(boxes_raw[:, 2].min()), float(boxes_raw[:, 2].max()),
            float(boxes_raw[:, 3].min()), float(boxes_raw[:, 3].max()),
        )

        # Model outputs already-sigmoided probabilities; take max as confidence
        scores = np.amax(class_scores, axis=1)  # max of [p_table, p_cell]
        class_ids = np.argmax(class_scores, axis=1)  # 0=table, 1=cell

        # Score percentiles for debugging
        for pct in [50, 75, 90, 95, 99]:
            val = float(np.percentile(scores, pct))
            logger.debug("[postprocess] score p%d=%.6f", pct, val)

        # Debug: class distribution at different score levels
        for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
            mask_t = scores > thresh
            if mask_t.sum() > 0:
                cid_t = class_ids[mask_t]
                n_table = int(np.sum(cid_t == 0))
                n_cell = int(np.sum(cid_t == 1))
                logger.debug("[postprocess] score>%.2f: total=%d, table(class0)=%d, cell(class1)=%d",
                           thresh, int(mask_t.sum()), n_table, n_cell)

        # Filter by confidence
        mask = scores > self.cfg.score_threshold
        n_pass_conf = int(mask.sum())
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        logger.debug(
            "[postprocess] confidence filter: threshold=%.4f, passed=%d/%d",
            self.cfg.score_threshold, n_pass_conf, len(output),
        )

        if len(scores) == 0:
            logger.info("[postprocess] no detections above confidence threshold %.4f", self.cfg.score_threshold)
            return []

        # Convert cxcywh -> xyxy for NMS
        cx = boxes_raw[:, 0]
        cy = boxes_raw[:, 1]
        w = boxes_raw[:, 2]
        h = boxes_raw[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        logger.debug(
            "[postprocess] after cxcywh->xyxy: x1=[%.1f,%.1f] y1=[%.1f,%.1f] x2=[%.1f,%.1f] y2=[%.1f,%.1f]",
            float(x1.min()), float(x1.max()),
            float(y1.min()), float(y1.max()),
            float(x2.min()), float(x2.max()),
            float(y2.min()), float(y2.max()),
        )

        # NMS needs xywh format
        boxes_xywh = [[float(v) for v in row] for row in np.stack([x1, y1, w, h], axis=1)]
        scores_list = [float(s) for s in scores]
        class_ids_list = [int(c) for c in class_ids]  # already 0 or 1 from argmax

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores_list,
            self.cfg.score_threshold,
            self.cfg.nms_threshold,
        )

        logger.debug("[postprocess] NMS: input=%d, output=%d", len(boxes_xywh), len(indices))

        if len(indices) == 0:
            logger.info("[postprocess] NMS removed all detections")
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        # Scale coordinates back to original image space
        results = []
        for idx in indices:
            bx1 = float(boxes_xyxy[idx, 0]) * scale_w
            by1 = float(boxes_xyxy[idx, 1]) * scale_h
            bx2 = float(boxes_xyxy[idx, 2]) * scale_w
            by2 = float(boxes_xyxy[idx, 3]) * scale_h
            score = float(scores[idx])
            class_id = class_ids_list[idx]

            x1_out = max(0, int(round(bx1)))
            y1_out = max(0, int(round(by1)))
            x2_out = min(src_w, int(round(bx2)))
            y2_out = min(src_h, int(round(by2)))

            if x2_out <= x1_out or y2_out <= y1_out:
                logger.debug("[postprocess] skip degenerate box: [%d,%d,%d,%d]", x1_out, y1_out, x2_out, y2_out)
                continue

            results.append(DetectedBox(
                bbox=(x1_out, y1_out, x2_out, y2_out),
                score=score,
                class_id=class_id,
            ))

        # Debug: log all final detections
        for i, d in enumerate(results[:20]):
            logger.debug(
                "[postprocess] det[%d]: class=%d, score=%.4f, bbox=%s",
                i, d.class_id, d.score, d.bbox,
            )
        if len(results) > 20:
            logger.debug("[postprocess] ... and %d more detections", len(results) - 20)

        if len(results) > self.cfg.max_detections:
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[: self.cfg.max_detections]

        return results

    def detect(self, img: np.ndarray) -> Tuple[List[TableCell], List[DetectedBox]]:
        """
        Detect table regions and cells in an image.

        Args:
            img: BGR image.

        Returns:
            (cells, table_boxes) where:
            - cells: List of TableCell objects (class_id == CLASS_CELL)
            - table_boxes: List of DetectedBox for table regions (class_id == CLASS_TABLE)
        """
        src_h, src_w = img.shape[:2]
        logger.info("Table detect: image size=%dx%d", src_w, src_h)

        t0 = time.perf_counter()

        # Preprocess
        tensor, scale_w, scale_h = self._preprocess(img)
        t_pre = time.perf_counter() - t0
        logger.info("Table preprocess: %.1fms, tensor=%s, scale=(%.4f, %.4f)", t_pre * 1000, tensor.shape, scale_w, scale_h)

        # Inference
        t1 = time.perf_counter()
        outputs = self.model.infer([tensor])
        t_infer = time.perf_counter() - t1
        logger.info("Table inference: %.1fms, outputs=%d", t_infer * 1000, len(outputs))
        for i, out in enumerate(outputs):
            logger.debug("  output[%d]: shape=%s, dtype=%s, min=%.6f, max=%.6f",
                        i, out.shape, out.dtype, float(out.min()), float(out.max()))

        # Post-process
        t2 = time.perf_counter()
        detections = self._postprocess(outputs[0], src_h, src_w, scale_w, scale_h)
        t_post = time.perf_counter() - t2
        logger.info("Table postprocess: %.1fms, %d detections", t_post * 1000, len(detections))

        # Separate by class
        cell_boxes = [d for d in detections if d.class_id == CLASS_CELL]
        table_boxes = [d for d in detections if d.class_id == CLASS_TABLE]
        other_boxes = [d for d in detections if d.class_id not in (CLASS_CELL, CLASS_TABLE)]

        logger.info(
            "Table detect results: total=%d, cells(class=%d)=%d, tables(class=%d)=%d, other=%d",
            len(detections), CLASS_CELL, len(cell_boxes), CLASS_TABLE, len(table_boxes), len(other_boxes),
        )

        # Filter duplicate cells
        cell_boxes = _filter_repeat_boxes(cell_boxes)

        logger.info("After filter_repeat: %d cells, %d tables", len(cell_boxes), len(table_boxes))

        cells = [
            TableCell(bbox=d.bbox, score=d.score)
            for d in cell_boxes
        ]
        cells.sort(key=lambda c: (c.bbox[1], c.bbox[0]))

        return cells, table_boxes

    def release(self) -> None:
        """Release model resources."""
        if self.model is not None:
            self.model.release()
            logger.info("Table model released")


# ---------------------------------------------------------------------------
# Cell-to-table assignment
# ---------------------------------------------------------------------------

def assign_cells_to_tables(
    cells: List[TableCell],
    table_boxes: List[DetectedBox],
) -> Dict[int, List[TableCell]]:
    """
    Assign cells to table regions based on center containment.

    If no table_boxes are provided, all cells are assigned to table index 0.

    Args:
        cells: Detected TableCell objects.
        table_boxes: Detected table region boxes.

    Returns:
        Dict mapping table index to list of cells.
    """
    if not table_boxes:
        # No table regions detected: treat all cells as one table
        return {0: list(cells)} if cells else {}

    region_cells: Dict[int, List[TableCell]] = {}

    for cell in cells:
        cx = (cell.bbox[0] + cell.bbox[2]) / 2
        cy = (cell.bbox[1] + cell.bbox[3]) / 2

        best_idx = -1
        for ti, tb in enumerate(table_boxes):
            if _box_contains_cx_cy(cx, cy, *tb.bbox):
                best_idx = ti
                break

        if best_idx >= 0:
            region_cells.setdefault(best_idx, []).append(cell)

    return region_cells


# ---------------------------------------------------------------------------
# Row / Column clustering
# ---------------------------------------------------------------------------

def find_rows(
    cells: List[TableCell],
    height_threshold: float,
) -> List[List[TableCell]]:
    """
    Cluster cells into rows based on top-edge Y coordinate alignment.
    Also computes rowstart/rowend for each cell (rowspan).
    """
    if not cells:
        return []

    cells.sort(key=lambda c: c.bbox[1])

    rows: List[List[TableCell]] = []
    current_row: List[TableCell] = [cells[0]]

    for cell in cells[1:]:
        cell_top = cell.bbox[1]
        in_current_row = any(
            abs(cell_top - rc.bbox[1]) < height_threshold
            for rc in current_row
        )

        if in_current_row:
            is_duplicate = any(
                _box_contains_cx_cy(
                    (cell.bbox[0] + cell.bbox[2]) / 2,
                    (cell.bbox[1] + cell.bbox[3]) / 2,
                    *rc.bbox,
                )
                for rc in current_row
            )
            if not is_duplicate:
                current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]

    rows.append(current_row)

    # Assign rowstart
    for row_idx, row in enumerate(rows):
        for cell in row:
            cell.rowstart = row_idx

    # Compute rowend
    for cell in cells:
        cell_bottom = cell.bbox[3]
        aligned = [
            other for other in cells
            if abs(cell_bottom - other.bbox[1]) < height_threshold and other is not cell
        ]
        if aligned:
            aligned.sort(key=lambda c: abs(cell_bottom - c.bbox[1]))
            cell.rowend = aligned[0].rowstart
        else:
            cell.rowend = len(rows)

    return rows


def find_cols(
    cells: List[TableCell],
    width_threshold: float,
) -> List[List[TableCell]]:
    """
    Cluster cells into columns based on left-edge X coordinate alignment.
    Also computes colstart/colend for each cell (colspan).
    """
    if not cells:
        return []

    cells.sort(key=lambda c: c.bbox[0])

    cols: List[List[TableCell]] = []
    current_col: List[TableCell] = [cells[0]]

    for cell in cells[1:]:
        cell_left = cell.bbox[0]
        in_current_col = any(
            abs(cell_left - cc.bbox[0]) < width_threshold
            for cc in current_col
        )

        if in_current_col:
            is_duplicate = any(
                _box_contains_cx_cy(
                    (cell.bbox[0] + cell.bbox[2]) / 2,
                    (cell.bbox[1] + cell.bbox[3]) / 2,
                    *cc.bbox,
                )
                for cc in current_col
            )
            if not is_duplicate:
                current_col.append(cell)
        else:
            cols.append(current_col)
            current_col = [cell]

    cols.append(current_col)

    # Assign colstart
    for col_idx, col in enumerate(cols):
        for cell in col:
            cell.colstart = col_idx

    # Compute colend
    for cell in cells:
        cell_right = cell.bbox[2]
        aligned = [
            other for other in cells
            if abs(cell_right - other.bbox[0]) < width_threshold and other is not cell
        ]
        if aligned:
            aligned.sort(key=lambda c: abs(cell_right - c.bbox[0]))
            cell.colend = aligned[0].colstart
        else:
            cell.colend = len(cols)

    return cols


# ---------------------------------------------------------------------------
# Text-to-cell assignment
# ---------------------------------------------------------------------------

def assign_text_to_cells(
    cells: List[TableCell],
    ocr_results: list,
    overlap_threshold: float = 0.5,
) -> list:
    """
    Assign OCR text lines to cells. Returns unassigned (outside) lines.

    Text is assigned to the cell whose bbox contains the text center.
    Fallback: assign to the cell with highest overlap ratio.
    """
    cell_aabbs = [
        (c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3]) for c in cells
    ]

    assigned = set()
    for ocr_result in ocr_results:
        pts = ocr_result.box.reshape(-1, 2)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

        best_cell = None
        best_overlap = 0.0

        for cell, (cx1, cy1, cx2, cy2) in zip(cells, cell_aabbs):
            if _box_contains_cx_cy(cx, cy, cx1, cy1, cx2, cy2):
                best_cell = cell
                break
            tx1, ty1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            tx2, ty2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            overlap = _compute_overlap_ratio(tx1, ty1, tx2, ty2, cx1, cy1, cx2, cy2)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cell = cell

        if best_cell is not None:
            best_cell.lines.append(ocr_result)
            assigned.add(id(ocr_result))

    # Sort lines within each cell by reading order
    for cell in cells:
        cell.lines.sort(key=lambda r: (r.box.reshape(-1, 2)[:, 1].mean(),
                                        r.box.reshape(-1, 2)[:, 0].mean()))

    return [r for r in ocr_results if id(r) not in assigned]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _build_cell_text(lines: list) -> str:
    """Build HTML content for a cell from its text lines."""
    if not lines:
        return "&nbsp;"

    parts = [lines[0].text]
    for line in lines[1:]:
        prev_line = lines[len(parts) - 1]
        prev_pts = prev_line.box.reshape(-1, 2)
        curr_pts = line.box.reshape(-1, 2)
        prev_h = prev_pts[:, 1].max() - prev_pts[:, 1].min()
        curr_h = curr_pts[:, 1].max() - curr_pts[:, 1].min()
        y_overlap = (min(prev_pts[:, 1].max(), curr_pts[:, 1].max())
                     - max(prev_pts[:, 1].min(), curr_pts[:, 1].min()))
        min_h = max(min(prev_h, curr_h), 1)
        if y_overlap > min_h * 0.5:
            parts.append("&nbsp;&nbsp;" + line.text)
        else:
            parts.append("<br />" + line.text)

    return "".join(parts)


def generate_html(table: TableStructure) -> str:
    """Generate HTML table using occupancy matrix algorithm."""
    rows = table.rows
    cols = table.cols
    cells = table.cells

    if rows == 0 or cols == 0:
        return "<html><body><table border='1'><tbody><tr><td>empty</td></tr></tbody></table></body></html>"

    occupied = [[False] * cols for _ in range(rows)]
    for cell in cells:
        for r in range(cell.rowstart, min(cell.rowend, rows)):
            for c in range(cell.colstart, min(cell.colend, cols)):
                occupied[r][c] = True

    html = "<html><body><table border='1'><tbody>"
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            if occupied[i][j]:
                cell_at_pos = None
                for cell in cells:
                    if cell.rowstart == i and cell.colstart == j:
                        cell_at_pos = cell
                        break
                if cell_at_pos is not None:
                    attrs = ""
                    if cell_at_pos.rowspan > 1:
                        attrs += f" rowspan={cell_at_pos.rowspan}"
                    if cell_at_pos.colspan > 1:
                        attrs += f" colspan={cell_at_pos.colspan}"
                    html += f"<td{attrs}>{_build_cell_text(cell_at_pos.lines)}</td>"
            else:
                html += "<td>&nbsp;</td>"
        html += "</tr>"
    html += "</tbody></table></body></html>"
    return html
