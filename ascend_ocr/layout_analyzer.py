"""
Layout analysis using PP-DocLayoutV3 model (DETR-based, 25 classes).

Input:
    - im_shape: (1, 2) - [800, 800] (target size, keep_ratio=false)
    - image: (1, 3, 800, 800) - resized (/255 only, no mean/std)
    - scale_factor: (1, 2) - [h_scale, w_scale] (different for h and w)

Output (3 tensors):
    - output[0]: float32[300, 7] - [label, score, x1, y1, x2, y2, order]
    - output[1]: float32[1] - number of detections
    - output[2]: float32[300, 200, 200] - segmentation masks (raw logits, need sigmoid)
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .model import AscendModel

logger = logging.getLogger(__name__)

# PP-DocLayoutV3 class labels (25 classes, alphabetical order from model config)
LAYOUT_CLASSES = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}


@dataclass
class LayoutRegion:
    """A detected layout region."""

    class_id: int
    class_name: str
    score: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    html: str = ""  # HTML content for table regions

    def __repr__(self) -> str:
        return (
            f"LayoutRegion(class={self.class_name!r}, score={self.score:.3f}, "
            f"bbox={self.bbox})"
        )


class LayoutAnalyzer:
    """Wraps the PP-DocLayoutV3 layout analysis model."""

    INPUT_SIZE = 800
    MASK_SIZE = 200  # Segmentation mask size

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        decrypt_callback: Optional[Callable[[str], bytes]] = None,
    ):
        logger.info("Loading layout model from %s", model_path)
        self.model = AscendModel(
            model_path, device_id=device_id, decrypt_callback=decrypt_callback
        )
        # Print model info
        for i in range(self.model._input_num):
            shape = self.model.get_input_shape(i)
            logger.info("Layout model input[%d] shape: %s", i, shape)
        for i in range(self.model._output_num):
            shape = self.model.get_output_shape(i)
            logger.info("Layout model output[%d] shape: %s", i, shape)

    def _preprocess(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Preprocess image for PP-DocLayoutV3.

        Returns:
            (image_tensor, im_shape, scale_factor, scale_h, scale_w)
        """
        src_h, src_w = img.shape[:2]
        target_h, target_w = self.INPUT_SIZE, self.INPUT_SIZE

        # Resize directly to 800x800 (keep_ratio: false from model config)
        # No aspect ratio preservation, no padding
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        logger.debug(
            "Layout preprocess: src=%dx%d, resized=%dx%d (keep_ratio=false)",
            src_w, src_h, target_w, target_h,
        )

        # Normalize: only /255 (PP-DocLayoutV3 uses mean=[0,0,0], std=[1,1,1])
        img_norm = resized.astype(np.float32) / 255.0

        # HWC -> CHW, add batch dim
        tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        # im_shape: (1, 2) = [height, width] = target size (since keep_ratio=false, no padding)
        im_shape = np.array([[target_h, target_w]], dtype=np.float32)

        # scale_factor: (1, 2) = [h_scale, w_scale]
        # With keep_ratio: false, h and w scale differently
        scale_h = target_h / src_h
        scale_w = target_w / src_w
        scale_factor = np.array([[scale_h, scale_w]], dtype=np.float32)

        logger.debug(
            "Layout preprocess output: tensor=%s, im_shape=%s, scale_factor=%s",
            tensor.shape, im_shape.shape, scale_factor.shape,
        )

        return tensor, im_shape, scale_factor, scale_h, scale_w

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        src_h: int,
        src_w: int,
        scale_h: float,
        scale_w: float,
        score_threshold: float = 0.5,
    ) -> List[LayoutRegion]:
        """
        Post-process layout model output.

        The model outputs 3 tensors:
        - output[0]: float32[N, 7] - detection results [class_id, score, x1, y1, x2, y2, ...]
        - output[1]: int32[M] - number of detections or indices
        - output[2]: int32[K, 200, 200] - segmentation masks

        Args:
            outputs: List of 3 output arrays.
            src_h: Original image height.
            src_w: Original image width.
            scale_h: Scale factor for height.
            scale_w: Scale factor for width.
            score_threshold: Minimum confidence to keep a region.

        Returns:
            List of LayoutRegion objects.
        """
        logger.info("Layout postprocess: %d outputs, src=%dx%d, scale=(%.4f, %.4f)",
                   len(outputs), src_w, src_h, scale_w, scale_h)

        # Log all output details
        for i, out in enumerate(outputs):
            logger.info("  output[%d]: shape=%s, dtype=%s, size=%d", i, out.shape, out.dtype, out.size)
            if out.size > 0 and out.size <= 100:
                logger.info("  output[%d] values:\n%s", i, out)

        # Parse detection output[0]: float32[N, 7]
        # Format from DocLayoutV3PostProcess: [label, score, x1, y1, x2, y2, order]
        det_output = outputs[0]
        if det_output.ndim == 3:
            det_output = det_output[0]  # Remove batch dim

        logger.info("Detection output: %d rows, columns=%d", det_output.shape[0], det_output.shape[1])

        # Log score distribution for debugging
        scores = det_output[:, 1]
        valid_score_count = int(np.sum(scores >= score_threshold))
        logger.info("Score distribution: min=%.4f, max=%.4f, mean=%.4f, above_threshold=%d",
                   scores.min(), scores.max(), scores.mean(), valid_score_count)

        # Diagnostic: log coordinate ranges to understand the space
        if det_output.shape[1] >= 6:
            x_coords = det_output[:, 2:6:2]  # columns 2,4 = x1,x2
            y_coords = det_output[:, 3:6:2]  # columns 3,5 = y1,y2
            logger.info("Coord ranges: x=[%.1f, %.1f], y=[%.1f, %.1f] (image=%dx%d)",
                       x_coords.min(), x_coords.max(),
                       y_coords.min(), y_coords.max(), src_w, src_h)
            # Log first few valid detections raw values
            for i in range(min(5, det_output.shape[0])):
                if det_output[i, 1] >= score_threshold:
                    logger.info("  det_raw[%d]: [%s]", i,
                               ", ".join(f"{v:.2f}" for v in det_output[i]))

        # Check if coordinates need decoding
        # Possible formats:
        #   1. Already in original image space (post-processing included in ONNX)
        #   2. Normalized [0,1] cxcywh (raw DETR output, need decode + scale)
        #   3. Scaled to model input space (800x800)
        x_max = float(det_output[:, 2].max()) if det_output.shape[1] > 2 else 0
        y_max = float(det_output[:, 3].max()) if det_output.shape[1] > 3 else 0
        coord_in_image_space = x_max <= src_w * 1.1 and y_max <= src_h * 1.1

        if not coord_in_image_space and det_output.shape[1] >= 6:
            logger.info("Coords not in image space (max=%.0f,%.0f, image=%dx%d), decoding...",
                       x_max, y_max, src_w, src_h)

            raw_bboxes = det_output[:, 2:6].copy()
            x_min_raw = float(raw_bboxes[:, 0].min())
            x_max_raw = float(raw_bboxes[:, 0].max())
            y_min_raw = float(raw_bboxes[:, 1].min())
            y_max_raw = float(raw_bboxes[:, 1].max())
            logger.info("Raw bbox ranges: x=[%.2f, %.2f], y=[%.2f, %.2f]",
                       x_min_raw, x_max_raw, y_min_raw, y_max_raw)

            if x_max_raw <= 1.0 and y_max_raw <= 1.0:
                # Format 2: normalized [0,1] cxcywh → xyxy → scale to image
                logger.info("Detected normalized [0,1] cxcywh format")
                xyxy = np.zeros_like(raw_bboxes)
                xyxy[:, 0] = raw_bboxes[:, 0] - raw_bboxes[:, 2] / 2
                xyxy[:, 1] = raw_bboxes[:, 1] - raw_bboxes[:, 3] / 2
                xyxy[:, 2] = raw_bboxes[:, 0] + raw_bboxes[:, 2] / 2
                xyxy[:, 3] = raw_bboxes[:, 1] + raw_bboxes[:, 3] / 2
                out_shape = np.array([src_w, src_h, src_w, src_h], dtype=np.float32)
                xyxy *= out_shape
            elif x_max_raw <= 800 * 1.1 and y_max_raw <= 800 * 1.1:
                # Format 3: in model input space (800x800), scale to original
                logger.info("Detected model-input-space format (800x800)")
                xyxy = raw_bboxes.copy()
                xyxy[:, 0] /= scale_w
                xyxy[:, 1] /= scale_h
                xyxy[:, 2] /= scale_w
                xyxy[:, 3] /= scale_h
            else:
                # Unknown format, try cxcywh decode with aggressive clipping
                logger.warning("Unknown coord format (max=%.0f), trying cxcywh decode with clipping", x_max_raw)
                xyxy = np.zeros_like(raw_bboxes)
                xyxy[:, 0] = raw_bboxes[:, 0] - raw_bboxes[:, 2] / 2
                xyxy[:, 1] = raw_bboxes[:, 1] - raw_bboxes[:, 3] / 2
                xyxy[:, 2] = raw_bboxes[:, 0] + raw_bboxes[:, 2] / 2
                xyxy[:, 3] = raw_bboxes[:, 1] + raw_bboxes[:, 3] / 2
                out_shape = np.array([src_w, src_h, src_w, src_h], dtype=np.float32)
                xyxy *= out_shape
                xyxy = np.clip(xyxy, 0, out_shape)

            # Apply sigmoid to scores if they look like logits
            raw_scores = det_output[:, 1]
            if raw_scores.max() > 1.0 or raw_scores.min() < 0.0:
                logger.info("Applying sigmoid to scores (logits detected, range=[%.2f, %.2f])",
                           raw_scores.min(), raw_scores.max())
                raw_scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -50, 50)))

            det_output[:, 1] = raw_scores
            det_output[:, 2:6] = xyxy
            logger.info("After decode: coord ranges: x=[%.1f, %.1f], y=[%.1f, %.1f]",
                       xyxy[:, 0].min(), xyxy[:, 2].max(),
                       xyxy[:, 1].min(), xyxy[:, 3].max())

        regions = []

        # Log all 3 outputs for debugging
        mask_output = outputs[2] if len(outputs) > 2 else None
        if mask_output is not None:
            logger.info("Mask output: shape=%s, dtype=%s, min=%.6f, max=%.6f, mean=%.6f, nnz=%d",
                       mask_output.shape, mask_output.dtype,
                       float(mask_output.min()), float(mask_output.max()),
                       float(mask_output.mean()), int(np.count_nonzero(mask_output)))

            # Also log per-channel stats for first few channels
            for ch in range(min(10, mask_output.shape[0])):
                ch_data = mask_output[ch]
                nz = int(np.count_nonzero(ch_data))
                if nz > 0:
                    logger.info("  mask[%d]: nnz=%d, min=%.4f, max=%.4f",
                               ch, nz, float(ch_data.min()), float(ch_data.max()))

        # Check detection output details
        det_valid = det_output[:, 1] >= score_threshold
        n_valid = int(np.sum(det_valid))
        if n_valid > 0:
            valid_rows = det_output[det_valid]
            logger.info("Valid detections (%d):", n_valid)
            for idx, row in enumerate(valid_rows):
                logger.info("  [%d] class=%.0f score=%.4f bbox=[%.1f,%.1f,%.1f,%.1f]",
                           idx, row[0], row[1], row[2], row[3], row[4], row[5])

        # Mask output: DETR outputs raw logits, need sigmoid + threshold
        mask_valid = False
        if mask_output is not None and mask_output.size > 0:
            # Log raw mask stats
            logger.info("Mask raw: dtype=%s, shape=%s, min=%.6f, max=%.6f, nnz=%d",
                       mask_output.dtype, mask_output.shape,
                       float(mask_output.min()), float(mask_output.max()),
                       int(np.count_nonzero(mask_output)))

            # Apply sigmoid to convert logits to probabilities
            mask_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(mask_output, -50, 50)))
            sigmoid_max = float(mask_sigmoid.max())
            sigmoid_nnz = int(np.count_nonzero(mask_sigmoid > 0.5))
            logger.info("Mask after sigmoid: max=%.6f, pixels>0.5=%d", sigmoid_max, sigmoid_nnz)

            if sigmoid_max > 0.5:
                mask_output = mask_sigmoid
                mask_valid = True

        if mask_valid:
            logger.info("Extracting regions from segmentation mask...")
            regions = self._extract_from_mask(mask_output, det_output, src_h, src_w, scale_h, scale_w, score_threshold)

        # Fallback: use detection output if mask is empty
        if not regions and n_valid > 0:
            logger.info("Mask empty/invalid, falling back to detection output...")
            for i in range(det_output.shape[0]):
                row = det_output[i]
                if len(row) < 6:
                    continue

                class_id = int(row[0])
                score = float(row[1])
                if score < score_threshold:
                    continue

                x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])

                # Try: coords might already be in original image space
                x1_orig = max(0, int(x1))
                y1_orig = max(0, int(y1))
                x2_orig = min(src_w, int(x2))
                y2_orig = min(src_h, int(y2))

                w, h = x2_orig - x1_orig, y2_orig - y1_orig
                if w < 10 or h < 10:
                    continue

                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 20:
                    continue

                class_name = LAYOUT_CLASSES.get(class_id, f"unknown_{class_id}")
                regions.append(LayoutRegion(
                    class_id=class_id,
                    class_name=class_name,
                    score=score,
                    bbox=(x1_orig, y1_orig, x2_orig, y2_orig),
                ))
                logger.info(
                    "  det[%d]: %s (id=%d), score=%.4f, bbox=[%d,%d,%d,%d] %dx%d",
                    i, class_name, class_id, score,
                    x1_orig, y1_orig, x2_orig, y2_orig, w, h,
                )

        if not regions:
            logger.warning("No regions found from mask or detection output!")

        # Sort by reading order (top to bottom, left to right)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        logger.info("Layout analysis found %d regions", len(regions))
        return regions

    def _extract_from_mask(
        self,
        mask: np.ndarray,
        det_output: np.ndarray,
        src_h: int,
        src_w: int,
        scale_h: float,
        scale_w: float,
        score_threshold: float = 0.5,
    ) -> List[LayoutRegion]:
        """
        Extract layout regions from segmentation mask.

        Each of the N mask channels corresponds to one detection in det_output.
        We iterate over each channel individually: threshold → find contours →
        extract each contour as a separate region.

        Coordinate mapping: mask(200x200) -> model(800x800) -> original image.
            orig_x = mask_x * (INPUT_SIZE / MASK_SIZE) / scale_w

        Args:
            mask: Segmentation mask, shape (N, 200, 200).
            det_output: Detection output, shape (N, 7).
            src_h: Original image height.
            src_w: Original image width.
            scale_h: Scale factor for height (new_h / src_h).
            scale_w: Scale factor for width (new_w / src_w).

        Returns:
            List of LayoutRegion objects.
        """
        logger.info("Extracting regions from segmentation mask, shape=%s", mask.shape)

        if mask.ndim != 3:
            logger.warning("Unexpected mask shape %s, skipping", mask.shape)
            return []

        num_channels = mask.shape[0]
        det_class_ids = det_output[:, 0].astype(int)  # class_id for each detection
        det_scores = det_output[:, 1]  # score for each detection

        # Scale factor: mask(200) -> model_input(800) -> original image
        mask_to_orig_x = self.INPUT_SIZE / self.MASK_SIZE / scale_w
        mask_to_orig_y = self.INPUT_SIZE / self.MASK_SIZE / scale_h

        # Minimum region size in mask space (~100px in original image)
        min_size_orig = 100
        min_w_mask = max(3, int(min_size_orig / mask_to_orig_x))
        min_h_mask = max(3, int(min_size_orig / mask_to_orig_y))

        logger.info("Mask-to-original scale: x=%.2f, y=%.2f, min_mask_size=%dx%d",
                    mask_to_orig_x, mask_to_orig_y, min_w_mask, min_h_mask)

        # Morphological kernel for noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Determine mask value range for proper thresholding
        mask_max = float(mask.max())
        if mask_max <= 1.0:
            mask_threshold = 0.5  # Binary mask in [0, 1]
        elif mask_max <= 10:
            mask_threshold = 1.0  # Small integer values (0-10)
        elif mask_max <= 255:
            mask_threshold = 127  # Probability map in [0, 255]
        else:
            mask_threshold = 1
        logger.info("Mask value range: dtype=%s, max=%.4f, threshold=%.4f", mask.dtype, mask_max, mask_threshold)

        # Maximum region area as fraction of total image area
        max_area_ratio = 0.5
        total_area = src_h * src_w

        regions = []
        for ch_idx in range(num_channels):
            class_id = int(det_class_ids[ch_idx]) if ch_idx < len(det_class_ids) else -1
            if class_id < 0 or class_id not in LAYOUT_CLASSES:
                continue

            # Skip channels with low detection score
            score = float(det_scores[ch_idx]) if ch_idx < len(det_scores) else 0.0
            if score < score_threshold:
                continue

            # Each channel is a mask for one detection
            ch_mask = mask[ch_idx]
            if ch_mask.max() < mask_threshold:
                continue  # No significant activation in this channel

            binary_mask = (ch_mask > mask_threshold).astype(np.uint8)

            # Morphological open to remove noise
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

            # Find contours - each contour is a separate region
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Skip tiny regions
                if w < min_w_mask or h < min_h_mask:
                    continue

                # Scale from mask space (200x200) to original image space
                x1_orig = max(0, int(x * mask_to_orig_x))
                y1_orig = max(0, int(y * mask_to_orig_y))
                x2_orig = min(src_w, int((x + w) * mask_to_orig_x))
                y2_orig = min(src_h, int((y + h) * mask_to_orig_y))

                # Validate bbox
                if x2_orig <= x1_orig or y2_orig <= y1_orig:
                    continue

                # Skip regions covering too much of the image (likely noise)
                region_area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
                if region_area > total_area * max_area_ratio:
                    continue

                class_name = LAYOUT_CLASSES[class_id]
                regions.append(LayoutRegion(
                    class_id=class_id,
                    class_name=class_name,
                    score=1.0,
                    bbox=(x1_orig, y1_orig, x2_orig, y2_orig),
                ))

        # Merge overlapping regions of the same class
        regions = self._merge_overlapping_regions(regions)

        logger.info("Mask extraction: %d regions after filtering and merging", len(regions))
        for r in regions:
            logger.info("  %s bbox=[%d,%d,%d,%d] size=%dx%d",
                       r.class_name, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
                       r.bbox[2]-r.bbox[0], r.bbox[3]-r.bbox[1])

        return regions

    @staticmethod
    def _merge_overlapping_regions(regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Merge overlapping regions of the same class (only true overlap, not distant blocks)."""
        if not regions:
            return regions

        from collections import defaultdict
        by_class = defaultdict(list)
        for r in regions:
            by_class[r.class_id].append(r)

        merged = []
        for class_id, class_regions in by_class.items():
            class_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

            used = [False] * len(class_regions)
            for i, r1 in enumerate(class_regions):
                if used[i]:
                    continue
                x1, y1, x2, y2 = r1.bbox
                for j in range(i + 1, len(class_regions)):
                    if used[j]:
                        continue
                    r2 = class_regions[j]
                    rx1, ry1, rx2, ry2 = r2.bbox
                    # Only merge if truly overlapping (no margin for distant blocks)
                    if rx1 <= x2 and rx2 >= x1 and ry1 <= y2 and ry2 >= y1:
                        x1 = min(x1, rx1)
                        y1 = min(y1, ry1)
                        x2 = max(x2, rx2)
                        y2 = max(y2, ry2)
                        used[j] = True

                used[i] = True
                merged.append(LayoutRegion(
                    class_id=class_id,
                    class_name=class_regions[i].class_name,
                    score=class_regions[i].score,
                    bbox=(x1, y1, x2, y2),
                ))

        return merged

    def analyze(
        self,
        img: np.ndarray,
        score_threshold: float = 0.5,
    ) -> List[LayoutRegion]:
        """
        Run layout analysis on an image.

        Args:
            img: BGR image.
            score_threshold: Minimum confidence to keep a region.

        Returns:
            List of detected LayoutRegion objects.
        """
        src_h, src_w = img.shape[:2]
        logger.info("Layout analysis start: image size=%dx%d", src_w, src_h)

        t0 = time.perf_counter()

        # Preprocess
        tensor, im_shape, scale_factor, scale_h, scale_w = self._preprocess(img)
        t_pre = time.perf_counter() - t0
        logger.info("Layout preprocess: %.1fms", t_pre * 1000)

        # Infer (input order: im_shape, image, scale_factor)
        t1 = time.perf_counter()
        outputs = self.model.infer([im_shape, tensor, scale_factor])
        t_infer = time.perf_counter() - t1
        logger.info("Layout inference: %.1fms", t_infer * 1000)

        # Log output details
        logger.info("Layout model returned %d outputs:", len(outputs))
        for i, out in enumerate(outputs):
            logger.info("  output[%d]: shape=%s, dtype=%s", i, out.shape, out.dtype)

        # Postprocess
        t2 = time.perf_counter()
        regions = self._postprocess(outputs, src_h, src_w, scale_h, scale_w, score_threshold)
        t_post = time.perf_counter() - t2
        logger.info("Layout postprocess: %.1fms", t_post * 1000)

        t_total = time.perf_counter() - t0
        logger.info(
            "Layout analysis complete: %d regions, total=%.1fms",
            len(regions), t_total * 1000,
        )

        return regions

    def release(self) -> None:
        """Release model resources."""
        if self.model is not None:
            self.model.release()
            logger.info("Layout model released")
