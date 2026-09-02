"""Text detection: wraps the detection OM model and DBNet post-processing."""

import logging
import math
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .config import DetConfig, OCRConfig
from .image_utils import perspective_transform
from .model import AscendModel
from .postprocess import postprocess_detection
from .preprocess import preprocess_for_detection

logger = logging.getLogger(__name__)


class TextDetector:
    """Wraps the text-detection OM model."""

    def __init__(
        self,
        model_path: str,
        cfg: Optional[DetConfig] = None,
        device_id: int = 0,
        decrypt_callback: Optional[Callable[[str], bytes]] = None,
    ):
        self.cfg = cfg or DetConfig()
        self.model = AscendModel(
            model_path, device_id=device_id, decrypt_callback=decrypt_callback
        )
        if not self.model.dynamic_input:
            shape = self.model.get_input_shape(0)
            if len(shape) == 4 and self.cfg.fixed_input_size is None:
                self.cfg.fixed_input_size = (int(shape[2]), int(shape[3]))
                logger.debug(
                    "Static det model, input shape %s: using fixed input size %dx%d",
                    list(shape), shape[3], shape[2],
                )

    def _pad_to_model_aspect(self, img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        静态 OM 检测模型（固定输入尺寸）会把任意图片各向异性拉伸到模型输入大小。
        宽高比与模型输入相差较大的图（如身份证等小卡片）文字几何失真，导致漏检。
        这里先按原图中心补白边，把宽高比补到与模型输入一致，使后续强制拉伸近似等比缩放。

        Returns:
            (补边后的图像, pad_x, pad_y)，pad_x/pad_y 为左/上补边像素数，用于还原坐标
        """
        det_h, det_w = self.cfg.fixed_input_size
        h, w = img.shape[:2]
        target_ratio = det_h / det_w
        cur_ratio = h / w
        # 宽高比已接近模型输入比例，无需补边，行为与原逻辑完全一致。
        # 阈值取 5%：只有偏差大的图（如接近方形或窄长条的小图）才补边
        if abs(cur_ratio - target_ratio) / target_ratio < 0.05:
            return img, 0, 0
        if cur_ratio < target_ratio:
            # 偏宽：上下补白边
            new_h = int(round(w * target_ratio))
            pad_y = (new_h - h) // 2
            padded = cv2.copyMakeBorder(img, pad_y, new_h - h - pad_y, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
            return padded, 0, pad_y
        # 偏高：左右补白边
        new_w = int(round(h / target_ratio))
        pad_x = (new_w - w) // 2
        padded = cv2.copyMakeBorder(img, 0, 0, pad_x, new_w - w - pad_x,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return padded, pad_x, 0

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Detect text boxes in the image.

        Args:
            img: BGR image.

        Returns:
            List of 4-point polygons (shape (4, 2)) in original image coordinates,
            ordered from top to bottom.
        """
        orig_h, orig_w = img.shape[:2]
        pad_x = pad_y = 0
        if self.cfg.fixed_input_size is not None:
            img, pad_x, pad_y = self._pad_to_model_aspect(img)
            if pad_x or pad_y:
                logger.debug(
                    "检测前补边: 原图 %dx%d, pad_x=%d, pad_y=%d",
                    orig_w, orig_h, pad_x, pad_y,
                )
        tensor, scale, _ = preprocess_for_detection(img, self.cfg)
        outputs = self.model.infer([tensor])
        pred = outputs[0]
        src_h, src_w = img.shape[:2]
        boxes = postprocess_detection(
            pred,
            src_shape=(src_h, src_w),
            scale=scale,
            cfg=self.cfg,
        )
        if pad_x or pad_y:
            # 把补边坐标系下的框平移回原图坐标系，并裁剪到原图范围内
            for box in boxes:
                box[:, 0] = np.clip(box[:, 0] - pad_x, 0, orig_w)
                box[:, 1] = np.clip(box[:, 1] - pad_y, 0, orig_h)
        logger.debug("Detected %d text boxes", len(boxes))
        return boxes

    def crop_text_lines(
        self, img: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Rectify each detected box into a horizontal text-line image.

        Args:
            img: Source BGR image.
            boxes: List of polygons; each is converted to a 4-point min-area
                rectangle internally to guarantee a valid perspective transform.

        Returns:
            List of (text_line_image, box) tuples.
        """
        crops = []
        for i, box in enumerate(boxes):
            t0 = time.perf_counter()
            pts = box.reshape(-1, 2)
            if pts.dtype != np.float32:
                pts = pts.astype(np.float32)
            if pts.shape[0] != 4:
                rect = cv2.minAreaRect(pts)
                pts = cv2.boxPoints(rect)

            # 用 math.hypot 替代 np.linalg.norm，减少 numpy 开销
            dx01 = float(pts[1, 0] - pts[0, 0])
            dy01 = float(pts[1, 1] - pts[0, 1])
            dx12 = float(pts[2, 0] - pts[1, 0])
            dy12 = float(pts[2, 1] - pts[1, 1])
            edge1 = math.hypot(dx01, dy01)
            edge2 = math.hypot(dx12, dy12)
            target_w = max(int(edge1), 1)
            target_h = max(int(edge2), 1)
            if edge2 > edge1:
                target_w, target_h = target_h, target_w

            # Ensure width is the long side for horizontal text reading.
            if target_h > target_w:
                target_w, target_h = target_h, target_w

            warped = perspective_transform(img, pts, (target_w, target_h))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "  裁剪 %2d: %dx%d, 耗时: %.1fms",
                i + 1, target_w, target_h, elapsed_ms,
            )
            crops.append((warped, pts))
        return crops

    def release(self) -> None:
        self.model.release()
