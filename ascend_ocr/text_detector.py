"""Text detection: wraps the detection OM model and DBNet post-processing."""

import logging
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
                logger.info(
                    "Static det model, input shape %s: using fixed input size %dx%d",
                    list(shape), shape[3], shape[2],
                )

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Detect text boxes in the image.

        Args:
            img: BGR image.

        Returns:
            List of 4-point polygons (shape (4, 2)) in original image coordinates,
            ordered from top to bottom.
        """
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
        logger.info("Detected %d text boxes", len(boxes))
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
        for box in boxes:
            pts = box.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] != 4:
                rect = cv2.minAreaRect(pts)
                pts = cv2.boxPoints(rect)

            # Compute target size from the two adjacent edges.
            edge1 = np.linalg.norm(pts[1] - pts[0])
            edge2 = np.linalg.norm(pts[2] - pts[1])
            target_w = max(int(edge1), 1)
            target_h = max(int(edge2), 1)
            if edge2 > edge1:
                target_w, target_h = target_h, target_w

            # Ensure width is the long side for horizontal text reading.
            if target_h > target_w:
                target_w, target_h = target_h, target_w

            warped = perspective_transform(img, pts, (target_w, target_h))
            crops.append((warped, pts))
        return crops

    def release(self) -> None:
        self.model.release()
