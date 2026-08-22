"""Large-angle classifier: predicts 0 / 90 / 180 / 270 degree rotation."""

import logging
from typing import Optional, Callable

import numpy as np

from .config import RotateConfig
from .image_utils import rotate_image
from .model import AscendModel
from .preprocess import preprocess_for_classification

logger = logging.getLogger(__name__)


class AngleClassifier:
    """Wraps the angle-classification OM model."""

    def __init__(
        self,
        model_path: str,
        cfg: Optional[RotateConfig] = None,
        device_id: int = 0,
        decrypt_callback: Optional[Callable[[str], bytes]] = None,
    ):
        self.cfg = cfg or RotateConfig()
        self.model = AscendModel(
            model_path, device_id=device_id, decrypt_callback=decrypt_callback
        )

    def classify(self, img: np.ndarray) -> tuple:
        """
        Predict the rotation angle of the image.

        Args:
            img: BGR image.

        Returns:
            (angle, confidence) - Rotation angle in degrees and confidence score.
        """
        tensor = preprocess_for_classification(img, self.cfg)
        outputs = self.model.infer([tensor])
        probs = outputs[0]
        if probs.ndim == 2:
            probs = probs[0]
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])
        label_id = max(0, min(label_id, len(self.cfg.label_list) - 1))
        angle = self.cfg.label_list[label_id]
        logger.debug("Angle classification: label_id=%d angle=%d confidence=%.3f", label_id, angle, confidence)
        return angle, confidence

    def rotate_to_upright(self, img: np.ndarray) -> tuple:
        """
        Classify and rotate the image so text is upright.

        Returns:
            (rotated_image, angle, confidence)
        """
        angle, confidence = self.classify(img)
        if angle != 0:
            img = rotate_image(img, angle)
        return img, angle, confidence

    def release(self) -> None:
        self.model.release()
