"""
High-level OCR engine that orchestrates classification, detection and recognition.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np

from .angle_classifier import AngleClassifier
from .config import ClsConfig, DetConfig, OCRConfig, RecConfig
from .exceptions import YuntuAscendOCRError
from .image_utils import draw_boxes, load_image
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR for a single text box."""

    box: np.ndarray
    text: str
    score: float

    def __repr__(self) -> str:
        return f"OCRResult(text={self.text!r}, score={self.score:.3f})"


class AscendOCR:
    """
    End-to-end OCR engine for Huawei Ascend.

    Usage::

        engine = AscendOCR(
            det_model="models/det.om",
            rec_model="models/rec.om",
            cls_model="models/cls.om",
            rec_char_dict="configs/ppocr_keys_v1.txt",
        )
        result = engine.ocr("./image.png")
        for item in result:
            print(item.box, item.text, item.score)
    """

    def __init__(self, config: Optional[OCRConfig] = None, **overrides):
        """
        Args:
            config: ``OCRConfig`` instance. If omitted, a default config is used.
            **overrides: Keyword shortcuts for common config fields, e.g.
                ``det_model=..., rec_model=..., cls_model=..., device_id=...``.
        """
        if config is None:
            config = OCRConfig()

        # Apply simple keyword overrides.
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise YuntuAscendOCRError(f"Unknown OCRConfig field: {key}")

        self.config = config
        self._validate_config()

        # Build sub-modules lazily so missing optional models don't crash init.
        self._detector: Optional[TextDetector] = None
        self._recognizer: Optional[TextRecognizer] = None
        self._classifier: Optional[AngleClassifier] = None

    def _validate_config(self) -> None:
        if self.config.det_model is None:
            raise YuntuAscendOCRError("det_model is required")
        if self.config.rec_model is None:
            raise YuntuAscendOCRError("rec_model is required")
        if self.config.use_angle_cls and self.config.cls_model is None:
            logger.warning(
                "use_angle_cls=True but cls_model is not provided; disabling angle classification"
            )
            self.config.use_angle_cls = False

    @property
    def detector(self) -> TextDetector:
        if self._detector is None:
            self._detector = TextDetector(
                self.config.det_model,
                cfg=self.config.det,
                device_id=self.config.device_id,
                decrypt_callback=self.config.decrypt_callback,
            )
        return self._detector

    @property
    def recognizer(self) -> TextRecognizer:
        if self._recognizer is None:
            self._recognizer = TextRecognizer(
                self.config.rec_model,
                char_dict_path=self.config.rec_char_dict,
                cfg=self.config.rec,
                device_id=self.config.device_id,
                decrypt_callback=self.config.decrypt_callback,
            )
        return self._recognizer

    @property
    def classifier(self) -> Optional[AngleClassifier]:
        if self._classifier is None and self.config.use_angle_cls:
            self._classifier = AngleClassifier(
                self.config.cls_model,
                cfg=self.config.cls,
                device_id=self.config.device_id,
                decrypt_callback=self.config.decrypt_callback,
            )
        return self._classifier

    def ocr(
        self,
        image: Union[str, bytes, np.ndarray],
        return_visualization: bool = False,
    ):
        """
        Run the full OCR pipeline on an image.

        Args:
            image: Local path, HTTP(S) URL, raw bytes, or numpy array.
            return_visualization: If True, also return an annotated image.

        Returns:
            If ``return_visualization`` is False: list of ``OCRResult``.
            If True: ``(list_of_OCRResult, visualization_image)``.
        """
        img = load_image(image)
        logger.info(
            "OCR start, image shape=%s, device=%d",
            img.shape,
            self.config.device_id,
        )

        # 1. Large-angle classification and rotation.
        angle = 0
        if self.config.use_angle_cls:
            cls = self.classifier
            if cls is not None:
                img, angle = cls.rotate_to_upright(img)
                logger.info("Image rotated by %d degrees to upright", angle)

        # 2. Text detection.
        boxes = self.detector.detect(img)
        if not boxes:
            logger.info("No text boxes detected")
            if return_visualization:
                return [], img
            return []

        # 3. Crop and rectify text lines.
        crops = self.detector.crop_text_lines(img, boxes)
        images = [crop for crop, _ in crops]
        boxes = [box for _, box in crops]

        # 4. Text recognition.
        rec_results = self.recognizer.recognize_batch(images)

        results = [
            OCRResult(box=box, text=text, score=score)
            for box, (text, score) in zip(boxes, rec_results)
        ]

        if return_visualization:
            vis = draw_boxes(
                img,
                [r.box for r in results],
                texts=[r.text for r in results],
            )
            return results, vis
        return results

    def ocr_text_only(
        self, image: Union[str, bytes, np.ndarray]
    ) -> List[str]:
        """Convenience method returning only the recognized text strings."""
        results = self.ocr(image)
        return [r.text for r in results]

    def release(self) -> None:
        """Release all underlying Ascend models."""
        if self._detector is not None:
            self._detector.release()
            self._detector = None
        if self._recognizer is not None:
            self._recognizer.release()
            self._recognizer = None
        if self._classifier is not None:
            self._classifier.release()
            self._classifier = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
