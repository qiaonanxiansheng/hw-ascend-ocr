"""
High-level OCR engine that orchestrates classification, detection and recognition.
"""

import logging
import time
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


def _rotate_boxes_back(boxes: List[np.ndarray], angle: int, orig_h: int, orig_w: int) -> List[np.ndarray]:
    """
    将旋转后图像的检测框坐标映射回原图坐标系。

    Args:
        boxes: 旋转后图像中的检测框列表，每个 shape (4, 2)
        angle: 图像被旋转的角度（逆时针，即 rotate_image 的参数）
        orig_h: 原图高度
        orig_w: 原图宽度

    Returns:
        映射回原图坐标系的检测框列表
    """
    if angle == 0 or not boxes:
        return boxes

    rotated = []
    for box in boxes:
        pts = box.reshape(-1, 2)
        if angle == 90:
            new_x = orig_w - 1 - pts[:, 1]
            new_y = pts[:, 0]
        elif angle == 180:
            new_x = orig_w - 1 - pts[:, 0]
            new_y = orig_h - 1 - pts[:, 1]
        elif angle == 270:
            new_x = pts[:, 1]
            new_y = orig_h - 1 - pts[:, 0]
        else:
            rotated.append(box)
            continue
        new_pts = pts.copy()
        new_pts[:, 0] = new_x
        new_pts[:, 1] = new_y
        rotated.append(new_pts)
    return rotated


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
        orig_img = load_image(image)
        orig_h, orig_w = orig_img.shape[:2]
        logger.info("OCR 开始, 输入图片: %s", orig_img.shape)

        # 预加载模型（懒加载），不计入 OCR 耗时
        _ = self.detector
        _ = self.recognizer
        if self.config.use_angle_cls:
            _ = self.classifier

        t_total = time.perf_counter()

        # 1. Large-angle classification and rotation.
        angle = 0
        img = orig_img
        if self.config.use_angle_cls:
            cls = self.classifier
            if cls is not None:
                t0 = time.perf_counter()
                img, angle, cls_conf = cls.rotate_to_upright(img)
                t_cls = time.perf_counter() - t0
                logger.info("[角度分类] 角度: %d°, 置信度: %.3f, 耗时: %.1fms", angle, cls_conf, t_cls * 1000)
            else:
                logger.debug("[角度分类] 分类器未加载，跳过")
        else:
            logger.debug("[角度分类] 已禁用，跳过")

        # 2. Text detection (on rotated image).
        t0 = time.perf_counter()
        boxes = self.detector.detect(img)
        t_det = time.perf_counter() - t0
        if not boxes:
            logger.info("[文字检测] 未检测到文字, 耗时: %.1fms", t_det * 1000)
            if return_visualization:
                return [], orig_img
            return []
        logger.info("[文字检测] 检测到 %d 个文字区域, 耗时: %.1fms", len(boxes), t_det * 1000)
        for idx, box in enumerate(boxes, 1):
            pts = box.reshape(-1, 2).astype(int).tolist()
            logger.debug("  %2d. 坐标: %s", idx, pts)

        # 3. Crop and rectify text lines (from rotated image).
        t0 = time.perf_counter()
        crops = self.detector.crop_text_lines(img, boxes)
        images = [crop for crop, _ in crops]
        boxes = [box for _, box in crops]
        t_crop = time.perf_counter() - t0
        logger.debug("[文字裁剪] 裁剪 %d 个文本行, 耗时: %.1fms", len(images), t_crop * 1000)

        # 4. Text recognition.
        t0 = time.perf_counter()
        rec_results = self.recognizer.recognize_batch(images)
        t_rec = time.perf_counter() - t0

        # 5. 将检测框坐标映射回原图坐标系。
        orig_boxes = _rotate_boxes_back(boxes, angle, orig_h, orig_w)

        results = [
            OCRResult(box=box, text=text, score=score)
            for box, (text, score) in zip(orig_boxes, rec_results)
        ]

        logger.info("[文字识别] 识别 %d 行, 耗时: %.1fms", len(results), t_rec * 1000)
        for idx, r in enumerate(results, 1):
            logger.debug("  %2d. 置信度: %.3f, 文字: %s", idx, r.score, r.text)
        t_total = time.perf_counter() - t_total
        logger.info("OCR 完成, 共 %d 行文字, 总耗时: %.1fms", len(results), t_total * 1000)

        if return_visualization:
            vis = draw_boxes(
                orig_img,
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
