"""
High-level OCR engine that orchestrates classification, detection and recognition.

Architecture:
    - _ocr_core(img): 纯检测+裁剪+识别，不含角度分类（内部方法，零额外开销）
    - ocr(image, use_rotate): 全文 OCR，可选大角度分类
    - layout_ocr(image, use_rotate): 版面分析 + OCR + 文本聚类
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .angle_classifier import AngleClassifier
from .config import OCRConfig
from .exceptions import AscendOCRError
from .image_utils import draw_boxes, load_image
from .layout_analyzer import LayoutAnalyzer, LayoutRegion
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer

logger = logging.getLogger(__name__)


# 不需要 OCR 的版面元素类型
_SKIP_OCR_CLASSES = frozenset({
    "image", "seal", "chart", "table", "watermark",
    "header_image", "footer_image",
})


def _rotate_pts_back(pts: np.ndarray, angle: int, orig_h: int, orig_w: int) -> np.ndarray:
    """
    将旋转坐标映射回原图坐标系（就地修改）。

    Args:
        pts: shape (N, 2) 的坐标数组，直接修改
        angle: 旋转角度
        orig_h, orig_w: 原图尺寸
    """
    if angle == 90:
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = orig_w - 1 - y
        pts[:, 1] = x
    elif angle == 180:
        pts[:, 0] = orig_w - 1 - pts[:, 0]
        pts[:, 1] = orig_h - 1 - pts[:, 1]
    elif angle == 270:
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = y
        pts[:, 1] = orig_h - 1 - x


@dataclass
class OCRResult:
    """Result of OCR for a single text box."""

    box: np.ndarray
    text: str
    score: float

    def __repr__(self) -> str:
        return f"OCRResult(text={self.text!r}, score={self.score:.3f})"


def _sort_results(results: List[OCRResult]) -> None:
    """按阅读顺序排序：先按 y（行），同行按 x（列）。原地排序。"""
    # 预计算排序 key，避免每次比较都 reshape
    keys = []
    for i, r in enumerate(results):
        pts = r.box.reshape(-1, 2)
        y, x = pts[0][1], pts[0][0]
        h = pts[3][1] - pts[0][1]
        keys.append((y // max(h, 1), x, i))  # i 作为 tiebreaker，避免比较 OCRResult
    keys.sort()
    results[:] = [results[k[2]] for k in keys]


def _box_overlap_ratio(
    bx1: float, by1: float, bx2: float, by2: float,
    rx1: int, ry1: int, rx2: int, ry2: int,
) -> float:
    """计算 bbox 交集占 bbox 面积的比例（纯标量运算，无 numpy 开销）。"""
    box_area = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    if box_area <= 0:
        return 0.0
    ix1 = max(bx1, rx1)
    iy1 = max(by1, ry1)
    ix2 = min(bx2, rx2)
    iy2 = min(by2, ry2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    return inter / box_area


def _cluster_text_to_regions(
    results: List[OCRResult],
    regions: List[LayoutRegion],
) -> Dict[int, List[OCRResult]]:
    """
    将 OCR 文本行聚类到对应的版面区域。

    对每个文本行，计算其与所有版面区域的交集占比，分配到占比最高的区域。
    占比低于 min_overlap 的文本行不分配。

    Args:
        results: OCR 识别结果列表
        regions: 版面区域列表

    Returns:
        {region_index: [OCRResult, ...]} 映射
    """
    min_overlap = 0.3
    clusters: Dict[int, List[OCRResult]] = {}

    # 预过滤：只保留需要聚类的文本类区域
    text_regions = [
        (ri, region.bbox)
        for ri, region in enumerate(regions)
        if region.class_name not in _SKIP_OCR_CLASSES
    ]

    # 预计算每个文本行的 AABB（避免在内层循环中重复 reshape+min/max）
    result_aabbs = []
    for r in results:
        pts = r.box.reshape(-1, 2)
        result_aabbs.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))

    for r, (bx1, by1, bx2, by2) in zip(results, result_aabbs):
        best_idx = -1
        best_overlap = 0.0
        for ri, (rx1, ry1, rx2, ry2) in text_regions:
            overlap = _box_overlap_ratio(bx1, by1, bx2, by2, rx1, ry1, rx2, ry2)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = ri
        if best_idx >= 0 and best_overlap >= min_overlap:
            clusters.setdefault(best_idx, []).append(r)

    return clusters


class AscendOCR:
    """
    End-to-end OCR engine for Ascend NPU.

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
                raise AscendOCRError(f"Unknown OCRConfig field: {key}")

        self.config = config
        self._validate_config()

        # Build sub-modules lazily so missing optional models don't crash init.
        self._detector: Optional[TextDetector] = None
        self._recognizer: Optional[TextRecognizer] = None
        self._classifier: Optional[AngleClassifier] = None
        self._layout_analyzer: Optional[LayoutAnalyzer] = None

    def _validate_config(self) -> None:
        if self.config.det_model is None:
            raise AscendOCRError("det_model is required")
        if self.config.rec_model is None:
            raise AscendOCRError("rec_model is required")
        if self.config.use_rotate and self.config.cls_model is None:
            logger.warning(
                "use_rotate=True but cls_model is not provided; disabling rotation"
            )
            self.config.use_rotate = False

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
        if self._classifier is None and self.config.use_rotate:
            self._classifier = AngleClassifier(
                self.config.cls_model,
                cfg=self.config.cls,
                device_id=self.config.device_id,
                decrypt_callback=self.config.decrypt_callback,
            )
        return self._classifier

    @property
    def layout_analyzer(self) -> Optional[LayoutAnalyzer]:
        if self._layout_analyzer is None and self.config.layout_model:
            self._layout_analyzer = LayoutAnalyzer(
                self.config.layout_model,
                device_id=self.config.device_id,
                decrypt_callback=self.config.decrypt_callback,
            )
        return self._layout_analyzer

    # ------------------------------------------------------------------
    # 核心 OCR 管线（不含角度分类，零额外开销）
    # ------------------------------------------------------------------

    def _ocr_core(self, img: np.ndarray) -> Tuple[List[OCRResult], float]:
        """
        纯 OCR 管线：检测 → 裁剪 → 识别。

        不含角度分类、不含模型加载检查、不含原图坐标映射。
        用于 layout_ocr 中对每个区域的 OCR，避免重复走角度分类。

        Args:
            img: 已经转正的 BGR 图像

        Returns:
            (results, t_rec_ms) 其中 results 已按阅读顺序排序
        """
        t_det_start = time.perf_counter()

        # 1. 文字检测
        boxes = self.detector.detect(img)
        if not boxes:
            return [], 0.0

        # 2. 裁剪文本行
        crops = self.detector.crop_text_lines(img, boxes)
        images = [crop for crop, _ in crops]
        boxes = [box for _, box in crops]

        t_det = time.perf_counter() - t_det_start

        # 3. 文字识别
        t0 = time.perf_counter()
        rec_results = self.recognizer.recognize_batch(images)
        t_rec = time.perf_counter() - t0

        # 4. 组装结果并排序
        results = [
            OCRResult(box=box, text=text, score=score)
            for box, (text, score) in zip(boxes, rec_results)
        ]
        _sort_results(results)

        return results, t_rec * 1000

    # ------------------------------------------------------------------
    # 全文 OCR
    # ------------------------------------------------------------------

    def ocr(
        self,
        image: Union[str, bytes, np.ndarray],
        use_rotate: Optional[bool] = None,
        return_visualization: bool = False,
    ):
        """
        Run the full OCR pipeline on an image.

        Args:
            image: Local path, HTTP(S) URL, raw bytes, or numpy array.
            use_rotate: 是否进行大角度分类并转正。None 时使用 config 默认值。
            return_visualization: If True, also return an annotated image.

        Returns:
            If ``return_visualization`` is False: ``(list_of_OCRResult, angle)``.
            If True: ``(list_of_OCRResult, angle, visualization_image)``.
        """
        if use_rotate is None:
            use_rotate = self.config.use_rotate

        orig_img = load_image(image)
        orig_h, orig_w = orig_img.shape[:2]
        logger.info("OCR 开始, 输入图片: %s", orig_img.shape)

        # 预加载模型（懒加载），不计入 OCR 耗时
        _ = self.detector
        _ = self.recognizer
        if use_rotate:
            _ = self.classifier

        t_total = time.perf_counter()

        # 1. 大角度分类和转正
        angle = 0
        img = orig_img
        if use_rotate:
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

        # 2. 核心 OCR（检测 + 识别）
        t0 = time.perf_counter()
        results, t_rec_ms = self._ocr_core(img)
        t_core = time.perf_counter() - t0

        if not results:
            logger.info("[文字检测] 未检测到文字, 耗时: %.1fms", t_core * 1000)
            if return_visualization:
                return [], angle, orig_img
            return [], angle

        logger.info("[文字检测] 检测到 %d 个文字区域", len(results))

        # 3. 将检测框坐标映射回原图坐标系
        if angle != 0:
            for r in results:
                _rotate_pts_back(r.box, angle, orig_h, orig_w)

        logger.info("[文字识别] 识别 %d 行, 识别耗时: %.1fms", len(results), t_rec_ms)
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
            return results, angle, vis
        return results, angle

    def ocr_text_only(
        self,
        image: Union[str, bytes, np.ndarray],
        use_rotate: Optional[bool] = None,
    ) -> List[str]:
        """Convenience method returning only the recognized text strings."""
        results, _angle = self.ocr(image, use_rotate=use_rotate)
        return [r.text for r in results]

    # ------------------------------------------------------------------
    # 版面分析 + OCR + 文本聚类
    # ------------------------------------------------------------------

    def layout_ocr(
        self,
        image: Union[str, bytes, np.ndarray],
        use_rotate: Optional[bool] = None,
        score_threshold: float = 0.5,
    ) -> Tuple[
        List[LayoutRegion],
        List[Tuple[LayoutRegion, List[OCRResult]]],
        Dict[int, List[OCRResult]],
        int,
    ]:
        """
        版面分析 + 全图 OCR + 文本聚类。

        流程：
        1. 大角度分类 → 转正整图（一次性）
        2. 版面分析（在转正后的图上）
        3. 全图文字检测 + 识别（一次性，不按区域裁剪）
        4. 文本聚类：将文本行分配到对应的版面元素
        5. 坐标映射回原图

        Args:
            image: Local path, HTTP(S) URL, raw bytes, or numpy array.
            use_rotate: 是否进行大角度分类并转正。None 时使用 config 默认值。
            score_threshold: Minimum confidence for layout regions.

        Returns:
            (regions, region_ocr_results, clusters, angle) where:
            - regions: 所有检测到的 LayoutRegion 列表。
            - region_ocr_results: [(LayoutRegion, [OCRResult, ...]), ...] 需要 OCR 的区域。
            - clusters: {region_index: [OCRResult, ...]} 文本聚类结果。
            - angle: 检测到的旋转角度。
        """
        if self.layout_analyzer is None:
            raise AscendOCRError("layout_model is not configured")

        if use_rotate is None:
            use_rotate = self.config.use_rotate

        orig_img = load_image(image)
        orig_h, orig_w = orig_img.shape[:2]
        logger.info("Layout OCR 开始, 输入图片: %s", orig_img.shape)

        # 预加载所有模型
        _ = self.detector
        _ = self.recognizer
        if use_rotate:
            _ = self.classifier
        _ = self.layout_analyzer

        t_total = time.perf_counter()

        # 1. 大角度分类 → 转正整图（只做一次）
        angle = 0
        img = orig_img
        if use_rotate:
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

        # 2. 版面分析（在转正后的图上）
        t0 = time.perf_counter()
        regions = self.layout_analyzer.analyze(img, score_threshold=score_threshold)
        t_layout = time.perf_counter() - t0
        logger.info("[版面分析] 检测到 %d 个区域, 耗时: %.1fms", len(regions), t_layout * 1000)

        # 3. 全图文字检测 + 识别（一次性）
        t0 = time.perf_counter()
        all_results, t_rec_ms = self._ocr_core(img)
        t_ocr = time.perf_counter() - t0
        logger.info("[全图OCR] 检测到 %d 行文字, 耗时: %.1fms", len(all_results), t_ocr * 1000)

        # 4. 文本聚类：将文本行分配到版面区域（在转正坐标系中进行）
        clusters = _cluster_text_to_regions(all_results, regions)

        # 5. 构建 region_ocr_results
        region_ocr_results: List[Tuple[LayoutRegion, List[OCRResult]]] = []
        for ri, region in enumerate(regions):
            region_lines = clusters.get(ri, [])
            if region_lines:
                region_ocr_results.append((region, region_lines))

        # 6. 将所有坐标映射回原图坐标系
        if angle != 0:
            for r in all_results:
                _rotate_pts_back(r.box, angle, orig_h, orig_w)

            for region in regions:
                x1, y1, x2, y2 = region.bbox
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                _rotate_pts_back(pts, angle, orig_h, orig_w)
                region.bbox = (
                    int(pts[:, 0].min()),
                    int(pts[:, 1].min()),
                    int(pts[:, 0].max()),
                    int(pts[:, 1].max()),
                )

        t_total_elapsed = time.perf_counter() - t_total
        total_lines = sum(len(r) for _, r in region_ocr_results)
        logger.info(
            "Layout OCR 完成: %d regions, %d lines, %d clustered, 总耗时: %.1fms",
            len(regions), total_lines, sum(len(v) for v in clusters.values()), t_total_elapsed * 1000,
        )

        return regions, region_ocr_results, clusters, angle

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

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
        if self._layout_analyzer is not None:
            self._layout_analyzer.release()
            self._layout_analyzer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
