"""
ascend_ocr
==========

A native, high-cohesion OCR engine for Ascend NPU.

Pipeline:
    1. Load image from local path or HTTP URL.
    2. Large-angle classification (0 / 90 / 180 / 270).
    3. Rotate image to 0 degrees if necessary.
    4. Text-box detection (DBNet post-processing).
    5. Crop and rectify each text box (perspective transform).
    6. Text recognition with CTC decoding.

Example
-------
>>> from ascend_ocr import AscendOCR
>>> engine = AscendOCR(
...     det_model="models/ppocrv5_server_det_Ascend910B3.om",
...     rec_model="models/ppocrv5_rec_Ascend910B3.om",
...     rotate_model="models/rotate.om",
...     rec_char_dict="configs/ppocr_keys_v1.txt",
... )
>>> results, angle = engine.ocr("https://example.com/image.png")
>>> for r in results:
...     print(r.box, r.text, r.score)
"""

from .ocr import AscendOCR, OCRResult
from .angle_classifier import AngleClassifier
from .layout_analyzer import LayoutAnalyzer, LayoutRegion
from .table_recognizer import TableRecognizer, TableStructure, TableCell
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .model import AscendModel
from .config import OCRConfig, DetConfig, RecConfig, RotateConfig, TableConfig, load_config

__version__ = "3.0.0"
__all__ = [
    "AscendOCR",
    "OCRResult",
    "AngleClassifier",
    "LayoutAnalyzer",
    "LayoutRegion",
    "TableRecognizer",
    "TableStructure",
    "TableCell",
    "TextDetector",
    "TextRecognizer",
    "AscendModel",
    "OCRConfig",
    "DetConfig",
    "RecConfig",
    "RotateConfig",
    "TableConfig",
    "load_config",
]
