"""Default hyper-parameters and configuration for the OCR pipeline."""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml


def default_char_dict_path() -> str:
    """Return the path to the bundled default character dictionary."""
    return os.path.join(os.path.dirname(__file__), "configs", "v6_rec_dict.txt")


@dataclass
class DetConfig:
    """
    Text-detection configuration (DBNet style).

    All algorithm choices are exposed as switches so you can tune for accuracy
    or speed depending on the scene.
    """

    # ------------------------- Resize / Pre-process -------------------------
    # Maximum side length of the resized image. Larger -> better accuracy, slower.
    limit_side_len: int = 960

    # Multiple to which both height and width are rounded (model requirement).
    limit_multiple: int = 32

    # Fixed input size (H, W) for static-shape OM models. When set, the image
    # is stretched to exactly this size and limit_side_len/limit_multiple are
    # ignored. TextDetector populates this automatically from the model.
    fixed_input_size: Optional[Tuple[int, int]] = None

    # Resize strategy:
    #   "pad"           - keep aspect ratio, pad with pad_color (default, best accuracy)
    #   "stretch"       - stretch to exact target size (fastest, may distort)
    #   "crop_center"   - center crop to target size (rarely used for OCR)
    resize_mode: str = "pad"

    # Padding color used when resize_mode == "pad".
    pad_color: Tuple[int, int, int] = (0, 0, 0)

    # Normalization preset:
    #   "imagenet" - mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], scale=1/255
    #   "ppocr"    - mean=[0.5,0.5,0.5],      std=[0.5,0.5,0.5],      scale=1/127.5
    #   "none"     - only divide by 255
    #   "custom"   - use the mean/std/scale fields below
    normalize_mode: str = "custom"

    # Custom normalization values (used when normalize_mode == "custom").
    # Set mean/std to [] to disable mean/std subtraction.
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    scale: float = 1.0 / 255.0

    # ------------------------- Post-process (DBNet) -------------------------
    # Binarization threshold for the probability map.
    thresh: float = 0.35

    # Minimum mean probability inside a candidate box to keep it.
    box_thresh: float = 0.25

    # Polygon expansion ratio. 1.0 = no expansion, 1.5~2.0 = standard DB unclip.
    unclip_ratio: float = 1.75

    # Minimum / maximum box area (in original image pixels) to keep.
    min_box_area: float = 10.0
    max_box_area: float = float("inf")

    # Minimum / maximum aspect ratio (w/h) of a box to keep.
    min_aspect_ratio: float = 0.05
    max_aspect_ratio: float = 50.0

    # Maximum number of detected boxes.
    max_candidates: int = 1000

    # Box representation:
    #   "minarearect" - 4-point minimum-area rectangle (recommended for recognition)
    #   "poly"        - raw approximated polygon (more points, slower)
    #   "quad"        - same as minarearect but explicitly 4-point
    box_type: str = "minarearect"

    # Whether to use pyclipper for accurate polygon un-clipping.
    # Falls back to a geometric approximation if pyclipper is unavailable.
    use_pyclipper: bool = True

    # Whether to dilate the binary map before finding contours.
    # Helps connect broken text strokes but may merge close boxes.
    use_dilate: bool = False
    dilate_kernel_size: int = 2

    # NMS threshold between candidate boxes. <0 means skip NMS.
    nms_threshold: float = -1.0

    # Reading-order sorting:
    #   "top2bottom" - strict top-to-bottom, then left-to-right
    #   "left2right" - strict left-to-right, then top-to-bottom
    #   "natural"    - group boxes into text lines then sort (default)
    sort_mode: str = "natural"

    # Controls how aggressively boxes are grouped into the same text line
    # when sort_mode == "natural". Smaller -> more lines, larger -> fewer lines.
    line_cluster_factor: float = 0.5


@dataclass
class RecConfig:
    """Text-recognition configuration."""

    # Fixed input shape (C, H, W). W is the target width after padding/cropping.
    rec_image_shape: tuple = (3, 48, 320)

    # Maximum width; longer images are down-scaled, shorter are padded.
    max_text_width: int = 320

    # Minimum width in pixels to guarantee enough CTC time steps.
    min_text_width: int = 96  # default = H * 2

    # Resize strategy:
    #   "fixed_height_pad"    - fixed height, pad to target width (default, preserves aspect)
    #   "fixed_size_stretch"  - stretch to exact rec_image_shape (fastest)
    #   "fixed_height_stretch"- fixed height, stretch width to target (may distort)
    resize_mode: str = "fixed_height_pad"

    # Padding alignment when resize_mode == "fixed_height_pad".
    #   "left" - text at left edge, pad right (PP-OCR training convention;
    #            verified on the v6 rec model: center padding causes
    #            dropped characters, left padding is correct)
    #   "center", "right"
    pad_align: str = "left"

    # Normalization preset. Same values as DetConfig.
    normalize_mode: str = "ppocr"

    # Custom normalization values (used when normalize_mode == "custom").
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    scale: float = 1.0 / 127.5

    # CTC blank index. 0 = first class (PP-OCR/PyTorch convention, dict chars
    # start at class 1). -1 = last class.
    blank_index: int = 0

    # Batch size for recognition. Dynamic-shape models always infer one-by-one.
    batch_size: int = 8

    # Whether to apply a 180-degree flip correction if recognition confidence is low.
    # Requires running the image twice; useful for upside-down text lines.
    use_direction_ensemble: bool = False
    direction_ensemble_thresh: float = 0.6


@dataclass
class RotateConfig:
    """Large-angle rotation classification configuration."""

    # Input size expected by the rotation model.
    rotate_image_shape: tuple = (3, 224, 224)

    # Class index -> rotation angle (degrees, counter-clockwise).
    # The order must match the model output order.
    label_list: List[int] = field(default_factory=lambda: [0, 180, 270, 90])

    # Minimum confidence to apply rotation. Below this threshold, skip rotation.
    rotate_min_confidence: float = 0.9

    # Normalization preset. Same values as DetConfig.
    normalize_mode: str = "ppocr"

    # Custom normalization values (used when normalize_mode == "custom").
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    scale: float = 1.0 / 127.5


@dataclass
class TableConfig:
    """Table cell detection configuration (YOLO-based)."""

    # Input size expected by the table model (H, W).
    input_size: int = 640

    # Confidence threshold for cell detections.
    # Model outputs are already probabilities; background detections have
    # max(class_scores) ~0.5, real detections ~0.7+. Use 0.6 as default.
    score_threshold: float = 0.6

    # NMS IoU threshold for suppressing overlapping detections.
    nms_threshold: float = 0.5

    # Maximum number of detections to keep after NMS.
    max_detections: int = 1000

    # Overlap ratio threshold for assigning text lines to cells.
    text_cell_overlap: float = 0.5

    # Normalization: only /255, no mean/std subtraction (YOLO convention).
    scale: float = 1.0 / 255.0


@dataclass
class OCRConfig:
    """Top-level OCR engine configuration."""

    # Model paths (OM files).
    det_model: Optional[str] = None
    rec_model: Optional[str] = None
    rotate_model: Optional[str] = None
    layout_model: Optional[str] = None
    table_model: Optional[str] = None

    # Character dictionary path for recognition.
    rec_char_dict: Optional[str] = None

    # NPU device id.
    device_id: int = 0

    # Optional callback used to decrypt an encrypted OM model.
    # Signature: decrypt_callback(path: str) -> bytes
    decrypt_callback: Optional[Callable[[str], bytes]] = None

    # Sub-configs.
    det: DetConfig = field(default_factory=DetConfig)
    rec: RecConfig = field(default_factory=RecConfig)
    rotate: RotateConfig = field(default_factory=RotateConfig)
    table: TableConfig = field(default_factory=TableConfig)

    # Whether to run large-angle classification (0/90/180/270) and auto-rotate.
    # Can be overridden per-call via the use_rotate parameter.
    use_rotate: bool = True

    # Whether to return debug visualizations.
    debug: bool = False


def _apply_dict(obj: Any, data: Dict[str, Any]) -> None:
    """Apply a dict to a dataclass instance, skipping None values."""
    for key, value in data.items():
        if value is None:
            continue
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(obj, key, value)


def load_config(yaml_path: str, models_root: str = "models") -> OCRConfig:
    """
    Load OCRConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.
        models_root: Root directory for models (default: "models").

    Returns:
        OCRConfig instance.

    Raises:
        FileNotFoundError: If yaml_path does not exist.
        ValueError: If YAML contains unknown config keys.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Build sub-configs
    det = DetConfig()
    rec = RecConfig()
    rotate = RotateConfig()
    table = TableConfig()

    if "det" in data and data["det"]:
        _apply_dict(det, data["det"])
    if "rec" in data and data["rec"]:
        _apply_dict(rec, data["rec"])
    if "rotate" in data and data["rotate"]:
        _apply_dict(rotate, data["rotate"])
    if "table" in data and data["table"]:
        _apply_dict(table, data["table"])

    # Build top-level config
    cfg = OCRConfig(det=det, rec=rec, rotate=rotate, table=table)

    # Handle chip-based model paths
    chip = data.get("chip")
    det_model = data.get("det_model")
    rec_model = data.get("rec_model")
    rotate_model = data.get("rotate_model")
    layout_model = data.get("layout_model")
    table_model = data.get("table_model")

    if chip:
        chip_dir = os.path.join(models_root, chip)
        if not det_model:
            det_model = os.path.join(chip_dir, "det.om")
        if not rec_model:
            rec_model = os.path.join(chip_dir, "rec.om")
        if not rotate_model:
            rotate_model = os.path.join(chip_dir, "rotate.om")
        if not layout_model:
            layout_model = os.path.join(chip_dir, "PP-DocLayoutV3.om")
        if not table_model:
            table_model = os.path.join(chip_dir, "table.om")

    # Apply top-level fields
    if det_model:
        cfg.det_model = det_model
    if rec_model:
        cfg.rec_model = rec_model
    if rotate_model:
        cfg.rotate_model = rotate_model
    if layout_model:
        cfg.layout_model = layout_model
    if table_model:
        cfg.table_model = table_model
    if "rec_char_dict" in data and data["rec_char_dict"]:
        cfg.rec_char_dict = data["rec_char_dict"]
    elif cfg.rec_char_dict is None:
        cfg.rec_char_dict = default_char_dict_path()
    if "device_id" in data and data["device_id"] is not None:
        cfg.device_id = data["device_id"]
    if "use_rotate" in data and data["use_rotate"] is not None:
        cfg.use_rotate = data["use_rotate"]
    if "debug" in data and data["debug"] is not None:
        cfg.debug = data["debug"]

    return cfg
