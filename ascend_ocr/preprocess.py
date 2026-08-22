"""
Model-specific preprocessing for detection, angle classification and recognition.

All functions return numpy arrays ready to be passed to ``AscendModel.infer``.
Algorithm choices are driven by the config dataclasses.
"""

from typing import Tuple

import cv2
import numpy as np

from .config import ClsConfig, DetConfig, RecConfig
from .exceptions import PreprocessError
from .image_utils import normalize_image, resize_with_aspect_ratio

import logging

logger = logging.getLogger(__name__)


def _get_norm_params(cfg):
    """Return (scale, mean, std) from a config with normalize_mode support."""
    mode = getattr(cfg, "normalize_mode", "custom").lower()
    if mode == "imagenet":
        return 1.0 / 255.0, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if mode == "ppocr":
        return 1.0 / 255.0, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if mode == "none":
        return 1.0 / 255.0, None, None
    # custom
    mean = tuple(cfg.mean) if cfg.mean else None
    std = tuple(cfg.std) if cfg.std else None
    return cfg.scale, mean, std


def preprocess_for_detection(
    img: np.ndarray, cfg: DetConfig
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize an image for the detection model and normalize it.

    Args:
        img: BGR image, shape (H, W, 3).
        cfg: Detection configuration.

    Returns:
        (input_tensor, scale_factors, resized_shape)
            input_tensor: float32 array shape (1, 3, new_h, new_w).
            scale_factors: (scale_h, scale_w) from original -> resized.
            resized_shape: (resized_h, resized_w) after resize, before normalization.
    """
    h, w = img.shape[:2]

    if cfg.fixed_input_size is not None:
        # Static-shape model: stretch to the exact declared input size.
        resize_h, resize_w = cfg.fixed_input_size
        resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        scale_h = resize_h / h
        scale_w = resize_w / w
        logger.debug(
            "Det preprocess: fixed input size, stretched %dx%d -> %dx%d",
            w, h, resize_w, resize_h,
        )
    else:
        max_side = max(h, w)
        ratio = cfg.limit_side_len / max_side
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # Round to multiple of limit_multiple.
        resize_h = max(
            cfg.limit_multiple, (resize_h // cfg.limit_multiple) * cfg.limit_multiple
        )
        resize_w = max(
            cfg.limit_multiple, (resize_w // cfg.limit_multiple) * cfg.limit_multiple
        )

        mode = cfg.resize_mode.lower()
        if mode == "pad":
            resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            scale_h = resize_h / h
            scale_w = resize_w / w
        elif mode == "stretch":
            resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            scale_h = resize_h / h
            scale_w = resize_w / w
        elif mode == "crop_center":
            # First scale so the smaller side fits, then center crop.
            scale = max(resize_h / h, resize_w / w)
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(
                img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
            )
            start_y = (scaled_h - resize_h) // 2
            start_x = (scaled_w - resize_w) // 2
            resized = scaled[start_y : start_y + resize_h, start_x : start_x + resize_w]
            scale_h = scaled_h / h
            scale_w = scaled_w / w
        else:
            raise PreprocessError(f"Unknown det.resize_mode: {cfg.resize_mode}")
        logger.debug(
            "Det preprocess: resized %dx%d -> %dx%d (mode=%s)",
            w, h, resize_w, resize_h, mode,
        )

    scale, mean, std = _get_norm_params(cfg)
    tensor = normalize_image(resized, scale=scale, mean=mean, std=std)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, (scale_h, scale_w), (resize_h, resize_w)


def preprocess_for_classification(img: np.ndarray, cfg: ClsConfig) -> np.ndarray:
    """
    Preprocess an image for the large-angle classifier.

    Args:
        img: BGR image.
        cfg: Classification configuration.

    Returns:
        Float32 tensor shape (1, 3, H, W).
    """
    _, target_h, target_w = cfg.cls_image_shape
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scale, mean, std = _get_norm_params(cfg)
    tensor = normalize_image(resized, scale=scale, mean=mean, std=std)
    return np.expand_dims(tensor, axis=0)


def preprocess_for_recognition(
    img: np.ndarray, cfg: RecConfig
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Preprocess a cropped text-line image for the recognition model.

    Args:
        img: BGR text-line image.
        cfg: Recognition configuration.

    Returns:
        (input_tensor, original_cropped_size)
            input_tensor: float32 array shape (1, 3, H, W).
            original_cropped_size: (h, w) of the resized text line before padding.
    """
    _, target_h, target_w = cfg.rec_image_shape
    h, w = img.shape[:2]

    mode = cfg.resize_mode.lower()
    if mode == "fixed_height_pad":
        # 1. Resize to target height, keep aspect ratio
        # 2. If width > target_w, resize to target_w (squeeze)
        # 3. If width < target_w, pad to target_w (left-aligned by default)
        ratio = target_h / h
        resized_w = max(int(w * ratio), 1)
        resized_w = max(resized_w, cfg.min_text_width)
        resized_w = min(resized_w, cfg.max_text_width)
        resized = cv2.resize(img, (resized_w, target_h), interpolation=cv2.INTER_LINEAR)

        if resized_w > target_w:
            # Image wider than model input: squeeze to target width
            resized = cv2.resize(resized, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        elif resized_w < target_w:
            # Image narrower than model input: pad (left-aligned)
            pad_total = target_w - resized_w
            pad_left, pad_right = 0, pad_total  # left padding
            resized = cv2.copyMakeBorder(
                resized,
                0,
                0,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

    elif mode == "fixed_size_stretch":
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_w = target_w

    elif mode == "fixed_height_stretch":
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_w = target_w

    else:
        raise PreprocessError(f"Unknown rec.resize_mode: {cfg.resize_mode}")

    scale, mean, std = _get_norm_params(cfg)
    tensor = normalize_image(resized, scale=scale, mean=mean, std=std)
    tensor = np.expand_dims(tensor, axis=0)
    logger.debug(
        "Rec preprocess: crop %dx%d -> tensor %s (mode=%s)",
        w, h, tensor.shape, cfg.resize_mode,
    )
    return tensor, (target_h, resized_w)
