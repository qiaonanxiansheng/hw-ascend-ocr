"""Custom exceptions used by the OCR package."""


class YuntuAscendOCRError(Exception):
    """Base exception for all OCR package errors."""


class ModelLoadError(YuntuAscendOCRError):
    """Raised when an Ascend OM model fails to load."""


class InferenceError(YuntuAscendOCRError):
    """Raised when model inference fails."""


class ImageLoadError(YuntuAscendOCRError):
    """Raised when an image cannot be loaded from path or URL."""


class PreprocessError(YuntuAscendOCRError):
    """Raised when image preprocessing fails."""


class PostprocessError(YuntuAscendOCRError):
    """Raised when model output post-processing fails."""
