"""Custom exceptions used by the OCR package."""


class AscendOCRError(Exception):
    """Base exception for all OCR package errors."""


class ModelLoadError(AscendOCRError):
    """Raised when an Ascend OM model fails to load."""


class InferenceError(AscendOCRError):
    """Raised when model inference fails."""


class ImageLoadError(AscendOCRError):
    """Raised when an image cannot be loaded from path or URL."""


class PreprocessError(AscendOCRError):
    """Raised when image preprocessing fails."""


class PostprocessError(AscendOCRError):
    """Raised when model output post-processing fails."""
