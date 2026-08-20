"""
Text recognition helpers: character dictionary loading and CTC decoding.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from .config import RecConfig
from .exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class CTCDecoder:
    """Greedy CTC decoder with blank merging."""

    def __init__(self, char_dict_path: Optional[str], blank_index: int = -1):
        """
        Args:
            char_dict_path: Path to a text file with one character per line.
                If None, a default Latin/number dictionary is used.
            blank_index: Index of the CTC blank label. 0 means the first class
                (PP-OCR/PyTorch convention: model class j maps to dict line
                j-1). -1 means the last class.
        """
        self.blank_index = blank_index
        self.char_list = self._load_char_dict(char_dict_path)
        logger.info(
            "Loaded character dictionary with %d classes from %s (blank_index=%d)",
            len(self.char_list),
            char_dict_path or "<built-in>",
            blank_index,
        )

    def _load_char_dict(self, path: Optional[str]) -> List[str]:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chars = [line.rstrip("\n") for line in f]
            if self.blank_index == 0:
                # Prepend blank token if not present.
                if not chars or chars[0] != "":
                    chars.insert(0, "")
            else:
                # Append blank token at the end if not present.
                if not chars or chars[-1] != "":
                    chars.append("")
            return chars

        if path:
            raise ModelLoadError(f"Recognition char dict not found: {path}")

        logger.warning(
            "No char dict provided; using built-in ASCII fallback dictionary. "
            "CJK characters will NOT be decoded. Pass char_dict_path for "
            "non-Latin models."
        )
        # Fallback: ASCII printable characters + a few CJK samples.
        chars = [""]  # blank
        chars.extend(chr(i) for i in range(32, 127))
        # Common digits and punctuation for Chinese OCR.
        chars.extend(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ".,;:!?-_'\"()[]{}@#$%&*+-/=<>|\\~`"
        )
        chars = list(dict.fromkeys(chars))  # dedupe, keep order
        return chars

    def decode(self, probs: np.ndarray) -> Tuple[str, float]:
        """
        Greedy decode a single CTC probability sequence.

        Args:
            probs: Array of shape (T, C) or (1, T, C) with class probabilities
                or raw logits. If logits are provided, argmax is used directly.

        Returns:
            (decoded_text, mean_confidence)
        """
        if probs.ndim == 3:
            probs = probs[0]
        if probs.ndim != 2:
            raise ValueError(f"Expected 2-D CTC output, got shape {probs.shape}")

        indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        blank = self.blank_index if self.blank_index >= 0 else len(self.char_list) - 1

        # 预过滤：用 numpy 找出非空白、非重复的位置，减少 Python 循环次数
        not_blank = indices != blank
        not_repeat = np.empty(len(indices), dtype=bool)
        not_repeat[0] = True
        not_repeat[1:] = indices[1:] != indices[:-1]
        valid = not_blank & not_repeat
        valid_indices = indices[valid]
        valid_confs = confidences[valid]

        # 构建文本（仍需 Python 循环查 char_list）
        char_list = self.char_list
        n_chars = len(char_list)
        text_chars = []
        confs = []
        for idx, conf in zip(valid_indices, valid_confs):
            if 0 <= idx < n_chars:
                text_chars.append(char_list[idx])
                confs.append(conf)

        text = "".join(text_chars)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return text, mean_conf

    def decode_batch(
        self, probs_batch: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Decode a batch of CTC outputs."""
        return [self.decode(p) for p in probs_batch]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
