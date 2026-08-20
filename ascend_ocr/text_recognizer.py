"""Text recognition: wraps the recognition OM model and CTC decoder."""

import logging
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import OCRConfig, RecConfig
from .image_utils import rotate_image
from .model import AscendModel
from .preprocess import preprocess_for_recognition
from .recognition import CTCDecoder, softmax

logger = logging.getLogger(__name__)


def _to_probs(arr: np.ndarray) -> np.ndarray:
    """Return per-class probabilities, skipping softmax if the model output
    is already normalized (some OM models embed a Softmax layer)."""
    if arr.min() >= 0.0 and np.allclose(arr.sum(axis=-1), 1.0, atol=1e-2):
        return arr
    return softmax(arr, axis=-1)


class TextRecognizer:
    """Wraps the text-recognition OM model."""

    def __init__(
        self,
        model_path: str,
        char_dict_path: Optional[str] = None,
        cfg: Optional[RecConfig] = None,
        device_id: int = 0,
        decrypt_callback: Optional[Callable[[str], bytes]] = None,
    ):
        self.cfg = cfg or RecConfig()
        self.model = AscendModel(
            model_path, device_id=device_id, decrypt_callback=decrypt_callback
        )
        if not self.model.dynamic_input:
            shape = self.model.get_input_shape(0)
            if len(shape) == 4:
                batch, _, mh, mw = (int(s) for s in shape)
                cur = self.cfg.rec_image_shape
                if (cur[1], cur[2]) != (mh, mw):
                    logger.debug(
                        "Static rec model, input shape %s: overriding "
                        "rec_image_shape (%d,%d,%d) -> (3,%d,%d)",
                        list(shape), cur[0], cur[1], cur[2], mh, mw,
                    )
                    self.cfg.rec_image_shape = (3, mh, mw)
                    self.cfg.max_text_width = mw
                if batch == 1 and self.cfg.batch_size != 1:
                    logger.debug(
                        "Static rec model with batch=1: disabling batching"
                    )
                    self.cfg.batch_size = 1
        self.decoder = CTCDecoder(
            char_dict_path, blank_index=self.cfg.blank_index
        )

    def recognize(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a single text-line image.

        Args:
            img: BGR text-line image.

        Returns:
            (text, confidence)
        """
        result = self._infer_one(img)

        if self.cfg.use_direction_ensemble:
            text, conf = result
            if conf < self.cfg.direction_ensemble_thresh:
                flipped = rotate_image(img, 180)
                flipped_text, flipped_conf = self._infer_one(flipped)
                if flipped_conf > conf:
                    logger.debug(
                        "Direction ensemble: flipped chosen (%.3f > %.3f)",
                        flipped_conf,
                        conf,
                    )
                    return flipped_text, flipped_conf
        return result

    def _log_rec_result(self, idx: int, text: str, conf: float, elapsed_ms: float) -> None:
        """Debug-level per-line recognition log."""
        logger.debug(
            "  %2d. 置信度: %.3f, 耗时: %.1fms, 文字: %s",
            idx, conf, elapsed_ms, text,
        )

    def _infer_one(self, img: np.ndarray) -> Tuple[str, float]:
        tensor, _ = preprocess_for_recognition(img, self.cfg)
        outputs = self.model.infer([tensor])
        logits = outputs[0]
        probs = _to_probs(logits)
        text, conf = self.decoder.decode(probs)
        return text, conf

    def recognize_batch(
        self, images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Recognize a batch of text-line images.

        Dynamic-shape recognition models infer each image separately because
        text-line widths vary. Static-shape models are padded and batched.
        """
        if self.model.dynamic_input or self.cfg.batch_size <= 1:
            results = []
            for i, img in enumerate(images):
                t0 = time.perf_counter()
                text, conf = self.recognize(img)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._log_rec_result(i + 1, text, conf, elapsed_ms)
                results.append((text, conf))
            return results

        results = []
        batch_size = max(1, self.cfg.batch_size)
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            if len(batch) == 1:
                t0 = time.perf_counter()
                text, conf = self.recognize(batch[0])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._log_rec_result(i + 1, text, conf, elapsed_ms)
                results.append((text, conf))
            else:
                t0 = time.perf_counter()
                batch_results = self._recognize_static_batch(batch)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                for j, (text, conf) in enumerate(batch_results):
                    self._log_rec_result(i + j + 1, text, conf, elapsed_ms / len(batch))
                results.extend(batch_results)
        return results

    def _recognize_static_batch(
        self, images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Batch inference when the recognition model has a fixed input width."""
        # Preprocess each image.
        processed = []
        for img in images:
            tensor, _ = preprocess_for_recognition(img, self.cfg)
            processed.append(tensor)

        # Pad to max width in batch.
        max_w = max(t.shape[3] for t in processed)
        padded = []
        for t in processed:
            _, c, h, w = t.shape
            if w < max_w:
                pad = np.zeros((1, c, h, max_w - w), dtype=np.float32)
                t = np.concatenate([t, pad], axis=3)
            padded.append(t)
        batch_tensor = np.concatenate(padded, axis=0)

        outputs = self.model.infer([batch_tensor])
        probs = _to_probs(outputs[0])
        return self.decoder.decode_batch(list(probs))

    def release(self) -> None:
        self.model.release()
