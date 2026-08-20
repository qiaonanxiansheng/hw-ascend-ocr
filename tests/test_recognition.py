"""Unit tests for CTC decoding."""

import numpy as np

from ascend_ocr.recognition import CTCDecoder


def test_ctc_decode_simple():
    # Decoder appends blank automatically at the end of the char list.
    decoder = CTCDecoder(None)
    decoder.char_list = ["a", "b", "c", ""]  # blank is last (index 3)

    # Sequence: blank, a, a, blank, b, c, c, blank
    probs = np.zeros((8, 4), dtype=np.float32)
    probs[0, 3] = 1.0  # blank
    probs[1, 0] = 1.0  # a
    probs[2, 0] = 1.0  # a
    probs[3, 3] = 1.0  # blank
    probs[4, 1] = 1.0  # b
    probs[5, 2] = 1.0  # c
    probs[6, 2] = 1.0  # c
    probs[7, 3] = 1.0  # blank

    text, score = decoder.decode(probs)
    assert text == "abc"
