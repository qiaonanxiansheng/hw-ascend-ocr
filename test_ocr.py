"""
Quick test: run OCR on docs/test.jpg and print results.

Usage:
    python test_ocr.py
    python test_ocr.py --det models/det/v6_small_det.om --rec models/rec/v6_small_rec.om
"""

import argparse
import logging

from ascend_ocr import AscendOCR
from ascend_ocr.config import default_char_dict_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="OCR test on docs/test.jpg")
    parser.add_argument("--det", default="models/det/v6_small_det.om")
    parser.add_argument("--rec", default="models/rec/v6_small_rec.om")
    parser.add_argument("--cls", default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    engine = AscendOCR(
        det_model=args.det,
        rec_model=args.rec,
        cls_model=args.cls,
        rec_char_dict=default_char_dict_path(),
        device_id=args.device,
    )

    results = engine.ocr("docs/test.jpg")

    print(f"\n{'='*50}")
    print(f"Detected {len(results)} text lines")
    print(f"{'='*50}")
    for idx, r in enumerate(results, 1):
        print(f"{idx:2d}. [{r.score:.3f}] {r.text}")

    engine.release()


if __name__ == "__main__":
    main()
