"""
Example: run OCR on a local image or URL with the AscendOCR engine.
"""

import argparse
import logging
import sys

sys.path.insert(0, "..")

from ascend_ocr import AscendOCR
from ascend_ocr.config import default_char_dict_path


def main():
    parser = argparse.ArgumentParser(description="AscendOCR example")
    parser.add_argument("image", help="Local path or HTTP(S) URL of the image")
    parser.add_argument("--det-model", required=True, help="Path to detection OM model")
    parser.add_argument("--rec-model", required=True, help="Path to recognition OM model")
    parser.add_argument("--cls-model", default=None, help="Path to angle classification OM model")
    parser.add_argument(
        "--rec-char-dict",
        default=default_char_dict_path(),
        help="Path to recognition char dict",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id")
    parser.add_argument("--vis", default="./ocr_result.jpg", help="Path to save visualization")
    parser.add_argument(
        "--save-crops",
        default=None,
        help="Directory to save each cropped text-line image (for debugging)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = AscendOCR(
        det_model=args.det_model,
        rec_model=args.rec_model,
        cls_model=args.cls_model,
        rec_char_dict=args.rec_char_dict,
        device_id=args.device,
    )

    results, angle, vis = engine.ocr(args.image, return_visualization=True)

    print("\n=== OCR Results ===")
    for idx, r in enumerate(results, 1):
        pts = r.box.reshape(-1, 2).astype(int).tolist()
        print(f"{idx}. [{r.score:.3f}] {r.text}")
        print(f"   box={pts}")

    if args.save_crops:
        import os

        import cv2

        from ascend_ocr.image_utils import load_image

        os.makedirs(args.save_crops, exist_ok=True)
        src = load_image(args.image)
        crops = engine.detector.crop_text_lines(src, [r.box for r in results])
        for idx, (crop, _) in enumerate(crops, 1):
            out = os.path.join(args.save_crops, f"line_{idx:02d}.png")
            cv2.imwrite(out, crop)
        print(f"\n{len(crops)} text-line crops saved to {args.save_crops}/")

    if args.vis:
        import cv2

        cv2.imwrite(args.vis, vis)
        print(f"\nVisualization saved to {args.vis}")

    engine.release()


if __name__ == "__main__":
    main()
