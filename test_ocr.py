"""
Quick test: run OCR on docs/test.jpg and print results.

Usage:
    python test_ocr.py                          # 使用 config.yaml 配置
    python test_ocr.py --config my_config.yaml  # 指定配置文件
    python test_ocr.py --chip 910B3             # 指定芯片型号
"""

import argparse
import logging
import os

from ascend_ocr import AscendOCR, load_config

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="OCR test on docs/test.jpg")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--chip", default=None, help="芯片型号（覆盖配置文件）")
    parser.add_argument("--det", default=None, help="检测模型路径（覆盖配置文件）")
    parser.add_argument("--rec", default=None, help="识别模型路径（覆盖配置文件）")
    parser.add_argument("--cls", default=None, help="分类模型路径")
    parser.add_argument("--device", type=int, default=None, help="NPU 设备 ID")
    parser.add_argument("--image", default="docs/test.jpg", help="测试图片路径")
    args = parser.parse_args()

    # 加载配置文件
    if os.path.exists(args.config):
        cfg = load_config(args.config)
        print(f"已加载配置: {args.config}")
    else:
        from ascend_ocr.config import OCRConfig, default_char_dict_path
        cfg = OCRConfig(rec_char_dict=default_char_dict_path())
        print(f"配置文件不存在: {args.config}，使用默认配置")

    # 命令行参数覆盖配置
    if args.chip:
        cfg.det_model = os.path.join("models", args.chip, "det.om")
        cfg.rec_model = os.path.join("models", args.chip, "rec.om")
        cfg.cls_model = os.path.join("models", args.chip, "cls.om")
    if args.det:
        cfg.det_model = args.det
    if args.rec:
        cfg.rec_model = args.rec
    if args.cls:
        cfg.cls_model = args.cls
    if args.device is not None:
        cfg.device_id = args.device

    # 根据 debug 配置设置日志级别
    log_level = logging.DEBUG if cfg.debug else logging.INFO
    logging.getLogger().setLevel(log_level)

    print(f"检测模型: {cfg.det_model}")
    print(f"识别模型: {cfg.rec_model}")
    print(f"设备 ID: {cfg.device_id}")

    engine = AscendOCR(cfg)

    results, angle = engine.ocr(args.image)

    print(f"\n{'='*50}")
    print(f"检测到 {len(results)} 行文字, 旋转角度: {angle}°")
    print(f"{'='*50}")
    for idx, r in enumerate(results, 1):
        pts = r.box.reshape(-1, 2).astype(int).tolist()
        print(f"{idx:2d}. [{r.score:.3f}] {r.text}")
        print(f"     坐标: {pts}")

    engine.release()


if __name__ == "__main__":
    main()
