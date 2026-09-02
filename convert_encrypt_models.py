#!/usr/bin/env python3
"""
ONNX -> OM 批量转换脚本(替代原 convert_onnx_to_om.sh, 已去除模型加密)

逻辑:
- 按芯片型号循环, 调用 atc 将 models/onnx/*.onnx 转为 models/{chip}/*.om
- 全部模型统一使用 --precision_mode=must_keep_origin_dtype
  --op_select_implmode=high_precision 保持原始精度

必须在装有 CANN(atc 命令可用) 的环境执行, 通常在目标昇腾机器/容器内运行:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

用法:
    python convert_encrypt_models.py              # 转换 config.yaml 中 chip 指定的芯片
    python convert_encrypt_models.py 310 910B3    # 只转换指定芯片(覆盖 config)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
ONNX_DIR = ROOT / "models" / "onnx"
OUTPUT_ROOT = ROOT / "models"
CONFIG_YAML = ROOT / "config.yaml"

# 模型定义: (onnx文件名, 输出名, input_shape, input_format(None 则不传))
MODELS = [
    # 1. 大角度分类模型
    ("rotate.onnx", "rotate", "images:1,3,512,512", None),
    # 2. 文字检测模型
    ("v6_small_det.onnx", "det", "x:1,3,960,672", None),
    # 3. 文字识别模型
    ("v6_small_rec.onnx", "rec", "x:1,3,48,960", None),
    # 4. 版面分析模型
    ("PP-DocLayoutV3.onnx", "PP-DocLayoutV3",
     "im_shape:1,2;image:1,3,800,800;scale_factor:1,2", "ND"),
    # 5. 表格识别模型
    ("table.onnx", "table", "images:1,3,640,640", "NCHW"),
]


def convert_model(onnx_file, output_name, input_shape, input_format,
                  soc_version, output_dir):
    """转换单个模型, 返回 True/False/None(跳过)"""
    onnx_path = ONNX_DIR / onnx_file
    output_path = output_dir / output_name  # atc 会自动追加 .om 后缀

    if not onnx_path.is_file():
        print(f"  跳过: {onnx_file} 不存在")
        return None

    print(f"  转换: {onnx_file} -> {output_name}.om")

    cmd = [
        "atc",
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_path}",
        f"--input_shape={input_shape}",
        f"--soc_version={soc_version}",
        "--precision_mode=must_keep_origin_dtype",
        "--op_select_implmode=high_precision",
        "--plugin=ByPass",
    ]
    if input_format:
        cmd.append(f"--input_format={input_format}")

    ret = subprocess.run(cmd, cwd=ROOT).returncode
    if ret == 0:
        print(f"  成功: {output_path}.om")
        return True
    print(f"  失败: {onnx_file} 转换出错")
    return False


def convert_chip(chip, stats):
    """转换单个芯片的所有模型"""
    soc_version = f"Ascend{chip}"
    output_dir = OUTPUT_ROOT / chip

    print()
    print("-" * 42)
    print(f"芯片: {chip} ({soc_version})")
    print(f"输出目录: {output_dir.relative_to(ROOT)}")
    print("-" * 42)

    output_dir.mkdir(parents=True, exist_ok=True)

    for onnx_file, output_name, input_shape, input_format in MODELS:
        result = convert_model(onnx_file, output_name, input_shape,
                               input_format, soc_version, output_dir)
        if result is None:
            stats["skip"] += 1
        else:
            stats["total"] += 1
            stats["success" if result else "fail"] += 1


def load_config_chips():
    """从 config.yaml 读取 chip 字段作为默认转换目标"""
    if not CONFIG_YAML.is_file():
        print(f"错误: 配置文件不存在: {CONFIG_YAML.relative_to(ROOT)}")
        sys.exit(1)
    with open(CONFIG_YAML, "r", encoding="utf-8") as f:
        chip = (yaml.safe_load(f) or {}).get("chip")
    if not chip:
        print(f"错误: {CONFIG_YAML.relative_to(ROOT)} 中未配置 chip")
        sys.exit(1)
    return [str(chip)]


def main():
    chips = sys.argv[1:] or load_config_chips()

    if not ONNX_DIR.is_dir():
        print(f"错误: ONNX 目录不存在: {ONNX_DIR.relative_to(ROOT)}")
        sys.exit(1)
    if shutil.which("atc") is None:
        print("错误: atc 命令未找到")
        sys.exit(1)

    stats = {"total": 0, "success": 0, "fail": 0, "skip": 0}

    print("=" * 42)
    print("ONNX -> OM 批量转换")
    print(f"目标芯片: {', '.join(chips)}")
    print("=" * 42)

    for chip in chips:
        convert_chip(chip, stats)

    print()
    print("=" * 42)
    print(f"转换完成: 成功 {stats['success']}/{stats['total']}, "
          f"失败 {stats['fail']}, 跳过 {stats['skip']}")
    print("=" * 42)

    if stats["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
