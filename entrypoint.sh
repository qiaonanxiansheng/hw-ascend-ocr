#!/bin/bash
#
# 容器入口脚本
#
# 流程:
#   1. 加载 CANN 环境
#   2. 根据 CONVERT_MODELS 环境变量决定是否转换模型:
#        auto   - (默认) models/{chip}/ 下缺少 .om 时才转换
#        always - 强制重新转换所有模型
#        never  - 不转换, 直接启动服务
#   3. 启动 OCR API 服务(或执行传入的其他命令)
#
# 模型来源: 挂载目录中的 models/onnx/*.onnx -> atc -> models/{chip}/*.om
# chip 从 config.yaml 读取, 转换和加载使用同一字段, 保证一致。
#
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh

CONVERT_MODELS="${CONVERT_MODELS:-auto}"

# 从 config.yaml 读取 chip 字段
CHIP=$(python -c "
import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    print((yaml.safe_load(f) or {}).get('chip') or '')
")
if [ -z "$CHIP" ]; then
    echo "[entrypoint] 错误: config.yaml 中未配置 chip 字段"
    exit 1
fi

MODEL_DIR="models/${CHIP}"
echo "[entrypoint] chip=${CHIP} 模型目录=${MODEL_DIR} CONVERT_MODELS=${CONVERT_MODELS}"

# 判断是否需要转换
need_convert=0
case "$CONVERT_MODELS" in
    always)
        need_convert=1
        ;;
    never)
        need_convert=0
        ;;
    auto)
        if [ ! -d "$MODEL_DIR" ] || ! ls "$MODEL_DIR"/*.om > /dev/null 2>&1; then
            need_convert=1
        fi
        ;;
    *)
        echo "[entrypoint] 错误: CONVERT_MODELS 只能是 auto / always / never, 当前值: $CONVERT_MODELS"
        exit 1
        ;;
esac

if [ "$need_convert" = "1" ]; then
    if ! ls models/onnx/*.onnx > /dev/null 2>&1; then
        echo "[entrypoint] 错误: 需要转换模型, 但 models/onnx/ 目录下没有 ONNX 文件"
        echo "[entrypoint] 请将 ONNX 模型放入挂载的 models/onnx/ 目录, 或设置 CONVERT_MODELS=never 并提供已转换的 OM 模型"
        exit 1
    fi
    echo "[entrypoint] 开始转换模型 (models/onnx/*.onnx -> ${MODEL_DIR}/*.om) ..."
    python convert_encrypt_models.py "$CHIP"
    echo "[entrypoint] 模型转换完成"
else
    echo "[entrypoint] 跳过模型转换"
fi

# 默认启动 OCR 服务; 若 docker run 传入了命令则执行传入的命令
if [ $# -eq 0 ]; then
    echo "[entrypoint] 启动 OCR 服务 (端口 13502) ..."
    exec uvicorn api.app:app --host 0.0.0.0 --port 13502
else
    exec "$@"
fi
