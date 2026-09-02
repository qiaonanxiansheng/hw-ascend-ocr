#!/bin/bash
#
# 容器入口脚本
#
# 流程:
#   1. 打印环境诊断信息(排查问题时把这段日志完整贴出来)
#   2. 修正 python3-config(atc 依赖, 见下方注释)
#   3. 根据 CONVERT_MODELS 环境变量决定是否转换模型:
#        auto   - (默认) models/{chip}/ 下缺少 .om 时才转换
#        always - 强制重新转换所有模型
#        never  - 不转换, 直接启动服务
#   4. 启动 OCR API 服务(或执行传入的其他命令)
#
# 模型来源: 挂载目录中的 models/onnx/*.onnx -> atc -> models/{chip}/*.om
# chip 从 config.yaml 读取, 转换和加载使用同一字段, 保证一致。
#
# 注意: CANN 基础镜像已自带环境变量(ASCEND_OPP_PATH 等), 无需 source set_env.sh
#
set -e

ENTRYPOINT_VERSION="2026-09-02-v8"
CONVERT_MODELS="${CONVERT_MODELS:-auto}"

# 从 config.yaml 读取 chip 字段
CHIP=$(python -c "
import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    print((yaml.safe_load(f) or {}).get('chip') or '')
" 2>&1) || { echo "[entrypoint] 错误: 读取 config.yaml 失败: $CHIP"; exit 1; }
if [ -z "$CHIP" ]; then
    echo "[entrypoint] 错误: config.yaml 中未配置 chip 字段"
    exit 1
fi

MODEL_DIR="models/${CHIP}"

# atc 转换时需要调用 python3-config 定位 Python 库。部分镜像只有带版本号的
# python3.x-config, 或 /usr/bin/python3-config 指向系统自带的旧版本 Python,
# atc 会报: Unable to load a valid Python SO -> OpsManager initialize failed
# 这里校验 python3-config 的 prefix 是否与 python3 一致, 不一致就在 python3
# 所在目录创建/修正软链(PATH 中 python3 目录优先), 换镜像/Python 版本也能自适应
PY3=$(command -v python3 || true)
PYCONFIG_FIX_MSG="无需修正"
if [ -n "$PY3" ]; then
    PY3_DIR=$(dirname "$PY3")
    PY3_VER=$("$PY3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    # 候选 config 工具: python3-config 或 python3.x-config
    CFG=""
    for c in "${PY3}-config" "${PY3_DIR}/python${PY3_VER}-config"; do
        if [ -x "$c" ]; then
            CFG="$c"
            break
        fi
    done
    if [ -n "$CFG" ]; then
        PY3_PREFIX=$("$PY3" -c "import sys; print(sys.prefix)" 2>/dev/null)
        CFG_PREFIX=$(python3-config --prefix 2>/dev/null || true)
        if [ -z "$CFG_PREFIX" ] || [ "$CFG_PREFIX" != "$PY3_PREFIX" ]; then
            ln -sf "$CFG" "${PY3_DIR}/python3-config"
            PYCONFIG_FIX_MSG="已修正 python3-config -> ${CFG} (原 prefix: ${CFG_PREFIX:-不存在}, 正确: ${PY3_PREFIX})"
        fi
    else
        PYCONFIG_FIX_MSG="未找到 ${PY3}-config 或 python${PY3_VER}-config, 无法修正"
    fi
fi

# acl 是 CANN 自带的 Python 推理库。它需要两个条件才能 import 成功:
#   1. acl 包所在目录(CANN 安装目录的 python/site-packages)在 PYTHONPATH 中
#   2. 依赖的动态库 libascendcl.so 所在目录(CANN 安装目录的 lib64)在 LD_LIBRARY_PATH 中
# 部分镜像两者都没配, 服务调用接口时会报:
#   The 'acl' package is not available on this machine.
#   或 ImportError: libascendcl.so: cannot open shared object file
# 这里自动探测并补全
ACL_FIX_MSG="无需修正"
if ! python3 -c "import acl" > /dev/null 2>&1; then
    # 1. 补 PYTHONPATH
    ACL_SITEPKG=$(find /usr/local/Ascend -maxdepth 5 -type d -name acl -path "*site-packages*" 2>/dev/null | head -1)
    if [ -n "$ACL_SITEPKG" ]; then
        export PYTHONPATH="$(dirname "$ACL_SITEPKG")${PYTHONPATH:+:$PYTHONPATH}"
        ACL_FIX_MSG="PYTHONPATH 补充: $(dirname "$ACL_SITEPKG")"
    fi
    # 2. 补 LD_LIBRARY_PATH (libascendcl.so)
    #    注意排除 devlib 目录: 里面是交叉编译用的桩库, 符号不全,
    #    误用会报 undefined symbol: acldumpUnregCallback
    if ! python3 -c "import acl" > /dev/null 2>&1; then
        ASCENDCL=""
        for d in "${ASCEND_HOME_PATH}/lib64" "${ASCEND_HOME_PATH}"/*/lib64; do
            if [ -f "$d/libascendcl.so" ]; then
                ASCENDCL="$d/libascendcl.so"
                break
            fi
        done
        if [ -z "$ASCENDCL" ]; then
            ASCENDCL=$(find /usr/local/Ascend -maxdepth 4 -name "libascendcl.so" ! -path "*devlib*" 2>/dev/null | head -1)
        fi
        if [ -n "$ASCENDCL" ]; then
            export LD_LIBRARY_PATH="$(dirname "$ASCENDCL")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ACL_FIX_MSG="${ACL_FIX_MSG}; LD_LIBRARY_PATH 补充: $(dirname "$ASCENDCL")"
        fi
    fi
    # 3. 最终验证
    if python3 -c "import acl" > /dev/null 2>&1; then
        ACL_FIX_MSG="${ACL_FIX_MSG}; import acl 验证通过"
    else
        ACL_FIX_MSG="${ACL_FIX_MSG}; import acl 仍失败"
    fi
fi

# ============================ 环境诊断输出 ============================
echo "==================== [entrypoint] 环境诊断 ===================="
echo "[diag] entrypoint 版本: ${ENTRYPOINT_VERSION}"
echo "[diag] chip: ${CHIP}  模型目录: ${MODEL_DIR}  CONVERT_MODELS: ${CONVERT_MODELS}"
echo "[diag] python3 路径: $(command -v python3 || echo 未找到)"
echo "[diag] python3 版本: $(python3 --version 2>&1)"
echo "[diag] python3 prefix: $(python3 -c 'import sys; print(sys.prefix)' 2>&1)"
echo "[diag] python3-config 路径: $(command -v python3-config || echo 未找到)"
echo "[diag] python3-config --prefix: $(python3-config --prefix 2>&1)"
echo "[diag] python3-config 修正: ${PYCONFIG_FIX_MSG}"
echo "[diag] import acl: $(python3 -c 'import acl; print("OK")' 2>&1)"
echo "[diag] acl 修正: ${ACL_FIX_MSG}"
echo "[diag] PYTHONPATH: ${PYTHONPATH:-未设置}"
echo "[diag] atc 路径: $(command -v atc || echo 未找到)"
echo "[diag] ASCEND_OPP_PATH: ${ASCEND_OPP_PATH:-未设置}"
echo "[diag] ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-未设置}"
echo "[diag] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-未设置}"
echo "[diag] PATH: ${PATH}"
echo "[diag] models/onnx 目录内容:"
ls -la models/onnx/ 2>&1 | sed 's/^/[diag]   /'
echo "[diag] ${MODEL_DIR} 目录内容:"
ls -la "${MODEL_DIR}/" 2>&1 | sed 's/^/[diag]   /'
echo "================================================================"

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
        echo "[entrypoint] 警告: 需要转换模型, 但 models/onnx/ 目录下没有 ONNX 文件"
        echo "[entrypoint] 请将 ONNX 模型放入挂载的 models/onnx/ 目录, 或设置 CONVERT_MODELS=never 并提供已转换的 OM 模型"
    else
        echo "[entrypoint] 开始转换模型 (models/onnx/*.onnx -> ${MODEL_DIR}/*.om) ..."
        if python convert_encrypt_models.py "$CHIP"; then
            echo "[entrypoint] 模型转换完成"
        else
            echo "[entrypoint] 警告: 模型转换失败, 仍继续启动服务(服务可能因缺少模型而无法正常工作)"
            echo "[entrypoint] 排查方法: docker exec -it <容器名> /bin/bash 进入容器后手动执行:"
            echo "[entrypoint]   python convert_encrypt_models.py $CHIP"
        fi
    fi
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
