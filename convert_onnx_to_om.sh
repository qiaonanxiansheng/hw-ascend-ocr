#!/bin/sh
#
# ONNX → OM 批量转换脚本
# 用法: ./convert_onnx_to_om.sh [芯片型号...]
# 示例: ./convert_onnx_to_om.sh                    # 使用默认芯片列表
#       ./convert_onnx_to_om.sh 310 910B3           # 只转换指定芯片
#

set -e

# ONNX 文件所在目录
ONNX_DIR="models/onnx"

# 输出根目录
OUTPUT_ROOT="models"

# 检查 ONNX 目录是否存在
if [ ! -d "$ONNX_DIR" ]; then
    echo "错误: ONNX 目录不存在: $ONNX_DIR"
    exit 1
fi

# 检查 atc 命令是否可用
if ! command -v atc > /dev/null 2>&1; then
    echo "错误: atc 命令未找到"
    exit 1
fi

# 转换函数
# 参数: onnx文件名 输出名 input_shape input_format soc_version 输出目录
convert_model() {
    onnx_file="$1"
    output_name="$2"
    input_shape="$3"
    input_format="$4"
    soc_version="$5"
    output_dir="$6"

    onnx_path="${ONNX_DIR}/${onnx_file}"
    output_path="${output_dir}/${output_name}"

    if [ ! -f "$onnx_path" ]; then
        echo "  跳过: ${onnx_file} 不存在"
        return 0
    fi

    echo "  转换: ${onnx_file} → ${output_name}.om"

    # 构建 atc 命令
    atc_cmd="atc --model=${onnx_path} --framework=5 --output=${output_path} --input_shape=\"${input_shape}\" --soc_version=${soc_version} --precision_mode=allow_fp32_to_fp16 --plugin=ByPass"

    # 如果指定了 input_format 则添加
    if [ -n "$input_format" ]; then
        atc_cmd="$atc_cmd --input_format=${input_format}"
    fi

    eval $atc_cmd
    ret=$?

    if [ $ret -eq 0 ]; then
        echo "  成功: ${output_path}.om"
    else
        echo "  失败: ${onnx_file} 转换出错"
    fi
    return $ret
}

# 转换单个芯片的所有模型
# 参数: 芯片型号
convert_chip() {
    chip="$1"
    soc_version="Ascend${chip}"
    output_dir="${OUTPUT_ROOT}/${chip}"

    echo ""
    echo "------------------------------------------"
    echo "芯片: ${chip} (${soc_version})"
    echo "输出目录: ${output_dir}"
    echo "------------------------------------------"

    mkdir -p "$output_dir"

    # 1. 大角度分类模型
    convert_model "rotate.onnx" "rotate" "images:1,3,512,512" "" "$soc_version" "$output_dir"
    total=$((total+1))
    [ $? -eq 0 ] && success=$((success+1)) || fail=$((fail+1))

    # 2. 文字检测模型
    convert_model "v6_small_det.onnx" "det" "x:1,3,960,672" "" "$soc_version" "$output_dir"
    total=$((total+1))
    [ $? -eq 0 ] && success=$((success+1)) || fail=$((fail+1))

    # 3. 文字识别模型
    convert_model "v6_small_rec.onnx" "rec" "x:1,3,48,960" "" "$soc_version" "$output_dir"
    total=$((total+1))
    [ $? -eq 0 ] && success=$((success+1)) || fail=$((fail+1))

    # 4. 版面分析模型
    convert_model "PP-DocLayoutV3.onnx" "PP-DocLayoutV3" "im_shape:1,2;image:1,3,800,800;scale_factor:1,2" "ND" "$soc_version" "$output_dir"
    total=$((total+1))
    [ $? -eq 0 ] && success=$((success+1)) || fail=$((fail+1))

    # 5. 表格识别模型
    convert_model "table.onnx" "table" "images:1,3,640,640" "NCHW" "$soc_version" "$output_dir"
    total=$((total+1))
    [ $? -eq 0 ] && success=$((success+1)) || fail=$((fail+1))
}

# 主流程
total=0
success=0
fail=0

echo "=========================================="
echo "ONNX → OM 批量转换"
echo "=========================================="

if [ $# -gt 0 ]; then
    # 使用命令行参数
    for chip in "$@"; do
        convert_chip "$chip"
    done
else
    # 默认芯片列表
    for chip in 310 310P3; do
        convert_chip "$chip"
    done
fi

echo ""
echo "=========================================="
echo "转换完成: 成功 ${success}/${total}, 失败 ${fail}"
echo "=========================================="
