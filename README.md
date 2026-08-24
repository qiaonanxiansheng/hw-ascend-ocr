# ascend-ocr

基于 Ascend NPU 的原生高性能 OCR 引擎，支持全文识别、版面分析和表格结构识别。

## 功能特性

- **端到端流水线**：角度分类 → 文字检测 → 文字识别
- **版面分析**：基于 PP-DocLayoutV3 的文档版面检测，自动将文本行聚类到对应版面区域，表格区域自动生成 HTML
- **表格结构识别**：基于 YOLO 的表格/单元格检测，自动识别行列结构、合并单元格，输出 HTML 表格
- **多种输入方式**：本地文件路径、HTTP/HTTPS URL、原始字节流、`numpy.ndarray`
- **大角度校正**：自动识别并旋转 0/90/180/270 度的图片
- **标准 OCR 管线**：DBNet 检测、透视变换矫正、CTC 解码
- **生产就绪**：完善的日志、异常处理、资源释放、进程级 ACL 单例管理
- **REST API**：基于 FastAPI 的 HTTP 服务，提供 `/api/ocr`、`/api/layout-ocr` 和 `/api/table-ocr` 接口

## 环境要求

- Ascend NPU 设备（310 / 310P / 910B3 等）
- Ascend CANN Toolkit（提供 `acl` 运行时库）
- Python >= 3.8

## 安装

```bash
pip install -r requirements.txt
```

> `acl` 包由 Ascend CANN Toolkit 在运行时提供，不要通过 pip 安装。

## 快速开始

### 全文 OCR

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import default_char_dict_path

engine = AscendOCR(
    det_model="models/det.om",
    rec_model="models/rec.om",
    rotate_model="models/rotate.om",
    rec_char_dict=default_char_dict_path(),
    device_id=0,
)

results, angle = engine.ocr("table.png")
for r in results:
    print(r.text, r.score)

engine.release()
```

### 版面分析 + OCR

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import OCRConfig

cfg = OCRConfig(
    det_model="models/det.om",
    rec_model="models/rec.om",
    layout_model="models/PP-DocLayoutV3.om",
    device_id=0,
)
engine = AscendOCR(cfg)

regions, region_ocr_results, clusters, angle = engine.layout_ocr("table.png")

for region, ocr_lines in region_ocr_results:
    print(f"[{region.class_name}] {region.bbox}")
    for line in ocr_lines:
        print(f"  {line.text} ({line.score:.3f})")

engine.release()
```

### 表格结构识别 + OCR

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import OCRConfig

cfg = OCRConfig(
    det_model="models/det.om",
    rec_model="models/rec.om",
    table_model="models/table.om",
    device_id=0,
)
engine = AscendOCR(cfg)

tables, outside_text, angle = engine.table_ocr("table.png")

for table in tables:
    print(f"表格: {table.rows}行 x {table.cols}列, {len(table.cells)}个单元格")
    print(table.html)
    for cell in table.cells:
        print(f"  [{cell.rowstart},{cell.colstart}] -> [{cell.rowend},{cell.colend}]", end="")
        for line in cell.lines:
            print(f"  {line.text}", end="")
        print()

if outside_text:
    print("表格外文字:")
    for line in outside_text:
        print(f"  {line.text}")

engine.release()
```

## 模型文件

模型按芯片类型存放在 `models/` 目录下：

```
models/
├── 310/
│   ├── det.om
│   ├── rec.om
│   ├── rotate.om
│   ├── PP-DocLayoutV3.om
│   └── table.om
├── 310P/
│   └── ...
└── 910B3/
    └── ...
```

| 模型 | 用途 |
|------|------|
| `det.om` | 文字检测（DBNet） |
| `rec.om` | 文字识别（SVTR + CTC） |
| `rotate.om` | 角度校正（可选） |
| `PP-DocLayoutV3.om` | 文档版面分析（可选） |
| `table.om` | 表格结构识别（YOLO，可选） |

## 配置

所有参数可在 `config.yaml` 中配置：

```yaml
chip: "310"           # 芯片型号，用于定位 models/{chip}/ 目录
device_id: 0
use_rotate: true      # 是否启用大角度分类并自动转正
```

也可通过代码直接配置：

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import OCRConfig, DetConfig, RecConfig, RotateConfig, TableConfig

cfg = OCRConfig(
    det_model="models/310/det.om",
    rec_model="models/310/rec.om",
    rotate_model="models/310/rotate.om",
    layout_model="models/310/PP-DocLayoutV3.om",
    table_model="models/310/table.om",
    use_rotate=True,
    det=DetConfig(
        limit_side_len=960,
        resize_mode="pad",
        thresh=0.3,
        box_thresh=0.6,
    ),
    rec=RecConfig(
        resize_mode="fixed_height_pad",
        pad_align="left",
        batch_size=8,
    ),
    rotate=RotateConfig(label_list=[0, 180, 270, 90]),
    table=TableConfig(
        input_size=640,
        score_threshold=0.6,
        nms_threshold=0.5,
    ),
)
engine = AscendOCR(cfg)
```

完整配置项见 `config.yaml`。

## REST API 部署

### 启动服务

```bash
uvicorn api.app:app --host 0.0.0.0 --port 13502
```

### 全文识别接口

`POST /api/ocr`

**请求参数（form-data）：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file_id` | string | 是 | 调用方自定义的文件标识 |
| `file` | file | 是 | 图片文件 |
| `use_rotate` | bool | 否 | 是否启用角度校正，不传则使用配置默认值 |
| `vis` | bool | 否 | 是否返回标注可视化图片（base64），默认 false |

**请求示例：**

```bash
curl -X POST \
  -F "file_id=test1" \
  -F "file=@table.png" \
  -F "use_rotate=true" \
  http://localhost:13502/api/ocr
```

**响应示例：**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "file_id": "test1",
    "rotate": 0,
    "lines": [
      {
        "index": 1,
        "coords": [[100, 50], [300, 50], [300, 80], [100, 80]],
        "text": "你好世界",
        "score": 0.9876
      }
    ],
    "text": "你好世界"
  }
}
```

### 版面分析接口

`POST /api/layout-ocr`

**请求参数（form-data）：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file_id` | string | 是 | 调用方自定义的文件标识 |
| `file` | file | 是 | 图片文件 |
| `use_rotate` | bool | 否 | 是否启用角度校正，不传则使用配置默认值 |
| `score_threshold` | float | 否 | 版面区域最低置信度，默认 0.5 |
| `vis` | bool | 否 | 是否返回标注可视化图片（base64），默认 false |

**请求示例：**

```bash
curl -X POST \
  -F "file_id=test1" \
  -F "file=@table.png" \
  -F "score_threshold=0.5" \
  http://localhost:13502/api/layout-ocr
```

**响应示例：**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "file_id": "test1",
    "rotate": 0,
    "regions": [
      {
        "index": 1,
        "class_id": 22,
        "class_name": "text",
        "score": 0.9395,
        "bbox": [387, 797, 2140, 1075],
        "lines": [
          {
            "index": 1,
            "coords": [[400, 800], [2100, 800], [2100, 850], [400, 850]],
            "text": "这是一段文字内容",
            "score": 0.9812
          }
        ],
        "text": "这是一段文字内容"
      }
    ],
    "total_lines": 1
  }
}
```

**响应字段说明：**

| 字段 | 说明 |
|------|------|
| `code` | 状态码，0 表示成功 |
| `message` | 状态消息 |
| `data.rotate` | 检测到的旋转角度（0/90/180/270） |
| `data.lines` | 识别到的文本行列表（按阅读顺序排序） |
| `data.text` | 全文内容（所有行用 `\n` 拼接） |
| `data.regions` | 版面区域列表（仅 layout-ocr） |
| `region.class_name` | 区域类型：text / title / table / image 等 |
| `region.bbox` | 区域边界框 `[x1, y1, x2, y2]` |
| `region.lines` | 该区域内的文本行 |
| `region.html` | 表格区域的 HTML 内容（仅 class_name="table" 时有值） |
| `line.coords` | 文本行四点坐标 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` |
| `line.score` | 识别置信度 |

### 表格结构识别接口

`POST /api/table-ocr`

**请求参数（form-data）：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file_id` | string | 是 | 调用方自定义的文件标识 |
| `file` | file | 是 | 图片文件 |
| `use_rotate` | bool | 否 | 是否启用角度校正，不传则使用配置默认值 |
| `vis` | bool | 否 | 是否返回标注可视化图片（base64），默认 false |

**请求示例：**

```bash
curl -X POST \
  -F "file_id=test1" \
  -F "file=@table.jpg" \
  http://localhost:13502/api/table-ocr
```

**响应示例：**

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "file_id": "test1",
    "rotate": 0,
    "tables": [
      {
        "index": 1,
        "bbox": [100, 200, 1900, 1500],
        "rows": 5,
        "cols": 4,
        "html": "<html><body><table border='1'><tbody><tr><td>姓名</td><td>年龄</td>...</tr></tbody></table></body></html>",
        "cells": [
          {
            "index": 1,
            "bbox": [100, 200, 500, 350],
            "score": 0.95,
            "rowstart": 0,
            "rowend": 1,
            "colstart": 0,
            "colend": 1,
            "rowspan": 1,
            "colspan": 1,
            "lines": [
              {
                "index": 1,
                "coords": [[120, 220], [480, 220], [480, 280], [120, 280]],
                "text": "姓名",
                "score": 0.9876
              }
            ],
            "text": "姓名"
          }
        ]
      }
    ],
    "outside_text": [
      {
        "index": 1,
        "coords": [[100, 50], [300, 50], [300, 80], [100, 80]],
        "text": "表格外的文字",
        "score": 0.9812
      }
    ]
  }
}
```

**响应字段说明：**

| 字段 | 说明 |
|------|------|
| `data.tables` | 检测到的表格列表 |
| `table.bbox` | 表格边界框 `[x1, y1, x2, y2]` |
| `table.rows` | 表格行数 |
| `table.cols` | 表格列数 |
| `table.html` | 表格的 HTML 内容 |
| `table.cells` | 单元格列表 |
| `cell.bbox` | 单元格边界框 `[x1, y1, x2, y2]` |
| `cell.rowstart/rowend` | 单元格起始/结束行号 |
| `cell.colstart/colend` | 单元格起始/结束列号 |
| `cell.rowspan/colspan` | 单元格跨行/跨列数 |
| `cell.lines` | 单元格内的文本行 |
| `cell.text` | 单元格的完整文本 |
| `data.outside_text` | 未分配到任何单元格的文本行 |

## Docker 部署

### 第一步：准备模型文件

模型按芯片类型存放在 `models/` 目录下，程序会根据 `config.yaml` 中的 `chip` 字段自动加载对应目录的模型：

```
models/
├── 310/                  ← chip: "310" 时加载这个目录
│   ├── det.om
│   ├── rec.om
│   ├── rotate.om
│   ├── PP-DocLayoutV3.om
│   └── table.om
├── 310P/                 ← chip: "310P" 时加载这个目录
│   └── ...
└── 910B3/                ← chip: "910B3" 时加载这个目录
    └── ...
```

将你转换好的 OM 模型放到对应的芯片目录下即可。

### 第二步：修改配置文件

编辑 `config.yaml`，修改 `chip` 字段为你的芯片型号：

```yaml
chip: "310"      # 改成你的芯片型号：310 / 310P / 910B3 等
device_id: 0     # NPU 设备 ID
```

程序会自动拼接模型路径：`models/{chip}/det.om`、`models/{chip}/rec.om` 等。如果你的模型文件名不同，可以手动指定：

```yaml
chip: "310"
det_model: "models/310/det.om"       # 手动指定，覆盖自动拼接
rec_model: "models/310/rec.om"
table_model: "models/310/table.om"
```

### 第三步：构建镜像

**1. 下载 Ascend CANN 基础镜像**

前往 [Ascend 官网资源页面](https://www.hiascend.com/developer/download/community/result?cann=8.0.0) 下载对应芯片的 CANN 镜像。

根据你的芯片型号和 Python 版本选择合适的镜像，例如：
- 310 芯片：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11`
- 310P 芯片：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11`
- 910B3 芯片：`cswr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.11`

**2. 构建 Docker 镜像**

```bash
# 使用默认基础镜像（需要本地已有）
docker build -t ascend-ocr:310p .

# 或指定从 Ascend 官网下载的基础镜像
docker build \
  --build-arg BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11 \
  -t ascend-ocr:310p .
```

> 将基础镜像地址替换为你从 Ascend 官网下载的实际镜像，标签中的 `310p` 改成你的芯片型号。

### 第四步：启动容器

以物理机 NPU 4 在容器中映射为 NPU 0 为例：

```bash
docker run -d --restart unless-stopped \
    --name ascend-ocr \
    -p 13502:13502 \
    --device=/dev/davinci4:/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /opt/ascend-ocr/config.yaml:/workspace/config.yaml \
    -v /opt/ascend-ocr/models:/workspace/models \
    -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common \
    --shm-size=8g \
    ascend-ocr:310p
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--device=/dev/davinci4:/dev/davinci0` | 将物理机 NPU 4 映射为容器内 NPU 0，按实际情况修改 |
| `--device=/dev/davinci_manager` | NPU 管理设备，必须挂载 |
| `--device=/dev/devmm_svm` | SVM 设备，必须挂载 |
| `--device=/dev/hisi_hdc` | HDC 设备，必须挂载 |
| `-v .../driver:ro` | 挂载 Ascend 驱动目录（只读） |
| `-v .../add-ons:ro` | 挂载 Ascend 插件目录（只读） |
| `-v .../ascend_install.info:ro` | 挂载安装信息（只读） |
| `-v .../npu-smi:ro` | 挂载 npu-smi 工具（只读） |
| `-v .../dcmi:ro` | 挂载 DCMI（只读） |
| `-v .../config.yaml` | 挂载配置文件，修改芯片型号后重启即生效 |
| `-v .../models` | 挂载模型目录，更换模型不用重建镜像 |
| `--shm-size=8g` | 共享内存，OCR 推理需要较大内存 |

> **NPU 设备映射：** `--device=/dev/davinci4:/dev/davinci0` 表示物理机的第 4 号 NPU 在容器里是第 0 个。如果你只有一块 NPU 且是第 0 号，写 `--device=/dev/davinci0` 即可。

容器启动后监听 `13502` 端口，日志输出到 stdout。

### 第五步：测试接口

```bash
# 全文识别
curl -X POST \
  -F "file_id=test1" \
  -F "file=@docs/table.png" \
  http://localhost:13502/api/ocr

# 版面分析（表格区域自动生成 HTML）
curl -X POST \
  -F "file_id=test1" \
  -F "file=@docs/table.png" \
  http://localhost:13502/api/layout-ocr

# 表格结构识别（独立接口，无需版面模型）
curl -X POST \
  -F "file_id=test1" \
  -F "file=@docs/table.png" \
  http://localhost:13502/api/table-ocr
```

## 模型转换

所有模型都是开源模型，转换流程：`原始模型` → `ONNX` → `OM`。

ONNX 模型可从 [ModelScope](https://modelscope.cn) 下载。

**核心步骤：**
1. 从开源社区下载原始模型（或自己训练）
2. 转成 ONNX 格式
3. 使用华为 `atc` 工具将 ONNX 转成 OM 格式

**切换芯片型号只需改一个参数：** 将下面所有命令中的 `--soc_version=Ascend310` 改成你的芯片型号即可，例如 `Ascend310P`、`Ascend910B3`。

> **注意：** `atc` 工具由 Ascend CANN Toolkit 提供，需要在安装了 CANN 的环境中执行（或在 Ascend 官方 Docker 镜像中执行）。

<details>
<summary><b>1. 大角度分类模型</b></summary>

**用途：** 识别图片旋转角度（0°/90°/180°/270°），自动转正

**ONNX 模型来源：** 开源角度分类模型，输入 `[1,3,512,512]`

```bash
atc --model=./rotate.onnx \
    --framework=5 \
    --output=rotate \
    --input_shape="images:1,3,512,512" \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

输出：`rotate.om`

</details>

<details>
<summary><b>2. 文字检测模型</b></summary>

**用途：** 检测图片中的文字区域（DBNet）

**ONNX 模型来源：** PaddleOCR `PP-OCRv6_small_det`，从 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 下载

```bash
# 第一步：PaddleOCR 模型 → ONNX
pip install paddlex
paddlex --paddle2onnx \
    --paddle_model_dir ./PP-OCRv6_small_det_infer \
    --onnx_model_dir ./models/det \
    --opset_version 11

# 第二步：ONNX → OM
atc --model=./v6_small_det.onnx \
    --framework=5 \
    --output=det \
    --input_shape="x:1,3,960,672" \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

输出：`det.om`

> **重要：** `--precision_mode=allow_fp32_to_fp16` 必须加，否则在某些设备上会导致检测不出文本框或误差极大。
>
> `--input_shape` 中的 960 是静态输入尺寸，建议根据业务场景定死。静态模型比动态模型速度快，常见值：960、1600。

</details>

<details>
<summary><b>3. 文字识别模型</b></summary>

**用途：** 识别裁剪后的文字图片（SVTR + CTC）

**ONNX 模型来源：** PaddleOCR `PP-OCRv6_small_rec`，从 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 下载

```bash
# 第一步：PaddleOCR 模型 → ONNX
paddlex --paddle2onnx \
    --paddle_model_dir ./PP-OCRv6_small_rec_infer \
    --onnx_model_dir ./models/rec \
    --opset_version 11

# 第二步：ONNX → OM
atc --model=./v6_small_rec.onnx \
    --framework=5 \
    --output=rec \
    --input_shape="x:1,3,48,960" \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

输出：`rec.om`

</details>

<details>
<summary><b>4. 版面分析模型（可选）</b></summary>

**用途：** 检测文档版面区域（文字、标题、表格、图片等），表格区域自动生成 HTML

**ONNX 模型来源：** PaddleOCR `PP-DocLayoutV3`，从 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 下载

```bash
atc --model=PP-DocLayoutV3.onnx \
    --framework=5 \
    --output=PP-DocLayoutV3 \
    --input_format=ND \
    --input_shape="im_shape:1,2;image:1,3,800,800;scale_factor:1,2" \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

输出：`PP-DocLayoutV3.om`

> 不需要版面分析功能可以不转换此模型。

</details>

<details>
<summary><b>5. 表格识别模型（可选）</b></summary>

**用途：** 检测表格区域和单元格，配合 OCR 生成结构化 HTML 表格

**ONNX 模型来源：** 使用 YOLO11 自行训练的表格/单元格检测模型，2 个类别：`table`（表格区域）和 `cell`（单元格）

```bash
atc --model=table.onnx \
    --framework=5 \
    --output=table \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --soc_version=Ascend310 \
    --precision_mode=allow_fp32_to_fp16
```

输出：`table.om`

> 模型输出格式：`[1, 6, 8400]`，其中 6 = `[cx, cy, w, h, table_score, cell_score]`。
>
> 不需要表格识别功能可以不转换此模型。

</details>

### 转换完成后

将所有 `.om` 文件放到对应的芯片目录下：

```bash
mkdir -p models/310
mv det.om rec.om rotate.om PP-DocLayoutV3.om table.om models/310/
```

然后修改 `config.yaml` 中的 `chip` 字段为你的芯片型号即可。

## 加密模型

如果 `.om` 文件是加密的，需要提供解密回调：

```python
def decrypt(path: str) -> bytes:
    with open(path, "rb") as f:
        cipher = f.read()
    return your_decrypt_function(cipher)

engine = AscendOCR(
    det_model="models/det.om.enc",
    rec_model="models/rec.om.enc",
    rotate_model="models/rotate.om.enc",
    decrypt_callback=decrypt,
)
```

## 性能调优建议

| 目标 | 推荐配置 |
|------|---------|
| 最高精度 | `det.resize_mode="pad"`, `det.use_pyclipper=True`, `det.unclip_ratio=1.6~2.0`, `rec.use_direction_ensemble=True` |
| 最快速度 | `det.resize_mode="stretch"`, `det.box_type="minarearect"`, `rec.resize_mode="fixed_size_stretch"`, `rec.batch_size=16+` |
| 小字/密集文字 | 增大 `det.limit_side_len`，启用 `det.use_dilate`，降低 `det.box_thresh` |
| 不需要角度校正 | 设置 `use_rotate=False` 跳过角度分类 |

## 项目结构

```
ascend_ocr/
├── __init__.py          # 公开 API
├── ocr.py               # AscendOCR 主引擎
├── model.py             # AscendCL 模型封装
├── angle_classifier.py  # 角度分类器（0/90/180/270）
├── text_detector.py     # 文字检测器（DBNet）
├── text_recognizer.py   # 文字识别器（SVTR + CTC）
├── layout_analyzer.py   # 文档版面分析（PP-DocLayoutV3）
├── table_recognizer.py  # 表格结构识别（YOLO）
├── preprocess.py        # 图像预处理
├── postprocess.py       # DBNet 检测后处理
├── recognition.py       # CTC 解码器
├── image_utils.py       # 图像加载、旋转、透视变换
├── acl_env.py           # 进程级 ACL 环境单例
├── config.py            # 配置数据类
└── exceptions.py        # 自定义异常

api/
├── __init__.py
├── app.py               # FastAPI 应用 + 生命周期管理
├── schemas.py           # Pydantic 响应模型
└── routers/
    ├── __init__.py
    ├── ocr.py           # POST /api/ocr
    ├── layout_ocr.py    # POST /api/layout-ocr
    └── table_ocr.py     # POST /api/table-ocr
```

### 处理流程

**全文 OCR（`engine.ocr`）：**

```
图片 → [角度分类] → [旋转校正] → 文字检测 → 裁剪 → 文字识别 → 结果
```

**版面分析 OCR（`engine.layout_ocr`）：**

```
图片 → [角度分类] → [旋转校正]
                       ↓
                 版面分析（PP-DocLayoutV3）
                       ↓
                 全图文字检测 + 识别（单次）
                       ↓
                 文本聚类（将文本行分配到版面区域）
                       ↓
                 表格区域 → 表格模型 → 单元格检测 → 文字分配 → HTML
                       ↓
                 坐标映射回原图 → 结果
```

**表格结构识别 OCR（`engine.table_ocr`）：**

```
图片 → [角度分类] → [旋转校正]
                       ↓
                 表格模型（YOLO）→ 检测表格区域 + 单元格
                       ↓
                 全图文字检测 + 识别（单次）
                       ↓
                 单元格分配到表格区域
                       ↓
                 每个表格：文字分配 → 行列聚类 → 生成 HTML
                       ↓
                 坐标映射回原图 → 结果
```

## 开源协议

本项目基于 MIT 协议开源，详见 [LICENSE](LICENSE)。

本项目包含来自开源 OCR 项目的代码，详见 [NOTICE](NOTICE)。
