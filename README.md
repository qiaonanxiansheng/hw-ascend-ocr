# ascend-ocr

基于 Ascend NPU 的原生高性能 OCR 引擎，支持全文识别和版面分析。

## 功能特性

- **端到端流水线**：角度分类 → 文字检测 → 文字识别
- **版面分析**：基于 PP-DocLayoutV3 的文档版面检测，自动将文本行聚类到对应版面区域
- **多种输入方式**：本地文件路径、HTTP/HTTPS URL、原始字节流、`numpy.ndarray`
- **大角度校正**：自动识别并旋转 0/90/180/270 度的图片
- **标准 OCR 管线**：DBNet 检测、透视变换矫正、CTC 解码
- **生产就绪**：完善的日志、异常处理、资源释放、进程级 ACL 单例管理
- **REST API**：基于 FastAPI 的 HTTP 服务，提供 `/api/ocr` 和 `/api/layout-ocr` 接口

## 环境要求

- Ascend NPU 设备（310 / 310P / 910B3 等）
- Ascend CANN Toolkit（提供 `acl` 运行时库）
- Python >= 3.8

## 安装

```bash
pip install .
```

如果需要使用 REST API 服务：

```bash
pip install ".[api]"
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

results, angle = engine.ocr("test.png")
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

regions, region_ocr_results, clusters, angle = engine.layout_ocr("test.png")

for region, ocr_lines in region_ocr_results:
    print(f"[{region.class_name}] {region.bbox}")
    for line in ocr_lines:
        print(f"  {line.text} ({line.score:.3f})")

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
│   └── PP-DocLayoutV3.om
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
from ascend_ocr.config import OCRConfig, DetConfig, RecConfig, RotateConfig

cfg = OCRConfig(
    det_model="models/310/det.om",
    rec_model="models/310/rec.om",
    rotate_model="models/310/rotate.om",
    layout_model="models/310/PP-DocLayoutV3.om",
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
  -F "file=@test.jpg" \
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
  -F "file=@test.jpg" \
  -F "score_threshold=0.6" \
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
| `line.coords` | 文本行四点坐标 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` |
| `line.score` | 识别置信度 |

## Docker 部署

```bash
docker build -t ascend-ocr .
docker run --device /dev/davinci0 -p 13502:13502 ascend-ocr
```

使用其他 CANN 基础镜像：

```bash
docker build --build-arg BASE_IMAGE=your-registry/cann:8.0.0-ubuntu22.04-py3.11 -t ascend-ocr .
```

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
    └── layout_ocr.py    # POST /api/layout-ocr
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
                 坐标映射回原图 → 结果
```

## 开源协议

本项目基于 MIT 协议开源，详见 [LICENSE](LICENSE)。

本项目包含来自开源 OCR 项目的代码，详见 [NOTICE](NOTICE)。
