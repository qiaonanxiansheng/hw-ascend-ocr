# ascend-ocr

A native, high-performance OCR engine for Ascend NPU, with optional layout analysis support.

## Features

- **End-to-end pipeline**: angle classification → text detection → text recognition
- **Layout analysis**: document layout detection via PP-DocLayoutV3, with text clustering into layout regions
- **Input support**: local file path, HTTP/HTTPS URL, raw bytes, or `numpy.ndarray`
- **Large-angle correction**: automatically rotates images by 0/90/180/270 degrees
- **Standard pipeline**: DBNet detection, perspective transform, CTC decoding
- **Production ready**: logging, exception handling, resource cleanup, process-level ACL singleton
- **REST API**: FastAPI-based HTTP service with `/api/ocr` and `/api/layout-ocr` endpoints

## Installation

```bash
pip install .
```

For the REST API service:

```bash
pip install ".[api]"
```

Dependencies are listed in `requirements.txt`. The `acl` package is provided by Ascend CANN at runtime and should not be installed via pip.

## Quick Start

### Basic OCR

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import default_char_dict_path

engine = AscendOCR(
    det_model="models/ppocrv5_server_det_Ascend910B3.om",
    rec_model="models/ppocrv5_rec_Ascend910B3.om",
    cls_model="models/angle_cls_Ascend910B3.om",
    rec_char_dict=default_char_dict_path(),
    device_id=0,
)

results, angle = engine.ocr("test.png")
for r in results:
    print(r.text, r.score)

engine.release()
```

### Layout Analysis + OCR

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

## Model Files

Models are organized by chip type under the `models/` directory:

```
models/
├── 310/
│   ├── det.om
│   ├── rec.om
│   ├── cls.om
│   └── PP-DocLayoutV3.om
├── 310P/
│   ├── det.om
│   ├── rec.om
│   ├── cls.om
│   └── PP-DocLayoutV3.om
└── 910B3/
    ├── det.om
    ├── rec.om
    ├── cls.om
    └── PP-DocLayoutV3.om
```

| Model | Purpose |
|-------|---------|
| `det.om` | Text detection (DBNet) |
| `rec.om` | Text recognition (SVTR + CTC) |
| `cls.om` | Angle classification (optional) |
| `PP-DocLayoutV3.om` | Document layout analysis (optional) |

## Configuration

All parameters can be set in `config.yaml`:

```yaml
chip: "310"           # Chip type, used to locate models/{chip}/
device_id: 0
use_rotate: true      # Auto-rotate based on angle classification
```

Or programmatically with `OCRConfig`:

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import OCRConfig, DetConfig, RecConfig, ClsConfig, default_char_dict_path

cfg = OCRConfig(
    det_model="models/310/det.om",
    rec_model="models/310/rec.om",
    cls_model="models/310/cls.om",
    layout_model="models/310/PP-DocLayoutV3.om",
    rec_char_dict=default_char_dict_path(),
    use_rotate=True,
    det=DetConfig(
        limit_side_len=1280,
        resize_mode="pad",
        normalize_mode="imagenet",
        thresh=0.3,
        box_thresh=0.6,
        unclip_ratio=1.6,
        box_type="minarearect",
    ),
    rec=RecConfig(
        resize_mode="fixed_height_pad",
        pad_align="left",
        normalize_mode="ppocr",
        batch_size=16,
    ),
    cls=ClsConfig(label_list=[0, 180, 270, 90]),
)
engine = AscendOCR(cfg)
```

See `config.yaml` for all configurable parameters with documentation.

## REST API

### Start the server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### OCR endpoint

```bash
curl -X POST \
  -F "file_id=test1" \
  -F "file=@test.jpg" \
  -F "use_rotate=true" \
  http://localhost:8000/api/ocr
```

Response:

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
        "text": "Hello World",
        "score": 0.9876
      }
    ],
    "text": "Hello World"
  }
}
```

### Layout OCR endpoint

```bash
curl -X POST \
  -F "file_id=test1" \
  -F "file=@test.jpg" \
  -F "score_threshold=0.5" \
  http://localhost:8000/api/layout-ocr
```

Response:

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
            "text": "Some text content",
            "score": 0.9812
          }
        ],
        "text": "Some text content"
      }
    ],
    "total_lines": 1
  }
}
```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_id` | string | required | Caller-defined file identifier |
| `file` | file | required | Image file to process |
| `use_rotate` | bool | null | Override angle classification (null = use config default) |
| `vis` | bool | false | Return base64-encoded visualization image |
| `score_threshold` | float | 0.5 | Minimum confidence for layout regions (layout-ocr only) |

## Architecture

```
ascend_ocr/
├── __init__.py          # Public API
├── ocr.py               # AscendOCR orchestrator
├── model.py             # AscendCL model wrapper
├── angle_classifier.py  # Angle classifier (0/90/180/270)
├── text_detector.py     # Text detector (DBNet)
├── text_recognizer.py   # Text recognizer (SVTR + CTC)
├── layout_analyzer.py   # Document layout analysis (PP-DocLayoutV3)
├── preprocess.py        # Image preprocessing
├── postprocess.py       # DBNet detection post-processing
├── recognition.py       # CTC decoder
├── image_utils.py       # Image loading, rotation, perspective transform
├── acl_env.py           # Process-level ACL environment singleton
├── config.py            # Dataclass configuration
└── exceptions.py        # Custom exceptions

api/
├── __init__.py
├── app.py               # FastAPI application with lifespan management
├── schemas.py           # Pydantic response models
└── routers/
    ├── __init__.py
    ├── ocr.py           # POST /api/ocr
    └── layout_ocr.py    # POST /api/layout-ocr
```

### Pipeline Flow

**OCR (`engine.ocr`)**:

```
Image → [Angle Classification] → [Rotate] → Text Detection → Crop → Recognition → Results
```

**Layout OCR (`engine.layout_ocr`)**:

```
Image → [Angle Classification] → [Rotate]
                                    ↓
                              Layout Analysis (PP-DocLayoutV3)
                                    ↓
                              Full-page Text Detection + Recognition
                                    ↓
                              Text Clustering (assign lines to regions)
                                    ↓
                              Coordinate Mapping → Results
```

## Encrypted Models

If your `.om` files are encrypted, provide a decrypt callback:

```python
def decrypt(path: str) -> bytes:
    with open(path, "rb") as f:
        cipher = f.read()
    return your_decrypt_function(cipher)

engine = AscendOCR(
    det_model="models/310/det.om.enc",
    rec_model="models/310/rec.om.enc",
    cls_model="models/310/cls.om.enc",
    decrypt_callback=decrypt,
)
```

## Docker

```bash
docker build -t ascend-ocr .
docker run --device /dev/davinci0 -p 8000:8000 ascend-ocr
```

To use a different CANN base image:

```bash
docker build --build-arg BASE_IMAGE=your-registry/cann:8.0.0-ubuntu22.04-py3.11 -t ascend-ocr .
```

## Performance & Accuracy Tips

| Goal | Recommended switches |
|------|---------------------|
| Best accuracy | `det.resize_mode="pad"`, `det.use_pyclipper=True`, `det.unclip_ratio=1.6~2.0`, `rec.use_direction_ensemble=True` |
| Maximum speed | `det.resize_mode="stretch"`, `det.box_type="minarearect"`, `rec.resize_mode="fixed_size_stretch"`, `rec.batch_size=16+` |
| Small / dense text | Increase `det.limit_side_len`, enable `det.use_dilate`, lower `det.box_thresh` |
| No angle correction needed | Set `use_rotate=False` to skip the classifier entirely |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project includes code derived from open-source OCR projects. See [NOTICE](NOTICE) for details.
