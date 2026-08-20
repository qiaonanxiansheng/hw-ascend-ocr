# ascend-ocr

A native, high-cohesion OCR engine for Huawei Ascend (NPU).

## Features

- **End-to-end pipeline**: angle classification → text detection → text recognition.
- **Input support**: local file path, HTTP/HTTPS URL, raw bytes, or `numpy.ndarray`.
- **Large-angle correction**: automatically rotates images by 0 / 90 / 180 / 270 degrees.
- **PaddleOCR-style processing**: DBNet detection post-processing, perspective transform rectification, CTC decoding.
- **Production ready**: logging, exception handling, resource cleanup, process-level ACL singleton.
- **Packaging**: can be built as a Python wheel (`pip install .`).

## Installation

```bash
pip install .
```

Dependencies are listed in `requirements.txt`. The `acl` package is provided by
Ascend CANN at runtime and should not be installed via pip.

## Quick Start

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

# Image can be a path, URL, bytes, or numpy array.
results = engine.ocr("https://example.com/image.png")
for r in results:
    print(r.text, r.score)

engine.release()
```

## Model Files

Models are organized by chip type under `models/` directory:

```
models/
├── 310/
│   ├── det.om
│   ├── rec.om
│   └── cls.om
├── 310P/
│   ├── det.om
│   ├── rec.om
│   └── cls.om
└── 910B3/
    ├── det.om
    ├── rec.om
    └── cls.om
```

| Model | Purpose |
|-------|---------|
| det.om | Text detection (DBNet) |
| rec.om | Text recognition (SVTR + CTC) |
| cls.om | Angle classification (optional) |

## Configuration

All parameters can be set in `config.yaml`:

```yaml
chip: "310"           # Chip type, used to locate models/{chip}/
device_id: 0
use_angle_cls: true
```

Or programmatically with `OCRConfig`:

```python
from ascend_ocr import AscendOCR
from ascend_ocr.config import (
    OCRConfig, DetConfig, RecConfig, ClsConfig, default_char_dict_path
)

cfg = OCRConfig(
    det_model="models/310/det.om",
    rec_model="models/310/rec.om",
    cls_model="models/310/cls.om",
    rec_char_dict=default_char_dict_path(),
    det=DetConfig(
        limit_side_len=1280,
        resize_mode="pad",          # "pad" | "stretch" | "crop_center"
        normalize_mode="imagenet",  # "imagenet" | "ppocr" | "none" | "custom"
        thresh=0.3,
        box_thresh=0.6,
        unclip_ratio=1.6,
        box_type="minarearect",     # "minarearect" | "poly" | "quad"
        use_pyclipper=True,
        use_dilate=False,
        nms_threshold=-1.0,
        sort_mode="natural",        # "natural" | "top2bottom" | "left2right"
    ),
    rec=RecConfig(
        resize_mode="fixed_height_pad",
        pad_align="left",           # "left" | "center" | "right"
        normalize_mode="ppocr",
        batch_size=16,
        use_direction_ensemble=False,
    ),
    cls=ClsConfig(label_list=[0, 180, 270, 90]),
)
engine = AscendOCR(cfg)
```

See `config.yaml` for all configurable parameters with documentation.

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

## Command Line Example

```bash
python examples/ocr_example.py \
    --det-model models/310/det.om \
    --rec-model models/310/rec.om \
    ./test.png
```

## Project Structure

```
ascend_ocr/
├── __init__.py          # Public API
├── ocr.py               # AscendOCR orchestrator
├── model.py             # AscendCL model wrapper
├── angle_classifier.py  # Angle classifier
├── text_detector.py     # Text detector
├── text_recognizer.py   # Text recognizer
├── preprocess.py        # Image preprocessing
├── postprocess.py       # DBNet detection post-processing
├── recognition.py       # CTC decoder
├── image_utils.py       # Image loading, rotation, perspective transform
├── acl_env.py           # Process-level ACL environment singleton
├── config.py            # Dataclass configuration
└── exceptions.py        # Custom exceptions
```

## Performance & Accuracy Tips

| Goal | Recommended switches |
|------|---------------------|
| Best accuracy | `det.resize_mode="pad"`, `det.use_pyclipper=True`, `det.unclip_ratio=1.6~2.0`, `rec.use_direction_ensemble=True` |
| Maximum speed | `det.resize_mode="stretch"`, `det.box_type="minarearect"`, `rec.resize_mode="fixed_size_stretch"`, `rec.batch_size=16+` |
| Small / dense text | Increase `det.limit_side_len`, enable `det.use_dilate`, lower `det.box_thresh` |
| No angle correction needed | Set `use_angle_cls=False` to skip the classifier entirely |

## Notes

- `acl.init()` / `acl.finalize()` are managed once per process via `acl_env.py`.
- Multiple `AscendOCR` / `AscendModel` instances can share the same ACL context.
- Replace the bundled character dictionary (`ascend_ocr/configs/ppocr_keys_v1.txt`) with the dictionary that matches your recognition model.
