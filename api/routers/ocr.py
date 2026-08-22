from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from api.schemas import OCRData, OCRLineResponse, OCRResponse

if TYPE_CHECKING:
    from ascend_ocr import AscendOCR, OCRResult

logger = logging.getLogger("api.ocr")

router = APIRouter(prefix="/api", tags=["ocr"])

# Will be set by app.py lifespan
engine: Optional["AscendOCR"] = None


def _to_line_response(idx: int, r: OCRResult) -> OCRLineResponse:
    coords = [[int(pt[0]), int(pt[1])] for pt in r.box.tolist()]
    return OCRLineResponse(
        index=idx,
        coords=coords,
        text=r.text,
        score=round(r.score, 4),
    )


@router.post("/ocr", response_model=OCRResponse)
async def ocr(
    file_id: str = Form(...),
    file: UploadFile = File(...),
    use_rotate: Optional[bool] = Form(None),
    vis: bool = Form(False),
):
    logger.info("POST /api/ocr  file_id=%s, file=%s, use_rotate=%s, vis=%s", file_id, file.filename, use_rotate, vis)

    if engine is None:
        logger.error("OCR engine not ready")
        return OCRResponse(code=503, message="OCR engine not ready")

    t0 = time.perf_counter()
    image_bytes = await file.read()
    logger.debug("读取文件完成, 大小=%d bytes", len(image_bytes))

    if vis:
        results, angle, vis_img = engine.ocr(image_bytes, use_rotate=use_rotate, return_visualization=True)
        _, buf = cv2.imencode(".jpg", vis_img)
        vis_b64 = base64.b64encode(buf).decode("ascii")
    else:
        results, angle = engine.ocr(image_bytes, use_rotate=use_rotate)
        vis_b64 = None

    if not results:
        logger.info("未检测到文字, 耗时=%.1fms", (time.perf_counter() - t0) * 1000)
        return OCRResponse(
            code=0,
            message="ok",
            data=OCRData(file_id=file_id, rotate=0, lines=[], text=""),
        )

    # 引擎内部已完成排序，直接用
    lines = [_to_line_response(i + 1, r) for i, r in enumerate(results)]
    full_text = "\n".join(r.text for r in results)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("OCR 完成, file_id=%s, rotate=%d, lines=%d, 耗时=%.1fms", file_id, angle, len(results), elapsed)

    return OCRResponse(
        code=0,
        message="ok",
        data=OCRData(
            file_id=file_id,
            rotate=angle,
            lines=lines,
            text=full_text,
            visualization=vis_b64,
        ),
    )
