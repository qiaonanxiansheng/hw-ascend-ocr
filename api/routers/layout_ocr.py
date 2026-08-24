from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Optional

import cv2
from fastapi import APIRouter, File, Form, UploadFile

from api.schemas import (
    LayoutOCRData,
    LayoutOCRResponse,
    LayoutRegionResponse,
    OCRLineResponse,
)

if TYPE_CHECKING:
    from ascend_ocr import AscendOCR, OCRResult
    from ascend_ocr.layout_analyzer import LayoutRegion

logger = logging.getLogger("api.layout_ocr")

router = APIRouter(prefix="/api", tags=["layout-ocr"])

# Will be set by app.py lifespan
engine: Optional["AscendOCR"] = None


def _to_line_response(idx: int, r: "OCRResult") -> OCRLineResponse:
    coords = [[int(pt[0]), int(pt[1])] for pt in r.box.tolist()]
    return OCRLineResponse(
        index=idx,
        coords=coords,
        text=r.text,
        score=round(r.score, 4),
    )


@router.post("/layout-ocr", response_model=LayoutOCRResponse)
async def layout_ocr(
    file_id: str = Form(...),
    file: UploadFile = File(...),
    use_rotate: Optional[bool] = Form(None),
    score_threshold: float = Form(0.5),
    vis: bool = Form(False),
):
    logger.info(
        "POST /api/layout-ocr  file_id=%s, file=%s, use_rotate=%s, score_threshold=%.2f, vis=%s",
        file_id, file.filename, use_rotate, score_threshold, vis,
    )

    if engine is None:
        logger.error("OCR engine not ready")
        return LayoutOCRResponse(code=503, message="OCR engine not ready")

    if engine.layout_analyzer is None:
        logger.error("Layout model not configured")
        return LayoutOCRResponse(code=503, message="Layout model not configured")

    t0 = time.perf_counter()
    image_bytes = await file.read()
    logger.debug("读取文件完成, 大小=%d bytes", len(image_bytes))

    try:
        regions, region_ocr_results, clusters, angle = engine.layout_ocr(
            image_bytes, use_rotate=use_rotate, score_threshold=score_threshold
        )
    except Exception as e:
        logger.error("Layout OCR 失败: %s", e, exc_info=True)
        return LayoutOCRResponse(code=500, message=str(e))

    # Build response: include ALL regions (table/seal etc. have empty lines)
    ocr_map = {id(region): ocr_results for region, ocr_results in region_ocr_results}
    region_responses = []
    total_lines = 0
    for idx, region in enumerate(regions):
        ocr_results = ocr_map.get(id(region), [])
        lines = [_to_line_response(i + 1, r) for i, r in enumerate(ocr_results)]
        full_text = "\n".join(r.text for r in ocr_results)
        total_lines += len(ocr_results)

        region_responses.append(
            LayoutRegionResponse(
                index=idx + 1,
                class_id=region.class_id,
                class_name=region.class_name,
                score=round(region.score, 4),
                bbox=list(region.bbox),
                lines=lines,
                text=full_text,
                html=region.html or None,
            )
        )

    # Visualization
    vis_b64 = None
    if vis:
        from ascend_ocr.image_utils import load_image

        img = load_image(image_bytes)
        vis_img = img.copy()
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{region.class_name} {region.score:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _, buf = cv2.imencode(".jpg", vis_img)
        vis_b64 = base64.b64encode(buf).decode("ascii")

    elapsed = (time.perf_counter() - t0) * 1000
    table_count = sum(1 for r in regions if r.class_name == "table" and r.html)
    logger.info(
        "Layout OCR 完成, file_id=%s, regions=%d, lines=%d, tables=%d, rotate=%d, 耗时=%.1fms",
        file_id, len(regions), total_lines, table_count, angle, elapsed,
    )

    return LayoutOCRResponse(
        code=0,
        message="ok",
        data=LayoutOCRData(
            file_id=file_id,
            rotate=angle,
            regions=region_responses,
            total_lines=total_lines,
            visualization=vis_b64,
        ),
    )
