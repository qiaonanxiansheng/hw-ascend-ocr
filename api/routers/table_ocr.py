from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Optional

import cv2
from fastapi import APIRouter, File, Form, UploadFile

from api.schemas import (
    OCRLineResponse,
    TableCellResponse,
    TableOCRData,
    TableOCRResponse,
    TableRegionResponse,
)

if TYPE_CHECKING:
    from ascend_ocr import AscendOCR, OCRResult
    from ascend_ocr.table_recognizer import TableStructure

logger = logging.getLogger("api.table_ocr")

router = APIRouter(prefix="/api", tags=["table-ocr"])

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


@router.post("/table-ocr", response_model=TableOCRResponse)
async def table_ocr(
    file_id: str = Form(...),
    file: UploadFile = File(...),
    use_rotate: Optional[bool] = Form(None),
    vis: bool = Form(False),
):
    logger.info(
        "POST /api/table-ocr  file_id=%s, file=%s, use_rotate=%s, vis=%s",
        file_id, file.filename, use_rotate, vis,
    )

    if engine is None:
        logger.error("OCR engine not ready")
        return TableOCRResponse(code=503, message="OCR engine not ready")

    if engine.table_recognizer is None:
        logger.error("Table model not configured")
        return TableOCRResponse(code=503, message="Table model not configured")

    t0 = time.perf_counter()
    image_bytes = await file.read()
    logger.debug("读取文件完成, 大小=%d bytes", len(image_bytes))

    try:
        tables, outside_text, angle = engine.table_ocr(
            image_bytes, use_rotate=use_rotate
        )
    except Exception as e:
        logger.error("Table OCR 失败: %s", e, exc_info=True)
        return TableOCRResponse(code=500, message=str(e))

    # Build table responses
    table_responses = []
    for tidx, table in enumerate(tables):
        cell_responses = []
        for cidx, cell in enumerate(table.cells):
            lines = [_to_line_response(i + 1, line) for i, line in enumerate(cell.lines)]
            cell_text = "\n".join(line.text for line in cell.lines)
            cell_responses.append(
                TableCellResponse(
                    index=cidx + 1,
                    bbox=list(cell.bbox),
                    score=round(cell.score, 4),
                    rowstart=cell.rowstart,
                    rowend=cell.rowend,
                    colstart=cell.colstart,
                    colend=cell.colend,
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                    lines=lines,
                    text=cell_text,
                )
            )

        table_responses.append(
            TableRegionResponse(
                index=tidx + 1,
                bbox=list(table.bbox),
                rows=table.rows,
                cols=table.cols,
                html=table.html,
                cells=cell_responses,
            )
        )

    # Outside text
    outside_lines = [_to_line_response(i + 1, r) for i, r in enumerate(outside_text)]

    # Visualization
    vis_b64 = None
    if vis:
        from ascend_ocr.image_utils import load_image

        img = load_image(image_bytes)
        vis_img = img.copy()
        for table in tables:
            for cell in table.cells:
                x1, y1, x2, y2 = cell.bbox
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"r{cell.rowstart}c{cell.colstart}"
                if cell.rowspan > 1 or cell.colspan > 1:
                    label += f"({cell.rowspan}x{cell.colspan})"
                cv2.putText(vis_img, label, (x1 + 2, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        _, buf = cv2.imencode(".jpg", vis_img)
        vis_b64 = base64.b64encode(buf).decode("ascii")

    elapsed = (time.perf_counter() - t0) * 1000
    total_cells = sum(len(t.cells) for t in tables)
    logger.info(
        "Table OCR 完成, file_id=%s, tables=%d, cells=%d, outside_lines=%d, rotate=%d, 耗时=%.1fms",
        file_id, len(tables), total_cells, len(outside_text), angle, elapsed,
    )

    return TableOCRResponse(
        code=0,
        message="ok",
        data=TableOCRData(
            file_id=file_id,
            rotate=angle,
            tables=table_responses,
            outside_text=outside_lines,
            visualization=vis_b64,
        ),
    )
