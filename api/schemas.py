from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class OCRLineResponse(BaseModel):
    index: int
    coords: List[List[int]]
    text: str
    score: float


class OCRData(BaseModel):
    file_id: str
    rotate: int
    lines: List[OCRLineResponse]
    text: str
    visualization: Optional[str] = None  # base64 编码的标注图片


class OCRResponse(BaseModel):
    code: int
    message: str
    data: Optional[OCRData] = None


class LayoutRegionResponse(BaseModel):
    index: int
    class_id: int
    class_name: str
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]
    lines: List[OCRLineResponse]
    text: str


class LayoutOCRData(BaseModel):
    file_id: str
    rotate: int
    regions: List[LayoutRegionResponse]
    total_lines: int
    visualization: Optional[str] = None


class LayoutOCRResponse(BaseModel):
    code: int
    message: str
    data: Optional[LayoutOCRData] = None
