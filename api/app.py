from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 配置日志输出到 stderr，docker logs 才能看到
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # 屏蔽 uvicorn 请求日志

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ascend_ocr import AscendOCR, load_config
from ascend_ocr.exceptions import (
    ImageLoadError,
    InferenceError,
    ModelLoadError,
    PreprocessError,
    AscendOCRError,
)
from api.routers import ocr as ocr_router_module
from api.routers import layout_ocr as layout_ocr_router_module
from api.routers import table_ocr as table_ocr_router_module
from api.routers.ocr import router as ocr_router
from api.routers.layout_ocr import router as layout_ocr_router
from api.routers.table_ocr import router as table_ocr_router
from api.schemas import OCRResponse

logger = logging.getLogger("api")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading OCR config from %s", CONFIG_PATH)
    config = load_config(str(CONFIG_PATH))
    engine = AscendOCR(config=config)
    ocr_router_module.engine = engine
    layout_ocr_router_module.engine = engine
    table_ocr_router_module.engine = engine
    logger.info("OCR engine ready, layout_model=%s, table_model=%s", config.layout_model, config.table_model)
    yield
    engine.release()
    ocr_router_module.engine = None
    layout_ocr_router_module.engine = None
    table_ocr_router_module.engine = None
    logger.info("OCR engine released")


app = FastAPI(title="Ascend OCR API", lifespan=lifespan)

app.include_router(ocr_router)
app.include_router(layout_ocr_router)
app.include_router(table_ocr_router)


@app.exception_handler(AscendOCRError)
async def ocr_error_handler(request: Request, exc: AscendOCRError):
    status_map = {
        ModelLoadError: 503,
        InferenceError: 500,
        ImageLoadError: 400,
        PreprocessError: 400,
    }
    status = status_map.get(type(exc), 500)
    logger.error("OCR 异常: %s %s -> %d %s", request.method, request.url.path, status, exc)
    resp = OCRResponse(code=status, message=str(exc))
    return JSONResponse(status_code=status, content=resp.model_dump())
