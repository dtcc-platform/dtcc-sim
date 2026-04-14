import importlib
import logging

from service.config import SERVICE_NAME, MODULE_PREFIX
from service.routes import router
from service.tasks import celery_app
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Import the DTCC package to trigger dataset auto-registration
importlib.import_module(MODULE_PREFIX)

app = FastAPI(title=SERVICE_NAME)
app.include_router(router)
app.celery = celery_app


def _error_message(detail, status_code: int) -> str:
    """Return a human-readable protocol error message."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        if isinstance(detail.get("error"), str):
            return detail["error"]
        if isinstance(detail.get("detail"), str):
            return detail["detail"]
    if status_code == 422:
        return "Validation error"
    return "Request failed"


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return all HTTP errors in the protocol's error envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": _error_message(exc.detail, exc.status_code),
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors."""
    logger.exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
