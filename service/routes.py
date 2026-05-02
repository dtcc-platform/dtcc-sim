import json
import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from service.config import SERVICE_NAME, SERVICE_VERSION, MODULE_PREFIX
import dtcc_core.datasets as datasets

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _get_service_datasets() -> Dict[str, Any]:
    """Return datasets belonging to this service (filtered by module prefix)."""
    return {
        name: ds
        for name, ds in datasets.list().items()
        if ds.__class__.__module__.startswith(MODULE_PREFIX)
    }


def _require_dataset(name: str):
    """Validate that a dataset name belongs to this service, or raise 404."""
    service_datasets = _get_service_datasets()
    if name not in service_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return service_datasets, service_datasets[name]


def _extract_formats(schema: dict) -> list[str]:
    """Extract supported format values from a dataset's JSON schema.

    Handles both Pydantic v2's const-in-anyOf (for Optional[Literal[...]])
    and plain enum (for Literal[...] without Optional).
    """
    fmt_prop = schema.get("properties", {}).get("format", {})
    formats = []
    if "enum" in fmt_prop:
        formats = [v for v in fmt_prop["enum"] if v is not None]
    elif "anyOf" in fmt_prop:
        for opt in fmt_prop["anyOf"]:
            if "const" in opt:
                formats.append(opt["const"])
            elif "enum" in opt:
                formats.extend(v for v in opt["enum"] if v is not None)
    return formats


def _task_state_payload(result) -> Dict[str, Any]:
    """Convert a Celery AsyncResult-like object into protocol JSON."""
    state = result.state

    if state == "PROGRESS":
        meta = result.info or {}
        return {
            "status": "running",
            "progress": meta.get("progress", 0),
            "message": meta.get("message", ""),
        }
    if state == "SUCCESS":
        meta = result.result or {}
        return {
            "status": "completed",
            "result_file": meta.get("result_file"),
            "size_bytes": meta.get("size_bytes"),
        }
    if state == "REVOKED":
        return {"status": "cancelled", "message": "Job cancelled by client"}
    if state == "FAILURE":
        return {
            "status": "failed",
            "error": str(result.result) if result.result else "Unknown error",
        }
    return {"status": "pending", "progress": 0, "message": ""}


@router.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@router.get("/datasets")
def list_datasets():
    service_datasets = _get_service_datasets()
    dataset_info = {}
    for name, ds in service_datasets.items():
        schema = ds.show_options()
        descriptor = ds.describe() if hasattr(ds, "describe") else {}
        formats = descriptor.get("supported_formats") or _extract_formats(schema)

        dataset_info[name] = {
            "name": name,
            "description": ds.description,
            "args_schema": schema,
            "data_category": descriptor.get("data_category", "simulation"),
            "result_kind": getattr(ds, "result_kind", "unknown"),
            "python_return_type": descriptor.get("python_return_type", "object"),
            "supported_formats": formats or ["bin"],
            "formats": descriptor.get("formats", []),
            "timeout_hint": getattr(ds, "timeout_hint", None),
        }

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "protocol_version": "1",
        "datasets": dataset_info,
    }


@router.post("/datasets/{name}/submit", status_code=201)
def submit_job(name: str, params: Dict[str, Any] = Body()):
    _, ds = _require_dataset(name)
    params = dict(params)

    # Enforce format (default to first supported format)
    if "format" not in params or params["format"] is None:
        formats = _extract_formats(ds.show_options())
        if formats:
            params["format"] = formats[0]

    # Validate using dataset's ArgsModel
    try:
        ds.ArgsModel(**params)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    # Submit Celery task
    from service.tasks import celery_app

    try:
        task = celery_app.send_task(f"dataset.{name}", args=[params])
    except Exception as e:
        logger.error(f"Broker unreachable when submitting {name}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: cannot reach task broker",
        )
    return {"task_id": task.id, "status": "pending"}


@router.get("/datasets/{name}/status/{task_id}")
async def job_status(name: str, task_id: str, request: Request):
    _require_dataset(name)
    from service.tasks import celery_app

    accept = request.headers.get("accept", "")

    if "text/event-stream" in accept:
        return StreamingResponse(
            _sse_status_stream(celery_app, task_id, request),
            media_type="text/event-stream",
        )

    # Non-SSE: return JSON snapshot (polling fallback)
    try:
        result = celery_app.AsyncResult(task_id)
    except Exception as e:
        logger.error(f"Broker unreachable when polling status for {task_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: cannot reach result backend",
        )
    return _task_state_payload(result)


async def _sse_status_stream(celery_app, task_id, request: Request | None = None):
    """Generate SSE events for job progress."""
    while True:
        if request is not None and await request.is_disconnected():
            return

        try:
            result = celery_app.AsyncResult(task_id)
        except Exception as e:
            logger.error(f"Broker unreachable during SSE stream for {task_id}: {e}")
            event = {
                "status": "failed",
                "error": "Service unavailable: cannot reach result backend",
            }
            yield f"data: {json.dumps(event)}\n\n"
            return

        event = _task_state_payload(result)
        yield f"data: {json.dumps(event)}\n\n"

        if event["status"] in {"completed", "cancelled", "failed"}:
            return

        await asyncio.sleep(0.5)


@router.post("/datasets/{name}/cancel/{task_id}")
def cancel_job(name: str, task_id: str):
    _require_dataset(name)
    from service.tasks import celery_app

    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    except Exception as e:
        logger.error(f"Broker unreachable when cancelling {task_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: cannot reach task broker",
        )
    return {"task_id": task_id, "status": "cancelling"}
