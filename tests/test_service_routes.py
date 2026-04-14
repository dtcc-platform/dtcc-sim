import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import BaseModel


class _BasicArgs(BaseModel):
    bounds: list[float]
    format: str | None = None


class _FakeAsyncResult:
    def __init__(self, state, info=None, result=None):
        self.state = state
        self.info = info
        self.result = result


class _FakeCeleryApp:
    def __init__(self):
        self.sent = []
        self.revoked = []
        self._results = {}
        self.control = SimpleNamespace(revoke=self._revoke)

    def send_task(self, name, args):
        self.sent.append((name, args))
        return SimpleNamespace(id="task-123")

    def AsyncResult(self, task_id):
        result = self._results[task_id]
        if callable(result):
            return result()
        return result

    def _revoke(self, *args, **kwargs):
        self.revoked.append((args, kwargs))


def _make_dataset(
    schema,
    *,
    module_name="dtcc_sim.datasets",
    description="A remote-ready dataset",
    result_kind="mesh",
    timeout_hint=600,
):
    dataset_type = type("FakeDataset", (), {})
    dataset_type.__module__ = module_name
    dataset = dataset_type()
    dataset.description = description
    dataset.result_kind = result_kind
    dataset.timeout_hint = timeout_hint
    dataset.ArgsModel = _BasicArgs
    dataset.show_options = lambda: schema
    return dataset


def _install_fake_tasks(monkeypatch, celery_app):
    fake_tasks = types.ModuleType("service.tasks")
    fake_tasks.celery_app = celery_app
    monkeypatch.setitem(sys.modules, "service.tasks", fake_tasks)


@pytest.fixture
def routes_env(monkeypatch):
    registry = {}
    fake_dtcc_core = types.ModuleType("dtcc_core")
    fake_datasets = types.ModuleType("dtcc_core.datasets")
    fake_datasets.list = lambda: registry
    fake_dtcc_core.datasets = fake_datasets

    monkeypatch.setitem(sys.modules, "dtcc_core", fake_dtcc_core)
    monkeypatch.setitem(sys.modules, "dtcc_core.datasets", fake_datasets)
    sys.modules.pop("service.routes", None)

    import service.routes as routes

    return importlib.reload(routes), registry


def test_extract_formats_supports_enum_and_anyof(routes_env):
    routes, _ = routes_env

    assert routes._extract_formats(
        {"properties": {"format": {"enum": ["pb", None, "xdmf"]}}}
    ) == ["pb", "xdmf"]
    assert routes._extract_formats(
        {
            "properties": {
                "format": {
                    "anyOf": [
                        {"const": "xdmf"},
                        {"enum": ["pb", None]},
                    ]
                }
            }
        }
    ) == ["xdmf", "pb"]


def test_list_datasets_only_exposes_service_owned_descriptors(routes_env):
    routes, registry = routes_env
    registry["urban_heat"] = _make_dataset(
        {
            "properties": {
                "format": {"anyOf": [{"const": "xdmf"}]},
            }
        },
        timeout_hint=900,
    )
    registry["external_dataset"] = _make_dataset(
        {"properties": {"format": {"enum": ["pb"]}}},
        module_name="other_service.datasets",
    )

    payload = routes.list_datasets()

    assert payload["service"] == "dtcc-sim"
    assert set(payload["datasets"].keys()) == {"urban_heat"}
    assert payload["datasets"]["urban_heat"]["supported_formats"] == ["xdmf"]
    assert payload["datasets"]["urban_heat"]["timeout_hint"] == 900


def test_submit_job_injects_default_format(routes_env, monkeypatch):
    routes, registry = routes_env
    registry["urban_heat"] = _make_dataset(
        {"properties": {"format": {"anyOf": [{"const": "xdmf"}]}}}
    )
    celery_app = _FakeCeleryApp()
    _install_fake_tasks(monkeypatch, celery_app)

    payload = routes.submit_job("urban_heat", {"bounds": [1.0, 2.0, 3.0, 4.0]})

    assert payload == {"task_id": "task-123", "status": "pending"}
    assert celery_app.sent == [
        (
            "dataset.urban_heat",
            [{"bounds": [1.0, 2.0, 3.0, 4.0], "format": "xdmf"}],
        )
    ]


def test_submit_job_returns_422_for_invalid_params(routes_env):
    routes, registry = routes_env
    registry["urban_heat"] = _make_dataset(
        {"properties": {"format": {"enum": ["xdmf"]}}}
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.submit_job("urban_heat", {})

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail


def test_job_status_returns_non_sse_snapshot(routes_env, monkeypatch):
    routes, registry = routes_env
    registry["urban_heat"] = _make_dataset(
        {"properties": {"format": {"enum": ["xdmf"]}}}
    )
    celery_app = _FakeCeleryApp()
    celery_app._results["task-7"] = _FakeAsyncResult(
        "SUCCESS",
        result={"result_file": "task-7.tar.gz", "size_bytes": 128},
    )
    _install_fake_tasks(monkeypatch, celery_app)

    request = SimpleNamespace(headers={})
    payload = asyncio.run(routes.job_status("urban_heat", "task-7", request))

    assert payload == {
        "status": "completed",
        "result_file": "task-7.tar.gz",
        "size_bytes": 128,
    }


def test_cancel_job_revokes_celery_task(routes_env, monkeypatch):
    routes, registry = routes_env
    registry["urban_heat"] = _make_dataset(
        {"properties": {"format": {"enum": ["xdmf"]}}}
    )
    celery_app = _FakeCeleryApp()
    _install_fake_tasks(monkeypatch, celery_app)

    payload = routes.cancel_job("urban_heat", "task-99")

    assert payload == {"task_id": "task-99", "status": "cancelling"}
    assert celery_app.revoked == [
        (("task-99",), {"terminate": True, "signal": "SIGTERM"})
    ]


def test_sse_stream_emits_pending_then_completed(routes_env, monkeypatch):
    routes, _ = routes_env
    results = [
        _FakeAsyncResult("PENDING"),
        _FakeAsyncResult(
            "SUCCESS",
            result={"result_file": "job-1.pb", "size_bytes": 42},
        ),
    ]

    celery_app = _FakeCeleryApp()
    celery_app._results["job-1"] = lambda: results.pop(0)

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", _no_sleep)

    async def _collect():
        chunks = []
        async for chunk in routes._sse_status_stream(celery_app, "job-1"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())
    payloads = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]

    assert payloads == [
        {"status": "pending", "progress": 0, "message": ""},
        {"status": "completed", "result_file": "job-1.pb", "size_bytes": 42},
    ]


def test_sse_stream_stops_when_client_disconnects(routes_env):
    routes, _ = routes_env
    calls = []

    class _DisconnectingRequest:
        async def is_disconnected(self):
            return True

    class _CountingCeleryApp:
        def AsyncResult(self, task_id):
            calls.append(task_id)
            return _FakeAsyncResult("PENDING")

    async def _collect():
        chunks = []
        async for chunk in routes._sse_status_stream(
            _CountingCeleryApp(), "job-2", _DisconnectingRequest()
        ):
            chunks.append(chunk)
        return chunks

    assert asyncio.run(_collect()) == []
    assert calls == []
