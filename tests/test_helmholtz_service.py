"""Broker-free checks for isolated Helmholtz task routing and worker startup."""

import json

from service import helmholtz_worker
from service.config import HELMHOLTZ_QUEUE


def test_helmholtz_routes_separately_from_heat():
    from service.tasks import celery_app

    router = celery_app.amqp.Router()
    helmholtz = router.route({}, "dataset.helmholtz_simulation")
    heat = router.route({}, "dataset.urban_heat_simulation")
    assert helmholtz["queue"].name == HELMHOLTZ_QUEUE
    assert heat["queue"].name == "dtcc-sim"
    assert helmholtz["queue"].name != heat["queue"].name


def test_worker_plans_without_starting(monkeypatch, capsys):
    def forbidden(*args):
        raise AssertionError("Planning must not start a worker")

    monkeypatch.setattr(helmholtz_worker.os, "execv", forbidden)
    assert helmholtz_worker.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["starts_worker"] is False
    assert report["queue"] == HELMHOLTZ_QUEUE
    command = helmholtz_worker.worker_command()
    assert command[command.index("--queues") + 1] == HELMHOLTZ_QUEUE
    assert "--pool=solo" in command
    assert "--concurrency=1" in command


def test_worker_refuses_bad_runtime(monkeypatch):
    from dtcc_sim.helmholtz import check_runtime

    monkeypatch.setattr(check_runtime, "inspect_runtime", lambda **kw: {"ready": False})
    monkeypatch.setattr(
        helmholtz_worker.os,
        "execv",
        lambda *args: (_ for _ in ()).throw(AssertionError("Worker must not start")),
    )
    assert helmholtz_worker.main(["--start"]) == 1


def test_worker_start_uses_current_python_and_checked_runtime(monkeypatch):
    from dtcc_sim.helmholtz import check_runtime

    checks, calls = [], []

    def inspect(**kwargs):
        checks.append(kwargs)
        return {"ready": True}

    monkeypatch.setattr(check_runtime, "inspect_runtime", inspect)
    monkeypatch.setattr(helmholtz_worker.os, "execv", lambda *args: calls.append(args))
    assert helmholtz_worker.main(["--start"]) == 0
    assert checks == [{"workers": 2, "mesh_generation": True}]
    assert calls == [
        (helmholtz_worker.sys.executable, helmholtz_worker.worker_command())
    ]
