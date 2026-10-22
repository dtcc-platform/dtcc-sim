"""Job-control checks without numerical execution or a broker."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from celery.exceptions import Ignore

from service import helmholtz_control as control


@pytest.fixture
def env(tmp_path, monkeypatch):
    calls = []
    backend = SimpleNamespace(
        get=lambda key: None,
        mark_as_revoked=lambda *args, **kwargs: calls.append(("revoked", args, kwargs)),
    )
    task = SimpleNamespace(
        app=SimpleNamespace(backend=backend), request=SimpleNamespace(id="test-job")
    )
    monkeypatch.setattr(control, "SHARED_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        control.os, "killpg", lambda *args: calls.append(("kill", args))
    )
    process = SimpleNamespace(pid=12345, returncode=None)
    process.poll = lambda: process.returncode

    def wait(timeout):
        process.returncode = -15
        return process.returncode

    process.wait = wait
    monkeypatch.setattr(control.subprocess, "Popen", lambda *args, **kwargs: process)
    return task, process, calls, tmp_path


def test_cancel_before_start_creates_no_job(env):
    task, _, calls, directory = env
    task.app.backend.get = lambda key: b"1"
    with pytest.raises(Ignore):
        control.run_managed_helmholtz(task, {}, "dtccpkg")
    assert calls[0][0] == "revoked"
    assert not list(directory.iterdir())


def test_running_cancel_stops_group_and_acknowledges(env):
    task, process, calls, _ = env
    requests = iter([None, b"1"])
    task.app.backend.get = lambda key: next(requests)
    with pytest.raises(Ignore):
        control.run_managed_helmholtz(task, {}, "dtccpkg")
    assert process.returncode == -15
    assert [call[0] for call in calls] == ["kill", "kill", "revoked"]


def test_timeout_stops_group_and_fails(env, monkeypatch):
    task, process, calls, _ = env
    times = iter([0, 2])
    monkeypatch.setattr(control.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(control, "JOB_TIMEOUT", 1)
    with pytest.raises(TimeoutError, match="exceeded 1s"):
        control.run_managed_helmholtz(task, {}, "dtccpkg")
    assert process.returncode == -15
    assert [call[0] for call in calls] == ["kill", "kill"]


def test_child_failure_is_not_reported_as_success(env):
    task, process, _, _ = env
    process.returncode = 1
    with pytest.raises(RuntimeError, match="exit code 1"):
        control.run_managed_helmholtz(task, {}, "dtccpkg")


def test_success_returns_child_metadata(env, monkeypatch):
    task, process, _, _ = env
    expected = {"result_file": "test-job.dtccpkg", "size_bytes": 42}

    def launch(command, **kwargs):
        from pathlib import Path

        request = Path(command[command.index("--request") + 1])
        request.with_name("response.json").write_text(json.dumps(expected))
        process.returncode = 0
        return process

    monkeypatch.setattr(control.subprocess, "Popen", launch)
    assert control.run_managed_helmholtz(task, {}, "dtccpkg") == expected


def test_backend_failure_stops_running_child(env):
    task, process, calls, _ = env
    count = 0

    def get(key):
        nonlocal count
        count += 1
        if count > 1:
            raise ConnectionError("Backend offline")

    task.app.backend.get = get
    with pytest.raises(ConnectionError, match="Backend offline"):
        control.run_managed_helmholtz(task, {}, "dtccpkg")
    assert process.returncode == -15
    assert [call[0] for call in calls] == ["kill", "kill"]


def test_cancellation_request_uses_backend_key():
    calls = []
    app = SimpleNamespace(backend=SimpleNamespace(set=lambda *args: calls.append(args)))
    control.request_cancellation(app, "one-job")
    assert calls == [("dtcc-sim:helmholtz:cancel:one-job", b"1")]


def test_parent_loss_uses_configured_backend_without_environment(monkeypatch):
    import celery
    from threading import Lock
    from service import config, helmholtz_job

    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    monkeypatch.setattr(config, "CELERY_RESULT_BACKEND", "redis://localhost:16379/1")
    calls = []
    app = SimpleNamespace(
        conf=SimpleNamespace(update=lambda **kwargs: None),
        backend=SimpleNamespace(mark_as_failure=lambda *args: calls.append(args)),
    )

    def create_app(name, *, backend):
        assert backend == config.CELERY_RESULT_BACKEND
        return app

    monkeypatch.setattr(celery, "Celery", create_app)
    helmholtz_job._report_parent_loss("interrupted-job", Lock())
    assert calls[0][0] == "interrupted-job"
    assert "interrupted" in str(calls[0][1])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group verification")
def test_stop_group_kills_descendant_ignoring_sigterm(tmp_path):
    ready = tmp_path / "child.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    process = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        child_pid = int(ready.read_text())
        control._stop_group(process)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/stat")
            if (
                not status.exists()
                or status.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Descendant survived job-group termination")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux hard-stop verification")
def test_hard_stop_leaves_partial_case_rejected_for_reuse(tmp_path):
    from dtcc_sim.helmholtz import HelmholtzSimulator

    case = tmp_path / "partial-case"
    program = """
import sys,time
from pathlib import Path
from dtcc_sim.helmholtz.simulator import _case_lease
case = Path(sys.argv[1])
case.mkdir()
with _case_lease(case):
    (case / "preconditioner_data.npz").write_bytes(b"partial")
    time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(case)], start_new_session=True
    )
    try:
        partial = case / "preconditioner_data.npz"
        deadline = time.monotonic() + 15
        while not partial.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.05)
        assert partial.exists()
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        simulator = HelmholtzSimulator(
            mesh_path=tmp_path / "mesh.xdmf", output_dir=case
        )
        with pytest.raises(FileExistsError, match="not empty"):
            simulator.prepare()
        with pytest.raises(FileNotFoundError, match="prepare"):
            simulator.solve()
        assert partial.read_bytes() == b"partial"
        assert (case / ".helmholtz.lock").is_file()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
