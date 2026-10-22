"""Supervise Helmholtz jobs independently of Celery's blocking solo pool."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from celery.exceptions import Ignore

from service.config import JOB_TIMEOUT, SHARED_RESULTS_DIR


def cancellation_key(task_id):
    return f"dtcc-sim:helmholtz:cancel:{task_id}"


def request_cancellation(app, task_id):
    """Store a cancellation request using the result backend's normal expiry."""
    app.backend.set(cancellation_key(task_id), b"1")


def _stop_group(process):
    """Stop the job session, including spawned patch workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # The group can outlive its leader if a patch worker ignores SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def run_managed_helmholtz(task, params, format_ext):
    """Run one job with cancellation, a wall-time limit, and retained logs."""
    backend = task.app.backend
    task_id = task.request.id

    def cancelled():
        return backend.get(cancellation_key(task_id)) is not None

    def acknowledge_cancel():
        backend.mark_as_revoked(task_id, reason="Helmholtz job cancelled by client")
        raise Ignore()

    if cancelled():
        acknowledge_cancel()
    directory = Path(SHARED_RESULTS_DIR) / ".helmholtz-jobs"
    directory.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="job-", dir=directory))
    request = job_dir / "request.json"
    response = job_dir / "response.json"
    request.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "params": params,
                "format": format_ext,
                "parent_pid": os.getpid(),
            }
        )
        + "\n"
    )
    command = [sys.executable, "-m", "service.helmholtz_job", "--request", str(request)]
    process = None
    stopped = False

    def stop():
        nonlocal stopped
        if process is not None and not stopped:
            _stop_group(process)
            stopped = True

    started = time.monotonic()
    try:
        with (job_dir / "run.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (job_dir / "process.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "pid": process.pid,
                        "parent_pid": os.getpid(),
                    }
                )
                + "\n"
            )
            while process.poll() is None:
                if cancelled():
                    stop()
                    acknowledge_cancel()
                if time.monotonic() - started >= JOB_TIMEOUT:
                    stop()
                    raise TimeoutError(
                        f"Helmholtz job exceeded {JOB_TIMEOUT}s; use a fresh case directory. Logs: {job_dir}"
                    )
                time.sleep(0.2)
        if process.returncode != 0 or not response.is_file():
            raise RuntimeError(
                f"Helmholtz job failed with exit code {process.returncode}; inspect {job_dir / 'run.log'}"
            )
        return json.loads(response.read_text())
    finally:
        stop()
