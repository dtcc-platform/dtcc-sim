"""Execute one supervised Helmholtz dataset job in its own process session."""

import argparse
import json
import os
import signal
from pathlib import Path
from threading import Event, Lock, Thread


def _report_parent_loss(task_id, state_lock):
    from celery import Celery
    from service.config import CELERY_RESULT_BACKEND

    app = Celery("helmholtz_watchdog", backend=CELERY_RESULT_BACKEND)
    app.conf.update(redis_socket_timeout=1, redis_socket_connect_timeout=1)
    with state_lock:
        app.backend.mark_as_failure(
            task_id,
            RuntimeError(
                "Helmholtz worker was interrupted; use a fresh case directory"
            ),
        )


def _watch_parent(parent_pid, task_id, finished, directory, parent_lost, state_lock):
    while not finished.wait(0.2):
        if os.getppid() == parent_pid:
            continue
        parent_lost.set()
        try:
            (directory / "interrupted.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "reason": "Supervisor process exited",
                    }
                )
                + "\n"
            )
            reporter = Thread(
                target=_report_parent_loss, args=(task_id, state_lock), daemon=True
            )
            reporter.start()
            reporter.join(timeout=2)
        finally:
            os.killpg(os.getpgrp(), signal.SIGKILL)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    values = json.loads(args.request.read_text())
    finished = Event()
    parent_lost = Event()
    state_lock = Lock()
    watcher = Thread(
        target=_watch_parent,
        args=(
            values["parent_pid"],
            values["task_id"],
            finished,
            args.request.parent,
            parent_lost,
            state_lock,
        ),
        daemon=True,
    )
    watcher.start()
    app = None
    try:
        from service.progress import progress_bridge
        from service.results import handle_result
        from service.tasks import _service_datasets, celery_app

        app = celery_app

        class Progress:
            def update_state(self, state, meta):
                with state_lock:
                    if not parent_lost.is_set():
                        celery_app.backend.store_result(values["task_id"], meta, state)

        with progress_bridge(Progress()):
            result = _service_datasets["helmholtz_simulation"](**values["params"])
            metadata = handle_result(result, values["format"], values["task_id"])
        temporary = args.request.with_name("response.json.tmp")
        temporary.write_text(json.dumps(metadata) + "\n")
        temporary.replace(args.request.with_name("response.json"))
        with state_lock:
            if parent_lost.is_set() or os.getppid() != values["parent_pid"]:
                parent_lost.set()
                raise RuntimeError("Supervisor exited before result completion")
            # A complete artifact remains discoverable if the worker exits before returning.
            celery_app.backend.mark_as_done(values["task_id"], metadata)
    except Exception as exc:
        if app is not None and not parent_lost.is_set():
            with state_lock:
                app.backend.mark_as_failure(values["task_id"], exc)
        raise
    finally:
        if parent_lost.is_set():
            watcher.join(timeout=3)
        finished.set()


if __name__ == "__main__":
    main()
