"""Live time-step reporting with retained worker logs and bounded execution."""

import json
import logging
import subprocess
import sys
import threading
from time import monotonic

import pytest

from dtcc_core import RectangleMesh
from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver
from dtcc_sim.shallow_water import solver as api


@pytest.fixture
def solver_logs(caplog, monkeypatch):
    monkeypatch.setattr(api._logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=api._logger.name)
    return caplog


def _worker_script(monkeypatch, source):
    run = subprocess.run

    def launch(command, **kwargs):
        return run([sys.executable, "-u", "-c", source], **kwargs)

    monkeypatch.setattr(api.subprocess, "run", launch)


def _steps(records):
    return [record.getMessage() for record in records
            if record.name == api._logger.name and record.getMessage().startswith("step=")]


def test_step_is_visible_before_worker_exits_and_failure_keeps_log(tmp_path, monkeypatch, solver_logs):
    acknowledgment = tmp_path / "progress-observed"
    first = "step=00001 t=0.01000 dt=1.000e-02"
    last = "step=00002 t=0.02000 dt=1.000e-02"

    class ObserveStep(logging.Handler):
        def emit(self, record):
            if record.getMessage() == first:
                acknowledgment.touch()

    handler = ObserveStep()
    api._logger.addHandler(handler)
    source = f"""
from pathlib import Path
import sys
import time
print('Preparing forms', flush=True)
sys.stdout.write('step=00001 t=0.')
sys.stdout.flush()
time.sleep(.15)
print('01000 dt=1.000e-02', flush=True)
deadline = time.monotonic() + 5
while not Path({str(acknowledgment)!r}).exists():
    if time.monotonic() > deadline:
        raise SystemExit(9)
    time.sleep(.01)
print('Numerical failure', file=sys.stderr, flush=True)
sys.stdout.write({last!r})
sys.stdout.flush()
raise SystemExit(7)
"""
    _worker_script(monkeypatch, source)
    solver = ShallowWaterSolver(mesh=RectangleMesh(1, 1), output_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="worker exited with 7"):
            solver.run(timeout=10)
    finally:
        api._logger.removeHandler(handler)

    assert _steps(solver_logs.records) == [first, last]
    case, = (tmp_path / ".shallow_water_runs").iterdir()
    assert (case / "solver.log").read_text() == f"Preparing forms\n{first}\nNumerical failure\n{last}"
    assert json.loads((case / "run.json").read_text())["status"] == "failed"
    assert not any(thread.name == "shallow-water-progress" for thread in threading.enumerate())


@pytest.mark.parametrize("chatty", [False, True])
def test_timeout_stops_progress_reporting_and_preserves_output(tmp_path, monkeypatch, solver_logs, chatty):
    step = "step=00001 t=0.01000 dt=1.000e-02"
    source = "import time\n"
    if chatty:
        source += f"print({step!r}, flush=True)\n"
    source += "time.sleep(60)\n"
    _worker_script(monkeypatch, source)
    solver = ShallowWaterSolver(mesh=RectangleMesh(1, 1), output_dir=tmp_path)
    started = monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        solver.run(timeout=1)
    assert monotonic() - started < 10
    case, = (tmp_path / ".shallow_water_runs").iterdir()
    assert json.loads((case / "run.json").read_text())["status"] == "failed"
    assert _steps(solver_logs.records) == ([step] if chatty else [])
    assert (case / "solver.log").read_text() == (step + "\n" if chatty else "")
    assert not any(thread.name == "shallow-water-progress" for thread in threading.enumerate())


@pytest.mark.usefixtures("real_worker")
@pytest.mark.simulation
def test_real_solve_reports_each_time_step_once(tmp_path, solver_logs):
    solver = ShallowWaterSolver(
        mesh=RectangleMesh(2, 2), output_dir=tmp_path,
        params=ShallowWaterParameters(amplitude=0., final_time=.001, n_outputs=2),
    )
    result = solver.run()
    logged = (result.case_dir / "solver.log").read_text().splitlines()
    steps = [line for line in logged if line.startswith("step=")]
    assert len(steps) == result.diagnostics["steps"] == 2
    assert _steps(solver_logs.records) == steps
