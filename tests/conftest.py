from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run expensive/manual-scale simulation tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        return

    skip_expensive = pytest.mark.skip(
        reason="expensive simulation tests require --run-expensive"
    )
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)


@pytest.fixture(scope="session")
def real_worker():
    """Probe the selected worker in isolation, without initializing caller MPI."""
    configured = os.environ.get("DTCC_SWE_PYTHON")
    executable = configured or sys.executable
    worker_source = Path(__file__).resolve().parents[1] / "dtcc_sim/shallow_water/_solver"
    probe = """
import importlib.util
required = ("dolfinx", "petsc4py", "mpi4py")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing optional runtime: " + ", ".join(missing))
    raise SystemExit(77)
from cecilia_swe.runtime import validate_runtime
validate_runtime()
"""
    environment = dict(os.environ, PYTHONPATH=str(worker_source), PYTHONNOUSERSITE="1")
    try:
        status = subprocess.run([executable, "-c", probe], env=environment,
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        pytest.fail(f"Cannot start shallow-water worker {executable}: {error}")
    detail = (status.stdout + status.stderr).strip()
    if status.returncode == 77 and not configured:
        pytest.skip(detail)
    if status.returncode:
        pytest.fail(f"Incompatible shallow-water worker {executable}: {detail}")
    return executable
