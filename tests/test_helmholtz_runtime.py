"""Runtime failure diagnostics without solver or worker execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from dtcc_sim.helmholtz import check_runtime


@pytest.mark.parametrize(
    "scalar,ranks,daemon,missing,expected",
    [
        (np.complex128, 1, False, None, None),
        (np.float64, 1, False, None, "complex PETSc"),
        (np.complex128, 2, False, None, "single MPI rank"),
        (np.complex128, 1, True, None, "patch worker process"),
        (np.complex128, 1, False, "dtcc_tetgen_wrapper", "import dtcc_tetgen_wrapper"),
    ],
)
def test_readiness_diagnostics(monkeypatch, scalar, ranks, daemon, missing, expected):
    modules = {
        "numpy": np,
        "petsc4py.PETSc": SimpleNamespace(ScalarType=scalar),
        "mpi4py.MPI": SimpleNamespace(COMM_WORLD=SimpleNamespace(size=ranks)),
        "dolfinx": SimpleNamespace(__version__="0.11.0"),
    }

    def import_module(name):
        if name == missing:
            raise ImportError("native library unavailable")
        return modules.get(name, SimpleNamespace())

    monkeypatch.setattr(check_runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(
        check_runtime.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(daemon=daemon),
    )
    report = check_runtime.inspect_runtime(workers=16, mesh_generation=True)
    failed = [entry["name"] for entry in report["checks"] if not entry["ok"]]
    assert failed == ([] if expected is None else [expected])
    assert report["ready"] == (expected is None)


def test_invalid_worker_count():
    with pytest.raises(ValueError, match="positive"):
        check_runtime.inspect_runtime(workers=0)
