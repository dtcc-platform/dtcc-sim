import importlib

import pytest

from dtcc_sim import __version__


def test_version_available():
    assert isinstance(__version__, str)


@pytest.mark.simulation
@pytest.mark.fenics
def test_fenics_environment_import_smoke():
    missing_backend = []
    for module_name in ("dolfinx", "mpi4py", "petsc4py"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                missing_backend.append(module_name)
                continue
            raise

    if missing_backend:
        missing = ", ".join(missing_backend)
        pytest.skip(
            "FEniCSx runtime is not active "
            f"(missing {missing}); create it with "
            "conda env create -f fenicsx-env.yml"
        )

    importlib.import_module("dtcc_core")
    importlib.import_module("dtcc_sim")


@pytest.mark.parametrize("level", ["WARNING", "INFO", "invalid"])
def test_fenics_log_level_configuration(level):
    import os
    import subprocess
    import sys

    pytest.importorskip("dolfinx")
    result = subprocess.run(
        [sys.executable, "-c",
         "import dtcc_sim; from dolfinx import log; print(log.get_log_level().name)"],
        env={**os.environ, "DTCC_FENICSX_LOG_LEVEL": level},
        capture_output=True, text=True,
    )
    if level == "invalid":
        assert result.returncode != 0
        assert "DTCC_FENICSX_LOG_LEVEL must be" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == level
