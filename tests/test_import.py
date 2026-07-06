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
            "conda env create -f environment-fenicsx.yml"
        )

    importlib.import_module("dtcc_core")
    importlib.import_module("dtcc_sim")
