"""Check the dedicated numerical runtime before preparing a mesh or output."""


def validate_runtime():
    setup = ("Create the worker with conda env create -f shallow-water-worker.yml "
             "and set DTCC_SWE_PYTHON to that environment's bin/python; "
             "see docs/shallow-water.md.")
    try:
        import dolfinx
        import numpy as np
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError(f"The shallow-water worker needs DOLFINx 0.10 and real PETSc. {setup}") from error
    if not dolfinx.__version__.startswith("0.10."):
        raise RuntimeError(f"Unsupported worker DOLFINx {dolfinx.__version__}; expected 0.10.x. {setup}")
    if np.dtype(PETSc.ScalarType) != np.dtype("float64"):
        raise RuntimeError(f"The shallow-water worker requires real Float64 PETSc. {setup}")
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The shallow-water worker supports one MPI rank")
