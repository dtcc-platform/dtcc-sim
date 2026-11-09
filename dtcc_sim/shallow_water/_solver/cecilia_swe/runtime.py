"""Check the numerical runtime before preparing a mesh or output."""


def validate_runtime():
    setup = ("Use the common real environment: conda env create -f fenicsx-env.yml, "
             "then conda activate fenicsx-env; "
             "see docs/shallow-water.md.")
    try:
        import dolfinx
        import numpy as np
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError(f"The shallow-water worker needs DOLFINx 0.11 and real PETSc. {setup}") from error
    if not dolfinx.__version__.startswith("0.11."):
        raise RuntimeError(f"Unsupported worker DOLFINx {dolfinx.__version__}; expected 0.11.x. {setup}")
    if np.dtype(PETSc.ScalarType) != np.dtype("float64"):
        raise RuntimeError(f"The shallow-water worker requires real Float64 PETSc. {setup}")
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The shallow-water worker supports one MPI rank")
