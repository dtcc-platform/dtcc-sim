"""Inspect Helmholtz runtime imports without generating a mesh or solving."""

import argparse
import importlib
import json
import multiprocessing
from importlib import metadata


def inspect_runtime(*, workers=1, mesh_generation=False):
    """Return individual readiness checks; import failures remain visible."""
    if workers < 1:
        raise ValueError("workers must be positive")
    checks = []

    def record(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    modules = {}
    required = [
        "numpy",
        "scipy",
        "dolfinx",
        "mpi4py.MPI",
        "petsc4py.PETSc",
        "pymetis",
        "threadpoolctl",
        "dtcc_core",
    ]
    if mesh_generation:
        required.extend(["dtcc_mesher", "dtcc_tetgen_wrapper"])
    for name in required:
        try:
            modules[name] = importlib.import_module(name)
            record(f"import {name}", True, "available")
        except (ImportError, OSError, RuntimeError) as exc:
            record(f"import {name}", False, f"{type(exc).__name__}: {exc}")

    if "numpy" in modules and "petsc4py.PETSc" in modules:
        np = modules["numpy"]
        scalar = modules["petsc4py.PETSc"].ScalarType
        record("complex PETSc", np.issubdtype(scalar, np.complexfloating), scalar)
    if "dolfinx" in modules:
        version = modules["dolfinx"].__version__
        record("supported DOLFINx", version == "0.11.0", version)
    if "mpi4py.MPI" in modules:
        size = modules["mpi4py.MPI"].COMM_WORLD.size
        record("single MPI rank", size == 1, f"{size} rank(s)")
    daemon = multiprocessing.current_process().daemon
    record(
        "patch worker process",
        workers == 1 or not daemon,
        f"requested workers={workers}, daemon={daemon}",
    )
    versions = {}
    for name in ("dtcc-sim", "dtcc-core", "dtcc-mesher", "dtcc-tetgen-wrapper"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "ready": all(check["ok"] for check in checks),
        "checks": checks,
        "versions": versions,
        "scope": "Imports and runtime configuration only; no mesh or solver execution",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mesh-generation", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = inspect_runtime(workers=args.workers, mesh_generation=args.mesh_generation)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
