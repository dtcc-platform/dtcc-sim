"""Python entry point for the two-level Helmholtz solver."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

from .parameters import HelmholtzParameters
from ._backend.params import Params, bind_parameters

if TYPE_CHECKING:
    from dtcc_core.model import VolumeMesh

_backend_lock = RLock()


def _solver_identity():
    """Fingerprint the shipped solver sources used to create offline data."""
    root = Path(__file__).parent
    paths = [root / "parameters.py", root / "simulator.py"]
    paths.extend(sorted((root / "_backend").rglob("*.py")))
    return {
        "compatibility_revision": 1,
        "sources": {str(path.relative_to(root)): _sha256(path) for path in paths},
    }


def _copy_parameters(params=None, parameters=None):
    if params is not None and parameters is not None:
        raise ValueError("Specify params or parameters, not both")
    selected = params if params is not None else parameters
    if selected is not None and not isinstance(selected, HelmholtzParameters):
        raise TypeError("params must be HelmholtzParameters")
    return deepcopy(selected if selected is not None else HelmholtzParameters())


def _simulation_mesh(result, *, output_path=None, unit=""):
    if not result.converged:
        raise RuntimeError("Helmholtz solve did not converge; inspect diagnostics")
    if not math.isfinite(result.relative_residual) or result.relative_residual < 0:
        raise RuntimeError("Helmholtz returned an invalid true residual")
    mesh = result.to_dtcc(unit=unit)
    if output_path is not None:
        from .results import _checked_export_path

        path = _checked_export_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(path)
    return mesh


@dataclass(frozen=True)
class HelmholtzResult:
    """Solution in finite-element DOF order and convergence diagnostics."""

    solution: Any
    iterations: int
    relative_residual: float
    converged: bool
    gmres_history: tuple[float, ...]
    output_dir: Path
    mesh_path: Path | None = None
    fe_degree: int = 1
    mesh_name: str = "mesh"
    prepared_identity: dict[str, Any] | None = None

    @classmethod
    def load(cls, output_dir):
        """Load a completed integrated case without repeating the numerical stages."""
        from .results import load_result

        return load_result(output_dir)

    def to_dtcc(self, *, unit=""):
        """Return the input DTCC mesh with real, imaginary, and amplitude fields."""
        from .results import result_to_dtcc

        return result_to_dtcc(self, unit=unit)

    def save_dtcc(self, path=None, *, unit=""):
        """Save the mesh and fields as VTU, XDMF, or DTCC without rerunning the solver."""
        from .results import save_dtcc_result

        return save_dtcc_result(self, path, unit=unit)


def _load_backend():
    from mpi4py import MPI
    from petsc4py import PETSc
    import numpy as np
    from ._backend import offline_setup, solve_twolevel

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "This Helmholtz backend requires one MPI rank; use workers for patch parallelism"
        )
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError("The Helmholtz solver requires a complex PETSc environment")
    return offline_setup, solve_twolevel


def _runtime_versions():
    import dolfinx
    import numpy as np
    import scipy
    from petsc4py import PETSc

    return {
        "dolfinx": dolfinx.__version__,
        "petsc": list(PETSc.Sys.getVersion()),
        "scalar_type": str(np.dtype(PETSc.ScalarType)),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "partitioner": _partitioner_identity(),
    }


def _partitioner_identity():
    """Identify the extension and the METIS code that it actually loads."""
    import ctypes
    from importlib.metadata import version

    try:
        from pymetis import _internal
    except ModuleNotFoundError as error:
        if error.name == "pymetis":
            return None
        raise

    extension = Path(_internal.__file__).resolve()
    provider = extension
    library = ctypes.CDLL(str(extension))
    try:
        partition = library.METIS_PartGraphKway
    except AttributeError:
        # A statically bundled implementation may hide its METIS symbols.
        pass
    else:
        class DlInfo(ctypes.Structure):
            _fields_ = [
                ("filename", ctypes.c_char_p),
                ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p),
                ("address", ctypes.c_void_p),
            ]

        try:
            dladdr = ctypes.CDLL(None).dladdr
        except AttributeError as error:
            raise RuntimeError("Cannot identify the loaded METIS library") from error
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        found = dladdr(ctypes.cast(partition, ctypes.c_void_p), ctypes.byref(info))
        if not found or not info.filename:
            raise RuntimeError("Cannot identify the loaded METIS library")
        provider = Path(os.fsdecode(info.filename)).resolve()

    return {
        "pymetis": version("pymetis"),
        "extension_sha256": _sha256(extension),
        "metis_provider_sha256": _sha256(provider),
    }


@contextmanager
def _configured_backend(params, threads):
    from threadpoolctl import threadpool_limits

    with _backend_lock:
        backend = _load_backend()
        with bind_parameters(params), threadpool_limits(limits=threads):
            yield backend


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_input_hashes(mesh_path, *, relative=False):
    mesh_path = Path(mesh_path).resolve()
    paths = {mesh_path}
    for item in ET.parse(mesh_path).iter():
        if (
            item.tag.rsplit("}", 1)[-1] == "DataItem"
            and item.get("Format", "").upper() == "HDF"
        ):
            reference = (item.text or "").strip().split(":", 1)[0]
            if not reference:
                raise ValueError("Empty HDF5 reference in the XDMF mesh")
            paths.add((mesh_path.parent / reference).resolve())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        str(path.relative_to(mesh_path.parent) if relative else path): _sha256(path)
        for path in sorted(paths)
    }


@contextmanager
def _case_lease(directory):
    """Exclude concurrent writers; a killed process leaves a lock for inspection."""
    lock = directory / ".helmholtz.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        raise FileExistsError(
            f"Case is locked by another or interrupted process: {lock}"
        ) from None
    try:
        with handle:
            handle.write(f"pid={os.getpid()}\n")
        yield lock
    finally:
        lock.unlink()


class HelmholtzSimulator:
    """Build a spectral preconditioner and solve on a supplied XDMF mesh.

    Construction and describe() do not load a mesh or start a simulation.
    prepare() performs the offline stage; solve() performs the online stage;
    run() performs both and returns numerical diagnostics. simulate() performs
    both and returns a DTCC VolumeMesh with solution fields. Output is confined
    to the chosen case directory. params is the preferred parameter keyword;
    parameters is a backwards-compatible alias.
    """

    def __init__(self, *, mesh_path, output_dir, params=None, parameters=None):
        self.mesh_path = Path(mesh_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.mesh_path.suffix.lower() not in (".xdmf", ".xmf"):
            raise ValueError("mesh_path must name an XDMF mesh")
        if (
            self.output_dir == self.mesh_path.parent
            or self.output_dir in self.mesh_path.parents
        ):
            raise ValueError(
                "Choose a separate output directory, not the input mesh directory or its parents"
            )
        self.parameters = _copy_parameters(params, parameters)
        self.result = None
        self.diagnostics = {}

    @property
    def params(self):
        """Simulation parameters, also available through the parameters alias."""
        return self.parameters

    @params.setter
    def params(self, value):
        self.parameters = _copy_parameters(value)

    @property
    def preconditioner_path(self):
        return self.output_dir / "preconditioner_data.npz"

    @property
    def manifest_path(self):
        return self.output_dir / "prepared.json"

    def describe(self):
        """Return input settings and planned artifacts without running either stage."""
        return {
            "mesh_path": str(self.mesh_path),
            "output_dir": str(self.output_dir),
            "parameters": asdict(self.parameters),
            "artifacts": {
                "preconditioner": str(self.preconditioner_path),
                "solution": str(self.output_dir / "solution.npy"),
                "gmres_history": str(self.output_dir / "gmres_history.txt"),
            },
        }

    def _params(self):
        values = asdict(self.parameters)
        values["patch_workers"] = values.pop("workers")
        values["n_modes_per_patch"] = values["n_modes_per_patch"] or 0
        return Params(
            **values,
            mesh_path=str(self.mesh_path),
            data_dir=str(self.output_dir),
            output_dir=str(self.output_dir),
            log_dir=str(self.output_dir),
            preconditioner_npz_path=str(self.preconditioner_path),
            solution_npy_path=str(self.output_dir / "solution.npy"),
            solution_xdmf_path=str(self.output_dir / "solution.xdmf"),
            gmres_history_path=str(self.output_dir / "gmres_history.txt"),
        )

    def _identity(self):
        identity = {
            "schema": 3,
            "solver": _solver_identity(),
            "parameters": asdict(self.parameters),
            "inputs": _mesh_input_hashes(self.mesh_path),
            "runtime": _runtime_versions(),
        }
        return json.loads(json.dumps(identity, sort_keys=True))

    def _verify_prepared_data(self, prepared_identity):
        identity = dict(prepared_identity)
        checksum = identity.pop("preconditioner_sha256", None)
        if checksum is None:
            raise ValueError(
                "Offline data has no integrity checksum; prepare a fresh case directory"
            )
        if identity != self._identity():
            raise ValueError(
                "Offline data does not match the mesh, parameters, runtime or solver "
                "implementation; prepare a fresh case directory"
            )
        if checksum != _sha256(self.preconditioner_path):
            raise ValueError(
                "Saved preconditioner checksum does not match prepared data; "
                "prepare a fresh case directory"
            )

    def prepare(self):
        """Build offline data in a new or empty directory; never replace an existing case."""
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {self.output_dir}")
        params = self._params()
        with _configured_backend(params, self.parameters.threads_per_worker) as (
            offline,
            _,
        ):
            identity = self._identity()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with _case_lease(self.output_dir) as lock:
                if any(path != lock for path in self.output_dir.iterdir()):
                    raise FileExistsError(
                        f"Output directory is not empty: {self.output_dir}"
                    )
                payload = offline.build_and_save_preconditioner_data()
                if not self.preconditioner_path.is_file():
                    raise RuntimeError("Offline setup did not write the preconditioner")
                if identity != self._identity():
                    raise RuntimeError(
                        "Mesh inputs changed during offline setup; choose a fresh case directory"
                    )
                identity["preconditioner_sha256"] = _sha256(self.preconditioner_path)
                temporary = self.manifest_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(identity, indent=2, sort_keys=True) + "\n"
                )
                temporary.replace(self.manifest_path)
                del payload
        return self.preconditioner_path

    def solve(self):
        """Solve using offline data whose input/configuration manifest still matches."""
        if not self.manifest_path.is_file() or not self.preconditioner_path.is_file():
            raise FileNotFoundError(
                "Call prepare() before solve(); matching offline data is required"
            )
        outputs = (
            "solution.npy",
            "solution.xdmf",
            "solution.h5",
            "gmres_history.txt",
            "diagnostics.json",
        )
        if any((self.output_dir / name).exists() for name in outputs):
            raise FileExistsError(
                "This case already has solve outputs; choose a fresh case directory"
            )
        with _configured_backend(
            self._params(), self.parameters.threads_per_worker
        ) as (_, online):
            with _case_lease(self.output_dir):
                if any((self.output_dir / name).exists() for name in outputs):
                    raise FileExistsError(
                        "This case already has solve outputs; choose a fresh case directory"
                    )
                prepared_identity = json.loads(self.manifest_path.read_text())
                self._verify_prepared_data(prepared_identity)
                raw = online.solve()
                self._verify_prepared_data(prepared_identity)
                result = HelmholtzResult(
                    **raw,
                    output_dir=self.output_dir,
                    mesh_path=self.mesh_path,
                    fe_degree=self.parameters.fe_degree,
                    mesh_name=self.parameters.mesh_name,
                    prepared_identity=prepared_identity,
                )
                diagnostics = {
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "relative_residual": result.relative_residual,
                    "gmres_history": result.gmres_history,
                }
                (self.output_dir / "diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2) + "\n"
                )
        return result

    def run(self):
        """Perform offline setup followed by the online solve."""
        self.prepare()
        return self.solve()

    def simulate(self, *, output_path=None, unit="") -> "VolumeMesh":
        """Run and return a DTCC VolumeMesh with real, imaginary and amplitude fields.

        The numerical result and convergence details remain on result and
        diagnostics. output_path optionally saves the returned mesh using its
        file extension. Vertex fields require P1 elements.
        """
        if self.parameters.fe_degree != 1:
            raise ValueError("simulate() returns vertex fields and requires FE degree 1")
        if output_path is not None:
            from .results import _checked_export_path

            output_path = _checked_export_path(output_path)
        self.result = self.run()
        self.diagnostics = {
            "converged": self.result.converged,
            "iterations": self.result.iterations,
            "relative_residual": self.result.relative_residual,
        }
        return _simulation_mesh(self.result, output_path=output_path, unit=unit)
