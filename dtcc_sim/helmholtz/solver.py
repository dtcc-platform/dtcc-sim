"""Python entry point for the two-level Helmholtz solver."""

from ..timing import phase as timing_phase, timed, timed_run
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, TYPE_CHECKING

from ..output import _output_directory, _export_city_surface
from ..meshing import _mesh_input_hashes, _sha256
from ..coordinates import input_crs, write_coordinate_frame
from .parameters import HelmholtzParameters
from ._backend.params import Params, bind_parameters

from dtcc_core.common import get_python_logger

_logger = get_python_logger("dtcc-sim")

if TYPE_CHECKING:
    from dtcc_core.model import VolumeMesh

_backend_lock = RLock()


def _solver_identity():
    """Fingerprint the shipped solver sources used to create offline data."""
    root = Path(__file__).parent
    paths = [root / "parameters.py", root / "solver.py"]
    paths.extend(sorted((root / "_backend").rglob("*.py")))
    return {
        "compatibility_revision": 1,
        "sources": {str(path.relative_to(root)): _sha256(path) for path in paths},
        "coordinate_frame_source": _sha256(root.parent / "coordinates.py"),
    }


def _copy_parameters(params=None, parameters=None):
    if params is not None and parameters is not None:
        raise ValueError("Specify params or parameters, not both")
    selected = params if params is not None else parameters
    if selected is not None and not isinstance(selected, HelmholtzParameters):
        raise TypeError("params must be HelmholtzParameters")
    return deepcopy(selected if selected is not None else HelmholtzParameters())


@timed("postprocessing")
def _simulation_mesh(result, *, output_path=None, unit=""):
    if not result.converged:
        raise RuntimeError("Helmholtz solve did not converge; inspect diagnostics")
    if not math.isfinite(result.relative_residual) or result.relative_residual < 0:
        raise RuntimeError("Helmholtz returned an invalid true residual")
    mesh = result.to_dtcc(unit=unit)
    if output_path is not None:
        from .results import _checked_export_path

        with timing_phase("export"):
            path = _checked_export_path(output_path, mesh_path=result.mesh_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            mesh.save(path)
            frame = getattr(mesh, "simulation_diagnostics", {}).get("coordinate_frame")
            if frame is not None:
                write_coordinate_frame(path, frame)
            _export_city_surface(path, native_mesh=mesh, frame=frame)
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
    coordinate_frame: dict[str, Any] | None = None

    run_timings: dict = field(default_factory=dict, compare=False)

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


class HelmholtzSolver:
    """Build a spectral preconditioner and solve on a supplied XDMF mesh.

    Construction and describe() do not load a mesh or start a simulation.
    prepare() performs the offline stage; solve() performs the online stage;
    run() performs both and returns numerical diagnostics. simulate() performs
    both and returns a DTCC VolumeMesh with solution fields. Output is confined
    to the chosen case directory. params is the preferred parameter keyword;
    parameters is a backwards-compatible alias.
    """

    def __init__(self, *, output_dir, mesh_path=None, mesh=None, params=None, parameters=None, mesh_crs=None):
        if (mesh_path is None) == (mesh is None):
            raise ValueError("Specify either mesh_path or mesh")
        self.output_dir = _output_directory(output_dir)
        if self.output_dir is None:
            raise ValueError("Helmholtz requires output_dir for numerical artifacts")
        self.mesh = mesh
        self.mesh_crs = input_crs(mesh, declared=mesh_crs)
        if mesh is not None:
            from dtcc_core.model import VolumeMesh

            if not isinstance(mesh, VolumeMesh):
                raise TypeError("mesh must be a DTCC VolumeMesh")
        self.mesh_path = (
            Path(mesh_path).expanduser().resolve()
            if mesh_path is not None else self.output_dir / "input.xdmf"
        )
        if self.mesh_path.suffix.lower() not in (".xdmf", ".xmf"):
            raise ValueError("mesh_path must name an XDMF mesh")
        if self.mesh is None and (
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
            "mesh_crs": self.mesh_crs,
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
            mesh_crs=self.mesh_crs,
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
            "schema": 4,
            "mesh_crs": self.mesh_crs,
            "solver": _solver_identity(),
            "parameters": asdict(self.parameters),
            "inputs": _mesh_input_hashes(self.mesh_path),
            "runtime": _runtime_versions(),
        }
        return json.loads(json.dumps(identity, sort_keys=True))

    def _verify_prepared_data(self, prepared_identity):
        identity = dict(prepared_identity)
        checksum = identity.pop("preconditioner_sha256", None)
        frame = identity.pop("coordinate_frame", None)
        if checksum is None:
            raise ValueError(
                "Offline data has no integrity checksum; call prepare() again"
            )
        if identity != self._identity():
            raise ValueError(
                "Offline data does not match the mesh, parameters, runtime or solver "
                "implementation; call prepare() again"
            )
        if checksum != _sha256(self.preconditioner_path):
            raise ValueError(
                "Saved preconditioner checksum does not match prepared data; "
                "call prepare() again"
            )
        if frame is not None:
            import numpy as np

            with np.load(self.preconditioner_path, allow_pickle=False) as payload:
                saved_frame = json.loads(str(payload["coordinate_frame_json"][0]))
            if frame != saved_frame:
                raise ValueError("Offline coordinate frame changed; call prepare() again")

    @timed_run
    def prepare(self):
        """Prepare offline data, replacing previous solver outputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with _case_lease(self.output_dir):
            return self._prepare()

    @timed("offline_preparation")
    def _prepare(self):
        """Build offline data, replacing this solver's previous case artifacts."""
        _logger.info("Preparing two-level preconditioner for Helmholtz equation")
        params = self._params()
        with _configured_backend(params, self.parameters.threads_per_worker) as (offline, _):
            if self.mesh is not None:
                self.mesh.save(self.mesh_path)
            identity = self._identity()
            # Only these named numerical artifacts belong to the solver.
            for name in (
                "prepared.json", "prepared.json.tmp", "preconditioner_data.npz",
                "solution.npy", "solution.xdmf", "solution.h5", "gmres_history.txt",
                "diagnostics.json", "mesh_provenance.json", "solution.vtu", "city.vtu",
                "solution.coordinates.json", "city.coordinates.json",
            ):
                (self.output_dir / name).unlink(missing_ok=True)
            payload = offline.build_and_save_preconditioner_data()
            if not self.preconditioner_path.is_file():
                raise RuntimeError("Offline setup did not write the preconditioner")
            if identity != self._identity():
                raise RuntimeError("Mesh inputs changed during offline setup")
            identity["preconditioner_sha256"] = _sha256(self.preconditioner_path)
            if "coordinate_frame_json" in payload:
                identity["coordinate_frame"] = json.loads(str(payload["coordinate_frame_json"][0]))
            temporary = self.manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            temporary.replace(self.manifest_path)
            del payload
        return self.preconditioner_path

    @timed_run
    def solve(self):
        """Solve with matching offline data, replacing previous solver outputs."""
        if not self.output_dir.is_dir():
            raise FileNotFoundError("Call prepare() before solve(); matching offline data is required")
        with _case_lease(self.output_dir):
            return self._solve()

    @timed("online_solver")
    def _solve(self):
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
            "solution.coordinates.json",
        )
        with _configured_backend(
            self._params(), self.parameters.threads_per_worker
        ) as (_, online):
            prepared_identity = json.loads(self.manifest_path.read_text())
            self._verify_prepared_data(prepared_identity)
            for name in outputs:
                (self.output_dir / name).unlink(missing_ok=True)
            _logger.info(
                f"Solving Helmholtz equation at {self.parameters.frequency_hz:g} Hz "
                f"(GMRES rtol={self.parameters.outer_rtol:.1e})"
            )
            raw = online.solve()
            if raw.get("coordinate_frame") != prepared_identity.get("coordinate_frame"):
                raise RuntimeError("Online coordinate frame does not match offline preparation")
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
            if result.coordinate_frame is not None:
                diagnostics["coordinate_frame"] = result.coordinate_frame
            with timing_phase("export"):
                (self.output_dir / "diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2) + "\n"
                )
        status = "converged" if result.converged else "FAILED to converge"
        log_result = _logger.info if result.converged else _logger.error
        log_result(
            f"Helmholtz solve {status} after {result.iterations} iterations; "
            f"true relative residual={result.relative_residual:.3e}"
        )
        return result

    @timed_run
    def run(self):
        """Perform offline setup followed by the online solve."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with _case_lease(self.output_dir):
            self._prepare()
            return self._solve()

    @timed_run
    def simulate(self, *, output_path=None, unit="") -> "VolumeMesh":
        """Run and return a DTCC VolumeMesh with real, imaginary and amplitude fields.

        The numerical result and convergence details remain on result and
        diagnostics. The returned mesh is saved as solution.vtu in output_dir;
        output_path selects another export destination. Vertex fields require P1 elements.
        """
        if self.parameters.fe_degree != 1:
            raise ValueError("simulate() returns vertex fields and requires FE degree 1")
        if output_path is None:
            output_path = self.output_dir / "solution.vtu"
        if output_path is not None:
            from .results import _checked_export_path

            output_path = _checked_export_path(output_path, mesh_path=self.mesh_path if self.mesh is None else None)
        self.result = self.run()
        self.diagnostics = {
            "converged": self.result.converged,
            "iterations": self.result.iterations,
            "relative_residual": self.result.relative_residual,
        }
        frame = getattr(self.result, "coordinate_frame", None)
        if frame is not None:
            self.diagnostics["coordinate_frame"] = frame
        return _simulation_mesh(self.result, output_path=output_path, unit=unit)
