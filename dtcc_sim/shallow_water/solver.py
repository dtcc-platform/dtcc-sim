"""Shallow-water solver with local assembly and native DTCC output."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from ..coordinates import MeshCoordinateFrame, input_crs, write_coordinate_frame
from ..output import _output_directory
from ..timing import phase, timed_run
from .parameters import ShallowWaterParameters
from .results import ShallowWaterResult
from ._solver.cecilia_swe.mesh_io import validate_mesh

_PACKAGE = Path(__file__).resolve().parent


def _reference():
    metadata = json.loads((_PACKAGE / "reference.json").read_text())
    for name, expected in metadata["sha256"].items():
        if hashlib.sha256((_PACKAGE / "_solver/cecilia_swe" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Reference source differs from its manifest: {name}")
    return metadata


class CeciliaSolver:
    """Run the experimental DG1 reference on a native triangular Mesh."""

    def __init__(self, *, mesh, params=None, output_dir, solver_python=None, mesh_crs=None):
        self.parameters = deepcopy(params if params is not None else ShallowWaterParameters())
        if not isinstance(self.parameters, ShallowWaterParameters):
            raise TypeError("params must be ShallowWaterParameters")
        self.output_dir = _output_directory(output_dir)
        if self.output_dir is None:
            raise ValueError("output_dir is required for transient numerical artifacts")
        self.mesh = mesh
        self.mesh_crs = input_crs(mesh, declared=mesh_crs)
        self.solver_python = solver_python or os.environ.get("DTCC_SWE_PYTHON") or sys.executable
        self.result = None
        self.diagnostics = {}

    @timed_run
    def run(self, *, timeout=None):
        """Return a reloadable transient result; preserve input coordinates on every exit."""
        self.result = None
        self.diagnostics = {}
        executable = shutil.which(os.path.expanduser(str(self.solver_python)))
        if executable is None:
            raise ValueError("solver_python or DTCC_SWE_PYTHON must identify a real-PETSc FEniCSx Python")
        with phase("preparation"):
            reference = _reference()
            data = validate_mesh(self.mesh.vertices, self.mesh.faces, self.mesh.markers)
            if not self.parameters.bed_from_mesh and not np.allclose(data["vertices"][:, 2], data["vertices"][0, 2], rtol=0, atol=1e-9):
                raise ValueError("Set bed_from_mesh=True to use nonhorizontal terrain")
            frame = MeshCoordinateFrame((*map(float, data["vertices"][:, :2].min(axis=0)), 0.), self.mesh_crs)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            runs = self.output_dir / ".shallow_water_runs"; runs.mkdir(exist_ok=True)
            case = Path(tempfile.mkdtemp(prefix="run-", dir=runs))
            self.mesh.save(case / "input.dtcc")
            data["vertices"] = np.array(data["vertices"], dtype=np.float64, copy=True)
            data["vertices"][:, :2] -= np.asarray(frame.origin[:2])
            with (case / "input_mesh.npz").open("xb") as stream:
                np.savez(stream, **data)
            settings = asdict(self.parameters)
            for axis, key in enumerate(("x_center", "y_center")):
                if settings[key] is not None:
                    settings[key] -= frame.origin[axis]
            settings.update(name="shallow-water", mesh_data_path=str(case / "input_mesh.npz"))
            config = case / "parameters.json"; config.write_text(json.dumps(settings, indent=2)+"\n")
            manifest = {"status": "running", "backend": "cecilia-reference", "experimental": True,
                        "reference": reference, "parameters": asdict(self.parameters),
                        "coordinate_frame": frame.as_dict(), "solver_python": str(Path(executable).resolve())}
            record = case / "run.json"; record.write_text(json.dumps(manifest, indent=2)+"\n")
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(_PACKAGE / "_solver"), PYTHONNOUSERSITE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CECILIA_EQUILIBRIUM_PRESSURE="1")
        try:
            with phase("solve"), (case / "solver.log").open("w") as log:
                process = subprocess.run([executable, "-u", "-m", "cecilia_swe", str(config),
                                          "--output-dir", str(case / "solver")],
                                         cwd=case, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=timeout, check=False)
            if process.returncode:
                raise RuntimeError(f"Shallow-water worker exited with {process.returncode}; see {case / 'solver.log'}. Set DTCC_SWE_PYTHON to a real-PETSc FEniCSx Python if required.")
            diagnostics = json.loads((case / "solver/diagnostics.json").read_text())
            worker = json.loads((case / "solver/configuration.json").read_text())
            if (not diagnostics["passed"] or not diagnostics["equilibrium_pressure"]
                    or diagnostics["scalar_type"] != "float64"
                    or worker["backend_sha256"] != reference["sha256"]["backend.py"]):
                raise RuntimeError("Worker output does not match a passing real-valued reference run")
        except Exception as error:
            manifest.update(status="failed", error=str(error))
            record.write_text(json.dumps(manifest, indent=2)+"\n")
            raise
        manifest["status"] = "completed"; record.write_text(json.dumps(manifest, indent=2)+"\n")
        self.result = ShallowWaterResult.load(case)
        self.diagnostics = self.result.diagnostics
        latest = self.output_dir / "shallow_water.json"
        temporary = latest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"run_directory": str(case.relative_to(self.output_dir))}, indent=2)+"\n")
        temporary.replace(latest)
        return self.result

    @timed_run
    def simulate(self):
        """Return a native Core Mesh and replace the standard solution exports."""
        self.result = self.run()
        with phase("postprocessing"):
            mesh = self.result.to_dtcc()
        with phase("export"):
            for suffix in ("vtu", "dtcc"):
                path = self.output_dir / ("solution."+suffix)
                mesh.save(path)
                write_coordinate_frame(path, self.diagnostics["coordinate_frame"])
            (self.output_dir / "diagnostics.json").write_text(json.dumps(self.diagnostics, indent=2)+"\n")
        return mesh
