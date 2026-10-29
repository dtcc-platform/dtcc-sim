"""Coordinate mesh preparation and the Helmholtz numerical stages."""

from ..timing import timed, timed_run
from copy import deepcopy
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ..output import _output_directory
from ..coordinates import input_crs
from .meshing import CityMeshCache, CityMeshParameters
from .solver import (
    HelmholtzSolver, _mesh_input_hashes, _copy_parameters, _simulation_mesh,
)

if TYPE_CHECKING:
    from dtcc_core.model import VolumeMesh


def _check_boundary_tags(path, parameters):
    from mpi4py import MPI
    from dolfinx.io import XDMFFile

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Helmholtz workflows require one MPI rank")
    with XDMFFile(MPI.COMM_WORLD, str(path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=parameters.mesh_name)
        dim = mesh.topology.dim
        mesh.topology.create_connectivity(dim - 1, dim)
        try:
            tags = xdmf.read_meshtags(mesh, name=parameters.facet_tag_name)
        except Exception:
            if parameters.boundary_mode == "mixed":
                raise ValueError(
                    "Mixed boundary conditions require readable boundary markers"
                ) from None
            return {"mode": "all_exterior", "available": None}
        available = set(int(v) for v in tags.values)
        report = {"mode": "tagged", "available": sorted(available)}
        for name, requested in (
            ("impedance", set(parameters.impedance_facet_tags)),
            (
                "dirichlet",
                set(parameters.dirichlet_facet_tags)
                if parameters.boundary_mode == "mixed"
                else set(),
            ),
        ):
            selected = requested.intersection(available)
            report[name] = {
                "requested": sorted(requested),
                "present": sorted(selected),
                "absent": sorted(requested - available),
            }
            if requested and not selected:
                warnings.warn(
                    f"No requested {name} boundary markers are present in {path}; "
                    "that boundary condition selects no facets",
                    UserWarning,
                    stacklevel=2,
                )
        return report


class HelmholtzWorkflow:
    """Resolve a city mesh or use an explicit XDMF, then prepare and solve.

    describe() is planning only. prepare_mesh() can download source data and
    generate a mesh, but does not run the solver. run() performs the full workflow.
    """

    def __init__(
        self,
        *,
        output_dir,
        params=None,
        parameters=None,
        mesh_parameters=None,
        mesh_cache_dir=None,
        mesh_path=None,
        mesh_h=None,
        mesh_crs=None,
    ):
        self.output_dir = _output_directory(output_dir)
        if self.output_dir is None:
            raise ValueError("Helmholtz requires output_dir for numerical artifacts")
        self.parameters = _copy_parameters(params, parameters)
        if (mesh_path is None) == (mesh_parameters is None):
            raise ValueError("Specify either mesh_path or mesh_parameters")
        if mesh_parameters is not None and not isinstance(
            mesh_parameters, CityMeshParameters
        ):
            raise TypeError("mesh_parameters must be CityMeshParameters")
        if mesh_parameters is not None and mesh_cache_dir is None:
            # A sibling cache survives replacement of solver output files.
            mesh_cache_dir = self.output_dir.with_name(self.output_dir.name + "-mesh-cache")
        if mesh_parameters is not None and mesh_h is not None:
            raise ValueError("Generated mesh h comes from mesh_parameters")
        if mesh_h is not None:
            import math

            if not math.isfinite(mesh_h) or mesh_h <= 0:
                raise ValueError("mesh_h must be finite and positive")
        self.mesh_parameters = deepcopy(mesh_parameters)
        self.mesh_crs = input_crs(generated=mesh_parameters is not None, declared=mesh_crs)
        self.mesh_h = mesh_h
        self.mesh_path = (
            Path(mesh_path).expanduser().resolve() if mesh_path is not None else None
        )
        self.cache = (
            CityMeshCache(mesh_cache_dir) if mesh_parameters is not None else None
        )
        self.simulator = None
        self.mesh_record = None
        self.result = None
        self.diagnostics = {}
        if self.mesh_path is not None:
            HelmholtzSolver(
                mesh_path=self.mesh_path,
                output_dir=self.output_dir,
                parameters=self.parameters,
                mesh_crs=self.mesh_crs,
            )
        if self.cache is not None:
            if (
                self.output_dir == self.cache.directory
                or self.output_dir in self.cache.directory.parents
                or self.cache.directory in self.output_dir.parents
            ):
                raise ValueError("Choose separate mesh-cache and case directories")

    @property
    def params(self):
        """Simulation parameters; parameters remains a supported alias."""
        return self.parameters

    @params.setter
    def params(self, value):
        self.parameters = _copy_parameters(value)

    def describe(self):
        from dataclasses import asdict

        mesh = (
            self.cache.describe(self.mesh_parameters)
            if self.cache
            else {
                "mesh_path": str(self.mesh_path),
                "h": self.mesh_h,
                "source": "external",
            }
        )
        return {
            "mesh": mesh,
            "parameters": asdict(self.parameters),
            "output_dir": str(self.output_dir),
        }

    @timed_run
    @timed("mesh_preparation")
    def prepare_mesh(self):
        if self.cache:
            path, record, reused = self.cache.resolve(self.mesh_parameters)
            record = dict(record, source="dtcc_city_volume_mesh", reused=reused)
        else:
            path = self.mesh_path
            record = {
                "source": "external",
                "h": self.mesh_h,
                "h_provenance": "user_supplied"
                if self.mesh_h is not None
                else "unspecified",
                "files": _mesh_input_hashes(path),
            }
        record["boundary_tags"] = _check_boundary_tags(path, self.parameters)
        self.simulator = HelmholtzSolver(
            mesh_path=path, output_dir=self.output_dir, parameters=self.parameters,
            mesh_crs=self.mesh_crs,
        )
        self.mesh_record = record
        return path

    @timed_run
    def prepare(self):
        self.prepare_mesh()
        path = self.simulator.prepare()
        self._write_mesh_provenance()
        return path

    @timed("export")
    def _write_mesh_provenance(self):
        (self.output_dir / "mesh_provenance.json").write_text(
            json.dumps(self.mesh_record, indent=2, sort_keys=True) + "\n"
        )

    @timed_run
    def solve(self):
        if self.simulator is None:
            self.prepare_mesh()
        self.simulator.parameters = deepcopy(self.parameters)
        return self.simulator.solve()

    @timed_run
    def run(self):
        self.prepare_mesh()
        result = self.simulator.run()
        self._write_mesh_provenance()
        return result

    @timed_run
    def simulate(self, *, output_path=None, unit="") -> "VolumeMesh":
        """Resolve the mesh, solve and return a DTCC VolumeMesh with solution fields."""
        if self.parameters.fe_degree != 1:
            raise ValueError("simulate() returns vertex fields and requires FE degree 1")
        if output_path is None:
            output_path = self.output_dir / "solution.vtu"
        if output_path is not None:
            from .results import _checked_export_path

            output_path = _checked_export_path(output_path, mesh_path=self.mesh_path)
        self.result = self.run()
        self.diagnostics = {
            "converged": self.result.converged,
            "iterations": self.result.iterations,
            "relative_residual": self.result.relative_residual,
        }
        frame = getattr(self.result, "coordinate_frame", None)
        if frame is not None:
            self.diagnostics["coordinate_frame"] = frame
        mesh = _simulation_mesh(self.result, output_path=output_path, unit=unit)
        return mesh
