"""Coordinate mesh preparation and the Helmholtz numerical stages."""

from copy import deepcopy
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from .meshing import CityMeshCache, CityMeshParameters
from .simulator import (
    HelmholtzSimulator, _mesh_input_hashes, _copy_parameters, _simulation_mesh,
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
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.parameters = _copy_parameters(params, parameters)
        if (mesh_path is None) == (mesh_parameters is None):
            raise ValueError("Specify either mesh_path or mesh_parameters")
        if mesh_parameters is not None and not isinstance(
            mesh_parameters, CityMeshParameters
        ):
            raise TypeError("mesh_parameters must be CityMeshParameters")
        if mesh_parameters is not None and mesh_cache_dir is None:
            raise ValueError("mesh_cache_dir is required for generated meshes")
        if mesh_parameters is not None and mesh_h is not None:
            raise ValueError("Generated mesh h comes from mesh_parameters")
        if mesh_h is not None:
            import math

            if not math.isfinite(mesh_h) or mesh_h <= 0:
                raise ValueError("mesh_h must be finite and positive")
        self.mesh_parameters = deepcopy(mesh_parameters)
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
            HelmholtzSimulator(
                mesh_path=self.mesh_path,
                output_dir=self.output_dir,
                parameters=self.parameters,
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
        self.simulator = HelmholtzSimulator(
            mesh_path=path, output_dir=self.output_dir, parameters=self.parameters
        )
        self.mesh_record = record
        return path

    def prepare(self):
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {self.output_dir}")
        self.prepare_mesh()
        path = self.simulator.prepare()
        (self.output_dir / "mesh_provenance.json").write_text(
            json.dumps(self.mesh_record, indent=2, sort_keys=True) + "\n"
        )
        return path

    def solve(self):
        if self.simulator is None:
            self.prepare_mesh()
        self.simulator.parameters = deepcopy(self.parameters)
        return self.simulator.solve()

    def run(self):
        self.prepare()
        return self.solve()

    def simulate(self, *, output_path=None, unit="") -> "VolumeMesh":
        """Resolve the mesh, solve and return a DTCC VolumeMesh with solution fields."""
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
