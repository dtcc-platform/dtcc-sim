"""DTCC dataset descriptor for the Helmholtz mesh-and-solver workflow."""

from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any, Literal
import math

from pydantic import Field, PrivateAttr, model_validator
from dtcc_core.datasets import DatasetBaseArgs, DatasetDescriptor

from .meshing import CityMeshParameters
from .parameters import HelmholtzParameters
from .._version import __version__


class HelmholtzSimulationArgs(DatasetBaseArgs):
    """Explicit mesh resolution and output case with grouped solver parameters."""

    h: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Target mesh size in meters; user-declared for explicit files",
    )
    output_dir: Path = Field(
        ...,
        description="New or empty case directory for diagnostics and numerical artifacts",
    )
    mesh_path: Path | None = Field(
        None, description="Existing XDMF mesh, instead of DTCC generation"
    )
    mesh_crs: str | None = Field(
        None,
        min_length=1,
        description="Caller-declared CRS of an existing XDMF; omit for unknown or local coordinates. Does not reproject.",
    )
    mesh_cache_dir: Path | None = Field(
        None, description="Required mesh-cache directory when generating from bounds"
    )
    domain_height: float = Field(
        80,
        gt=0,
        allow_inf_nan=False,
        description="Generated city domain height in meters",
    )
    min_building_detail: float = Field(
        0.5, gt=0, allow_inf_nan=False, description="Minimum building detail in meters"
    )
    mesh_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional DTCC city_volume_mesh generation arguments",
    )
    parameters: HelmholtzParameters = Field(
        default_factory=HelmholtzParameters,
        description="Physical, source, boundary, preconditioner, solver, and worker settings",
    )
    field_unit: str = Field(
        "", description="Unit of the modeled solution, if known; amplitude is not SPL"
    )
    format: Literal["dtccpkg", "dtcc", "vtu"] | None = Field(
        None,
        description="dtccpkg preserves fields and provenance; dtcc/vtu are standalone meshes; omit for VolumeMesh",
    )

    _run_metadata: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_workflow(self):
        if len(self.bounds) != 4 or not all(math.isfinite(x) for x in self.bounds):
            raise ValueError(
                "Helmholtz bounds require four finite coordinates; generated meshes use EPSG:3006"
            )
        if self.mesh_crs is not None:
            self.mesh_crs = self.mesh_crs.strip()
            if not self.mesh_crs:
                raise ValueError("mesh_crs must be nonempty when supplied")
        if self.parameters.fe_degree != 1:
            raise ValueError(
                "The Helmholtz dataset returns vertex fields and requires FE degree 1"
            )
        if self.mesh_path is not None:
            if self.mesh_path.suffix.lower() not in (".xdmf", ".xmf"):
                raise ValueError("mesh_path must name an XDMF mesh")
            if self.mesh_options or self.mesh_cache_dir is not None:
                raise ValueError(
                    "mesh_cache_dir and mesh_options apply only to generated meshes"
                )
        else:
            if self.mesh_crs is not None:
                raise ValueError("mesh_crs applies only to supplied XDMF meshes")
            if self.mesh_cache_dir is None:
                raise ValueError("mesh_cache_dir is required when generating a mesh")
            self.city_parameters().resolved_arguments()
        return self

    def city_parameters(self):
        if "strict_live" in self.mesh_options:
            raise ValueError("Set strict_live directly, not in mesh_options")
        return CityMeshParameters(
            bounds=tuple(self.bounds),
            h=self.h,
            domain_height=self.domain_height,
            min_building_detail=self.min_building_detail,
            options=dict(self.mesh_options, strict_live=self.strict_live),
        )


class HelmholtzSimulationDataset(DatasetDescriptor):
    """Complex Helmholtz field on a supplied or generated city volume mesh."""

    name = "helmholtz_simulation"
    title = "Helmholtz Simulation"
    description = (
        "Complex scalar Helmholtz simulation using a spectral overlapping "
        "preconditioner. Select an XDMF mesh or generate a DTCC city mesh "
        "for the requested h. Returns real, imaginary, and amplitude vertex fields."
    )
    ArgsModel = HelmholtzSimulationArgs
    data_category = "simulation"
    result_kind = "mesh"
    python_return_type = "dtcc_core.model.VolumeMesh"
    timeout_hint = 3600
    generated_by = {"package": "dtcc-sim", "version": __version__}
    provider = [
        {"name": "DTCC Sim", "role": "processor"},
        {"name": "FEniCSx/dolfinx", "role": "solver"},
    ]
    source = [
        {
            "name": "dtcc_core.datasets.city_volume_mesh or supplied XDMF",
            "role": "computational_domain",
        },
        {
            "name": "Complex scalar Helmholtz finite-element model",
            "role": "simulation_model",
            "equation": "-div(A grad u) - k^2 V^2 u with configured forcing and boundary terms",
            "unknown": "complex scalar u",
            "unit": "caller specified or unspecified",
        },
    ]
    license = "MIT simulation code; generated meshes and outputs inherit upstream source terms."
    collection_period = (
        "Computed on demand for the requested mesh and solver configuration."
    )
    default_crs = "EPSG:3006"
    data_types = ["volume_mesh", "complex_helmholtz_field", "finite_element_solution"]
    geographic_coverage = "Supplied XDMF coordinates with caller-declared CRS, or city_volume_mesh source coverage in EPSG:3006."
    update_frequency = "on-demand simulation run"
    derived_from = [
        {
            "name": "city_volume_mesh or supplied XDMF",
            "relationship": "checksummed computational mesh, selected for the requested h",
        }
    ]
    processing_steps = [
        "Validate explicit h, case directory, mesh request, and solver parameters",
        "Reuse a matching cached mesh or generate a DTCC city mesh; explicit files must exist",
        "Check boundary tags and warn if the unchanged Gaussian source center is outside the mesh",
        "Build local spectral problems and the two-level preconditioner",
        "Solve with GMRES and save convergence history and true residual",
        "Map P1 values to mesh vertices and attach real, imaginary, and amplitude fields",
    ]
    presentation_headline = "Complex Helmholtz Field"
    presentation_summary = (
        "A simulated complex field and its amplitude over a tetrahedral volume mesh."
    )
    presentation_legend = {
        "title": "Helmholtz fields",
        "entries": [
            {"label": "helmholtz_real", "meaning": "Real part of u"},
            {"label": "helmholtz_imaginary", "meaning": "Imaginary part of u"},
            {"label": "helmholtz_amplitude", "meaning": "Magnitude abs(u), not SPL"},
        ],
    }
    view_hints = {
        "preferred_geometry": "volume_mesh",
        "default_color_attribute": "helmholtz_amplitude",
        "fields": [
            {"name": name, "dim": 1, "unit": ""}
            for name in ("helmholtz_real", "helmholtz_imaginary", "helmholtz_amplitude")
        ],
    }
    presentation_warnings = [
        "The unchanged Gaussian source rule may place its center outside the mesh.",
        "Consult diagnostics.json for the true residual; GMRES convergence uses the transformed system.",
        "Amplitude is not a sound-pressure level, and physical units depend on the model.",
    ]
    presentation_limitations = [
        "Requires complex PETSc/DOLFINx, one MPI rank, and P1 vertex-field output.",
        "Dense eigenproblem fallback retains threshold selection; ARPACK cap behavior is preserved.",
        "Two saved h=5 city comparisons do not validate every mesh or physical configuration.",
        "Explicit mesh h is user declared; generated h is a meshing target, not an edge-length guarantee.",
    ]

    def create_context(self, args):
        context = super().create_context(args)
        context.metadata.crs = (
            ["EPSG:3006"]
            if args.mesh_path is None
            else [args.mesh_crs]
            if args.mesh_crs is not None
            else []
        )
        if args._run_metadata is not None:
            context.provenance.processing_steps.append(deepcopy(args._run_metadata))
            context.presentation.annotations.append(
                {
                    "name": "Helmholtz convergence",
                    "converged": args._run_metadata["converged"],
                    "iterations": args._run_metadata["iterations"],
                    "true_relative_residual": args._run_metadata[
                        "true_relative_residual"
                    ],
                }
            )
        hints = deepcopy(context.presentation.view_hints or {})
        for field in hints.get("fields", []):
            field["unit"] = args.field_unit
        context.presentation.view_hints = hints
        return context

    def build(self, args):
        from .workflow import HelmholtzWorkflow

        import multiprocessing as mp

        args._run_metadata = None

        if args.parameters.workers > 1 and mp.current_process().daemon:
            raise RuntimeError(
                "Parallel Helmholtz requires a non-daemonic worker; use a Celery solo pool or workers=1"
            )
        mesh_args = (
            {"mesh_path": args.mesh_path, "mesh_h": args.h}
            if args.mesh_path is not None
            else {
                "mesh_parameters": args.city_parameters(),
                "mesh_cache_dir": args.mesh_cache_dir,
            }
        )
        workflow = HelmholtzWorkflow(
            output_dir=args.output_dir, parameters=args.parameters, **mesh_args
        )
        result = workflow.run()
        if not result.converged:
            raise RuntimeError(
                f"Helmholtz GMRES did not converge; inspect {result.output_dir}/diagnostics.json"
            )
        if not math.isfinite(result.relative_residual) or result.relative_residual < 0:
            raise RuntimeError("Helmholtz returned an invalid true residual")
        history = [float(value) for value in result.gmres_history]
        if not all(math.isfinite(value) for value in history):
            raise RuntimeError("Helmholtz returned a nonfinite GMRES history")
        args._run_metadata = {
            "name": "Helmholtz completed run",
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "true_relative_residual": float(result.relative_residual),
            "gmres_history": history,
            "case_directory": str(result.output_dir),
            "prepared_identity": deepcopy(result.prepared_identity),
            "mesh_provenance": deepcopy(workflow.mesh_record),
        }
        mesh = result.to_dtcc(unit=args.field_unit)
        # Attach before serialization as well as the base class's public-call hook.
        mesh.dataset_context = self.create_context(args)
        if args.format == "dtccpkg":
            from dtcc_core.datasets.package import export_model_package

            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "helmholtz.dtccpkg"
                export_model_package(mesh, path, canonical=True)
                return path.read_bytes()
        if args.format:
            return self.export_to_bytes(mesh, args.format)
        return mesh
