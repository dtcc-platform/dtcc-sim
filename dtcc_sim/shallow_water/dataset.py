"""Urban flood dataset backed by a triangular shallow-water solver."""

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import Field, PrivateAttr, field_validator, model_validator
from dtcc_core.datasets import DatasetBaseArgs, DatasetDescriptor

from .._version import __version__
from ..output import _output_directory
from ..timing import timed_run
from .parameters import ShallowWaterParameters


class UrbanFloodSimulationArgs(DatasetBaseArgs):
    """Water settings and a native input mesh or a flat city-mesh request."""

    h: float = Field(10., gt=0, allow_inf_nan=False, description="Generated mesh target size in meters")
    min_building_detail: float = Field(.5, gt=0, allow_inf_nan=False)
    output_dir: Path
    mesh_path: Path | None = Field(None, description="Native .dtcc triangular mesh instead of live city generation")
    mesh_crs: str | None = None
    solver_python: str | None = None
    parameters: ShallowWaterParameters = Field(default_factory=ShallowWaterParameters)
    format: Literal["dtccpkg", "dtcc", "vtu"] | None = None
    _run_metadata: dict | None = PrivateAttr(default=None)

    @field_validator("output_dir", mode="before")
    @classmethod
    def output_directory(cls, value):
        directory = _output_directory(value)
        if directory is None:
            raise ValueError("output_dir is required")
        return directory

    @model_validator(mode="after")
    def scenario(self):
        if len(self.bounds) != 4 or not all(math.isfinite(v) for v in self.bounds):
            raise ValueError("Bounds must be four finite EPSG:3006 coordinates")
        if self.bounds[2] <= self.bounds[0] or self.bounds[3] <= self.bounds[1]:
            raise ValueError("Bounds must have positive horizontal extent")
        if self.mesh_path is not None and self.mesh_path.suffix.lower() != ".dtcc":
            raise ValueError("mesh_path must be a native .dtcc mesh")
        if self.mesh_path is None and self.mesh_crs not in (None, "EPSG:3006"):
            raise ValueError("Generated city meshes use EPSG:3006")
        return self


class UrbanFloodSimulationDataset(DatasetDescriptor):
    """Experimental water-depth and velocity fields on a 2D city mesh."""

    name = "urban_flood_simulation"
    title = "Experimental Urban Flood Simulation"
    description = "Transient shallow-water fields on a supplied native mesh or generated flat city mesh."
    ArgsModel = UrbanFloodSimulationArgs
    data_category = "simulation"
    result_kind = "mesh"
    python_return_type = "dtcc_core.model.Mesh"
    timeout_hint = 3600
    generated_by = {"package": "dtcc-sim", "version": __version__}
    default_crs = "EPSG:3006"
    source = [{"name": "city_flat_mesh or supplied native DTCC mesh", "role": "computational_domain"},
              {"name": "Experimental shallow-water reference", "role": "simulation_model"}]
    data_types = ["triangular_mesh", "water_depth", "water_velocity"]
    presentation_headline = "Experimental Water Depth"
    presentation_summary = "Depth and depth-weighted velocity from an idealized transient shallow-water scenario."
    presentation_warnings = ["Experimental terrain/wetting-drying extensions; not a validated flood forecast.",
                             "Reflective outer and building boundaries; building interiors remain separate simulated regions."]
    presentation_limitations = ["Generated city geometry has a flat bed, not measured terrain.",
                               "No observed flood calibration; mesh-sensitive shoreline behavior remains."]
    view_hints = {"preferred_geometry": "mesh", "default_color_attribute": "water_depth",
                  "fields": [{"name": "water_depth", "unit": "m", "dim": 1},
                             {"name": "water_velocity", "unit": "m/s", "dim": 3}]}

    def create_context(self, args):
        context = super().create_context(args)
        if args._run_metadata is not None:
            frame = args._run_metadata["coordinate_frame"]
            context.metadata.crs = [frame["input_crs"]] if frame["input_crs"] else []
            context.provenance.processing_steps.append(deepcopy(args._run_metadata))
        return context

    @timed_run
    def build(self, args):
        from dtcc_core import datasets
        from dtcc_core.io import load_mesh
        from dtcc_core.model import Bounds
        from .solver import CeciliaSolver

        args._run_metadata = None
        if args.mesh_path is None:
            mesh = datasets.city_flat_mesh(bounds=Bounds(*args.bounds), max_mesh_size=args.h,
                                           min_building_detail=args.min_building_detail,
                                           center_on_origin=False, strict_live=args.strict_live)
            crs = "EPSG:3006"
            source = {"dataset": "city_flat_mesh", "bounds": list(args.bounds),
                      "max_mesh_size": args.h, "min_building_detail": args.min_building_detail,
                      "bed": "flat", "strict_live": args.strict_live}
        else:
            mesh = load_mesh(args.mesh_path)
            crs = args.mesh_crs
            source = {"mesh_path": str(args.mesh_path.resolve())}
        solver = CeciliaSolver(mesh=mesh, params=args.parameters, output_dir=args.output_dir,
                                    solver_python=args.solver_python, mesh_crs=crs)
        result = solver.simulate()
        args._run_metadata = {"name": "Experimental shallow-water completed run", **solver.diagnostics,
                              "parameters": asdict(args.parameters), "mesh_source": source}
        result.dataset_context = self.create_context(args)
        result.save(args.output_dir / "solution.dtcc")
        if args.format == "dtccpkg":
            from dtcc_core.datasets.package import export_model_package
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "flood.dtccpkg"
                export_model_package(result, path, canonical=True)
                return path.read_bytes()
        if args.format:
            return self.export_to_bytes(result, args.format)
        return result
