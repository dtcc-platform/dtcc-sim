"""
DTCC Sim Datasets

This module provides simulation results as datasets that integrate with
the dtcc-core dataset system. When dtcc_sim is imported, these datasets
automatically register themselves with dtcc_core.datasets.
"""

from typing import Optional, Literal
from pydantic import Field

from dtcc_core.datasets import DatasetDescriptor, DatasetBaseArgs

from .traffic import TrafficAssignmentSimulator, TrafficAssignmentParameters


BoundaryConditionType = Literal["dirichlet", "neumann", "robin"]
WindWeatherAggregation = Literal["nearest", "mean", "median"]
WindEquationSet = Literal["navier_stokes", "stokes"]
WindSimulationMode = Literal["steady", "statistical_steady"]
WindWallModel = Literal["noslip", "friction"]
WindInletProfile = Literal["uniform", "power_law", "log_law"]
WindSideTopBoundary = Literal["open", "slip"]


class UrbanHeatSimulationArgs(DatasetBaseArgs):
    """Arguments for urban heat simulation dataset."""

    # PDE coefficients
    kappa: float = Field(1.0, description="Thermal diffusivity coefficient", gt=0)
    sigma: float = Field(0.0, description="Relaxation/reaction coefficient", ge=0)
    T_ambient: float = Field(0.0, description="Ambient temperature")
    degree: int = Field(1, description="Polynomial degree for FE space", ge=1, le=3)

    # Boundary conditions - simplified
    wall_bc_type: BoundaryConditionType = Field(
        "robin",
        description="Wall boundary condition type.",
    )
    wall_value: float = Field(1.0, description="Wall temperature")
    wall_h: float = Field(
        5.0,
        description="Wall Robin heat-transfer coefficient.",
        ge=0,
    )
    wall_flux: float = Field(0.0, description="Wall Neumann heat flux")
    roof_bc_type: BoundaryConditionType = Field(
        "robin",
        description="Roof boundary condition type.",
    )
    roof_value: float = Field(1.0, description="Roof temperature")
    roof_h: float = Field(
        5.0,
        description="Roof Robin heat-transfer coefficient.",
        ge=0,
    )
    roof_flux: float = Field(0.0, description="Roof Neumann heat flux")
    ground_bc_type: BoundaryConditionType = Field(
        "robin",
        description="Ground boundary condition type.",
    )
    ground_value: float = Field(0.0, description="Ground temperature")
    ground_h: float = Field(
        2.0,
        description="Ground Robin heat-transfer coefficient.",
        ge=0,
    )
    ground_flux: float = Field(0.0, description="Ground Neumann heat flux")
    open_bc_type: BoundaryConditionType = Field(
        "dirichlet",
        description="Open/far-field boundary condition type.",
    )
    open_value: float = Field(0.0, description="Open boundary temperature")
    open_h: float = Field(
        1.0,
        description="Open-boundary Robin heat-transfer coefficient.",
        ge=0,
    )
    open_flux: float = Field(0.0, description="Open-boundary Neumann heat flux")

    # Mesh parameters
    mesh_max_mesh_size: float = Field(25.0, description="Max mesh size in meters")
    mesh_domain_height: float = Field(80.0, description="Domain height in meters")

    format: Optional[Literal["xdmf"]] = Field(None, description="Output format")


class UrbanHeatSimulationDataset(DatasetDescriptor):
    """Urban heat simulation as a dataset.

    This dataset provides steady-state thermal analysis of urban environments by solving
    the heat diffusion equation with relaxation in a 3D city volume. It integrates mesh
    generation from geographic bounds with finite element simulation using FEniCSx.

    Physical Model:
        -∇·(κ∇T) + σ(T - T_ambient) = 0

    where T is the air temperature field, κ is thermal diffusivity, σ is the relaxation
    coefficient modeling atmospheric mixing, and T_ambient is the background temperature.

    Boundary conditions model convective heat transfer (Robin BC) or prescribed temperatures
    (Dirichlet BC) on building surfaces, ground, and domain boundaries.

    Applications:
        - Urban heat island effect quantification
        - Building energy analysis and thermal comfort assessment
        - Climate adaptation and urban planning studies
        - Evaluation of heat mitigation strategies

    Workflow:
        1. Takes a bounding box as input (geographic coordinates)
        2. Auto-generates a 3D tetrahedral volume mesh including buildings and terrain
        3. Solves the steady-state heat equation using finite elements
        4. Returns a DTCC VolumeMesh with the temperature field attached

    Example:
        >>> import dtcc_core.datasets as datasets
        >>> import dtcc_sim.datasets  # Register simulation datasets
        >>>
        >>> # Basic usage - default parameters
        >>> T = datasets.urban_heat_simulation(
        ...     bounds=[minx, miny, maxx, maxy]
        ... )
        >>>
        >>> # Heat wave scenario
        >>> T = datasets.urban_heat_simulation(
        ...     bounds=[minx, miny, maxx, maxy],
        ...     T_ambient=20.0,      # 20°C ambient air
        ...     wall_value=35.0,     # Hot building surfaces
        ...     ground_value=28.0,   # Solar-heated ground
        ...     kappa=2.0,           # Enhanced mixing
        ...     wall_h=8.0           # Moderate convection
        ... )
    """

    name = "urban_heat_simulation"
    title = "Urban Heat Simulation"
    description = (
        "Steady-state urban heat simulation using FEniCSx. Solves "
        "-div(kappa grad T) + sigma (T - T_ambient) = 0 in a 3D urban "
        "volume, with configurable Dirichlet, Neumann, or Robin boundary "
        "conditions for walls, roofs, ground, and open boundaries. Returns a "
        "DTCC VolumeMesh with a temperature Field in degC."
    )
    ArgsModel = UrbanHeatSimulationArgs
    data_category = "simulation"
    result_kind = "mesh"
    python_return_type = "dtcc_core.model.VolumeMesh"
    timeout_hint = 600
    multi_file_formats = ("xdmf",)
    provider = [
        {"name": "DTCC Sim", "role": "processor"},
        {"name": "FEniCSx/dolfinx", "role": "solver"},
    ]
    source = [
        {
            "name": "dtcc_core.datasets.city_volume_mesh",
            "role": "upstream_dataset",
            "source_terms_status": "requires_review",
        },
        {
            "name": "Urban heat diffusion/reaction finite-element model",
            "role": "simulation_model",
            "equation": "-div(kappa grad T) + sigma (T - T_ambient) = 0",
            "unknown": "temperature field T",
            "unit": "degC",
        },
    ]
    license = (
        "MIT-licensed simulation code; generated outputs inherit upstream "
        "city-volume-mesh source/license review requirements."
    )
    collection_period = (
        "Computed on demand for the requested parameters; this is not an "
        "observational time series."
    )
    default_crs = "EPSG:3006"
    data_types = ["volume_mesh", "temperature_field", "finite_element_solution"]
    geographic_coverage = (
        "Areas where the upstream city_volume_mesh dataset can construct a "
        "valid volume mesh from the requested bounds."
    )
    update_frequency = "on-demand simulation run"
    derived_from = [
        {
            "name": "city_volume_mesh",
            "relationship": "computational domain generated from requested bounds",
        }
    ]
    processing_steps = [
        (
            "Validate heat-equation coefficients, boundary values, mesh "
            "settings, and output format"
        ),
        (
            "Generate a DTCC city volume mesh from requested bounds when no "
            "mesh is provided"
        ),
        "Load the volume mesh and boundary markers into FEniCSx",
        "Collapse boundary markers into wall, roof, ground, and open categories",
        "Assemble the steady diffusion/reaction finite-element problem",
        "Apply Dirichlet, Neumann, or Robin boundary conditions by surface category",
        "Solve the linear system with PETSc through the dtcc_sim FEniCS wrapper",
        "Attach a temperature Field to the DTCC VolumeMesh or serialize XDMF output",
    ]
    presentation_headline = "Steady Urban Heat Field"
    presentation_summary = (
        "A simplified steady-state temperature field over a generated 3D urban "
        "volume mesh."
    )
    presentation_narrative = [
        {
            "heading": "What is solved",
            "body": (
                "The model solves a scalar diffusion/reaction equation for "
                "temperature. Kappa controls effective heat diffusion and "
                "sigma relaxes the field toward T_ambient."
            ),
        },
        {
            "heading": "Boundary conditions",
            "body": (
                "Walls, roofs, ground, and open boundaries each support "
                "Dirichlet fixed values, Neumann fluxes, or Robin convective "
                "exchange. Defaults use Robin on walls, roofs, and ground, "
                "and Dirichlet on open boundaries."
            ),
        },
        {
            "heading": "How to interpret it",
            "body": (
                "The result is a planning-scale scenario field. It is useful "
                "for comparing parameter choices and checking solver plumbing, "
                "not for predicting measured urban air temperature."
            ),
        },
    ]
    key_points = [
        "unknown: scalar temperature field T",
        "units: degC for the attached temperature Field",
        "default element: first-order Lagrange finite elements",
        "default mesh: city_volume_mesh with 25 m max mesh size and 80 m domain height",
        "solver diagnostics record DOF count, boundary counts, and solution range",
    ]
    presentation_legend = {
        "title": "Temperature field",
        "entries": [
            {"label": "temperature", "meaning": "solved scalar field in degC"},
            {"label": "walls/roofs", "meaning": "building surface categories"},
            {"label": "ground", "meaning": "terrain/ground boundary category"},
            {"label": "open", "meaning": "far-field domain boundary category"},
        ],
    }
    view_hints = {
        "preferred_geometry": "volume_mesh",
        "field": "temperature",
        "field_unit": "degC",
        "table_role": "simulation_field",
        "default_color_attribute": "temperature",
    }
    presentation_warnings = [
        (
            "This is a simplified steady-state model; it does not include "
            "transient weather, radiation, humidity, vegetation physiology, or "
            "validated urban energy balance."
        ),
        (
            "City-scale runs depend on generated mesh quality and should be "
            "treated as manual/expensive validation until benchmarked."
        ),
    ]
    presentation_limitations = [
        "Planning/scenario tool, not a measured-temperature prediction product.",
        "Boundary values are scenario inputs and must be calibrated externally.",
        "Solver residual/KSP details are not exposed by the current wrapper.",
        "Outputs inherit upstream city-volume-mesh source and geometry limitations.",
    ]

    def build(self, args):
        from .urban_heat import UrbanHeatSimulator, UrbanHeatParameters

        bounds = self.parse_bounds(args.bounds)
        params = UrbanHeatParameters(
            kappa=args.kappa,
            sigma=args.sigma,
            T_ambient=args.T_ambient,
            degree=args.degree,
            wall_bc_type=args.wall_bc_type,
            wall_value=args.wall_value,
            wall_h=args.wall_h,
            wall_flux=args.wall_flux,
            roof_bc_type=args.roof_bc_type,
            roof_value=args.roof_value,
            roof_h=args.roof_h,
            roof_flux=args.roof_flux,
            ground_bc_type=args.ground_bc_type,
            ground_value=args.ground_value,
            ground_h=args.ground_h,
            ground_flux=args.ground_flux,
            open_bc_type=args.open_bc_type,
            open_value=args.open_value,
            open_h=args.open_h,
            open_flux=args.open_flux,
            mesh_max_mesh_size=args.mesh_max_mesh_size,
            mesh_domain_height=args.mesh_domain_height,
        )
        sim = UrbanHeatSimulator(bounds=bounds, params=params)
        result = sim.simulate()
        if args.format:
            if sim.solution is None:
                raise RuntimeError(
                    "urban_heat_simulation did not produce a FEniCS solution "
                    f"for format={args.format!r} serialization."
                )
            return self.export_to_bytes(sim.solution, args.format)
        return result


class AirQualityFieldArgs(DatasetBaseArgs):
    """Arguments for air quality field reconstruction dataset."""

    # Phenomenon to reconstruct
    phenomenon: str = Field(
        "NO2",
        description="Phenomenon name (e.g., NO2, PM10)",
        min_length=1,
    )

    # Reconstruction weights
    lambda_smooth: float = Field(
        1.0, description="Smoothness regularization weight (gradient penalty)", gt=0
    )
    alpha: float = Field(
        1e-3, description="Background anchoring weight (mass penalty)", ge=0
    )
    data_weight: float = Field(
        100.0, description="Data fidelity weight (point observations)", gt=0
    )
    background_value: Optional[float] = Field(
        None, description="Background field value (None = use mean of observations)"
    )

    # Function space
    degree: int = Field(
        1,
        description="Polynomial degree (currently only 1 supported)",
        ge=1,
        le=1,
    )

    # Mesh parameters
    mesh_max_mesh_size: float = Field(
        25.0, description="Max mesh size in meters", gt=0
    )
    mesh_domain_height: float = Field(
        80.0, description="Domain height in meters", gt=0
    )
    mesh_raster_cell_size: float = Field(
        2.0, description="Terrain raster cell size", gt=0
    )
    mesh_raster_radius: float = Field(
        3.0, description="Terrain raster interpolation radius", gt=0
    )

    # Air quality dataset parameters
    airquality_crs: str = Field("EPSG:3006", description="CRS for air quality data")
    airquality_timeout_s: float = Field(
        10.0, description="API timeout in seconds", gt=0
    )
    airquality_max_stations: int = Field(
        250, description="Max stations to fetch", gt=0
    )
    airquality_drop_missing: bool = Field(
        True, description="Drop stations with no data"
    )
    airquality_base_url: str = Field(
        "https://datavardluft.smhi.se/52North/api", description="SMHI API base URL"
    )

    # Robustness options
    z_offset: float = Field(
        0.0, description="Vertical offset to add to sensor z-coordinates"
    )

    format: Optional[Literal["xdmf"]] = Field(None, description="Output format")


class AirQualityFieldDataset(DatasetDescriptor):
    """Smooth reconstruction of air quality fields from sparse sensor measurements.

    This dataset provides continuous, smooth 3D air quality fields by combining sparse
    sensor measurements (from SMHI air quality stations) with PDE-based smoothing. It
    uses Tikhonov regularization to interpolate between sensors while maintaining
    smoothness and physical plausibility.

    Mathematical Model:
        Minimizes: E(u) = ½w Σᵢ(u(xᵢ) - yᵢ)² + ½λ∫|∇u|²dx + ½α∫(u - u_bg)²dx

    where:
        - u is the reconstructed concentration field
        - xᵢ, yᵢ are sensor locations and measurements
        - w controls data fidelity (how closely field matches observations)
        - λ penalizes gradients (enforces smoothness)
        - α anchors field to background value u_bg (prevents unbounded growth)

    This produces a smooth field that:
        - Passes through or near sensor measurements (controlled by data_weight)
        - Varies smoothly between measurements (controlled by lambda_smooth)
        - Remains bounded and realistic (controlled by alpha and background_value)

    Applications:
        - Air quality mapping and visualization
        - Exposure assessment for health studies
        - Environmental monitoring and compliance
        - Urban planning and pollution mitigation
        - Validation of atmospheric dispersion models

    Workflow:
        1. Takes a bounding box as input (geographic coordinates)
        2. Fetches air quality sensor data from SMHI API (dtcc_core.datasets.airquality)
        3. Auto-generates a 3D tetrahedral volume mesh including buildings and terrain
        4. Solves the regularized reconstruction problem using finite elements (FEniCSx)
        5. Returns a dtcc-core VolumeMesh with the reconstructed field attached as a Field

    Parameters Guide:
        - lambda_smooth: Higher values → smoother field, more deviation from sensors
        - data_weight: Higher values → field closer to sensor values, less smooth
        - alpha: Small but nonzero prevents unbounded oscillations
        - background_value: Expected concentration in areas without sensors (default: mean)

    Example:
        >>> import dtcc_core.datasets as datasets
        >>> import dtcc_sim.datasets  # Register simulation datasets
        >>>
        >>> # Basic usage - reconstruct NO2 concentrations
        >>> volume_mesh = datasets.air_quality_field(
        ...     bounds=[665000, 6575000, 685000, 6595000],  # 20x20 km in Göteborg
        ...     phenomenon="NO2"
        ... )
        >>>
        >>> # Access the reconstructed field
        >>> field = volume_mesh.fields[0]
        >>> print(f"Field: {field.name}, unit: {field.unit}")
        >>> print(f"Values range: {field.values.min():.2f} - {field.values.max():.2f}")
        >>>
        >>> # Adjust reconstruction parameters
        >>> volume_mesh = datasets.air_quality_field(
        ...     bounds=[665000, 6575000, 685000, 6595000],
        ...     phenomenon="PM10",
        ...     lambda_smooth=0.5,    # Less smoothing
        ...     data_weight=200.0,    # Closer fit to sensors
        ...     background_value=10.0 # Expected background PM10
        ... )
    """

    name = "air_quality_field"
    title = "Air-Quality Reconstruction Field"
    description = (
        "Smooth concentration-field reconstruction from sparse air-quality "
        "station observations. Fetches measured station values from "
        "dtcc_core.datasets.air_quality, builds a city volume mesh, and solves "
        "a Tikhonov-regularized finite-element interpolation problem. Returns "
        "a DTCC VolumeMesh with a scalar reconstructed concentration Field; "
        "the field is derived from, but is not itself, a measured station record."
    )
    ArgsModel = AirQualityFieldArgs
    data_category = "simulation"
    result_kind = "mesh"
    python_return_type = "dtcc_core.model.VolumeMesh"
    timeout_hint = 300
    multi_file_formats = ("xdmf",)
    provider = [
        {"name": "DTCC Sim", "role": "processor"},
        {"name": "FEniCSx/dolfinx", "role": "solver"},
        {
            "name": "SMHI datavardluft via dtcc_core.datasets.air_quality",
            "role": "source_provider",
            "source_terms_status": "requires_review",
        },
    ]
    source = [
        {
            "name": "dtcc_core.datasets.air_quality",
            "role": "upstream_observation_dataset",
            "source_terms_status": "requires_review",
        },
        {
            "name": "dtcc_core.datasets.city_volume_mesh",
            "role": "upstream_mesh_dataset",
            "source_terms_status": "requires_review",
        },
        {
            "name": "Tikhonov smooth reconstruction finite-element model",
            "role": "simulation_model",
            "objective": (
                "0.5*w*sum_i (u(x_i)-y_i)^2 + "
                "0.5*lambda*integral |grad u|^2 dx + "
                "0.5*alpha*integral (u-u_bg)^2 dx"
            ),
            "unknown": "scalar reconstructed concentration field u",
            "unit": "inherited from upstream station observations",
        },
    ]
    license = (
        "MIT-licensed simulation code; generated outputs inherit SMHI "
        "air-quality and city-volume-mesh source/license review requirements."
    )
    collection_period = (
        "Computed on demand from the latest station values returned by the "
        "upstream air_quality dataset for the requested phenomenon."
    )
    default_crs = "EPSG:3006"
    data_types = [
        "volume_mesh",
        "station_observations",
        "reconstructed_concentration_field",
        "finite_element_solution",
    ]
    geographic_coverage = (
        "Areas where city_volume_mesh can construct a valid mesh and the "
        "upstream air_quality dataset returns stations for the requested "
        "phenomenon."
    )
    update_frequency = "on-demand reconstruction from upstream station snapshot"
    derived_from = [
        {
            "name": "air_quality",
            "relationship": "station coordinates, values, timestamps, and units",
        },
        {
            "name": "city_volume_mesh",
            "relationship": "finite-element reconstruction domain",
        },
    ]
    processing_steps = [
        (
            "Validate phenomenon, regularization weights, mesh settings, "
            "air-quality provider settings, and output format"
        ),
        "Fetch measured station observations from dtcc_core.datasets.air_quality",
        "Validate that station coordinates, values, and a non-empty unit are available",
        "Generate a DTCC city volume mesh from requested bounds when needed",
        "Load the mesh into FEniCSx and create a first-order Lagrange space",
        "Use the mean station value as background when background_value is not set",
        "Locate station points in the mesh and skip only points outside the domain",
        "Assemble data-fidelity, smoothness, and background-anchor terms",
        "Solve the linear Tikhonov reconstruction system with PETSc",
        "Attach the reconstructed scalar Field with the upstream observation unit",
    ]
    presentation_headline = "Derived Air-Quality Field"
    presentation_summary = (
        "A smoothed finite-element concentration field derived from sparse "
        "station observations."
    )
    presentation_narrative = [
        {
            "heading": "Measured versus reconstructed",
            "body": (
                "The upstream air_quality dataset contains station measurements. "
                "This dataset creates a continuous field from those points using "
                "regularization; values between stations are model-derived."
            ),
        },
        {
            "heading": "Regularization controls",
            "body": (
                "data_weight pulls the field toward observations, "
                "lambda_smooth penalizes gradients, and alpha anchors the "
                "field toward background_value or the station mean."
            ),
        },
        {
            "heading": "How to interpret it",
            "body": (
                "Use this as a visualization and scenario layer. It is not a "
                "regulatory dispersion model and should be checked against "
                "station density, timestamps, and provider coverage."
            ),
        },
    ]
    key_points = [
        "input: measured station values from air_quality",
        "output: derived scalar concentration Field on a VolumeMesh",
        "unit: copied from upstream station attributes",
        "default background: mean of valid station values",
        "city-scale validation remains provider- and mesh-sensitive",
    ]
    presentation_legend = {
        "title": "Reconstructed concentration",
        "entries": [
            {
                "label": "field value",
                "meaning": "regularized concentration estimate in upstream units",
            },
            {
                "label": "station observations",
                "meaning": "sparse measured values used as input constraints",
            },
        ],
    }
    view_hints = {
        "preferred_geometry": "volume_mesh",
        "field_parameter": "phenomenon",
        "default_color_attribute": "phenomenon",
        "table_role": "derived_simulation_field",
        "upstream_observation_dataset": "air_quality",
    }
    presentation_warnings = [
        (
            "Sparse or stale station observations can produce visually smooth "
            "fields that are not locally accurate."
        ),
        (
            "This is Tikhonov smoothing/interpolation, not an atmospheric "
            "dispersion, chemistry, emissions, or regulatory compliance model."
        ),
        (
            "Stations outside the generated mesh are skipped; the run fails if "
            "no usable station constraints remain."
        ),
    ]
    presentation_limitations = [
        (
            "Accuracy depends on station density, station timestamps, and "
            "provider coverage."
        ),
        (
            "The model does not include wind, emissions, street-canyon "
            "physics, or chemistry."
        ),
        (
            "Regularization weights are scenario parameters and require "
            "domain calibration."
        ),
        "Outputs inherit upstream air_quality and city_volume_mesh limitations.",
    ]

    def build(self, args):
        import dtcc_core.datasets as datasets

        bounds = self.parse_bounds(args.bounds)

        # Fetch air quality sensor data
        sensors = datasets.air_quality(
            bounds=bounds,
            phenomenon=args.phenomenon,
            crs=args.airquality_crs,
            timeout_s=args.airquality_timeout_s,
            max_stations=args.airquality_max_stations,
            drop_missing=args.airquality_drop_missing,
            base_url=args.airquality_base_url,
        )

        # Extract point coordinates and values
        point_coords, point_values = sensors.to_arrays(field_name=args.phenomenon)
        point_coords = self._validate_point_coordinates(point_coords)
        point_values = self._validate_point_values(point_values, args.phenomenon)
        if point_coords.shape[0] != point_values.size:
            raise RuntimeError(
                "air_quality_field requires one station value per coordinate; "
                f"got {point_coords.shape[0]} coordinates and "
                f"{point_values.size} values."
            )

        # Get unit from sensors
        field_unit = self._single_observation_unit(sensors, args.phenomenon)

        from .smooth_reconstruction import (
            SmoothReconstructionSimulator,
            SmoothReconstructionParameters,
        )

        # Create reconstruction parameters
        params = SmoothReconstructionParameters(
            degree=args.degree,
            lambda_smooth=args.lambda_smooth,
            alpha=args.alpha,
            data_weight=args.data_weight,
            background_value=args.background_value,
            mesh_max_mesh_size=args.mesh_max_mesh_size,
            mesh_domain_height=args.mesh_domain_height,
            mesh_raster_cell_size=args.mesh_raster_cell_size,
            mesh_raster_radius=args.mesh_raster_radius,
            z_offset=args.z_offset,
        )

        # Create simulator with point data
        sim = SmoothReconstructionSimulator(
            bounds=bounds,
            point_coords=point_coords,
            point_values=point_values,
            field_name=args.phenomenon,
            field_unit=field_unit,
            params=params,
        )

        volume_mesh = sim.simulate()

        if args.format:
            if sim.solution is None:
                raise RuntimeError(
                    "air_quality_field did not produce a FEniCS solution "
                    f"for format={args.format!r} serialization."
                )
            return self.export_to_bytes(sim.solution, args.format)
        return volume_mesh

    @staticmethod
    def _validate_point_coordinates(point_coords):
        import numpy as np

        coords = np.asarray(point_coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(
                "air_quality_field requires upstream station coordinates as an "
                f"Nx3 array; got shape {coords.shape}."
            )
        if coords.shape[0] == 0:
            raise RuntimeError(
                "air_quality_field requires at least one station coordinate for "
                "the requested phenomenon and bounds."
            )
        if not np.all(np.isfinite(coords)):
            raise RuntimeError("air_quality_field station coordinates must be finite.")
        return coords

    @staticmethod
    def _validate_point_values(point_values, phenomenon: str):
        import numpy as np

        values = np.asarray(point_values, dtype=float).reshape(-1)
        if values.size == 0:
            raise RuntimeError(
                "air_quality_field requires at least one station value for "
                f"phenomenon {phenomenon!r}."
            )
        if not np.any(np.isfinite(values)):
            raise RuntimeError(
                "air_quality_field station values are all NaN or non-finite for "
                f"phenomenon {phenomenon!r}."
            )
        return values

    @staticmethod
    def _single_observation_unit(sensors, phenomenon: str) -> str:
        units = []
        for station in sensors.stations():
            attributes = getattr(station, "attributes", {}) or {}
            unit = str(attributes.get("unit") or "").strip()
            if unit:
                units.append(unit)

        unique_units = sorted(set(units))
        if not unique_units:
            raise RuntimeError(
                "air_quality_field requires a non-empty unit from upstream "
                f"air_quality stations for phenomenon {phenomenon!r}."
            )
        if len(unique_units) > 1:
            raise RuntimeError(
                "air_quality_field requires one consistent unit for phenomenon "
                f"{phenomenon!r}; got {unique_units}."
            )
        return unique_units[0]


# ---------------------------------------------------------------------------
# Traffic Simulation Dataset
# ---------------------------------------------------------------------------


class TrafficSimulationArgs(DatasetBaseArgs):
    """Arguments for static road traffic simulation."""

    trips_per_employed: float = Field(
        2.0,
        ge=0.0,
        description="Daily vehicle-trip production per employed resident.",
    )
    peak_hour_factor: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of daily trips assigned to the simulated peak hour.",
    )
    population_attraction_weight: float = Field(
        0.25,
        ge=0.0,
        description="Attraction weight for DeSO population.",
    )
    employment_attraction_weight: float = Field(
        1.0,
        ge=0.0,
        description="Attraction weight for employed residents.",
    )
    gravity_gamma: float = Field(
        0.08,
        ge=0.0,
        description="Gravity model decay per minute of free-flow travel time.",
    )
    alpha: float = Field(0.15, ge=0.0, description="BPR alpha parameter.")
    beta: float = Field(4.0, gt=0.0, description="BPR beta parameter.")
    capacity_per_lane: float = Field(
        1800.0,
        gt=0.0,
        description="Nominal hourly road capacity per lane.",
    )
    background_flow_fraction: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Exogenous background traffic as a fraction of each directed arc capacity."
        ),
    )
    default_speed_kmh: float = Field(
        40.0,
        gt=0.0,
        description="Fallback speed for roads without OSM maxspeed/highway data.",
    )
    default_lanes: float = Field(
        2.0,
        gt=0.0,
        description="Fallback physical lane count for roads without OSM lanes data.",
    )
    bidirectional: bool = Field(
        True,
        description="Use reverse arcs for non-oneway road segments.",
    )
    exclude_self_trips: bool = Field(
        True,
        description="Exclude OD demand from a zone to itself.",
    )
    max_iterations: int = Field(
        30,
        ge=1,
        description="Maximum Frank-Wolfe assignment iterations.",
    )
    relative_gap_tolerance: float = Field(
        1.0e-4,
        ge=0.0,
        description="Relative assignment gap convergence tolerance.",
    )
    line_search_iterations: int = Field(
        32,
        ge=1,
        description="Bisection iterations for the Frank-Wolfe line search.",
    )
    statistics_year: Optional[int] = Field(
        None,
        description="Reference year for DeSO statistics; defaults to latest per topic.",
    )
    format: Optional[Literal["pb"]] = Field(None, description="Output format")


class TrafficSimulationDataset(DatasetDescriptor):
    """Static traffic assignment as a DTCC simulation dataset.

    Builds roads and DeSO statistics from the requested bounds, creates a
    synthetic gravity-model OD matrix, and solves a Wardrop user-equilibrium
    road assignment with BPR volume-delay costs using Frank-Wolfe.
    """

    name = "traffic_simulation"
    title = "Static Traffic Assignment"
    description = (
        "Static road traffic assignment on a DTCC RoadNetwork. Builds roads "
        "and DeSO statistics for the requested bounds, creates a synthetic "
        "gravity-model OD matrix from employed residents and attraction "
        "weights, and solves a Wardrop user-equilibrium assignment with BPR "
        "link costs using Frank-Wolfe. Returns a RoadNetwork with flow, "
        "capacity, travel_time, speed, and volume_capacity_ratio attributes."
    )
    ArgsModel = TrafficSimulationArgs
    data_category = "simulation"
    result_kind = "road_network"
    python_return_type = "dtcc_core.model.RoadNetwork"
    timeout_hint = 600
    provider = [{"name": "DTCC Sim", "role": "processor"}]
    source = [
        {
            "name": "dtcc_core.datasets.roads",
            "role": "upstream_road_network",
            "source_terms_status": "requires_review",
        },
        {
            "name": "dtcc_core.datasets.deso",
            "role": "upstream_zone_statistics",
            "required_statistics": ["population", "employment"],
            "source_terms_status": "requires_review",
        },
        {
            "name": "Synthetic gravity-demand and BPR assignment model",
            "role": "simulation_model",
            "demand_model": (
                "production = employed_residents * trips_per_employed * "
                "peak_hour_factor; attractions combine population and employment"
            ),
            "cost_model": "BPR travel time t=t0*(1+alpha*(flow/capacity)^beta)",
            "solver": "Frank-Wolfe user-equilibrium assignment",
        },
    ]
    license = (
        "MIT-licensed simulation code; generated outputs inherit upstream "
        "roads, OpenStreetMap, DeSO, and statistics source/license review "
        "requirements."
    )
    collection_period = (
        "Computed on demand from the current upstream roads dataset and the "
        "requested or latest DeSO statistics year."
    )
    default_crs = "EPSG:3006"
    data_types = [
        "road_network",
        "synthetic_od_matrix",
        "traffic_flow_attributes",
        "assignment_diagnostics",
    ]
    geographic_coverage = (
        "Areas where the upstream roads and DeSO datasets return a connected "
        "drivable road graph and zone statistics."
    )
    update_frequency = "on-demand static assignment run"
    derived_from = [
        {
            "name": "roads",
            "relationship": (
                "road topology, lengths, classes, lanes, speeds, and one-way tags"
            ),
        },
        {
            "name": "deso",
            "relationship": "zone centroids, population, and employment statistics",
        },
    ]
    processing_steps = [
        "Validate demand, BPR, capacity, lane, speed, and convergence parameters",
        "Fetch roads and DeSO statistics for the requested bounds",
        (
            "Build a directed drivable graph from road edges and "
            "one-way/bidirectional settings"
        ),
        "Compute free-flow times, directional capacities, and fallback speeds/lanes",
        "Connect DeSO zone centroids to nearest active graph vertices",
        "Build a synthetic gravity-model OD demand matrix from DeSO statistics",
        "Assign demand with Frank-Wolfe and BPR volume-delay costs",
        "Write flow, capacity, travel-time, speed, V/C, and exclusion attributes",
        "Record assignment diagnostics including demand, convergence, and graph size",
    ]
    presentation_headline = "Synthetic Peak-Hour Traffic"
    presentation_summary = (
        "A static user-equilibrium assignment over the road graph using "
        "synthetic DeSO-based demand."
    )
    presentation_narrative = [
        {
            "heading": "Demand model",
            "body": (
                "Trip productions come from employed residents, "
                "trips_per_employed, and peak_hour_factor. Attractions combine "
                "population and employment with configurable weights, then a "
                "gravity decay distributes trips between zones."
            ),
        },
        {
            "heading": "Road costs",
            "body": (
                "Each directed road arc uses free-flow time from length and "
                "speed, capacity from lanes and road class, and BPR congestion "
                "costs controlled by alpha and beta."
            ),
        },
        {
            "heading": "How to interpret it",
            "body": (
                "Outputs are scenario attributes for planning and workflow "
                "testing. They are not calibrated counts or real-time traffic "
                "conditions."
            ),
        },
    ]
    key_points = [
        "input roads: topology, lengths, highway class, maxspeed, lanes, oneway",
        "input zones: DeSO population and employment statistics",
        "outputs: flow, capacity, travel_time, speed, and volume_capacity_ratio",
        "solver: Frank-Wolfe static user-equilibrium assignment",
        "demand is synthetic and must be calibrated before operational use",
    ]
    presentation_legend = {
        "title": "Traffic assignment attributes",
        "entries": [
            {"label": "flow", "meaning": "assigned plus background vehicles/hour"},
            {"label": "capacity", "meaning": "directional road capacity vehicles/hour"},
            {
                "label": "travel_time",
                "meaning": "congested edge travel time in seconds",
            },
            {"label": "speed", "meaning": "edge speed implied by travel_time in km/h"},
            {"label": "V/C", "meaning": "volume-to-capacity ratio"},
        ],
    }
    view_hints = {
        "preferred_geometry": "road_network",
        "default_color_attribute": "volume_capacity_ratio",
        "secondary_attributes": ["flow", "speed", "travel_time", "capacity"],
        "table_role": "simulation_network",
        "upstream_datasets": ["roads", "deso"],
    }
    presentation_warnings = [
        (
            "The OD matrix is synthetic and DeSO-based; it is not calibrated "
            "against traffic counts, route choice surveys, or observed OD data."
        ),
        (
            "Road speeds, lanes, one-way flags, and capacities depend on "
            "upstream road attributes and documented fallback parameters."
        ),
        (
            "This is a static peak-hour-style assignment, not dynamic traffic, "
            "incident response, signal timing, public transport, or freight modeling."
        ),
    ]
    presentation_limitations = [
        "Not suitable for operational traffic prediction without calibration.",
        "Zone centroid connectors are nearest-road approximations.",
        "Disconnected road components can leave demand unassigned.",
        (
            "Outputs inherit upstream roads and DeSO source, coverage, and "
            "license limitations."
        ),
    ]

    def build(self, args):
        import dtcc_core.datasets as datasets

        bounds = self.parse_bounds(args.bounds)
        roads = datasets.roads(bounds=bounds)
        deso = datasets.deso(
            bounds=bounds,
            statistics=["population", "cars", "employment"],
            statistics_year=args.statistics_year,
        )
        params = TrafficAssignmentParameters(
            trips_per_employed=args.trips_per_employed,
            peak_hour_factor=args.peak_hour_factor,
            population_attraction_weight=args.population_attraction_weight,
            employment_attraction_weight=args.employment_attraction_weight,
            gravity_gamma=args.gravity_gamma,
            alpha=args.alpha,
            beta=args.beta,
            capacity_per_lane=args.capacity_per_lane,
            background_flow_fraction=args.background_flow_fraction,
            default_speed_kmh=args.default_speed_kmh,
            default_lanes=args.default_lanes,
            bidirectional=args.bidirectional,
            exclude_self_trips=args.exclude_self_trips,
            max_iterations=args.max_iterations,
            relative_gap_tolerance=args.relative_gap_tolerance,
            line_search_iterations=args.line_search_iterations,
        )
        sim = TrafficAssignmentSimulator(
            roads=roads,
            zones=deso,
            params=params,
        )
        result = sim.simulate()
        result.attributes["simulation_diagnostics"] = sim.diagnostics

        if args.format == "pb":
            return result.to_proto().SerializeToString()
        return result


__all__ = [
    "UrbanHeatSimulationArgs",
    "UrbanHeatSimulationDataset",
    "AirQualityFieldArgs",
    "AirQualityFieldDataset",
    "TrafficSimulationArgs",
    "TrafficSimulationDataset",
    "UrbanWindSimulationArgs",
    "UrbanWindSimulationDataset",
]


# ---------------------------------------------------------------------------
# Urban Wind Simulation Dataset
# ---------------------------------------------------------------------------


class UrbanWindSimulationArgs(DatasetBaseArgs):
    """Arguments for urban wind simulation dataset."""

    # Wind
    use_weather: bool = Field(
        False,
        description="Fetch reference wind speed and direction from SMHI weather data.",
    )
    weather_aggregation: WindWeatherAggregation = Field(
        "nearest",
        description="Weather aggregation: nearest, mean, or median.",
    )
    wind_speed: float = Field(5.0, description="Reference wind speed [m/s]", ge=0)
    wind_dir_deg: float = Field(
        270.0,
        description="Wind direction (met convention: FROM, 0=N 90=E 180=S 270=W)",
    )

    # Mesh
    mesh_max_mesh_size: float = Field(25.0, description="Max mesh size [m]")
    mesh_domain_height: float = Field(80.0, description="Domain height [m]")

    # Solver
    rho: float = Field(1.2, description="Air density [kg/m3]", gt=0)
    nu: float = Field(1.5e-5, description="Kinematic viscosity [m2/s]", gt=0)
    nu_t: float = Field(
        0.0,
        description="Eddy viscosity placeholder [m2/s]; default is laminar.",
        ge=0,
    )
    equations: WindEquationSet = Field(
        "navier_stokes",
        description="Equation set: 'navier_stokes' (IPCS) or 'stokes' (stationary)",
    )
    simulation_mode: WindSimulationMode = Field(
        "steady",
        description="Stopping mode: steady or statistical_steady.",
    )
    dt: float = Field(0.2, description="Pseudo-time step [s]", gt=0)
    max_steps: int = Field(2000, description="Maximum number of time steps", gt=0)
    steady_tolerance: float = Field(
        1e-4,
        description="Relative velocity-change tolerance for steady mode.",
        ge=0,
    )
    divergence_tolerance: float = Field(
        5e-3,
        description="L2(div u) tolerance for steady-mode stopping.",
        ge=0,
    )
    flux_imbalance_tolerance: float = Field(
        5e-2,
        description="Net open-boundary flux imbalance tolerance.",
        ge=0,
    )
    min_steps: int = Field(50, description="Minimum steps before early stop", ge=0)
    steady_window: int = Field(
        5,
        description="Consecutive converged steps required before stopping.",
        ge=1,
    )

    # BC model
    side_top_boundary: WindSideTopBoundary = Field(
        "open",
        description="Side/top boundary model: open or slip.",
    )
    wall_model: WindWallModel = Field(
        "noslip", description="Wall model: 'noslip' or 'friction'"
    )
    beta_wall: float = Field(0.5, description="Friction coefficient (friction model)")
    inlet_profile: WindInletProfile = Field(
        "uniform", description="Inlet profile: 'uniform', 'power_law', 'log_law'"
    )
    inlet_ramp_steps: int = Field(
        0,
        description="Initial pseudo-time steps used to ramp inlet velocity.",
        ge=0,
    )
    z0: float = Field(0.5, description="Roughness length [m] for log-law", gt=0)
    u_ref_height: float = Field(
        10.0, description="Reference measurement height [m]", gt=0
    )
    power_law_alpha: float = Field(0.2, description="Power-law exponent", gt=0)
    closed_cavity: bool = Field(
        False,
        description="Use a pressure null-space instead of an outlet boundary.",
    )

    format: Optional[Literal["pb"]] = Field(None, description="Output format")


class UrbanWindSimulationDataset(DatasetDescriptor):
    """Urban wind CFD simulation as a dataset.

    Solves the incompressible Navier-Stokes equations on a DTCC city volume mesh
    using the IPCS fractional-step method.  The solver marches in pseudo-time
    until a steady state is reached and returns a dtcc-core ``VolumeMesh`` with
    attached ``velocity``, ``pressure``, and ``speed`` fields.

    Example:
        >>> import dtcc_core.datasets as datasets
        >>> import dtcc_sim.datasets
        >>>
        >>> result = datasets.urban_wind_simulation(
        ...     bounds=[xmin, ymin, xmax, ymax],
        ...     wind_speed=8.0,
        ...     wind_dir_deg=240.0,
        ... )
    """

    name = "urban_wind_simulation"
    title = "Urban Wind Simulation"
    description = (
        "Incompressible urban wind simulation using FEniCSx. The default "
        "solves pseudo-time Navier-Stokes with an IPCS ABCN fractional-step "
        "scheme; a stationary mixed Stokes mode is also available. Output "
        "pressure is kinematic pressure p/rho in m^2/s^2. Returns a DTCC "
        "VolumeMesh with velocity, pressure, and speed Fields."
    )
    ArgsModel = UrbanWindSimulationArgs
    data_category = "simulation"
    result_kind = "mesh"
    python_return_type = "dtcc_core.model.VolumeMesh"
    timeout_hint = 1800
    provider = [
        {"name": "DTCC Sim", "role": "processor"},
        {"name": "FEniCSx/dolfinx", "role": "solver"},
        {
            "name": "dtcc_core.datasets.weather",
            "role": "optional_source_provider",
            "used_when": "use_weather=True",
        },
    ]
    source = [
        {
            "name": "dtcc_core.datasets.city_volume_mesh",
            "role": "upstream_dataset",
            "source_terms_status": "requires_review",
        },
        {
            "name": "Urban wind finite-element CFD model",
            "role": "simulation_model",
            "equation": (
                "du/dt + (u dot grad)u = -grad(p/rho) + nu_eff Delta u; "
                "div u = 0"
            ),
            "unknowns": "velocity u and kinematic pressure p/rho",
            "units": {
                "velocity": "m/s",
                "pressure": "m^2/s^2",
                "speed": "m/s",
            },
        },
        {
            "name": "SMHI weather observations through dtcc_core.datasets.weather",
            "role": "optional_reference_wind_source",
            "source_terms_status": "requires_review",
        },
    ]
    license = (
        "MIT-licensed simulation code; generated outputs inherit upstream "
        "city-volume-mesh and optional weather source/license review requirements."
    )
    collection_period = (
        "Computed on demand. When use_weather=True, the wind forcing comes from "
        "the latest weather observations returned by the upstream weather dataset."
    )
    default_crs = "EPSG:3006"
    data_types = [
        "volume_mesh",
        "velocity_field",
        "kinematic_pressure_field",
        "speed_field",
        "finite_element_solution",
    ]
    geographic_coverage = (
        "Areas where city_volume_mesh can construct a valid volume mesh and, "
        "when use_weather=True, where the weather provider returns wind data."
    )
    update_frequency = "on-demand simulation run"
    derived_from = [
        {
            "name": "city_volume_mesh",
            "relationship": "computational CFD domain generated from requested bounds",
        },
        {
            "name": "weather",
            "relationship": "optional source of reference wind speed and direction",
            "used_when": "use_weather=True",
        },
    ]
    processing_steps = [
        (
            "Validate wind forcing, inlet profile, wall model, convergence "
            "settings, mesh settings, and output format"
        ),
        "Generate a DTCC city volume mesh from requested bounds when needed",
        "Load the mesh and boundary markers into FEniCSx",
        (
            "Optionally fetch weather wind speed and direction; fail if "
            "use_weather=True and no usable wind data is returned"
        ),
        "Select inlet and outlet bbox faces from meteorological wind direction",
        (
            "Collapse raw boundary markers into wall, roof, ground, inlet, "
            "outlet, side, and top categories"
        ),
        "Build the selected uniform, power-law, or log-law inlet velocity profile",
        "Solve Navier-Stokes with IPCS pseudo-time stepping or stationary Stokes",
        "Record convergence, divergence, flux-imbalance, CFL, and field diagnostics",
        (
            "Attach velocity, kinematic pressure, and speed Fields or "
            "serialize protobuf output"
        ),
    ]
    presentation_headline = "Urban Wind CFD Field"
    presentation_summary = (
        "A finite-element airflow scenario over a generated 3D urban volume mesh."
    )
    presentation_narrative = [
        {
            "heading": "What is solved",
            "body": (
                "The default model solves incompressible Navier-Stokes in "
                "pseudo-time until the configured convergence checks pass or "
                "max_steps is reached. Stationary Stokes mode omits the "
                "convective term."
            ),
        },
        {
            "heading": "Wind direction and profile",
            "body": (
                "wind_dir_deg uses meteorological convention: the direction "
                "the wind comes from. The solver converts that to a flow "
                "vector and applies a uniform, power-law, or log-law inlet "
                "profile using u_ref_height and z0/power_law_alpha."
            ),
        },
        {
            "heading": "How to interpret it",
            "body": (
                "The result is an engineering scenario field for solver and "
                "workflow validation. Use diagnostics and mesh-sensitivity "
                "checks before treating a city-scale run as CFD evidence."
            ),
        },
    ]
    key_points = [
        "unknowns: velocity u and kinematic pressure p/rho",
        "fields: velocity dim=3 m/s, pressure dim=1 m^2/s^2, speed dim=1 m/s",
        "default wind convention: 270 degrees means wind from west, flowing east",
        "default wall model: no-slip on buildings, roofs, and ground",
        "city-scale CFD validation is expensive/manual until benchmarked",
    ]
    presentation_legend = {
        "title": "Wind fields",
        "entries": [
            {"label": "velocity", "meaning": "solved vector field in m/s"},
            {"label": "speed", "meaning": "velocity magnitude in m/s"},
            {
                "label": "pressure",
                "meaning": "kinematic pressure p/rho in m^2/s^2",
            },
            {
                "label": "inlet/outlet",
                "meaning": "bbox faces selected from wind direction",
            },
            {"label": "walls/ground", "meaning": "solid no-slip or friction surfaces"},
        ],
    }
    view_hints = {
        "preferred_geometry": "volume_mesh",
        "fields": [
            {"name": "velocity", "dim": 3, "unit": "m/s"},
            {"name": "pressure", "dim": 1, "unit": "m^2/s^2"},
            {"name": "speed", "dim": 1, "unit": "m/s"},
        ],
        "default_color_attribute": "speed",
        "table_role": "simulation_field",
        "pressure_type": "kinematic",
    }
    presentation_warnings = [
        (
            "This is not a certified CFD product. Mesh quality, boundary "
            "placement, wind forcing, turbulence assumptions, and solver "
            "settings can dominate the result."
        ),
        (
            "The default nu_t=0 is a laminar/simplified setting. Urban "
            "turbulence, atmospheric stratification, thermal buoyancy, and "
            "rough-wall calibration are not validated by default."
        ),
        (
            "City-scale runs can be slow and mesh-sensitive; treat them as "
            "manual/expensive validation until benchmark and convergence "
            "studies have been run."
        ),
    ]
    presentation_limitations = [
        "Planning/scenario field, not a measured wind or regulatory CFD product.",
        "Output pressure is kinematic pressure p/rho, not absolute pressure in Pa.",
        "Default weather use is optional and requires usable upstream wind data.",
        "Small solver smoke tests do not validate city-scale recirculation physics.",
        "Outputs inherit upstream city-volume-mesh source and geometry limitations.",
    ]

    def build(self, args):
        from .urban_wind import UrbanWindSimulator, UrbanWindParameters

        bounds = self.parse_bounds(args.bounds)
        params = UrbanWindParameters(
            rho=args.rho,
            nu=args.nu,
            nu_t=args.nu_t,
            equations=args.equations,
            simulation_mode=args.simulation_mode,
            use_weather=args.use_weather,
            weather_aggregation=args.weather_aggregation,
            wind_speed=args.wind_speed,
            wind_dir_deg=args.wind_dir_deg,
            mesh_max_mesh_size=args.mesh_max_mesh_size,
            mesh_domain_height=args.mesh_domain_height,
            dt=args.dt,
            max_steps=args.max_steps,
            steady_tolerance=args.steady_tolerance,
            divergence_tolerance=args.divergence_tolerance,
            flux_imbalance_tolerance=args.flux_imbalance_tolerance,
            min_steps=args.min_steps,
            steady_window=args.steady_window,
            side_top_boundary=args.side_top_boundary,
            wall_model=args.wall_model,
            beta_wall=args.beta_wall,
            inlet_profile=args.inlet_profile,
            inlet_ramp_steps=args.inlet_ramp_steps,
            z0=args.z0,
            u_ref_height=args.u_ref_height,
            power_law_alpha=args.power_law_alpha,
            closed_cavity=args.closed_cavity,
        )
        sim = UrbanWindSimulator(bounds=bounds, params=params)
        result = sim.simulate()

        if args.format:
            if not hasattr(result, "save"):
                raise RuntimeError(
                    "urban_wind_simulation expected a DTCC VolumeMesh for "
                    f"format={args.format!r} serialization, but the solver "
                    f"returned {type(result).__name__}."
                )
            return self.export_to_bytes(result, args.format)
        return result
