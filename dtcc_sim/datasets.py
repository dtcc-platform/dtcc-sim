"""
DTCC Sim Datasets

This module provides simulation results as datasets that integrate with
the dtcc-core dataset system. When dtcc_sim is imported, these datasets
automatically register themselves with dtcc_core.datasets.
"""

from typing import Optional, Literal
from pydantic import Field

from dtcc_core.datasets import DatasetDescriptor, DatasetBaseArgs

from .urban_heat import UrbanHeatSimulator, UrbanHeatParameters
from .smooth_reconstruction import (
    SmoothReconstructionSimulator,
    SmoothReconstructionParameters,
)


class UrbanHeatSimulationArgs(DatasetBaseArgs):
    """Arguments for urban heat simulation dataset."""

    # PDE coefficients
    kappa: float = Field(1.0, description="Thermal diffusivity coefficient", gt=0)
    sigma: float = Field(0.0, description="Relaxation/reaction coefficient", ge=0)
    T_ambient: float = Field(0.0, description="Ambient temperature")
    degree: int = Field(1, description="Polynomial degree for FE space", ge=1, le=3)

    # Boundary conditions - simplified
    wall_value: float = Field(1.0, description="Wall temperature")
    roof_value: float = Field(1.0, description="Roof temperature")
    ground_value: float = Field(0.0, description="Ground temperature")
    open_value: float = Field(0.0, description="Open boundary temperature")

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
        4. Returns the temperature field as a FEniCSx Function

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
    description = (
        "Steady-state urban heat simulation using FEniCSx. Solves the heat diffusion "
        "equation -∇·(κ∇T) + σ(T - T_ambient) = 0 in a 3D urban volume with convective "
        "boundary conditions on buildings, ground, and domain boundaries. Models air "
        "temperature distribution for urban heat island analysis, thermal comfort assessment, "
        "and climate adaptation studies. Automatically generates volume mesh from geographic "
        "bounds and returns temperature field as FEniCSx Function."
    )
    ArgsModel = UrbanHeatSimulationArgs

    def build(self, args):
        bounds = self.parse_bounds(args.bounds)
        params = UrbanHeatParameters(
            kappa=args.kappa,
            sigma=args.sigma,
            T_ambient=args.T_ambient,
            degree=args.degree,
            wall_value=args.wall_value,
            roof_value=args.roof_value,
            ground_value=args.ground_value,
            open_value=args.open_value,
            mesh_max_mesh_size=args.mesh_max_mesh_size,
            mesh_domain_height=args.mesh_domain_height,
        )
        sim = UrbanHeatSimulator(bounds=bounds, params=params)
        T = sim.simulate()
        if args.format:
            return self.export_to_bytes(T, args.format)
        return T


class AirQualityFieldArgs(DatasetBaseArgs):
    """Arguments for air quality field reconstruction dataset."""

    # Phenomenon to reconstruct
    phenomenon: str = Field("NO2", description="Phenomenon name (e.g., NO2, PM10)")

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
    degree: int = Field(1, description="Polynomial degree (currently only 1 supported)")

    # Mesh parameters
    mesh_max_mesh_size: float = Field(25.0, description="Max mesh size in meters")
    mesh_domain_height: float = Field(80.0, description="Domain height in meters")
    mesh_raster_cell_size: float = Field(2.0, description="Terrain raster cell size")
    mesh_raster_radius: float = Field(
        3.0, description="Terrain raster interpolation radius"
    )

    # Air quality dataset parameters
    airquality_crs: str = Field("EPSG:3006", description="CRS for air quality data")
    airquality_timeout_s: float = Field(10.0, description="API timeout in seconds")
    airquality_max_stations: int = Field(250, description="Max stations to fetch")
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
    description = (
        "Smooth reconstruction of air quality fields from sparse sensor measurements using "
        "PDE-based Tikhonov regularization. Fetches air quality data from SMHI API, generates "
        "a 3D urban volume mesh, and solves a regularized interpolation problem to create "
        "smooth, continuous concentration fields. Minimizes: ½w Σᵢ(u(xᵢ)-yᵢ)² + ½λ∫|∇u|²dx + "
        "½α∫(u-ubg)²dx where w controls data fidelity, λ enforces smoothness, and α anchors "
        "to background. Returns dtcc-core VolumeMesh with reconstructed field as Field. "
        "Supports multiple pollutants (NO2, PM10, O3, etc.) with configurable regularization."
    )
    ArgsModel = AirQualityFieldArgs

    def build(self, args):
        import dtcc_core.datasets as datasets

        bounds = self.parse_bounds(args.bounds)

        # Fetch air quality sensor data
        sensors = datasets.airquality(
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

        # Get unit from sensors
        field_unit = ""
        stations = sensors.stations()
        if stations:
            field_unit = stations[0].attributes.get("unit", "")

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
            return self.export_to_bytes(volume_mesh, args.format)
        return volume_mesh


__all__ = [
    "UrbanHeatSimulationArgs",
    "UrbanHeatSimulationDataset",
    "AirQualityFieldArgs",
    "AirQualityFieldDataset",
]
