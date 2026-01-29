"""
DTCC Sim Datasets

This module provides simulation results as datasets that integrate with
the dtcc-core dataset system. When dtcc_sim is imported, these datasets
automatically register themselves with dtcc_core.datasets.
"""

from typing import Optional, Literal
from pydantic import Field

try:
    from dtcc_core.datasets import DatasetDescriptor, DatasetBaseArgs

    DTCC_CORE_AVAILABLE = True
except ImportError:
    DTCC_CORE_AVAILABLE = False

    # Define minimal stubs for when dtcc-core is not available
    class DatasetBaseArgs:
        pass

    class DatasetDescriptor:
        pass


from .urbanheat import UrbanHeatSimulator, UrbanHeatParameters


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


__all__ = ["UrbanHeatSimulationArgs", "UrbanHeatSimulationDataset"]
