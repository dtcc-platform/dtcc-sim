"""Heat equation coefficients and surface boundary conditions."""

from ..meshing import CityMeshSettings
from enum import IntEnum
from typing import Union
from pydantic import BaseModel, Field


class BndCat(IntEnum):
    """Boundary category enumeration for urban surfaces."""

    WALL = 1
    ROOF = 2
    GROUND = 3
    OPEN = 4


# -----------------------------------------------------------------------------
# Boundary condition specifications
# -----------------------------------------------------------------------------


class DirichletBCSpec(BaseModel):
    """Dirichlet boundary condition: T = value"""

    type: str = "dirichlet"
    value: float = Field(..., description="Temperature value")


class NeumannBCSpec(BaseModel):
    """Neumann boundary condition: -k dT/dn = flux"""

    type: str = "neumann"
    flux: float = Field(..., description="Heat flux (outward normal)")


class RobinBCSpec(BaseModel):
    """Robin boundary condition: -k dT/dn = h(T - value)"""

    type: str = "robin"
    h: float = Field(..., description="Heat transfer coefficient")
    value: float = Field(..., description="Reference temperature")


BCSpec = Union[DirichletBCSpec, NeumannBCSpec, RobinBCSpec]


# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------


class HeatParameters(CityMeshSettings):
    """Parameters for the steady heat equation.

    This uses Pydantic BaseModel for validation and JSON schema generation,
    making it compatible with the dtcc-core dataset system.
    """

    # PDE coefficients
    kappa: float = Field(1.0, description="Thermal diffusivity coefficient", gt=0)
    sigma: float = Field(
        0.0,
        description="Relaxation/reaction coefficient (for sigma*(T - T_ambient) term)",
        ge=0,
    )
    T_ambient: float = Field(
        0.0,
        description="Ambient temperature (reference for relaxation and open boundaries)",
    )

    # Function space
    degree: int = Field(
        1, description="Polynomial degree for finite element space", ge=1, le=3
    )

    # Boundary conditions (using dict representation for Pydantic compatibility)
    # Default: hot buildings, ambient open boundary (anomaly form)
    wall_bc_type: str = Field(
        "robin", description="BC type for walls: dirichlet, neumann, or robin"
    )
    wall_value: float = Field(
        1.0, description="Temperature/reference value for wall BC"
    )
    wall_h: float = Field(
        5.0, description="Heat transfer coeff for wall (Robin BC only)"
    )
    wall_flux: float = Field(0.0, description="Heat flux for wall (Neumann BC only)")

    roof_bc_type: str = Field("robin", description="BC type for roofs")
    roof_value: float = Field(
        1.0, description="Temperature/reference value for roof BC"
    )
    roof_h: float = Field(
        5.0, description="Heat transfer coeff for roof (Robin BC only)"
    )
    roof_flux: float = Field(0.0, description="Heat flux for roof (Neumann BC only)")

    ground_bc_type: str = Field("robin", description="BC type for ground")
    ground_value: float = Field(
        0.0, description="Temperature/reference value for ground BC"
    )
    ground_h: float = Field(
        2.0, description="Heat transfer coeff for ground (Robin BC only)"
    )
    ground_flux: float = Field(
        0.0, description="Heat flux for ground (Neumann BC only)"
    )

    open_bc_type: str = Field("dirichlet", description="BC type for open boundaries")
    open_value: float = Field(
        0.0, description="Temperature/reference value for open BC"
    )
    open_h: float = Field(
        1.0, description="Heat transfer coeff for open (Robin BC only)"
    )
    open_flux: float = Field(0.0, description="Heat flux for open (Neumann BC only)")

    def get_bc_spec(self, surface: str) -> BCSpec:
        """Get boundary condition specification for a surface type.

        Args:
            surface: One of 'wall', 'roof', 'ground', 'open'

        Returns:
            BCSpec object (Dirichlet, Neumann, or Robin)
        """
        bc_type = getattr(self, f"{surface}_bc_type")
        value = getattr(self, f"{surface}_value")
        h = getattr(self, f"{surface}_h")
        flux = getattr(self, f"{surface}_flux")

        if bc_type == "dirichlet":
            return DirichletBCSpec(value=value)
        elif bc_type == "neumann":
            return NeumannBCSpec(flux=flux)
        elif bc_type == "robin":
            return RobinBCSpec(h=h, value=value)
        else:
            raise ValueError(f"Unknown BC type: {bc_type}")
