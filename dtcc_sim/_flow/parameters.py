"""Physical inputs and stationary flow solver configuration."""

from ..meshing import CityMeshSettings
from typing import Literal, Mapping

import numpy as np
from pydantic import ConfigDict, Field, model_validator


class _FlowParameters(CityMeshSettings):
    """Shared physical, mesh, boundary and numerical acceptance settings."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rho: float = Field(1.2, gt=0, description="Air density [kg/m³]; pressure output is kinematic")
    nu: float = Field(1.5e-5, gt=0, description="Kinematic viscosity [m²/s]")
    nu_t: float = Field(0., ge=0, description="Explicit effective eddy viscosity [m²/s]")
    use_weather: bool = False
    weather_aggregation: Literal["nearest", "mean", "median"] = "nearest"
    wind_speed: float = Field(5., ge=0, description="Reference inlet speed [m/s]")
    wind_dir_deg: float = Field(270., description="Meteorological direction FROM [degrees]")
    mesh_refinement_levels: int = Field(0, ge=0, le=2, description="Maximum explicit local refinements of pressure-deficient cell stars")
    velocity_degree: Literal[2] = 2
    pressure_degree: Literal[1] = 1
    side_top_boundary: Literal["open", "slip"] = "open"
    wall_model: Literal["noslip", "friction"] = "noslip"
    beta_wall: float = Field(.5, ge=0, description="Tangential friction coefficient")
    gamma_normal: float | None = Field(None, gt=0, description="Normal Nitsche penalty; default 100 ν_eff/h")
    inlet_profile: Literal["uniform", "power_law", "log_law"] = "uniform"
    z0: float = Field(.5, gt=0)
    u_ref_height: float = Field(10., gt=0)
    power_law_alpha: float = Field(.2, gt=0)
    inlet_marker: int | None = None
    outlet_marker: int | None = None
    closed_cavity: bool = Field(False, description="Suppress automatic outlet selection")
    grad_div_gamma: float = Field(0., ge=0, description="Consistent grad-div stabilization")
    residual_rtol: float = Field(1e-8, gt=0, lt=1)
    residual_atol: float = Field(1e-9, gt=0)
    divergence_tolerance: float = Field(.1, gt=0, description="Maximum hmin × RMS(div u) / RMS(u)")
    flux_imbalance_tolerance: float = Field(1e-4, gt=0, lt=1, description="Maximum absolute net flux / integrated absolute boundary flux")

    @model_validator(mode="before")
    @classmethod
    def stationary_options(cls, data):
        if isinstance(data, Mapping):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                raise ValueError(
                    f"Stationary flow configuration does not accept {', '.join(unknown)}. "
                    "Configure physical inputs, residual tolerances and nonlinear_rtol, "
                    "nonlinear_atol, nonlinear_max_it or explicit PETSc options."
                )
        return data

    @property
    def nu_eff(self) -> float:
        """Effective kinematic viscosity [m²/s]."""
        return self.nu + self.nu_t

    @property
    def wind_vector_xy(self) -> tuple[float, float]:
        """Horizontal unit vector pointing in the flow direction."""
        angle = np.deg2rad(270. - self.wind_dir_deg)
        return float(np.cos(angle)), float(np.sin(angle))
