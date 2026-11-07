"""Physical inputs and stationary flow solver configuration."""

from ..meshing import CityMeshSettings
from typing import Literal, Mapping

import numpy as np
from pydantic import ConfigDict, Field, model_validator

KAPPA = 0.41
"""Von Kármán constant used by the atmospheric boundary-layer closures."""


class _FlowParameters(CityMeshSettings):
    """Shared physical, mesh, boundary and numerical acceptance settings."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rho: float = Field(1.2, gt=0, description="Air density [kg/m³]; pressure output is kinematic")
    nu: float = Field(1.5e-5, gt=0, description="Kinematic viscosity [m²/s]")
    nu_t: float = Field(0., ge=0, description="Explicit effective eddy viscosity [m²/s] for turbulence_model='constant'")
    turbulence_model: Literal["constant", "abl", "mixing_length"] = Field(
        "constant",
        description="Eddy viscosity: explicit constant nu_t, equilibrium ABL profile u*·l_m, or Prandtl mixing length l_m²|S|",
    )
    mixing_length_limit: float | None = Field(None, gt=0, description="Blackadar asymptotic mixing length [m]; None keeps l_m = κ(d + z_w)")
    abl_top_shear: bool = Field(True, description="Apply the equilibrium shear stress u*² along the wind on a slip top for ABL closures")
    use_weather: bool = False
    weather_aggregation: Literal["nearest", "mean", "median"] = "nearest"
    wind_speed: float = Field(5., ge=0, description="Reference inlet speed [m/s]")
    wind_dir_deg: float = Field(270., description="Meteorological direction FROM [degrees]")
    mesh_refinement_levels: int = Field(0, ge=0, le=2, description="Maximum explicit local refinements of pressure-deficient cell stars")
    velocity_degree: Literal[2] = 2
    pressure_degree: Literal[1] = 1
    side_top_boundary: Literal["open", "slip"] = "open"
    wall_model: Literal["noslip", "friction", "log_law"] = "noslip"
    beta_wall: float = Field(.5, ge=0, description="Tangential friction coefficient")
    wall_function_height: float = Field(2., gt=0, description="Height of the computational wall above the roughness surface for wall_model='log_law' [m]")
    gamma_normal: float | None = Field(None, gt=0, description="Weak-wall normal Nitsche penalty [m/s]; default nitsche_penalty (ν_eff/h_F + wind_speed)")
    nitsche_penalty: float = Field(100., gt=0, description="Dimensionless weak-wall normal penalty factor when gamma_normal is unset")
    inlet_profile: Literal["uniform", "power_law", "log_law"] = "uniform"
    z0: float = Field(.5, gt=0, description="Aerodynamic roughness length [m]")
    u_ref_height: float = Field(10., gt=0, description="Height of wind_speed above the ground [m]")
    power_law_alpha: float = Field(.2, gt=0)
    inlet_marker: int | None = None
    outlet_marker: int | None = None
    closed_cavity: bool = Field(False, description="Suppress automatic outlet selection")
    grad_div_gamma: float = Field(0., ge=0, description="Consistent grad-div stabilization")
    residual_rtol: float = Field(1e-8, gt=0, lt=1)
    residual_atol: float = Field(1e-9, gt=0)
    divergence_tolerance: float = Field(.1, gt=0, description="Maximum hmin × RMS(div u) / RMS(u)")
    flux_imbalance_tolerance: float = Field(1e-4, gt=0, lt=1, description="Maximum absolute net flux / integrated absolute boundary flux")
    ground_leakage_tolerance: float = Field(.02, gt=0, lt=1, description="Maximum ∫ground |u·n| / inflow volume flux")
    building_leakage_tolerance: float = Field(.02, gt=0, lt=1, description="Maximum ∫walls and roofs |u·n| / inflow volume flux")

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

    @model_validator(mode="after")
    def atmospheric_consistency(self):
        if self.turbulence_model != "constant":
            if self.nu_t:
                raise ValueError(
                    f"nu_t applies only to turbulence_model='constant'; "
                    f"turbulence_model={self.turbulence_model!r} computes the eddy viscosity"
                )
            if self.inlet_profile != "log_law":
                raise ValueError(
                    f"turbulence_model={self.turbulence_model!r} requires inlet_profile='log_law' "
                    "so the inflow is in equilibrium with the closure; use turbulence_model='constant' "
                    "for other profiles"
                )
        if self.wall_model == "log_law" and self.inlet_profile != "log_law":
            raise ValueError("wall_model='log_law' requires inlet_profile='log_law'; "
                             "choose wall_model='noslip' or 'friction' for other profiles")
        if getattr(self, "equations", None) == "stokes" and self.turbulence_model == "mixing_length":
            raise ValueError(
                "turbulence_model='mixing_length' depends on the velocity gradient and needs "
                "Navier–Stokes Newton iteration; use turbulence_model='abl' with Stokes"
            )
        return self

    @property
    def nu_eff(self) -> float:
        """Constant effective kinematic viscosity [m²/s] for turbulence_model='constant'."""
        return self.nu + self.nu_t

    @property
    def uses_wall_distance(self) -> bool:
        """Whether the closure, wall law or inlet use computed wall distances."""
        return self.turbulence_model != "constant" or self.wall_model == "log_law"

    @property
    def wall_offset(self) -> float:
        """Log-law offset z_w [m]: z0 plus the wall-function height for log-law walls."""
        return self.z0 + (self.wall_function_height if self.wall_model == "log_law" else 0.)

    @property
    def friction_velocity(self) -> float:
        """Inflow friction velocity u* [m/s] from wind_speed at u_ref_height."""
        return KAPPA * self.wind_speed / float(np.log((self.u_ref_height + self.wall_offset) / self.z0))

    @property
    def wind_vector_xy(self) -> tuple[float, float]:
        """Horizontal unit vector pointing in the flow direction."""
        angle = np.deg2rad(270. - self.wind_dir_deg)
        return float(np.cos(angle)), float(np.sin(angle))
