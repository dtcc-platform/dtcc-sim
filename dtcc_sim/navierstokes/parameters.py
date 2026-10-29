"""Stationary NavierStokes equation parameters."""

from typing import Any, Literal
from pydantic import Field
from .._flow.parameters import _FlowParameters


class NavierStokesParameters(_FlowParameters):
    """Stationary Navier–Stokes configuration with full convection."""

    equations: Literal["navier_stokes"] = Field("navier_stokes", frozen=True)
    nonlinear_rtol: float = Field(1e-9, gt=0, lt=1)
    nonlinear_atol: float = Field(1e-10, gt=0)
    nonlinear_max_it: int = Field(40, gt=0)
    backflow_beta: float = Field(0., ge=0, description="Weak damping where open-boundary flow enters the domain")
    petsc_options: dict[str, Any] | None = None
