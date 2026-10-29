"""Stationary Stokes equation parameters."""

from typing import Any, Literal
from pydantic import Field
from .._flow.parameters import _FlowParameters


class StokesParameters(_FlowParameters):
    """Linear stationary Stokes configuration."""

    equations: Literal["stokes"] = Field("stokes", frozen=True)
    petsc_stokes: dict[str, Any] | None = None
