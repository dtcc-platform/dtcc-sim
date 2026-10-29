"""Stationary Stokes equation solver."""

from .._flow.solver import _FlowSolver
from .parameters import StokesParameters


class StokesSolver(_FlowSolver):
    """Solve the stationary Stokes equations."""

    parameter_type = StokesParameters
