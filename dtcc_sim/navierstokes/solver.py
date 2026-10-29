"""Stationary NavierStokes equation solver."""

from .._flow.solver import _FlowSolver
from .parameters import NavierStokesParameters


class NavierStokesSolver(_FlowSolver):
    """Solve the stationary NavierStokes equations."""

    parameter_type = NavierStokesParameters
