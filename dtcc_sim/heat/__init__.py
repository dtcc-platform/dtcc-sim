"""Steady heat diffusion with surface boundary conditions."""

from importlib import import_module
from .parameters import (
    BndCat, HeatParameters, DirichletBCSpec, NeumannBCSpec, RobinBCSpec,
)

__all__ = [
    "HeatSolver", "HeatParameters", "BndCat", "DirichletBCSpec",
    "NeumannBCSpec", "RobinBCSpec",
]


def __getattr__(name):
    if name != "HeatSolver":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".solver", __name__), name)
