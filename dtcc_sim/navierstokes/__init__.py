"""Stationary NavierStokes equations."""

from importlib import import_module
from .parameters import NavierStokesParameters

__all__ = ["NavierStokesParameters", "NavierStokesSolver"]


def __getattr__(name):
    if name != "NavierStokesSolver":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".solver", __name__), name)
