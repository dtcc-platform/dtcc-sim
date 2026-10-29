"""Stationary Stokes equations."""

from importlib import import_module
from .parameters import StokesParameters

__all__ = ["StokesParameters", "StokesSolver"]


def __getattr__(name):
    if name != "StokesSolver":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".solver", __name__), name)
