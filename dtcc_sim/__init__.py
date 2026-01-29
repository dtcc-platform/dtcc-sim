"""DTCC Sim package - Urban simulation tools."""

from ._version import __version__
from .urbanheat import (
    UrbanHeatSimulator,
    UrbanHeatParameters,
    BndCat,
    DirichletBCSpec,
    NeumannBCSpec,
    RobinBCSpec,
)
from .datasets import (
    UrbanHeatSimulationArgs,
    UrbanHeatSimulationDataset,
)

# Set default log level to INFO for FEniCSx
from .fenics import set_log_level, INFO
set_log_level(INFO)

__all__ = [
    "__version__",
    "UrbanHeatSimulator",
    "UrbanHeatParameters",
    "BndCat",
    "DirichletBCSpec",
    "NeumannBCSpec",
    "RobinBCSpec",
    "UrbanHeatSimulationArgs",
    "UrbanHeatSimulationDataset",
]
