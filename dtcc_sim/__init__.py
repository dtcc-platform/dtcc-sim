"""DTCC Sim package - Urban simulation tools."""

from ._version import __version__
from .urban_heat import (
    UrbanHeatSimulator,
    UrbanHeatParameters,
    BndCat,
    DirichletBCSpec,
    NeumannBCSpec,
    RobinBCSpec,
)
from .smooth_reconstruction import (
    SmoothReconstructionSimulator,
    SmoothReconstructionParameters,
)
from .urban_wind import (
    UrbanWindSimulator,
    UrbanWindParameters,
    BndCat as WindBndCat,
)
from .datasets import (
    UrbanHeatSimulationArgs,
    UrbanHeatSimulationDataset,
    AirQualityFieldArgs,
    AirQualityFieldDataset,
    UrbanWindSimulationArgs,
    UrbanWindSimulationDataset,
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
    "SmoothReconstructionSimulator",
    "SmoothReconstructionParameters",
    "UrbanWindSimulator",
    "UrbanWindParameters",
    "WindBndCat",
    "UrbanHeatSimulationArgs",
    "UrbanHeatSimulationDataset",
    "AirQualityFieldArgs",
    "AirQualityFieldDataset",
    "UrbanWindSimulationArgs",
    "UrbanWindSimulationDataset",
]
