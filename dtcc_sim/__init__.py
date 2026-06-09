"""DTCC Sim package - Urban simulation tools."""

import os

from ._version import __version__
from .traffic import (
    TrafficAssignmentSimulator,
    TrafficAssignmentParameters,
)
from .datasets import (
    UrbanHeatSimulationArgs,
    UrbanHeatSimulationDataset,
    AirQualityFieldArgs,
    AirQualityFieldDataset,
    TrafficSimulationArgs,
    TrafficSimulationDataset,
    UrbanWindSimulationArgs,
    UrbanWindSimulationDataset,
)

_fenics_exports = []
try:
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

    # Default FEniCSx log level (override with DTCC_FENICSX_LOG_LEVEL).
    # Note: FEniCSx backend messages use FEniCSx-native formatting.
    from .fenics import set_log_level, DEBUG, INFO, WARNING, ERROR

    _fenicsx_log_name = os.getenv("DTCC_FENICSX_LOG_LEVEL", "INFO").strip().upper()
    _fenicsx_log_level = {
        "DEBUG": DEBUG,
        "INFO": INFO,
        "WARNING": WARNING,
        "ERROR": ERROR,
    }.get(_fenicsx_log_name, INFO)
    set_log_level(_fenicsx_log_level)
    _fenics_exports = [
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
    ]
except ModuleNotFoundError as exc:
    if exc.name != "dolfinx":
        raise

__all__ = [
    "__version__",
    "TrafficAssignmentSimulator",
    "TrafficAssignmentParameters",
    "UrbanHeatSimulationArgs",
    "UrbanHeatSimulationDataset",
    "AirQualityFieldArgs",
    "AirQualityFieldDataset",
    "TrafficSimulationArgs",
    "TrafficSimulationDataset",
    "UrbanWindSimulationArgs",
    "UrbanWindSimulationDataset",
] + _fenics_exports
