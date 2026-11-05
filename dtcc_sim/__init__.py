"""DTCC Sim package - Urban simulation tools."""

import os

from ._version import __version__
from .meshing import CityMeshParameters, CityMeshCache, CityMeshSettings
from .helmholtz import HelmholtzSolver, HelmholtzParameters
from .shallow_water import CeciliaSolver, ShallowWaterParameters
from .traffic import (
    TrafficAssignmentSimulator,
    TrafficAssignmentParameters,
)
from .datasets import (
    UrbanFloodSimulationArgs,
    UrbanFloodSimulationDataset,
    UrbanNoiseSimulationArgs,
    UrbanNoiseSimulationDataset,
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
    from .heat import (
        HeatSolver,
        HeatParameters,
        BndCat,
        DirichletBCSpec,
        NeumannBCSpec,
        RobinBCSpec,
    )
    from .smooth_reconstruction import (
        SmoothReconstructionSimulator,
        SmoothReconstructionParameters,
    )
    from .stokes import StokesSolver, StokesParameters
    from .navierstokes import NavierStokesSolver, NavierStokesParameters
    from ._flow.solver import BndCat as WindBndCat

    # Default FEniCSx log level (override with DTCC_FENICSX_LOG_LEVEL).
    # Note: FEniCSx backend messages use FEniCSx-native formatting.
    from .fenics import set_log_level, DEBUG, INFO, WARNING, ERROR

    _fenicsx_log_name = os.getenv("DTCC_FENICSX_LOG_LEVEL", "WARNING").strip().upper()
    _fenicsx_log_levels = {
        "DEBUG": DEBUG,
        "INFO": INFO,
        "WARNING": WARNING,
        "ERROR": ERROR,
    }
    if _fenicsx_log_name not in _fenicsx_log_levels:
        raise ValueError("DTCC_FENICSX_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
    set_log_level(_fenicsx_log_levels[_fenicsx_log_name])
    _fenics_exports = [
        "HeatSolver",
        "HeatParameters",
        "BndCat",
        "DirichletBCSpec",
        "NeumannBCSpec",
        "RobinBCSpec",
        "SmoothReconstructionSimulator",
        "SmoothReconstructionParameters",
        "StokesSolver",
        "StokesParameters",
        "NavierStokesSolver",
        "NavierStokesParameters",
        "WindBndCat",
    ]
except ModuleNotFoundError as exc:
    if exc.name != "dolfinx":
        raise

__all__ = [
    "CityMeshParameters", "CityMeshCache", "CityMeshSettings",
    "__version__",
    "HelmholtzSolver",
    "HelmholtzParameters",
    "CeciliaSolver",
    "ShallowWaterParameters",
    "TrafficAssignmentSimulator",
    "TrafficAssignmentParameters",
    "UrbanFloodSimulationArgs",
    "UrbanFloodSimulationDataset",
    "UrbanNoiseSimulationArgs",
    "UrbanNoiseSimulationDataset",
    "UrbanHeatSimulationArgs",
    "UrbanHeatSimulationDataset",
    "AirQualityFieldArgs",
    "AirQualityFieldDataset",
    "TrafficSimulationArgs",
    "TrafficSimulationDataset",
    "UrbanWindSimulationArgs",
    "UrbanWindSimulationDataset",
] + _fenics_exports
