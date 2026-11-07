"""Experimental shallow-water equation solver on triangular DTCC meshes."""

from .parameters import ShallowWaterParameters
from .solver import ShallowWaterSolver
from .results import ShallowWaterResult

__all__ = ["ShallowWaterParameters", "ShallowWaterSolver", "ShallowWaterResult"]
