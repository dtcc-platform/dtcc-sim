"""Stationary cavity flow with a smooth tangential lid [m/s]."""

import numpy as np
from dtcc_core import BoxMesh
from dtcc_sim import NavierStokesParameters, NavierStokesSolver

mesh = BoxMesh(16, 16, 16)
mesh.boundary_markers[mesh.boundary_markers != -2] = -1


def lid(x):
    return np.array([1.6*x[0]*(1-x[0])*x[1]*(1-x[1]),
                     np.zeros(x.shape[1]), np.zeros(x.shape[1])])


params = NavierStokesParameters(nu=.1, wind_speed=.1, inlet_marker=-2,
                               closed_cavity=True, grad_div_gamma=1.)
solver = NavierStokesSolver(mesh=mesh, params=params, inlet_expression=lid,
                           output_dir="output/driven_cavity")
result = solver.simulate()
result.info()
result.plot(field="speed")
