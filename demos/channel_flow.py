"""Stationary planar Poiseuille flow with a custom inlet [m/s]."""

import numpy as np
from dtcc_core import BoxMesh
from dtcc_sim import NavierStokesParameters, NavierStokesSolver

mesh = BoxMesh(16, 4, 4)
mesh.vertices[:, 0] *= 4.
mesh.boundary_markers[mesh.boundary_markers == -2] = -1


def inlet(x):
    return np.array([4*x[2]*(1-x[2]), np.zeros(x.shape[1]), np.zeros(x.shape[1])])


params = NavierStokesParameters(nu=.1, wind_speed=1., side_top_boundary="slip")
solver = NavierStokesSolver(mesh=mesh, params=params, inlet_expression=inlet,
                           output_dir="output/channel_flow")
result = solver.simulate()
result.info()
result.plot(field="speed")
