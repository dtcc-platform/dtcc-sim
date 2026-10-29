"""Stationary low-Reynolds-number flow around a box obstruction."""

from dtcc_sim import NavierStokesParameters, NavierStokesSolver
from dtcc_sim.navierstokes.domains import bluff_body_mesh

mesh = bluff_body_mesh()
params = NavierStokesParameters(nu=.5, wind_speed=.1,
                               inlet_profile="power_law", power_law_alpha=1.,
                               u_ref_height=2., side_top_boundary="slip",
                               mesh_refinement_levels=1,
                               petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                              "pc_factor_mat_solver_type": "mumps"})
solver = NavierStokesSolver(mesh=mesh, params=params, output_dir="output/bluff_body")
result = solver.simulate()
result.info()
result.plot(field="speed")
