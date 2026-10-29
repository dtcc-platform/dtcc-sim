"""Stationary Poseidon airflow at explicitly specified effective viscosity."""

from dtcc_sim import NavierStokesParameters, NavierStokesSolver

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x-L/2, center_y-L/2, center_x+L/2, center_y+L/2)
params = NavierStokesParameters(
    mesh_max_mesh_size=12.5, mesh_domain_height=80., wind_speed=5., wind_dir_deg=270., nu_t=2.,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
    side_top_boundary="slip", inlet_profile="log_law", grad_div_gamma=3.,
    backflow_beta=2.,
)
solver = NavierStokesSolver(bounds=bounds, params=params,
                           output_dir="output/urban_wind_simulation")
result = solver.simulate()
result.info()
result.plot(field="speed")
