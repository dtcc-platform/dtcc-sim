"""Linear stationary Poseidon airflow with explicitly selected Stokes equations."""

from dtcc_sim import StokesParameters, StokesSolver

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x-L/2, center_y-L/2, center_x+L/2, center_y+L/2)
params = StokesParameters(
    mesh_max_mesh_size=12.5, mesh_domain_height=80., wind_speed=5., wind_dir_deg=270., nu_t=2.,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
    side_top_boundary="slip", inlet_profile="log_law",
)
solver = StokesSolver(bounds=bounds, params=params,
                      output_dir="output/urban_wind_stokes_simulation")
result = solver.simulate()
result.info()
result.plot(field="speed")
