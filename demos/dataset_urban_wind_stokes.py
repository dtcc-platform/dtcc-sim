"""Stationary Stokes wind fields through the dataset API."""

import dtcc_core.datasets as datasets
import dtcc_sim.datasets

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x-L/2, center_y-L/2, center_x+L/2, center_y+L/2)
result = datasets.urban_wind_simulation(
    bounds=bounds, equations="stokes", wind_speed=5., nu_t=2.,
    wind_dir_deg=270., inlet_profile="log_law", side_top_boundary="slip",
    mesh_max_mesh_size=12.5, mesh_domain_height=80.,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
    output_dir="output/dataset_urban_wind_stokes",
)
result.info()
result.plot(field="speed")
