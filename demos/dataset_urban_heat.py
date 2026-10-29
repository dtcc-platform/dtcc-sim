"""Stationary urban temperature through the dataset API."""

import dtcc_core.datasets as datasets
import dtcc_sim.datasets

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

result = datasets.urban_heat_simulation(
    bounds=bounds,
    output_dir="output/dataset_urban_heat",
    wall_value=25.0,
    ground_value=20.0,
    open_value=22.0,
    T_ambient=22.0,
    mesh_max_mesh_size=12.5,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
    mesh_domain_height=80.0,
)
result.info()
result.plot(field="temperature")
