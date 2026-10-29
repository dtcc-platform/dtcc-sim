"""Reconstruct Stockholm NO2 concentrations from available street observations."""

import dtcc_core.datasets as datasets
import dtcc_sim.datasets

center_x = 674295.0
center_y = 6581875.0
L = 750.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

result = datasets.air_quality_field(
    bounds=bounds,
    output_dir="output/dataset_air_quality_field",
    phenomenon="NO2",
    lambda_smooth=1.0,
    alpha=1e-3,
    data_weight=100.0,
    mesh_max_mesh_size=100.0,
    mesh_lod="lod1", mesh_cache_dir="output/city-mesh-cache",
    mesh_domain_height=80.0,
    airquality_timeout_s=30.0,
    airquality_max_stations=250,
)
result.info()
result.plot(field="NO2")
