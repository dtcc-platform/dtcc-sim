"""Simulate wave amplitude around city buildings through the noise dataset API."""

import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim.helmholtz import HelmholtzParameters

# Chalmers Johanneberg in Gothenburg (EPSG:3006); 250 m × 250 m domain
center_x = 319970.0
center_y = 6397910.0
L = 250.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)
pointcloud = dtcc.io.data.download_pointcloud(bounds=dtcc.Bounds(*bounds))
terrain = dtcc.builder.build_terrain_raster(pointcloud, cell_size=2.0, radius=3.0)
source_z = terrain.get_value(center_x, center_y) + 2.0
output_dir = "output/dataset_urban_noise"

noise = dtcc.datasets.urban_noise_simulation(
    bounds=bounds, h=12.5, domain_height=80, min_building_detail=.5,
    mesh_options={"lod": "lod2", "flat_ground": False, "repair_boundary_corners": True},
    mesh_cache_dir="output/city-mesh-cache",
    output_dir=output_dir,
    parameters=HelmholtzParameters(
        source_location_mode="manual", source_x=center_x, source_y=center_y,
        source_z=source_z,
        frequency_hz=5, rho=1.10, n_subdomains=2, workers=1,
        overlap_layers=1, oversample_layers=1, threads_per_worker=1,
        pou_mode="geometric_hats",
    ),
)
noise.info()
noise.plot(field="helmholtz_amplitude")
