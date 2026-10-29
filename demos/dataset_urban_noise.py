"""Simulate wave amplitude around city buildings through the noise dataset API."""

import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim.helmholtz import HelmholtzParameters

# Poseidon in Gothenburg (EPSG:3006); 200 m × 200 m domain
center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)
output_dir = "output/dataset_urban_noise"

noise = dtcc.datasets.urban_noise_simulation(
    bounds=bounds, h=12.5, domain_height=80, min_building_detail=.5,
    mesh_options={"lod": "lod2", "flat_ground": False, "repair_boundary_corners": True},
    mesh_cache_dir="output/city-mesh-cache",
    output_dir=output_dir,
    parameters=HelmholtzParameters(
        frequency_hz=5, rho=1.10, n_subdomains=2, workers=1,
        overlap_layers=1, oversample_layers=1, threads_per_worker=1,
        pou_mode="geometric_hats",
    ),
)
noise.info()
noise.plot(field="helmholtz_amplitude")
