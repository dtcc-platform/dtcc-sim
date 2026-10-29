"""Reconstruct a temperature field from synthetic observations over Poseidon."""

import numpy as np
from dtcc_sim import SmoothReconstructionParameters, SmoothReconstructionSimulator

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)
n_observations = 10
terrain_elevation = 30.0
rng = np.random.RandomState(42)

point_coords = np.zeros((n_observations, 3))
point_coords[:, 0] = center_x + (rng.rand(n_observations) - 0.5) * (0.8 * L)
point_coords[:, 1] = center_y + (rng.rand(n_observations) - 0.5) * (0.8 * L)
point_coords[:, 2] = terrain_elevation + 5.0 + rng.rand(n_observations) * 10.0
point_values = 20.0 + 5.0 * np.sin((point_coords[:, 0] - center_x) / L * np.pi) * np.cos(
    (point_coords[:, 1] - center_y) / L * np.pi
) + rng.randn(n_observations) * 0.5
params = SmoothReconstructionParameters(
    lambda_smooth=1.0, alpha=1e-3, data_weight=100.0, background_value=20.0,
    mesh_max_mesh_size=12.5, mesh_domain_height=80.0,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
)
solver = SmoothReconstructionSimulator(
    bounds=bounds, params=params, point_coords=point_coords, point_values=point_values,
    field_name="temperature", field_unit="°C", output_dir="output/smooth_reconstruction",
)
result = solver.simulate()
result.info()
result.plot(field="temperature")
