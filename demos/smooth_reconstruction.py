"""Reconstruct a temperature field from synthetic observations over Chalmers Johanneberg."""

import numpy as np
import dtcc_core as dtcc
from dtcc_sim import SmoothReconstructionParameters, SmoothReconstructionSimulator

center_x = 319970.0
center_y = 6397910.0
L = 250.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)
n_observations = 10
pointcloud = dtcc.io.data.download_pointcloud(bounds=dtcc.Bounds(*bounds))
terrain = dtcc.builder.build_terrain_raster(pointcloud, cell_size=2.0, radius=3.0)
rng = np.random.RandomState(42)

point_coords = np.zeros((n_observations, 3))
point_coords[:, 0] = center_x + (rng.rand(n_observations) - 0.5) * (0.8 * L)
point_coords[:, 1] = center_y + (rng.rand(n_observations) - 0.5) * (0.8 * L)
ground = np.array([terrain.get_value(x, y) for x, y in point_coords[:, :2]])
point_coords[:, 2] = ground + 5.0 + rng.rand(n_observations) * 10.0
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
