"""Stationary urban heatwave with hot surfaces and 20 °C ambient air."""

from dtcc_sim import HeatParameters, HeatSolver

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

params = HeatParameters(
    kappa=2.0,  # Higher thermal diffusivity (enhanced mixing)
    sigma=0.1,  # Relaxation to ambient air
    T_ambient=20.0,  # Ambient temperature
    wall_value=30.0,  # Hot building surfaces
    wall_h=5.0,  # Convective heat transfer coefficient
    roof_value=30.0,  # Hot roofs (solar heated)
    roof_h=5.0,
    ground_value=25.0,  # Warm ground
    ground_h=2.0,
    open_value=20.0,  # Far-field ambient temperature
    wall_bc_type="robin",
    roof_bc_type="robin",
    ground_bc_type="robin",
    open_bc_type="dirichlet",
    mesh_max_mesh_size=12.5,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
    mesh_domain_height=80.0,
)

solver = HeatSolver(bounds=bounds, params=params, output_dir="output/urban_heat_heatwave")
result = solver.simulate()
result.info()
result.plot(field="temperature")
