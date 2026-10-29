"""Stationary urban cooling with 5 °C ambient air and 0 °C ground."""

from dtcc_sim import HeatParameters, HeatSolver

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

params = HeatParameters(
    kappa=1.0,
    sigma=0.05,
    T_ambient=5.0,  # Cold ambient
    wall_value=10.0,  # Buildings warmer than ambient
    roof_value=10.0,
    ground_value=0.0,  # Cold ground
    open_value=5.0,
    wall_bc_type="robin",
    roof_bc_type="robin",
    ground_bc_type="dirichlet",
    open_bc_type="dirichlet",
    mesh_max_mesh_size=12.5,
    mesh_lod="lod2", mesh_cache_dir="output/city-mesh-cache",
    mesh_domain_height=80.0,
)

solver = HeatSolver(bounds=bounds, params=params, output_dir="output/urban_heat_cooling")
result = solver.simulate()
result.info()
result.plot(field="temperature")
