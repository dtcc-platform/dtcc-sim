"""
Urban wind simulation — real-world city area
=============================================

Runs an urban wind CFD simulation on a real-world area in Gothenburg,
Sweden.  Uses dtcc-core to build a 3D city volume mesh from geographic
data and solves the incompressible Navier–Stokes equations with the
IPCS fractional-step scheme.  The solution is saved as XDMF for
inspection in ParaView.
"""

from pathlib import Path
from mpi4py import MPI

import dtcc_core.datasets as datasets
from dtcc_sim import NavierStokesSolver, NavierStokesParameters

# Output directory
output_dir = Path("output/urban_wind_simulation")

# Poseidon in Gothenburg (EPSG:3006)
center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

# Simulation parameters
params = NavierStokesParameters(
    mesh_max_mesh_size=12.5,
    wind_speed=5.0,
    wind_dir_deg=270.0,
    nu_t=2.0,
    side_top_boundary="slip",
    inlet_profile="log_law",
    dt=0.1,
    adaptive_dt=True,
    dt_max=0.1,
    max_steps=500,
    simulation_mode="statistical_steady",
    stat_warmup_steps=60,
    stat_window=20,
    stat_tolerance=2.0e-2,
    stat_divergence_tolerance=1.0e-1,
    stat_flux_imbalance_tolerance=1.0e-1,
    velocity_relaxation=0.3,
    spike_rel_threshold=0.95,
    spike_relaxation=0.2,
    spike_min_step=120,
    grad_div_gamma=3.0,
    backflow_beta=2.0,
    inlet_ramp_time=20.0,
    log_every_steps=1,
    log_initial_steps=0,
)

# Surface mesh for visualization
if MPI.COMM_WORLD.rank == 0:
    surface_mesh = datasets.city_surface_mesh(
        bounds=bounds,
        max_mesh_size=params.mesh_max_mesh_size,
        raster_cell_size=params.mesh_raster_cell_size,
        raster_radius=params.mesh_raster_radius,
    )
    surface_mesh.offset_to_origin()
    surface_mesh_path = output_dir / "urban_wind_simulation_surface_mesh.vtu"
    surface_mesh.save(surface_mesh_path)

# Run wind simulation
sim = NavierStokesSolver(bounds=bounds, params=params, output_dir="output/urban_wind_simulation")
sim.simulate()
