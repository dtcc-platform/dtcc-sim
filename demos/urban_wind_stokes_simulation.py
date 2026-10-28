"""
Urban wind simulation (Stokes) — real-world city area
======================================================

Runs an urban flow simulation on a real-world area in Gothenburg, Sweden.
Uses dtcc-core to build a 3D city volume mesh and solves the stationary
incompressible Stokes equations. The solution is saved as XDMF for
inspection in ParaView.
"""

from mpi4py import MPI

import dtcc_core as dtcc
from dtcc_sim import StokesSolver, StokesParameters

# Center coordinates (Poseidon statue in Gothenburg)
center_x = 319995.962899
center_y = 6399009.716755

# Meshing parameters
H = 80.0   # domain height
L = 200.0  # horizontal domain size in meters
h = 12.5   # target maximum mesh edge length in meters
d = 1.0    # min building detail

# Define bounds
bounds = dtcc.Bounds(center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

# Simulation parameters
params = StokesParameters(
    wind_speed=5.0,
    wind_dir_deg=270.0,
    mesh_max_mesh_size=h,
    mesh_domain_height=H,
    mesh_min_building_detail=d,
    nu_t=2.0,
    side_top_boundary="slip",
    inlet_profile="log_law",
    log_every_steps=1,
    log_initial_steps=0,
)

# Surface mesh for visualization
if MPI.COMM_WORLD.rank == 0:
    surface_mesh = dtcc.datasets.city_surface_mesh(
        bounds=bounds,
        max_mesh_size=h,
        min_building_detail=d,
    )
    surface_mesh.offset_to_origin()
    surface_mesh.save("output/urban_wind_stokes_simulation_surface_mesh.vtu")

# Run flow simulation
sim = StokesSolver(bounds=bounds, params=params, output_dir="output/urban_wind_stokes_simulation")
sim.simulate()
