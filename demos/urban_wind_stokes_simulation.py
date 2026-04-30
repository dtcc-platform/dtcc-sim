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
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

# Center coordinates (Poseidon statue in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755

# Meshing parameters
H = 80.0   # domain height
L = 100.0  # domain size (use 500m to get same domain as in dtcc-core demo build_meshes.py)
h = 10.0   # max mesh size (needs to be smaller than 25m to get a sensible solution)
d = 1.0    # min building detail

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Simulation parameters
params = UrbanWindParameters(
    equations="stokes",
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
sim = UrbanWindSimulator(bounds=bounds, params=params)
sim.simulate(output_path="output/urban_wind_stokes_simulation.xdmf")
