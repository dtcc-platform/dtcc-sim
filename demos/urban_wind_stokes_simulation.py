"""
Urban wind simulation (Stokes) — real-world city area
======================================================

Runs an urban flow simulation on a real-world area in Gothenburg, Sweden.
Uses dtcc-core to build a 3D city volume mesh and solves the stationary
incompressible Stokes equations. The solution is saved as XDMF for
inspection in ParaView.
"""

from pathlib import Path
from mpi4py import MPI

import dtcc_core.datasets as datasets
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

# Output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Gothenburg city centre (SWEREF 99 TM)
x0 = 319_995.96
y0 = 6_399_009.72
L = 200.0
bounds = (x0, y0, x0 + L, y0 + L)

# Simulation parameters
params = UrbanWindParameters(
    equations="stokes",
    wind_speed=5.0,
    wind_dir_deg=270.0,
    nu_t=2.0,
    side_top_boundary="slip",
    inlet_profile="log_law",
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
    surface_mesh_path = output_dir / "urban_wind_stokes_simulation_surface_mesh.vtu"
    surface_mesh.save(surface_mesh_path)

# Run flow simulation
sim = UrbanWindSimulator(bounds=bounds, params=params)
sim.simulate(output_path=str(output_dir / "urban_wind_stokes_simulation.xdmf"))
