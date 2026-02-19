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

# ---- output directory ----
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# ---- simulation ----
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

# Gothenburg city centre (SWEREF 99 TM)
x0 = 319_995.96
y0 = 6_399_009.72
L = 200  # side length [m]
bounds = (x0, y0, x0 + L, y0 + L)

params = UrbanWindParameters(
    # Wind from the west, 5 m/s
    wind_speed=5.0,
    wind_dir_deg=270.0,
    # Mesh
    mesh_max_mesh_size=25.0,
    mesh_domain_height=80.0,
    # Turbulence — eddy viscosity is essential for city-scale flow.
    # Without it the effective Re ~ 10^7 and the Galerkin
    # discretisation is unstable.  A simple constant ν_t ≈ 1 m²/s
    # gives Re_eff ~ 1000 which is well-resolved on this mesh.
    nu_t=2.0,
    # Solver
    dt=0.1,
    adaptive_dt=True,
    dt_min=5e-4,
    dt_max=0.1,
    cfl_reduce_safety=0.95,
    cfl_increase_factor=1.03,
    cfl_hard_limit=8.0,
    max_steps=500,
    simulation_mode="statistical_steady",
    steady_tolerance=1e-2,
    divergence_tolerance=2e-1,
    flux_imbalance_tolerance=5e-2,
    min_steps=30,
    stat_warmup_steps=120,
    stat_window=40,
    stat_tolerance=2e-2,
    stat_divergence_tolerance=1e-1,
    stat_flux_imbalance_tolerance=1e-1,
    convection_linearization="picard",
    velocity_relaxation=0.3,
    spike_rel_threshold=0.95,
    spike_relaxation=0.2,
    spike_min_step=120,
    grad_div_gamma=3.0,
    backflow_beta=2.0,
    cfl_target=2.0,
    petsc_velocity={
        "ksp_type": "gmres",
        "ksp_rtol": 1e-4,
        "ksp_max_it": 500,
        "ksp_gmres_restart": 150,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    },
    petsc_pressure={
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    },
    log_every_steps=1,
    log_initial_steps=0,
    # BCs
    wall_model="noslip",
    side_top_boundary="slip",
    inlet_profile="log_law",
    inlet_ramp_steps=120,
    inlet_ramp_time=40.0,
    z0=0.5,
    u_ref_height=10.0,
)

# ---- surface mesh for visualization ----
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
    print(f"Saved surface mesh to: {surface_mesh_path}")


# --- run wind simulation ---
sim = UrbanWindSimulator(bounds=bounds, params=params)
result = sim.simulate(output_path=str(output_dir / "urban_wind_simulation.xdmf"))

if MPI.COMM_WORLD.rank == 0:
    print(f"\nDone — output saved to:")
    print(f"  {output_dir / 'urban_wind_simulation_velocity.xdmf'}")
    print(f"  {output_dir / 'urban_wind_simulation_pressure.xdmf'}")
    print(f"Open in ParaView to inspect velocity and pressure fields.")
