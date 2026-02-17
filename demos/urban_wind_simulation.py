"""
Urban wind simulation — real-world city area
=============================================

Runs an urban wind CFD simulation on a real-world area in Gothenburg,
Sweden.  Uses dtcc-core to build a 3D city volume mesh from geographic
data and solves the incompressible Navier–Stokes equations with the
IPCS fractional-step scheme.  The solution is saved as XDMF for
inspection in ParaView.
"""

import os
from pathlib import Path

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
    # Solver
    dt=0.5,
    max_steps=500,
    steady_tolerance=1e-4,
    min_steps=50,
    convection_linearization="picard",
    # BCs
    wall_model="noslip",
    inlet_profile="log_law",
    z0=0.5,
    u_ref_height=10.0,
)

sim = UrbanWindSimulator(bounds=bounds, params=params)
result = sim.simulate(output_path=str(output_dir / "urban_wind_simulation.xdmf"))

print(f"\nDone — output saved to:")
print(f"  {output_dir / 'urban_wind_simulation_velocity.xdmf'}")
print(f"  {output_dir / 'urban_wind_simulation_pressure.xdmf'}")
print(f"Open in ParaView to inspect velocity and pressure fields.")
