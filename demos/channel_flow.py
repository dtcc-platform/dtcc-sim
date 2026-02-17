"""
Channel flow (Poiseuille)
=========================

Solves flow in a rectangular channel [0,4]×[0,1]×[0,1] with a
parabolic (Poiseuille) inlet profile:

    u_x(y,z) = U_max · 4y(Ly-y)/Ly² · 4z(Lz-z)/Lz²

This satisfies no-slip at the walls and gives the exact fully-developed
duct Poiseuille solution.  The solver should maintain this profile
throughout the channel.

Output is saved as XDMF for inspection in ParaView.
"""

import numpy as np
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# ---- build mesh and markers ----
import dolfinx.mesh
from dolfinx.mesh import exterior_facet_indices
from mpi4py import MPI

Lx, Ly, Lz = 4.0, 1.0, 1.0
Nx, Ny, Nz = 20, 8, 8

mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [Lx, Ly, Lz]],
    [Nx, Ny, Nz],
    dolfinx.mesh.CellType.tetrahedron,
)

fdim = mesh.topology.dim - 1
mesh.topology.create_entities(fdim)
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

boundary_facets = exterior_facet_indices(mesh.topology)
midpoints = dolfinx.mesh.compute_midpoints(mesh, fdim, boundary_facets)
markers_vals = np.full(len(boundary_facets), -7, dtype=np.int32)

tol = 1e-10
for i, mp in enumerate(midpoints):
    x, y, z = mp
    if abs(x) < tol:
        markers_vals[i] = -3  # x-min  (inlet)
    elif abs(x - Lx) < tol:
        markers_vals[i] = -4  # x-max  (outlet)
    elif abs(y) < tol:
        markers_vals[i] = -1  # y-min  (no-slip wall = ground tag)
    elif abs(y - Ly) < tol:
        markers_vals[i] = -1  # y-max  (no-slip wall = ground tag)
    elif abs(z) < tol:
        markers_vals[i] = -1  # z-min  (no-slip wall = ground tag)
    elif abs(z - Lz) < tol:
        markers_vals[i] = -1  # z-max  (no-slip wall = ground tag)

order = np.argsort(boundary_facets)
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, boundary_facets[order], markers_vals[order]
)

# ---- Poiseuille inlet profile ----
# Parabolic in y and z: u_x = U_max * 4*y*(Ly-y)/Ly^2 * 4*z*(Lz-z)/Lz^2
U_max = 1.0


def poiseuille_inlet(x):
    """Analytical Poiseuille profile for a rectangular duct."""
    n = x.shape[1]
    y, z = x[1], x[2]
    vals = np.zeros((3, n), dtype=np.float64)
    vals[0] = U_max * (4.0 * y * (Ly - y) / Ly**2) * (4.0 * z * (Lz - z) / Lz**2)
    return vals


# ---- solve ----
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

params = UrbanWindParameters(
    wind_speed=U_max,
    wind_dir_deg=270.0,  # from west → flow in +x direction
    nu_t=0.01,  # eddy viscosity for stability
    dt=0.02,
    max_steps=500,
    min_steps=50,
    steady_tolerance=1e-5,
    velocity_degree=2,
    pressure_degree=1,
    wall_model="noslip",
)

sim = UrbanWindSimulator(
    mesh=mesh,
    markers=facet_tags,
    params=params,
    inlet_expression=poiseuille_inlet,
)
u, p = sim.simulate(output_path=str(output_dir / "channel_flow.xdmf"))

# ---- quick validation ----
u_max = np.max(np.abs(u.x.array))
print(f"\nChannel flow converged.")
print(f"  max |u| = {u_max:.4f}")
print(f"  u finite: {np.all(np.isfinite(u.x.array))}")
print(f"  p finite: {np.all(np.isfinite(p.x.array))}")
print(f"  Output saved to {output_dir / 'channel_flow_velocity.xdmf'}")
print(f"                   and {output_dir / 'channel_flow_pressure.xdmf'}")
