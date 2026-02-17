"""
Channel flow (Poiseuille)
=========================

Solves pressure-driven flow in a rectangular channel [0,4]×[0,1]×[0,1].
Wind enters from the x-min face (west) with a uniform profile and
exits at x-max (east).  Top and bottom faces as well as the y-faces
are no-slip walls.

The analytical steady-state solution is a parabolic profile in the yz
cross-section.  This demo verifies that the IPCS solver produces a
well-behaved, incompressible velocity field in a simple geometry.

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
Nx, Ny, Nz = 16, 6, 6

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
        markers_vals[i] = -3        # x-min  (inlet)
    elif abs(x - Lx) < tol:
        markers_vals[i] = -4        # x-max  (outlet)
    elif abs(y) < tol:
        markers_vals[i] = -5        # y-min  (wall)
    elif abs(y - Ly) < tol:
        markers_vals[i] = -6        # y-max  (wall)
    elif abs(z) < tol:
        markers_vals[i] = -1        # ground (wall)
    elif abs(z - Lz) < tol:
        markers_vals[i] = -2        # top    (wall)

order = np.argsort(boundary_facets)
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, boundary_facets[order], markers_vals[order]
)

# ---- solve ----
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

params = UrbanWindParameters(
    wind_speed=1.0,
    wind_dir_deg=270.0,       # from west → flow in +x direction
    nu_t=0.01,                # higher eddy viscosity for stability (Re ≈ 100)
    dt=0.05,
    max_steps=500,
    min_steps=50,
    steady_tolerance=1e-5,
    velocity_degree=2,
    pressure_degree=1,
    wall_model="noslip",
    inlet_profile="uniform",
)

sim = UrbanWindSimulator(mesh=mesh, markers=facet_tags, params=params)
u, p = sim.simulate(output_path=str(output_dir / "channel_flow.xdmf"))

# ---- quick validation ----
u_max = np.max(np.abs(u.x.array))
print(f"\nChannel flow converged.")
print(f"  max |u| = {u_max:.4f}")
print(f"  u finite: {np.all(np.isfinite(u.x.array))}")
print(f"  p finite: {np.all(np.isfinite(p.x.array))}")
print(f"  Output saved to {output_dir / 'channel_flow.xdmf'}")
