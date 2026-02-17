"""
Lid-driven cavity flow (3-D)
=============================

Classic benchmark: a unit cube [0,1]³ with no-slip walls on all faces
except the top (z=1), which moves at constant tangential velocity in the
+x direction (the "lid").

The solver is set up by using the UrbanWindSimulator in direct-mesh
mode.  The lid velocity is imposed by using the *inlet_marker* override
to mark the top face as the inlet, and setting wind direction so that
the inlet velocity points in the +x direction.  The outlet is placed
on one of the side faces (x-max) with zero-pressure outflow.

This is a well-known recirculating flow — you should see a large
primary vortex in the xz mid-plane when viewed in ParaView.

Output is saved as XDMF.
"""

import numpy as np
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# ---- build mesh and markers ----
import dolfinx.mesh
from dolfinx.mesh import exterior_facet_indices
from mpi4py import MPI

N = 10  # cells per direction
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    [N, N, N],
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
        markers_vals[i] = -3        # x-min
    elif abs(x - 1.0) < tol:
        markers_vals[i] = -4        # x-max  → outlet
    elif abs(y) < tol:
        markers_vals[i] = -5        # y-min
    elif abs(y - 1.0) < tol:
        markers_vals[i] = -6        # y-max
    elif abs(z) < tol:
        markers_vals[i] = -1        # ground
    elif abs(z - 1.0) < tol:
        markers_vals[i] = -2        # top  → lid (inlet)

order = np.argsort(boundary_facets)
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, boundary_facets[order], markers_vals[order]
)

# ---- solve ----
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

# Override markers: top=-2 is the lid (inlet), x-max=-4 is the outlet.
# Wind direction 270° (from west) → flow in +x → lid slides in +x.
params = UrbanWindParameters(
    wind_speed=1.0,
    wind_dir_deg=270.0,       # flow in +x direction
    inlet_marker=-2,          # top face is the lid
    outlet_marker=-4,         # x-max face is the outlet
    nu_t=0.01,                # higher eddy viscosity for stability (Re ≈ 100)
    dt=0.02,
    max_steps=1000,
    min_steps=100,
    steady_tolerance=1e-5,
    velocity_degree=2,
    pressure_degree=1,
    wall_model="noslip",
    inlet_profile="uniform",
)

sim = UrbanWindSimulator(mesh=mesh, markers=facet_tags, params=params)
u, p = sim.simulate(output_path=str(output_dir / "driven_cavity.xdmf"))

# ---- quick validation ----
u_max = np.max(np.abs(u.x.array))
print(f"\nDriven cavity converged.")
print(f"  max |u| = {u_max:.4f}")
print(f"  u finite: {np.all(np.isfinite(u.x.array))}")
print(f"  p finite: {np.all(np.isfinite(p.x.array))}")
print(f"  Output saved to {output_dir / 'driven_cavity.xdmf'}")
