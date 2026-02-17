"""
Bluff body flow
===============

Flow past a rectangular obstruction in a wind tunnel.  The domain is
[0,6]×[0,2]×[0,2] with a solid cube [2,3]×[0.5,1.5]×[0,1] sitting
on the ground.  Uniform inflow from x-min, zero-pressure outflow at
x-max, no-slip on the ground and the obstacle surfaces, and slip
(zero-stress) on the remaining outer walls.

The mesh is built with Gmsh (boolean cut) and converted to dolfinx.
Boundary markers are assigned geometrically using midpoint positions,
matching the dtcc convention: wall faces get marker 0 (building wall),
roof face gets marker 1 (building roof), and bounding-box faces get
negative markers (-1 ground, -2 top, -3 xmin, -4 xmax, -5 ymin,
-6 ymax).

Output is saved as XDMF for inspection in ParaView — you should see
a separation bubble and wake behind the obstacle.
"""

import numpy as np
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# ---- build mesh with Gmsh ----
try:
    import gmsh
except ImportError:
    raise ImportError(
        "This demo requires Gmsh (pip install gmsh).  "
        "See https://gmsh.info for details."
    )

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)
gmsh.model.add("bluff_body")

# Domain dimensions
Lx, Ly, Lz = 6.0, 2.0, 2.0

# Obstacle (cube sitting on the ground)
ox0, oy0, oz0 = 2.0, 0.5, 0.0
ox1, oy1, oz1 = 3.0, 1.5, 1.0

# Create outer box
outer = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
# Create obstacle box
obstacle = gmsh.model.occ.addBox(ox0, oy0, oz0, ox1 - ox0, oy1 - oy0, oz1 - oz0)

# Boolean cut: domain = outer - obstacle
gmsh.model.occ.cut([(3, outer)], [(3, obstacle)])
gmsh.model.occ.synchronize()

# Volume physical group (required for 3D meshing)
volumes = gmsh.model.getEntities(dim=3)
vol_tags = [v[1] for v in volumes]
gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name="fluid")

# We also need a physical group for ALL surfaces so they appear in the mesh
surfaces = gmsh.model.getEntities(dim=2)
surf_tags = [s[1] for s in surfaces]
gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="boundary")

# Mesh size
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
gmsh.model.mesh.generate(3)

# ---- convert to dolfinx ----
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

mesh, cell_tags, raw_facet_tags = model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
)
gmsh.finalize()

# ---- assign boundary markers geometrically ----
# Classify each boundary facet by its midpoint position.
# dtcc convention: -3=xmin, -4=xmax, -5=ymin, -6=ymax, -1=ground, -2=top
# Building:  0=wall, 1=roof  (1 building → N=1)
fdim = mesh.topology.dim - 1
mesh.topology.create_entities(fdim)
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

from dolfinx.mesh import exterior_facet_indices
import dolfinx.mesh as dmesh

boundary_facets = exterior_facet_indices(mesh.topology)
midpoints = dmesh.compute_midpoints(mesh, fdim, boundary_facets)
markers_vals = np.full(len(boundary_facets), -7, dtype=np.int32)

tol = 1e-6
for i, mp in enumerate(midpoints):
    x, y, z = mp

    # Check if this midpoint is on the obstacle surface
    on_obs = (
        ox0 - tol <= x <= ox1 + tol
        and oy0 - tol <= y <= oy1 + tol
        and oz0 - tol <= z <= oz1 + tol
    )

    if on_obs:
        # Obstacle roof (top face of obstacle at z ≈ oz1)
        if abs(z - oz1) < tol:
            markers_vals[i] = 1  # building roof (N=1 → roof marker = 1)
        else:
            markers_vals[i] = 0  # building wall (marker = 0)
    elif abs(x) < tol:
        markers_vals[i] = -3  # x-min (inlet)
    elif abs(x - Lx) < tol:
        markers_vals[i] = -4  # x-max (outlet)
    elif abs(y) < tol:
        markers_vals[i] = -5  # y-min (side)
    elif abs(y - Ly) < tol:
        markers_vals[i] = -6  # y-max (side)
    elif abs(z) < tol:
        markers_vals[i] = -1  # ground
    elif abs(z - Lz) < tol:
        markers_vals[i] = -2  # top

order = np.argsort(boundary_facets)
facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], markers_vals[order])

n_wall = np.sum(markers_vals == 0)
n_roof = np.sum(markers_vals == 1)
print(f"Obstacle facets: {n_wall} wall, {n_roof} roof")

# ---- solve ----
from dtcc_sim import UrbanWindSimulator, UrbanWindParameters

params = UrbanWindParameters(
    wind_speed=1.0,
    wind_dir_deg=270.0,  # from west → flow in +x
    nu_t=0.5,  # high eddy viscosity for stable low-Re flow
    dt=0.01,
    max_steps=500,
    min_steps=50,
    steady_tolerance=1e-5,
    velocity_degree=2,
    pressure_degree=1,
    wall_model="noslip",
    inlet_profile="uniform",
)

sim = UrbanWindSimulator(mesh=mesh, markers=facet_tags, params=params)
u, p = sim.simulate(output_path=str(output_dir / "bluff_body.xdmf"))

# ---- quick validation ----
u_max = np.max(np.abs(u.x.array))
print(f"\nBluff body flow converged.")
print(f"  max |u| = {u_max:.4f}")
print(f"  u finite: {np.all(np.isfinite(u.x.array))}")
print(f"  p finite: {np.all(np.isfinite(p.x.array))}")
print(f"  Output saved to {output_dir / 'bluff_body_velocity.xdmf'}")
print(f"                   and {output_dir / 'bluff_body_pressure.xdmf'}")
