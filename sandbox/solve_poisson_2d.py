from fenics import *

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
mesh, cell_tags = load_mesh_with_markers("output/flat_mesh_gbg.xdmf")

# -----------------------------------------------------------
# Convert cell markers to facet (edge) markers
# -----------------------------------------------------------
markers = cell_markers_to_facet_markers(mesh, cell_tags)

# -----------------------------------------------------------
# Get the number of buildings from the boundary markers
# -----------------------------------------------------------
max_marker = int(markers.values.max())
num_buildings = max_marker + 1
info(f"Number of buildings: {num_buildings}")

# ------------------------------------------------------------
# Function space
# -----------------------------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)

# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
bcs = []

# Boundary conditions on the buildings (markers 0, 1, 2, ...)
for i in range(num_buildings):
    bc = DirichletBC(V, 1.0, markers=markers, marker_value=i)
    bcs.append(bc)

# Boundary condition on the ground (marker -2)
bc_ground = DirichletBC(V, 0.0, markers=markers, marker_value=-2)
bcs.append(bc_ground)

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(mesh, 0.0)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# ------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------
opts = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
u = solve(a == L, bcs=bcs, petsc_options=opts)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
u.save("output/solution_poisson_2d.xdmf")
