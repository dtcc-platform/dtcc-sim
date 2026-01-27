from fenics import *
import numpy as np
import h5py

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Problem parameters
# ------------------------------------------------------------
kappa = 1.0  # molecular diffusivity (m²/s)
T = 10.0  # final time (s)
skip = 1  # skip time steps when saving
supg = False  # toggle streamline‑upwind term

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
mesh, markers = load_mesh_with_markers("output/volume_mesh_gbg.xdmf")
xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)

# -----------------------------------------------------------
# Get the number of buildings from the boundary markers
# -----------------------------------------------------------
max_marker = int(markers.values.max())
num_buildings = (max_marker + 1) // 2
info(f"Number of buildings: {num_buildings}")

# ------------------------------------------------------------
# Load velocity field from resampled CFD data
# ------------------------------------------------------------
info("Loading velocity field from resampled CFD data (direct HDF5)...")
W = FunctionSpace(mesh, "Lagrange", 1, dim=3)
u = Function(W)
with h5py.File("output/flowfield/velocity.h5", "r") as h5:
    vel = h5["/Function/f/0"][...]
u.x.array[0::3] = vel[:, 0]
u.x.array[1::3] = vel[:, 1]
u.x.array[2::3] = vel[:, 2]

# ------------------------------------------------------------
# Compute CFL speed
# ------------------------------------------------------------
U_max_local = float(np.linalg.norm(vel, axis=1).max())
U_max = MPI.COMM_WORLD.allreduce(U_max_local, op=MPI.MAX)
info(f"Maximum velocity magnitude: {U_max:.3f} m/s")

# ------------------------------------------------------------
# Set time step (CFL‑like heuristic)
# ------------------------------------------------------------
h = mesh.hmin()
info(f"hmin = {h :.3g}")
dt = 0.9 * h / (U_max + kappa / h)
num_steps = round(T / dt + 0.5)
dt = T / num_steps
info(f"Using dt = {dt :.3g} and {num_steps} time steps")

# ------------------------------------------------------------
# Function space
# ------------------------------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)

# ------------------------------------------------------------
# Source term
# ------------------------------------------------------------
A = 1.0  # amplitude
sigma = 5.0  # spatial extent (m)
x0 = np.array((0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.9 * zmin + 0.1 * zmax))
_t = Constant(mesh, 0.0)
_x = SpatialCoordinate(mesh)
r2 = sum((_x[i] - x0[i]) ** 2 for i in range(3))
f = A * exp(-r2 / (2 * sigma**2))

# ------------------------------------------------------------
# Boundary conditions
#
# Buildings & ground  → homogeneous Dirichlet (perfect sink)
# Everything else     → default Neumann (do nothing)
# ------------------------------------------------------------
c_D = 0.0
bcs = []
for i in range(num_buildings):
    bc_wall = DirichletBC(V, c_D, markers=markers, marker_value=i)
    bc_roof = DirichletBC(V, c_D, markers=markers, marker_value=2 * i)
    bcs.append(bc_wall)
    bcs.append(bc_roof)

# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

# Create functions for solution at two time levels
c_0 = Function(V)  # c at time step n-1
c_1 = Function(V)  # c at time step n

# Initial condition
c_0.x.array[:] = 0.0
c_1.x.array[:] = 0.0

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
c = TrialFunction(V)
v = TestFunction(V)

a = c / dt * v * dx + dot(u, grad(c)) * v * dx + kappa * inner(grad(c), grad(v)) * dx
L = c_0 / dt * v * dx + f * v * dx

# ------------------------------------------------------------
# SUPG stabilization
# ------------------------------------------------------------
if supg:

    h_K = CellDiameter(mesh)
    u_norm = sqrt(dot(u, u) + 1.0e-12)
    tau = 1.0 / (2 * u_norm / h_K + 4 * kappa / h_K**2)

    a += tau * dot(u, grad(v)) * (c / dt + dot(u, grad(c)) - kappa * div(grad(c))) * dx
    L += tau * dot(u, grad(v)) * (c_0 / dt + f) * dx

# ------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------
opts = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

problem = LinearProblem(a, L, u=c_1, bcs=bcs, petsc_options=opts)

# -------------------------------------------------------------
# Time‑stepping loop
# -------------------------------------------------------------
c_1.save("output/timeseries_advdiff.xdmf", t=0.0)

t = 0.0
for n in range(num_steps):
    t += dt
    info(f"t = {t :.3f}: ||c|| = {np.linalg.norm(c_1.x.array[:]):.3e}")

    # Update time for right-hand side expression
    _t.value = t

    # Solve for c_1
    problem.solve()

    # Shift solution for next time step
    c_0.x.array[:] = c_1.x.array

    # Save
    if n % skip == 0:
        c_1.save("output/timeseries_advdiff.xdmf", t=t)

# ------------------------------------------------------------
# Post‑processing & output
# ------------------------------------------------------------

# Save final solution
c_1.save("output/solution_advdiff.xdmf")

# Save the loaded velocity field for visualization
u.save("output/velocity_advdiff.xdmf")
