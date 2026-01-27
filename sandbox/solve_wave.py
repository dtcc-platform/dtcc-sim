from fenics import *

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Problem parameters
# ------------------------------------------------------------
c = 343.0  # speed of sound (m/s)
T = 1.0  # final time (s)
skip = 10  # skip time steps when saving

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
mesh, markers = load_mesh_with_markers("output/volume_mesh_gbg.xdmf")
xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)

# ------------------------------------------------------------
# Set time step based on CFL condition
# ------------------------------------------------------------
h = mesh.hmin()
info(f"hmin = {h :.3g}")
dt = 0.1 * h / c  # 0.2 and above blows up
num_steps = round(T / dt + 0.5)
dt = T / num_steps
info(f"Using dt = {dt :.3g} and {num_steps} time steps based on CFL condition")

# ------------------------------------------------------------
# Function space
# ------------------------------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)

# ------------------------------------------------------------
# Source term
# ------------------------------------------------------------
A = 100.0  # amplitude of source
sigma = 5.0  # spatial extent (m)
tau = 0.05  # temporal extent (s)
x0 = np.array((0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.9 * zmin + 0.1 * zmax))
t0 = 0.1
_x = SpatialCoordinate(mesh)
_t = Constant(mesh, 0.0)
r2 = sum((_x[i] - x0[i]) ** 2 for i in range(3))
t2 = (_t - t0) ** 2
f = A * exp(-r2 / (2 * sigma**2)) * exp(-t2 / (2 * tau**2))

# ------------------------------------------------------------
# Boundary conditions
#
# We use rigid boundary conditions on the ground and buildings
# which is implemented as homogeneous Neumann = do nothing.
#
# We use absorbing boundary conditions on the domain bounding box
# which is implemented as ∂_n u = - (1/c) ∂_t u (Sommerfeld).
# ------------------------------------------------------------
ds = NeumannBC(mesh, markers=markers, marker_value=[-2, -3, -4, -5, -6])

# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

# Create functions for solution at three time levels
u_0 = Function(V)  # u at time step n-1
u_1 = Function(V)  # u at time step n
u_2 = Function(V)  # u at time step n+1 (unknown to solve for)

# Initial conditions (u=0, velocity=0)
u_0.x.array[:] = 0.0
u_1.x.array[:] = 0.0

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

alpha = Constant(mesh, c**2 * dt**2 / 6)
beta = Constant(mesh, c * dt)
gamma = Constant(mesh, dt**2)

a = u * v * dx + alpha * inner(grad(u), grad(v)) * dx + beta * u * v * ds
L = (2 * u_1 - u_0) * v * dx - alpha * inner(grad(4 * u_1 + u_0), grad(v)) * dx
L += gamma * f * v * dx + beta * u_1 * v * ds

# ------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------
direct = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

opts = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

problem = LinearProblem(a, L, u=u_2, bcs=[], petsc_options=opts)

# -------------------------------------------------------------
# Time-stepping loop
# -------------------------------------------------------------

# Save initial solution
u_2.save("output/timeseries_wave.xdmf", t=0.0)

t = 0.0
for n in range(num_steps):
    t += dt
    info(f"t = {t}: ||x|| = {np.linalg.norm(u_2.x.array[:])}")

    # Update time for right-hand side expression
    _t.value = t

    # Solve for u_2
    problem.solve()

    # Shift solutions for next time step
    u_0.x.array[:] = u_1.x.array
    u_1.x.array[:] = u_2.x.array

    # Save solution
    if n % skip == 0:
        u_2.save("output/timeseries_wave.xdmf", t=t)

# ------------------------------------------------------------
# Post-processing & output
# ------------------------------------------------------------

# Save final solution
u_2.save("output/solution_wave.xdmf")
