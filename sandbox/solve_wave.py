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
# ------------------------------------------------------------
ds_absorb = NeumannBC(mesh, markers=markers, marker_value=[-2, -3, -4, -5, -6])

# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------
u_0 = Function(V)  # u^{n-1}
u_1 = Function(V)  # u^{n}
u_2 = Function(V)  # u^{n+1}

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

a = u * v * dx + alpha * inner(grad(u), grad(v)) * dx + beta * u * v * ds_absorb
L = (2 * u_1 - u_0) * v * dx - alpha * inner(grad(4 * u_1 + u_0), grad(v)) * dx
L += gamma * f * v * dx + beta * u_1 * v * ds_absorb

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

problem = LinearProblem(a, L, u=u_2, bcs=[], petsc_options=opts)

# -------------------------------------------------------------
# Time-stepping loop
# -------------------------------------------------------------
u_2.save("output/timeseries_wave.xdmf", t=0.0)

t = 0.0
for n in range(num_steps):
    t += dt
    info(f"t = {t}: ||x|| = {np.linalg.norm(u_2.x.array[:])}")

    _t.value = t
    problem.solve()

    u_0.x.array[:] = u_1.x.array
    u_1.x.array[:] = u_2.x.array

    if n % skip == 0:
        u_2.save("output/timeseries_wave.xdmf", t=t)

u_2.save("output/solution_wave.xdmf")
