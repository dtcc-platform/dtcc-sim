from fenics import *

# More verbose output
set_log_level(INFO)

# ------------------------------------------------------------
# Problem parameters
# ------------------------------------------------------------
c = 343.0  # speed of sound (m/s)
f = 1.0  # frequency (Hz)
k = 2.0 * np.pi * f / c

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
# mesh, markers = load_mesh_with_markers("output/volume_mesh_gbg.xdmf")
mesh, markers = load_mesh_with_markers("tmp/gbg_volume_mesh.xdmf")
xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)

# ------------------------------------------------------------
# Check if we resolve the wavelength
# ------------------------------------------------------------
h = mesh.hmin()
info(f"h = {h :.3g}")
info(f"kh = {k * h :.3g}")
if k * h > 0.9:
    error(f"Mesh too coarse for {f:.0f} Hz with P1 elements (k h = {k * h:.2f} > 0.9).")
    exit(1)

# ------------------------------------------------------------
# Function space: mixed (Re, Im) → R²
# -----------------------------------------------------------
W = FunctionSpace(mesh, (("Lagrange", 1), ("Lagrange", 1)))

# ------------------------------------------------------------
# Source term
# ------------------------------------------------------------
A = 1.0  # amplitude
sigma = 5.0  # spatial extent (m)
x0 = np.array((0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.9 * zmin + 0.1 * zmax))
_x = SpatialCoordinate(mesh)
r2 = sum((_x[i] - x0[i]) ** 2 for i in range(3))
s = A * exp(-r2 / (2 * sigma**2))

# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
ds = NeumannBC(mesh, markers=markers, marker_value=[-2, -3, -4, -5, -6])

# Dirichlet condition for anchor point (one dof)
bc = DirichletBC(W.sub(0), 0.0, dofs=[0])
# bcs = [bc] # does not seem to help much
bcs = []

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
(p_re, p_im) = TrialFunctions(W)
(q_re, q_im) = TestFunctions(W)

a = (
    inner(grad(p_re), grad(q_re)) * dx
    + inner(grad(p_im), grad(q_im)) * dx
    - k**2 * (p_re * q_re + p_im * q_im) * dx
    + k * (p_im * q_re - p_re * q_im) * ds
)

L = -s * q_re * dx

# ------------------------------------------------------------
# GLS stabilization (does not seem to help much)
# ------------------------------------------------------------
tau = 0.0
if tau > 0.0:

    h_K = CellDiameter(mesh)
    tau = 0.1 * h_K**2

    r_p_re = div(grad(p_re)) + k**2 * p_re
    r_p_im = div(grad(p_im)) + k**2 * p_im
    r_q_re = div(grad(q_re)) + k**2 * q_re
    r_q_im = div(grad(q_im)) + k**2 * q_im

    a += tau * (r_p_re * r_q_re + r_p_im * r_q_im) * dx
    L += -tau * s * r_q_re * dx

# ------------------------------------------------------------
# Shifted form for preconditioning
# ------------------------------------------------------------
alpha = 0.2
beta = 0.65
diag_shift = (1.0 - alpha**2) * k**2 * (p_re * q_re + p_im * q_im) * dx
rot_shift = beta * k**2 * (p_im * q_re - p_re * q_im) * dx
a_pc = a + diag_shift + rot_shift
a_pc = a

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
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    "ksp_max_it": 10000,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_cycle_type": "W",
    "pc_hypre_boomeramg_max_iter": 4,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    "pc_hypre_boomeramg_agg_nl": 4,
}

# Set up linear problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts)

# Add preconditioner
A_pc = assemble_matrix(a_pc, bcs=bcs)
A_pc.assemble()
problem.solver.setOperators(problem.A, A_pc)

# Solve linear problem
p = problem.solve()

# ------------------------------------------------------------
# Post-processing & output
# ------------------------------------------------------------

# Interpolate magnitude of complex solution
V = FunctionSpace(mesh, "Lagrange", 1)
expr = Expression(sqrt(p[0] ** 2 + p[1] ** 2), V.element.interpolation_points())
p_abs = interpolate(expr, V)

# Save solution
p_abs.save("output/solution_helmholtz.xdmf")
