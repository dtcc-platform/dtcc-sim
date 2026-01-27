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
mesh, markers = load_mesh_with_markers("output/volume_mesh_gbg.xdmf")
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
# Function space: mixed (Re, Im)
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
ds_absorb = NeumannBC(mesh, markers=markers, marker_value=[-2, -3, -4, -5, -6])

# Dirichlet condition for anchor point (one dof)
bc_anchor = DirichletBC(W.sub(0), 0.0, dofs=[0])
bcs = []  # optionally: [bc_anchor]

# ------------------------------------------------------------
# Variational problem
# ------------------------------------------------------------
(p_re, p_im) = TrialFunctions(W)
(q_re, q_im) = TestFunctions(W)

a = (
    inner(grad(p_re), grad(q_re)) * dx
    + inner(grad(p_im), grad(q_im)) * dx
    - k**2 * (p_re * q_re + p_im * q_im) * dx
    + k * (p_im * q_re - p_re * q_im) * ds_absorb
)

L = -s * q_re * dx

# ------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------

p = solve(a == L, bcs=bcs, petsc_options=opts)

# ------------------------------------------------------------
# Post-processing & output
# ------------------------------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)
p_abs = interpolate(sqrt(p[0] ** 2 + p[1] ** 2), V)
p_abs.save("output/solution_helmholtz.xdmf")
