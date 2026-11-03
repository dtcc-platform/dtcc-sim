# Stationary incompressible flow

`NavierStokesSolver` solves velocity and kinematic pressure together with
Taylor–Hood P2/P1 elements. Its momentum equation is

`(u · ∇)u - ν_eff Δu + ∇p = 0`, with `div(u) = 0` and
`ν_eff = nu + nu_t`.

With `turbulence_model="abl"` or `"mixing_length"`, `nu_t` is an algebraic
atmospheric boundary-layer eddy viscosity and the viscous term becomes
`-div(2 ν_eff ε(u))`; see the [atmospheric boundary layer](atmospheric-boundary-layer.md).
The viscosity and inlet forcing remain at the requested values. Density does
not enter this kinematic-pressure equation. A stationary field is neither a time
average nor a validated turbulence prediction. Some scenarios have no numerically
accessible stationary solution at the selected mesh and physical settings.

## Configuration

```python
from dtcc_core import BoxMesh
from dtcc_sim import NavierStokesParameters, NavierStokesSolver

parameters = NavierStokesParameters(
    wind_speed=0.1, nu=0.1, side_top_boundary="slip",
    inlet_profile="power_law", power_law_alpha=1.0, u_ref_height=1.0,
    nonlinear_rtol=1e-9, nonlinear_atol=1e-10, nonlinear_max_it=40,
)
solver = NavierStokesSolver(
    mesh=BoxMesh(3, 3, 3), params=parameters, output_dir="output/wind",
)
result = solver.simulate()
```

`StokesSolver` takes `StokesParameters` and omits convection. The dataset
`urban_wind_simulation` selects it with `equations="stokes"`. The dataset
defaults to the [atmospheric boundary-layer](atmospheric-boundary-layer.md)
closure with log-law walls and inflow and slip sides and top. Both APIs accept
native tagged DTCC volume meshes, tagged DOLFINx meshes, mesh paths or city bounds.
Wind direction follows the meteorological FROM convention. Every lateral
bounding-box face the wind enters through receives the inflow profile, every
face it leaves through is a traction outlet, and faces parallel to the wind
are side faces. An oblique wind therefore has two inflow and two outflow faces.
Explicit `inlet_marker` and `outlet_marker` or `closed_cavity=True` select
single faces.

Time steps, CFL controls, ramps, statistical windows and pressure-correction
options are invalid arguments. Configure nonlinear tolerances and
`petsc_options` for Navier–Stokes, or `petsc_stokes` for Stokes. Unknown arguments
raise a validation error in the direct and dataset APIs. The service exposes the
same dataset schema.

## Boundaries and pressure

The natural open-boundary condition is gradient traction
`ν_eff ∂u/∂n - p n = 0`, also with the symmetric variable-viscosity stress. Pressure is determined by this condition; an outlet does
not also receive a pressure Dirichlet constraint. Flat side/top slip imposes zero
normal velocity and natural tangential traction. No-slip fixes all velocity
components on buildings, roofs and ground.

Friction walls use tangential Robin friction; `wall_model="log_law"` uses a
rough-wall quadratic drag. Axis-aligned facets receive a strong normal
constraint. Other facets use consistent normal Nitsche terms with a default
penalty `nitsche_penalty (ν_eff/h_F + wind_speed)`, `nitsche_penalty=100`,
where `h_F = 3|K|/|F|` is the cell height normal to the facet. The velocity term
limits leakage under stagnation pressure; the normal height keeps flat cells
next to small terrain facets coercive. No penetration acts along an averaged
wall normal: area-weighted vertex means of the facet normals, interpolated
linearly, with facet normals kept at sharp edges and corners. Ground and
building leakage gates bound the flow across these walls along that normal.
`gamma_normal` explicitly controls that penalty. On inlet edges shared with
friction or log-law walls, the inflow is projected onto the wall tangent planes.

The assembled Jacobian is tested for the constant-pressure nullspace. A true
nullspace receives residual projection and zero-mean pressure export. Traction
boundaries receive no artificial gauge. There is no pressure-mass term in the
physical residual; the pressure mass matrix is used only for preconditioning.
For explicit LU on a closed domain, MUMPS null-pivot handling can be selected
with `mat_mumps_icntl_24=1` and `mat_mumps_icntl_25=0`.

City meshes generated from bounds request Core's
`repair_boundary_corners=True` before finite-element conversion and MPI
partitioning. This removes geometric singleton boundary corners independently
of the wind direction and boundary categories. It adds one vertex and three
cells per selected tetrahedron, preserving original boundary faces and labels.
The generated mesh and native output include these interior vertices. Core's
processing history records the actual repair counts.

The solver does not implicitly repair a supplied native, file-based or distributed
mesh. For a native `VolumeMesh`, call `mesh.repair_boundary_corners()` before
passing it to the solver, or generate it with the repair flag enabled. Cached
meshes must be created from a request that includes `repair_boundary_corners=True`;
loading an existing file does not regenerate it. No distributed mesh is gathered
for repair. Corner removal is not a general stability or quality guarantee, so
the algebraic pressure/gauge, residual and mass checks remain in force.

A pressure basis function uncoupled from all free velocity tests indicates an
invalid constrained discretization. The solver rejects it. Explicitly setting
`mesh_refinement_levels=1` permits one local refinement of the affected cell
stars; the maximum is two. Facet tags and geometry are preserved. Native DTCC
fields are evaluated at the original input vertices, so output coordinates and
field associations remain unchanged. Refinement and its cell counts are logged.

## Solve and acceptance

A shared mixed formulation supplies Stokes initialization, followed by PETSc
SNES Newton with backtracking at full convection. The default linear method is
FGMRES with a block Schur preconditioner: overlapping Schwarz/local ILU(0) for
velocity and BoomerAMG for the algebraic pressure Schur approximation. It uses
diagonal pressure preconditioning for a verified constant-pressure nullspace.
It uses the Jacobian directly. Newton linear tolerances adapt through Eisenstat–Walker
version 2, capped at 0.1; Stokes initialization stays at `1e-10`. Explicit PETSc
LU is available. Failed linear or nonlinear solves raise; there is no automatic
solver fallback. See the [linear-solver measurements](flow-linear-solvers.md)
for settings, comparisons and scaling limitations.

Acceptance requires positive SNES and applicable KSP reasons, finite fields,
an independently assembled residual including Dirichlet mismatch and pressure
compatibility, dimensionless divergence, total boundary flux balance and
ground and building leakage.
Defaults are:

- Residual: `max(1e-9, 1e-8 × max(initial residual norm, 1))`, using the
  full-convection residual at the Stokes initial field for Navier–Stokes.
- Dimensionless divergence: `hmin × RMS(div u) / max(RMS(u), 1e-14) <= 0.1`.
- Flux imbalance: `abs(∫boundary u·n) / max(∫boundary abs(u·n), 1e-14) <= 1e-4`.
- Ground leakage: `∫ground abs(u·m) / max(inflow volume flux, 1e-14) <= 0.02`,
  with `m` the enforced wall normal,
  and the same for building walls and roofs. Strong no-slip walls have none;
  see the [atmospheric boundary layer](atmospheric-boundary-layer.md#impermeability)
  for weak walls on curved terrain.

Taylor–Hood incompressibility is weak. Pointwise divergence need not vanish.
Underresolved cavity flows can fail the divergence gate despite a small
algebraic residual. Mesh resolution and explicitly requested consistent
`grad_div_gamma` stabilization must be assessed without loosening acceptance.

`solver.diagnostics` records initialization, nonlinear residual history,
linear convergence reasons and tolerances, effective PETSc options, field statistics, pressure normalization, quality
checks and elapsed times. DTCC logging reports progress and failure.

The solve uses coordinates relative to the domain minimum internally; custom
inlet callbacks receive the input/map frame. Built-in wind profiles use height
above the domain base. Numerical output sampling also uses the local frame.
Input meshes are restored exactly on
success and failure. Returned functions, native models and exported meshes use
the input coordinate frame, including meshes refined for pressure support.
The fixed translation and declared CRS are recorded in the coordinate-frame
diagnostics. An unknown CRS is not inferred from coordinate magnitudes.

## Outputs

Native output has vertex fields `velocity` (3 components, m/s), `pressure`
(1 component, m²/s²) and `speed` (1 component, m/s). CRS and native coordinates
are preserved. DOLFINx input returns `(velocity, pressure)` functions.

An output directory receives `solution_velocity.xdmf` and
`solution_pressure.xdmf` with HDF5 companions. XDMF samples velocity into P1;
quality diagnostics use the P2 solution. Dataset byte output remains `dtcc`.
Repeated successful runs replace owned outputs and preserve unrelated files.
Numerical failures preserve existing solution files and propagate through the
dataset and service as failures.

See [numerical verification](wind-verification.md) for reference errors,
measurements and city limitations.
