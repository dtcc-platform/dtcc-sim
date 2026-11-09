# DTCC Sim Design

## Purpose

DTCC Sim provides urban simulation and field reconstruction through direct
Python APIs and the shared DTCC Dataset contract.

## Place in the DTCC Platform

Sim owns specialized physical models, numerical solvers and their runtime
requirements. DTCC Core owns the semantic models, dataset registry, I/O and
Package contracts; reuse them rather than creating parallel representations.
Application orchestration and catalog operation belong in DTCC Twin.

## Design principles

- Return native Core models with explicit field associations, units and CRS.
  Keep numerical backend objects and diagnostics available through solver APIs.
- Separate scenario parameters, numerical solving and dataset integration.
  Dataset names describe domain products; solver names describe equations.
  See the [solver API](docs/solvers.md) for the public names and mesh contract.
- Make the common call direct, with documented domain defaults and explicit
  advanced controls. Validate required inputs and fail with actionable errors.
- Keep heavy solver dependencies isolated from capabilities that do not need
  them. Document real versus complex PETSc and supported DOLFINx requirements.
- Use DOLFINx 0.11.0 across all FEniCSx solvers and isolated numerical workers,
  matching the shared environment recipes. Real and complex PETSc use separate
  environments with the same DOLFINx version. Port solvers to this common
  baseline before integration rather than introducing per-solver version pins.
- Distinguish numerical convergence from physical validity. Record assumptions,
  sources, boundary conditions and limitations; do not label amplitude as dB.

## Architecture

- Equation packages use `<equation>/solver.py` and `<equation>/parameters.py`:
  `helmholtz`, `navierstokes`, `stokes`, `heat` and `shallow_water`. Shared incompressible-flow
  implementation belongs in the private `_flow` package. Top-level imports
  expose the public solver and parameter classes.
- **Parameters and solvers:** validate scenarios, solve equations and report
  convergence and numerical diagnostics.
- **Dataset adapters:** use Core's registry and Context to return semantic
  models with the request, actual provenance and limitations.
- **I/O and presentation:** use Core model persistence and visualization;
  operational launchers and service code remain separate from numerical APIs.

## Trust and quality

Use Float64 local coordinates throughout finite-element assembly, solving and
numerical postprocessing. Record the fixed local-to-input translation and the
declared CRS. Public source positions, observation points and spatial callbacks
use input coordinates and are transformed consistently at the numerical boundary.
Built-in wind profiles use height above the domain base.

Stationary incompressible flow uses flexible GMRES with a velocity/pressure
Schur split. The velocity preconditioner uses overlapping additive Schwarz
with local ILU(0), or ILU(1) with reverse Cuthill–McKee ordering for
eddy-viscosity closures; the algebraic pressure Schur approximation uses BoomerAMG
for traction-determined pressure. A verified constant-pressure nullspace uses
diagonal pressure preconditioning to avoid amplifying the arbitrary gauge mode.
Build these blocks from the actual Jacobian without assembling an unused
auxiliary matrix. Navier–Stokes uses bounded Eisenstat–Walker tolerances during
Newton iteration and a tight linear tolerance during Stokes initialization.
Independent full residual, flux, divergence and wall-leakage checks determine
acceptance. Lateral faces are classified from the wind direction: every face
the wind enters through is an inflow and every face it leaves through an
outlet, so oblique winds keep their crosswind component. Weak walls impose no
penetration along averaged vertex normals, keeping facet normals at sharp
edges; boundary data that depend on neighbouring facets are assembled across
MPI ranks so that results do not depend on the partition.
Explicit PETSc overrides remain available, including LU; failed solves raise
without changing methods or relaxing acceptance.

The default constant `nu_t` keeps the gradient viscous form. Atmospheric
boundary-layer closures use the symmetric stress with spatially varying
viscosity, a Poisson wall distance, an equilibrium log-law inflow at
terrain-relative height, top shear and an optional log-law wall. Closure,
inflow, wall and top are chosen together so that the log-law profile is an
exact continuous solution over flat terrain; horizontal homogeneity is its
verification. See [atmospheric boundary layer](docs/atmospheric-boundary-layer.md).

Scripts behave the same serially and under `mpirun`. Distributed solves
gather native results on rank zero; other ranks receive a stand-in whose
output methods do nothing, and informational logging comes from rank zero.

Preserve the input coordinate frame in returned models, backend functions and
exported meshes, including refined meshes. Restore supplied meshes exactly on
success and failure; retain original coordinates for serialization. Geographic
demo bounds use EPSG:3006; a numerical translation does not reproject a CRS.

Preserve fields and their meaning across persistence. Use Core Packages when
run context and provenance must travel with the model. Create output directories internally and replace previous solver outputs on
repeated runs, preserving unrelated files. Keep mesh caches separate from results.

Use small offline contract tests and real numerical regression tests in the
appropriate environment. Mocked tests do not validate solvers. Live-data and
expensive runs are explicit opt-ins. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Shared city meshes

City mesh settings and the validated mesh cache belong in `dtcc_sim.meshing`.
Equation solvers and application datasets use the same Core volume-mesh API,
fully resolved geometry settings and cache identity. Cache identity includes
LOD, roof reconstruction method, terrain settings, corner repair and mesher
versions; files are verified before reuse. An explicit `mesh_cache_dir` shares
meshes across runs and numerical runtimes without sharing solution artifacts.
Generated meshes retain their request and cache-reuse provenance.

The Gothenburg city demos use a terrain-following 250 × 250 × 80 m domain,
12.5 m target size, 0.5 m minimum building detail, catalogue LOD2 roofs and
boundary-corner repair when exposed by Core. They share
`output/city-mesh-cache`. The air-quality
sensor demo retains its Stockholm bounds, observation-coverage settings and
LOD1 geometry. Its larger LOD2 volume currently fails during TetGen generation;
meshing failures remain explicit, without automatic LOD fallback.
Synthetic equation-verification demos retain their analytical domains.

## City geometry output

Spatial solvers export `city.vtu` beside saved solution fields when physical
boundary tags are available. Reuse the simulation mesh's ground and building
faces; exclude the artificial domain sides and top. Keep geometry in the input
coordinate frame and retain boundary markers and CRS metadata. Native and
backend mesh inputs use the same surface selection. Distributed backend meshes
collect owned exterior facets on rank zero for one serial surface export.
Untagged inputs produce no context surface; repeated runs remove stale owned
context files. Surface export belongs to the run's export timing.

## Run timing

Solver and simulation-dataset entry points share a context-local wall-clock
collector. Nested calls contribute to one run report. Each phase records elapsed
time, time excluding children and call counts; repeated runs start fresh.
Core progress phases can contribute through `phase_timing`, while Sim retains
coarse mesh timing with Core versions that lack this optional hook.

Successful MPI runs report the maximum duration across ranks for each path,
with only rank zero printing and writing the report. Exception unwinding uses
local timings without adding collectives. Timings are observational: reporting
failures do not change numerical success or replace an original exception.

## Evolution

Preserve public behavior unless a deliberate migration changes it. Reuse
existing Core contracts, keep changes focused, and avoid speculative layers.
Demos teach these APIs; follow [demo guidelines](docs/demo-guidelines.md).
