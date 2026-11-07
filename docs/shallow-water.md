# Experimental shallow-water integration

`ShallowWaterSolver` exposes Cecilia Huang's triangular DG1 shallow-water
solver with explicit parameters and native Core face fields. The retained
reference includes experimental terrain, wetting/drying and equilibrium-pressure
extensions. It is not a calibrated flood forecast. The faster research variant
is not selected or switched to automatically.

## Direct solver

```python
from dtcc_core import RectangleMesh
from dtcc_sim import ShallowWaterSolver, ShallowWaterParameters

solver = ShallowWaterSolver(
    mesh=RectangleMesh(12, 12),
    params=ShallowWaterParameters(
        h0=0., compact_pulse=True, amplitude=.2, sigma=.12,
        final_time=.04, n_outputs=4,
    ),
    output_dir="output/water",
)
water = solver.simulate()
water.info()
solver.result.plot()
```

The input is a native triangular `Mesh`, rather than the tetrahedral
`VolumeMesh` used by the 3D solvers. Coordinates must use metric horizontal
units. Nonnegative face markers designate buildings; negative markers designate
ground. Reflecting interfaces separate simulated building interiors from ground.
The outer boundary also reflects. There are no rainfall, infiltration, open
inflow/outflow or drainage-network terms in this interface.

`h0` sets uniform initial depth. A Gaussian pulse of height `amplitude` and
width `sigma` is added at `(x_center, y_center)` in input coordinates, defaulting
to the domain center. `compact_pulse=True` subtracts the Gaussian's value at
three sigma and clips at zero. Initial momentum is zero. For a mesh with bed
elevation in vertex Z, explicitly set `bed_from_mesh=True`; with
`water_surface=H`, background depth becomes `max(H-bed, 0)` instead of `h0`.
Nonhorizontal mesh input is rejected unless bed use is enabled.

`max_steps` limits accepted time steps, defaults to 5,000, and accepts `None`
to disable that limit. Exhaustion raises an explicit error with the current
time and configured limit. `n_outputs` specifies evenly spaced output intervals:
steps are capped at the next scheduled output time, yielding the initial state
and `n_outputs` subsequent snapshots on a successful run. Output scheduling
can therefore require more time steps than the CFL limit alone.

The solver copies Float64 XY into a local computational frame and translates
source coordinates consistently. Bed elevations retain their vertical datum.
The supplied mesh is never translated in place, including on failure. Results
retain exact input vertices, face order, markers and CRS; `mesh_crs=` can declare
an otherwise unknown CRS, without reprojection. Diagnostics and
`solution.coordinates.json` record the local-to-input translation.

## Runtime

DTCC Sim uses Python 3.12+. The isolated numerical worker needs real-valued
Float64 PETSc and DOLFINx 0.10.x; the validated worker uses Python 3.11 and
PETSc 3.24.5. The standard repository environment uses DOLFINx 0.11 and cannot
serve as this worker. Create the dedicated environment from the checked-in
[worker recipe](../shallow-water-worker.yml), then configure the worker while
keeping the DTCC caller in its Python 3.12+ environment:

```sh
conda env create -f shallow-water-worker.yml
export DTCC_SWE_PYTHON="$(conda run -n dtcc-shallow-water-worker python -c 'import sys; print(sys.executable)')"
python demos/shallow_water_pulse.py
python demos/dataset_urban_flood.py
```

Otherwise, the caller's Python is used as the worker. Advanced callers can set
`solver_python=` explicitly. Both executables must be on the same machine.
No SSH login or server submission is hidden inside the solver API. The worker
uses one MPI rank and one OMP/OpenBLAS thread; its source and equilibrium-pressure
selection are pinned. Other equation solvers and their runtime defaults are unchanged.
Before mesh preparation, the worker checks the DOLFINx version, PETSc scalar
type and MPI size. Incompatible runtimes fail with setup instructions rather
than continuing into numerical assembly or snapshot extraction.

## City dataset

```python
import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim import ShallowWaterParameters
from dtcc_sim.shallow_water import ShallowWaterResult

center_x = 319970.
center_y = 6397910.
L = 250.
bounds = (center_x - L/2, center_y - L/2, center_x + L/2, center_y + L/2)
pulse_x = center_x + 35.
pulse_y = center_y - 20.
output_dir = "output/city_water"

water = dtcc.datasets.urban_flood_simulation(
    bounds=bounds,
    h=10., min_building_detail=.5, strict_live=True,
    output_dir=output_dir,
    parameters=ShallowWaterParameters(
        h0=0., amplitude=.2, sigma=7., compact_pulse=True,
        x_center=pulse_x, y_center=pulse_y, final_time=20., cfl=.08, n_outputs=4,
    ),
)
water.info()
ShallowWaterResult.load(output_dir).plot()
```

The city demo uses the 250 × 250 m Chalmers Johanneberg domain in EPSG:3006,
with bounds `(319845, 6397785, 320095, 6398035)`. The compact pulse is placed
on open ground at `(320005, 6397890)`, offset from the domain centre to approach
nearby buildings. It is 0.2 m high with a 7 m Gaussian width, truncated at three
widths; the 20-second scenario saves the initial state and four five-second
intervals. Its explicit CFL setting is 0.08. This is a longer numerical demo,
not a rainfall-driven or calibrated flood scenario.

Generated geometry comes from Core's `city_flat_mesh` in EPSG:3006, including
building-footprint markers. It is flat, not measured terrain, and does not use
the 3D volume-mesh cache. A supplied native `.dtcc` input can be selected with
`mesh_path=` and, if needed, `mesh_crs=`; it does not download or regenerate
geometry. The dataset returns a native Mesh when `format` is omitted.
`format="dtccpkg"` preserves fields and request/provenance context;
`format="dtcc"` and `format="vtu"` return standalone mesh bytes.
Context records the actual mesh source, parameters, numerical diagnostics,
coordinate frame and experimental limitations.

## Results and persistence

`simulate()` returns a Core Mesh, while `solver.result` provides the transient
raw result and `solver.diagnostics` records numerical checks and timings.
The diagnostic `maximum_speed_over_time` includes the gravity-wave contribution
`√(g h)` as well as fluid velocity; it is a signal-speed bound.
Face fields are `water_depth` (m), `water_momentum` (m²/s), `water_velocity`
(m/s), `bed_elevation` (m) and `water_surface_elevation` (m).
Depth and momentum are conservative DG1 cell means. Velocity is mean momentum
divided by mean depth, with zero at zero depth; dry water-surface values equal bed.

The library creates output directories. Repeated runs replace `solution.vtu`,
`solution.dtcc`, diagnostics, coordinate sidecars and the latest-run pointer,
while preserving unrelated files. Each raw run and worker log is retained
under `.shallow_water_runs` for inspection; failed runs are marked failed and
are not published as the latest successful result.

```python
from dtcc_sim.shallow_water import ShallowWaterResult
result = ShallowWaterResult.load("output/water")
print(result.snapshot_times, result.diagnostics)
initial_water = result.to_dtcc(snapshot=0)
result.export_series("output/water/series")
axes = result.plot(snapshot=0, show=False, vmax=.2)
```

`ShallowWaterResult.plot()` shows a filled, equal-scale XY depth map using the
actual conservative face values, with no interpolation between cells. It labels
the saved simulation time, metre axes, depth colour bar and a metric scale bar.
Building triangles form neutral footprint context and their interiors are hidden
from the depth scale. Coordinates and CRS stay in the input frame. The generic
Core `Mesh.plot()` remains available as its standard model preview.

The final snapshot is selected by default. Use `snapshot=` to select another
saved state and a common positive `vmax` when comparing maps. `show=False`
returns the Matplotlib 2D axes for headless rendering or customization; pass
`ax=` to draw into an existing 2D axes, or `theme="dark"` for the Core dark
theme. An all-dry map is labelled explicitly.
Plotting a completed result needs Matplotlib in the caller environment and
does not require another numerical-worker run.

The series contains native DTCC/VTU snapshots and `times.json`; actual saved
times are scheduled evenly and include the initial and final states. Repeated
series exports replace owned snapshots and remove stale indexed snapshots,
preserving unrelated files. Raw DG coordinates are local; use the recorded
frame for independent numerical postprocessing. Presentation exports use the
input frame.

## Numerical scope and source

The shipped MIT-licensed source is identified in `reference.json`, including
the original thesis repository, commit and every retained worker-file hash.
The active spatial operator is `terrain_flux.build_rhs_form`: hydrostatic
reconstruction with a Rusanov (local Lax–Friedrichs) interface flux and
hydrostatic pressure corrections. Outer edges and building interfaces use
reflected normal momentum. Volume terms include the bed-slope source; partially
dry cells use the retained local equilibrium-depth pressure/source correction.
The state is triangular DG1 in conservative depth and both momenta.

Each stage applies wall-aware TVB limiting to momentum and, on fully wet cells,
free-surface slopes. Mean-preserving depth scaling enforces nodal positivity;
dry-node momentum is zeroed and the convex neighbor-velocity limiter bounds
momentum. The time integrator is three-stage SSPRK3 with a CFL step capped at
the next output time. Negative cell means trigger step halving, with at most
13 attempts. The accepted-step limit is independent of those retries.
Unused alternative flat-bed flux and worker export implementations are removed;
`ShallowWaterResult` owns supported native result conversion and export.
TVB limiting applies throughout the domain, including on meshes with building
markers; its neighbor stencils exclude reflecting building interfaces. The
existing wet/dry treatment still restricts free-surface slope limiting in
partially dry cells. Runtime validation, a configurable step limit, scheduled
output times and the domain-wide TVB correction are recorded in the manifest.
Identically zero TVB polynomials bypass the local least-squares solves; nonzero
polynomials retain the same limiting procedure and wall-aware stencils.
Local assembly and the public DTCC adapter do not
claim to reproduce projected-coordinate floating-point results bitwise.
Completed diagnostics and dataset-package processing provenance include the
full `reference` identity, upstream commit and retained worker-file hashes.

Synthetic-slope tests and short city runs check conservation and nonnegative
depth. The independent [reflecting-basin wave benchmark](../tools/verification/shallow_water/README.md)
checks smooth evolving depth and momentum with mesh and time refinement, plus
reference-resolution and amplitude-halving checks. This does not establish
shoreline accuracy or convergence for real city flooding.
Measured terrain, flood observations and quantitative city-flood prediction
remain unvalidated. See the [integration validation record](validation/shallow-water-integration.md)
for runtime versions, test commands and the city-output audit.
