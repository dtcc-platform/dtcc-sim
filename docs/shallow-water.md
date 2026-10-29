# Experimental shallow-water integration

`CeciliaSolver` exposes Cecilia Huang's triangular DG1 shallow-water
solver with explicit parameters and native Core face fields. The retained
reference includes experimental terrain, wetting/drying and equilibrium-pressure
extensions. It is not a calibrated flood forecast. The faster research variant
is not selected or switched to automatically.

## Direct solver

```python
from dtcc_core import RectangleMesh
from dtcc_sim import CeciliaSolver, ShallowWaterParameters

solver = CeciliaSolver(
    mesh=RectangleMesh(12, 12),
    params=ShallowWaterParameters(
        h0=0., compact_pulse=True, amplitude=.2, sigma=.12,
        final_time=.04, n_outputs=4,
    ),
    output_dir="output/water",
)
water = solver.simulate()
water.info()
water.plot(field="water_depth")
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

The solver copies Float64 XY into a local computational frame and translates
source coordinates consistently. Bed elevations retain their vertical datum.
The supplied mesh is never translated in place, including on failure. Results
retain exact input vertices, face order, markers and CRS; `mesh_crs=` can declare
an otherwise unknown CRS, without reprojection. Diagnostics and
`solution.coordinates.json` record the local-to-input translation.

## Runtime

DTCC Sim uses Python 3.12+. The isolated numerical worker needs real-valued
PETSc and DOLFINx 0.10; the validated worker uses Python 3.11 and PETSc 3.24.5.
If the caller uses a different runtime, configure once before running demos:

```sh
export DTCC_SWE_PYTHON=/path/to/real-fenicsx-environment/bin/python
python demos/shallow_water_pulse.py
python demos/dataset_urban_flood.py
```

Otherwise, the caller's Python is used as the worker. Advanced callers can set
`solver_python=` explicitly. Both executables must be on the same machine.
No SSH login or server submission is hidden inside the solver API. The worker
uses one MPI rank and one OMP/OpenBLAS thread; its source and equilibrium-pressure
selection are pinned. Other equation solvers and their runtime defaults are unchanged.

## City dataset

```python
import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim import ShallowWaterParameters

water = dtcc.datasets.urban_flood_simulation(
    bounds=(319895.962899, 6398909.716755, 320095.962899, 6399109.716755),
    h=10., min_building_detail=.5, strict_live=True,
    output_dir="output/city_water",
    parameters=ShallowWaterParameters(
        h0=0., amplitude=.2, sigma=10., compact_pulse=True,
        final_time=.5, n_outputs=4,
    ),
)
water.info()
water.plot(field="water_depth")
```

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
```

The series contains native DTCC/VTU snapshots and `times.json`; actual saved
times follow accepted steps and include the initial and final states. Repeated
series exports replace owned snapshots and remove stale indexed snapshots,
preserving unrelated files. Raw DG coordinates are local; use the recorded
frame for independent numerical postprocessing. Presentation exports use the
input frame.

## Numerical scope and source

The shipped MIT-licensed source is identified in `reference.json`, including
the original thesis repository, commit and every retained worker-file hash.
The numerical operator, limiter and RK3 time integrator are unchanged from
the retained reference. Local assembly and the public DTCC adapter do not
claim to reproduce projected-coordinate floating-point results bitwise.

Synthetic-slope tests and short city runs check conservation and nonnegative
depth. These checks do not establish shoreline accuracy or mesh convergence.
Measured terrain, flood observations and quantitative city-flood prediction
remain unvalidated. See the [integration validation record](validation/shallow-water-integration.md)
for runtime versions, test commands and the city-output audit.
