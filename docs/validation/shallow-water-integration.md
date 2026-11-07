# Shallow-water integration validation

Validated on 2026-10-29 against DTCC Sim `develop` at
`17ccc7f63e47ceae83e04cecf16e4059535ff217` and the project's exact Core pin
`6f983244b630a2ac820e9e9734a8d653aece9f38`.

The DTCC caller used Python 3.12 with complex PETSc. Numerical work ran in a
separate Python 3.11 worker with DOLFINx 0.10.0 and real PETSc 3.24.5,
using one MPI rank and one OMP/OpenBLAS thread. All numerical runs and
scientific postprocessing were performed on the server.

## Installed package and regression checks

`uv build` produced a wheel and source distribution. The wheel was installed
in an isolated target directory without replacing the shared environment.
Every retained worker file matched `reference.json`; both distributions
contained the manifest and MIT license.

The initial selection passed all **64 tests** against the installed wheel,
with `DTCC_SWE_PYTHON` identifying the real-valued worker:

```sh
python -m pytest tests/test_shallow_water.py tests/test_shallow_water_dataset.py \
  tests/test_dtcc_core_contract.py tests/test_helmholtz_dataset.py \
  tests/test_traffic.py tests/test_timing.py -o pythonpath= -q
```

The installed wheel and exact pinned Core were on `PYTHONPATH`; the working
source tree was excluded from package import resolution.

Checks include native field round-trips, DTCC package context preservation,
repeated output replacement without deleting unrelated files, indexed transient
exports, worker failure isolation, exact input-coordinate preservation,
bitwise state invariance under an exactly representable XY translation,
short-run conservation and preservation of a fully wet sloping lake at rest.

## Demos and city output

Both shallow-water demos ran twice through the actual solver and Core
`info()` / `plot()` methods, using Matplotlib's Agg backend for headless
verification. The live city checks were explicitly enabled:

```sh
DTCC_RUN_DEMOS=1 python -m pytest tests/test_shallow_water_demos.py --run-expensive
python -m pytest tests/test_demo_catalog.py -q
```

All 19 demo catalog checks passed. Dataset QA and import checks also passed.

The city demo used live `city_flat_mesh` geometry around Poseidon in Gothenburg:
996 vertices, 1,861 triangles, and 12 building markers. Both negative ground
markers share the flow domain; their individual masses may exchange. Outer
boundaries and ground/building interfaces reflect. The two runs used the same
idealized compact pulse for 0.5 seconds on a flat bed.

An independent installed-wheel output audit confirmed:

- All fields finite and water depth nonnegative; zero rejected steps.
- Reported global relative mass error zero in both runs.
- Ground mass and each enclosed building-region mass preserved within
  `1e-9 * max(abs(initial_mass), 1)`.
- Native output vertices, triangles, markers and field values round-trip
  exactly. Results and saved native output declare EPSG:3006.
- The provider's input mesh had no stored CRS; the dataset explicitly declared
  EPSG:3006. Numerical XY was local; presentation coordinates remained projected.
- Core's native `plot(field="water_depth")` produced a saved preview.

The [machine-readable city audit](shallow-water-city.json) records the frame,
fields, snapshot times and regional masses.

## Installed-wheel revalidation on 2026-10-30

The saved final wheel was checked again without changing the numerical worker.
All 65 installed package files matched the wheel, and all 11 retained worker
files matched the pinned source manifest. The wheel SHA256 was
`3940ccff82cb7b3819b59f618b0f9fb384d9ceb76ed495a48dd2d70e036ddd2b`.
The caller used Python 3.12.15; the separate worker used Python 3.11,
DOLFINx 0.10.0 and real PETSc 3.24.5.

The three shallow-water test modules passed **12 tests**, with **one skipped**
in 83.94 seconds. The skipped test was the opt-in live-data city demo. All
real-worker API, native-file dataset and offline pulse-demo checks were enabled.
An import assertion verified that tests used the installed wheel rather than
the source checkout. To repeat this selection, place the installed wheel and
compatible Core on `PYTHONPATH`, set `DTCC_SWE_PYTHON` to the real-valued worker,
and run from the source checkout:

```sh
python -m pytest tests/test_shallow_water.py tests/test_shallow_water_dataset.py \
  tests/test_shallow_water_demos.py --import-mode=importlib -o pythonpath= \
  -p no:cacheprovider -q
```

These checks cover failure isolation, repeatable runs, input-coordinate and CRS
preservation, conservation, sloping-bed still water, native field round-trips,
dataset/package context and an offline demo run twice. The earlier broader
regression selection and live-city runs above are separate historical evidence;
they were not repeated by this selection.

A separate fresh run used a saved native city mesh with a synthetic sloping bed:
829 vertices, 1,527 triangles, a compact pulse, and a final time of 0.05 seconds.
It completed five accepted steps with no rejected attempts. Independent checks
of all three snapshots confirmed:

- Exact input vertices, face order and markers, with CRS preserved.
- Exported depth, momentum and bed matching raw DG cell means.
- DTCC and VTU field round-trips and consistent derived velocity/surface fields.
- Water conservation in 13 reflecting regions with 194 building/ground interfaces.
- Maximum regional volume drift of `7.11e-14` and global relative mass error
  of `6.04e-16`.

The [compact revalidation audit](shallow-water-revalidation.json) records these
results. This saved-input check did not repeat live geometry acquisition.
Raw snapshots, worker logs and numerical-experiment archives remain on the
validation server; they are not repository fixtures. Core plotting was tested
headlessly; interactive desktop viewer behavior was not checked.

## API and worker-control checks on 2026-11-07

The public entry point is `ShallowWaterSolver`. A server regression selection
covering the shallow-water API/dataset/demos, Core contracts, timing, demo
catalog and imports passed **55 tests**, with the opt-in live city demo skipped.
The actual DOLFINx 0.10 real-valued worker was enabled. New regressions check
explicit step-limit failure, all six snapshots for five requested intervals
when a CFL step would otherwise span them, incompatible-runtime rejection
before mesh preparation, and exact numerical reference metadata in a reloaded
dataset package.

An isolated wheel was then built and installed. Import assertions verified the
new public name and pinned source manifest; two additional real-worker tests
passed for scheduled output/step limits and dataset-package provenance. The
dedicated `shallow-water-worker.yml` resolved successfully in a Linux Conda
dry run; this was not a fresh installation of the resolved environment.

The source selection used:

```sh
python -m pytest tests/test_shallow_water.py tests/test_shallow_water_dataset.py \
  tests/test_shallow_water_demos.py tests/test_dtcc_core_contract.py \
  tests/test_timing.py tests/test_demo_catalog.py tests/test_import.py \
  --import-mode=importlib -o pythonpath= -p no:cacheprovider -q
```

This first batch did not resolve the city-mesh limiter defect, common city
demo domain/scenario, visualization, independent dynamic accuracy benchmark or
backend cleanup.

## City-mesh limiter correction on 2026-11-07

TVB limiting is enabled throughout the domain, with the same wall-aware edge
neighbors used by velocity limiting. A building marker no longer disables
TVB limiting away from its walls. Existing wet/dry restrictions on free-surface
slope limiting remain in effect; building interfaces still reflect.

A real-worker regression runs an 8-by-8 unit-square mesh for 0.01 seconds with
a compact pulse centered at (0.25, 0.25), then repeats it with one distant,
initially dry building triangle. Before the correction, both runs reported
success but differed near the pulse by up to 0.0481 m in raw nodal depth. The
new regression failed against that version. After the correction, depth,
momentum and regularized velocity near the pulse matched exactly in this case.
The test also verifies that ground-cell limiter stencils exclude the building.

The broader real-worker selection above passed **56 tests**, with the opt-in
live city demo skipped. Two installed-wheel checks then passed: the
distant-building regression and an active-pulse conservation test with a wet,
enclosed building region. Independent volume integration over the snapshots
gave maximum building/ground drifts of `1.73e-18` and `9.71e-17`, respectively.
Installed-package imports and pinned worker hashes were verified.

At that stage the common city-demo domain/scenario, visualization, independent dynamic
accuracy/refinement benchmark and backend cleanup remained outstanding. These
regressions address the reviewed limiter defect; they do not establish general
dynamic accuracy or flood-prediction validity.

## Johanneberg city scenario on 2026-11-07

The city demo and its workflow example use the requested Chalmers Johanneberg
domain: EPSG:3006, centre `(319970, 6397910)`, `L = 250` m and bounds
`(319845, 6397785, 320095, 6398035)`, derived as centre ± `L / 2`. The actual
Core flat-city mesh has 1,621 vertices and 3,108 triangles. Core resolved real
city data using its source-tile cache.

The compact pulse starts on open ground at `(320005, 6397890)`, with amplitude
0.2 m, width 7 m, zero background depth and zero initial momentum. The scenario
runs for 20 seconds at CFL 0.08 and saves states at 0, 5, 10, 15 and 20 seconds.
Its footprint initially excludes building interiors. Both complete runs
took 1,006 accepted steps with no rejected attempts. Independent integration
over all five snapshots in each run found maximum regional volume drift `7.11e-14 m³`
across 19 reflecting regions with 450 building/ground interfaces.

The saved states show propagation onto initially dry ground, including beside
buildings. Counts below use face-mean depth greater than `1e-4 m`:

| Time (s) | Wet ground faces | Newly wet ground faces | Newly wet beside buildings |
|---:|---:|---:|---:|
| 0 | 50 | 0 | 0 |
| 5 | 63 | 10 | 4 |
| 10 | 88 | 35 | 9 |
| 15 | 132 | 79 | 21 |
| 20 | 186 | 133 | 32 |

The live-demo test explicitly checks the centre, bounds, CRS, duration, saved
times, building exclusion and propagation to new ground next to buildings.
The actual demo passed both executions and preserved an unrelated notes file.
The opt-in test took about 31 minutes on the validation server, including city
preparation and plotting; runtime depends on mesh detail and hardware.

```sh
DTCC_RUN_DEMOS=1 python -m pytest \
  tests/test_shallow_water_demos.py::test_city_dataset_demo_runs_twice \
  --run-expensive -q
```

To keep this longer scenario practical, identically zero TVB polynomials skip
the local least-squares calculations. A 0.5-second comparison on the same city
mesh produced byte-identical final state, initial state, bed, coordinates and
DOF maps before and after this shortcut. Observed run times were 99.47 and
14.46 seconds; this is a single-run comparison, not a scaling benchmark.
The built wheel's short-case arrays also matched the source run. The wider
regression selection passed 57 tests, with the opt-in live case tested separately.

The bed remains flat and boundaries idealized and reflecting. These checks
establish a more informative city demonstration, not general dynamic accuracy
or flood calibration. Filled-map plotting support, an independent dynamic
accuracy/refinement benchmark and the remaining backend cleanup are separate
outstanding review items. Verification images and raw outputs remain on the
server; no visual-quality claim is made for the current Core preview.

## Limits

These are integration and short numerical health checks, not flood prediction
validation. The generated-city case used a flat bed; the saved-input case used
a synthetic slope. The pulse is synthetic; building interiors
are simulated regions. Rainfall, infiltration, drainage, observed flood data,
measured terrain accuracy and shoreline convergence remain outside this result.

The workflow supports metric triangular native meshes, explicit synthetic-bed
tests and conservative face fields. There is no automatic CRS reprojection,
open inflow/outflow boundary interface or drainage-network coupling. The worker
runs on the caller's host with one MPI rank; parallel scaling is not established.
Building interfaces are idealized impermeable reflections, not a roof/runoff
model. Exported face means omit subcell DG structure, which remains available
in the raw transient results.

Moving-shoreline accuracy, independent mesh/time convergence and validation
against observed flooding remain unresolved. The retained baseline remains
experimental. These checks support an integration demonstration, not
quantitative flood forecasting or an improved wet/dry algorithm.

The numerical source is pinned by the manifest, including integration controls
for runtime validation, maximum steps and scheduled outputs. It is not an
unmodified copy of the thesis commit. Its terrain, wetting/drying and
equilibrium extensions are identified in the workflow and source manifest.
Three inherited trailing-whitespace lines remain in the pinned backend; the
adapter, tests and documentation pass the whitespace check.
