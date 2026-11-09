# Shallow-water validation

## Source and runtime

Implementation: `97649faa17adb7c0c6909a5d364dc3155fb78139`.
The numerical reference is `equilibrium-terrain-dg1-dolfinx-0.11`;
[`reference.json`](../../dtcc_sim/shallow_water/reference.json) identifies the
MIT-licensed upstream commit and all 12 retained worker-file hashes. Installed
Core is the declared revision `350a2faefdc2977588d67156a6a86062333fa2a5`.
The [machine-readable audit](shallow-water-011.json) records commands, source
hashes, numerical errors, package identity and complete city-run measurements.

Validation on 2026-11-09 used an isolated clone of the standard real
`fenicsx-env`, Python 3.12.16, DOLFINx 0.11.0, real Float64 PETSc 3.25.6 and
MPICH 5.0.1 (MPI 5.0), on macOS arm64. The worker used one rank, one numerical
thread and local Float64 coordinates. The shared environment was preserved.
This host required `FI_PROVIDER=tcp`: its default `en1` provider failed MPI
finalization with an OFI I/O error. This setting selects transport, without
changing numerical parameters or acceptance criteria.

## Verification

| Check | Passed | Skipped |
| --- | ---: | ---: |
| Complete uv suite, no worker override | 343 | 91 |
| Complete uv suite, explicit real 0.11 worker | 355 | 79 |
| Required selection in real 0.11, no worker override | 99 | 1 |
| Installed-wheel solver/package selection, no worker override | 24 | 0 |
| Live city test, two complete demo executions | 1 | 0 |
| Hosted Ubuntu real 0.11 runtime selection | 30 | 1 |

The full worker-enabled suite skips 66 checks requiring caller-side FEniCSx,
three requiring `pymetis`, two requiring `xlrd` and eight opt-in live-data
checks. Without a worker, 12 additional optional shallow-water numerical/demo
checks skip in uv. The focused selection's sole skip is the city test, executed
separately with both opt-ins. Numerical tests execute in the standard real
runtime with `DTCC_SWE_PYTHON` unset.

Coverage includes conservation, positivity, fully and partially wet sloping
lakes, wall isolation, the distant dry-building regression, translation, exact
input/CRS preservation, scheduled snapshots, step-limit failure, unrelated-file
preservation, independent raw-DG/component-to-native-face mapping, native field
and package round trips, and source provenance. An actual complex128-PETSc
0.11 caller completed a solve using an explicitly selected real 0.11 worker.
Actual subprocess checks reject missing dependencies, complex PETSc, two MPI
ranks and a missing explicitly configured worker. Unit guards additionally
cover 0.10 rejection, 0.11.1 acceptance and non-Float64 PETSc rejection.

The fresh [wave study](../../tools/verification/shallow_water/README.md) passed
all mesh-error, time-refinement, amplitude-halving and reference-resolution
criteria with their prescribed thresholds. Mesh subdivisions 8, 16 and 24 gave
relative depth L2 errors of 26.1696%, 12.9806% and 8.3738%; corresponding momentum
errors were 34.0542%, 14.1056% and 8.4549% in each direction. Successive time-step
differences decreased in every component. The audit includes normalized errors,
amplitude differences, reference uncertainty and step counts.

Both actual demos executed twice with no interpreter override. Each city run
used 1,621 vertices and 3,108 triangles, took 1,006 accepted steps, rejected no
steps and saved times 0, 5, 10, 15 and 20 s. Fields were finite and nonnegative;
relative total mass error was `9.74e-16`. Across 19 reflecting regions, maximum
absolute volume drift was `3.56e-14 m³`. At 20 s, 133 initially dry ground faces
exceeded `1e-4 m` depth, including 32 beside buildings. The two-run test took
329.67 s. Both runs preserved an unrelated output file. Genuine city snapshots,
the city final map and the offline map were visually inspected for face values,
building context, CRS/time labels, units and scale bars.

The exact city scenario is EPSG:3006, centre `(319970, 6397910)`, `L = 250 m`,
bounds `(319845, 6397785, 320095, 6398035)`, pulse centre `(320005, 6397890)`,
height 0.2 m, width 7 m, final time 20 s, CFL 0.08 and four output intervals.
Mesh target size is 10 m with 0.5 m minimum building detail and a flat bed.

Wheel and source distributions build successfully. From outside the checkout,
Sim and Core import from the isolated wheel environment's site-packages; Core's
VCS identity and the declared dependency agree. All worker hashes and wheel
code/manifest bytes match source. Default-interpreter numerical execution and
the moved-package consumer pass with `PYTHONPATH` and `DTCC_SWE_PYTHON` unset.
The consumer deletes the numerical run and input, reopens the package and
verifies native geometry, fields, units, associations, CRS, affine and source
provenance. Its native map was visually inspected.

## Reproduction

Use the [common real environment](../installation.md#fenicsx-environment):

```sh
conda activate fenicsx-env
unset DTCC_SWE_PYTHON
python -m pytest tests/test_shallow_water.py tests/test_shallow_water_dataset.py \
  tests/test_shallow_water_demos.py tests/test_shallow_water_plotting.py \
  tests/test_shallow_water_wave_reference.py tests/test_demo_catalog.py \
  tests/test_city_demo_domains.py tests/test_dataset_qa.py tests/test_import.py \
  tests/test_core_pin_update.py tests/test_helmholtz_installer.py
python -m tools.verification.shallow_water.wave \
  --output output/verification/shallow-water-wave-011
DTCC_RUN_DEMOS=1 python -m pytest \
  tests/test_shallow_water_demos.py::test_city_dataset_demo_runs_twice \
  --run-expensive -q
uv build
```

Use a fresh wave output directory. For package checks, install the wheel and
its declared Core revision into an isolated real 0.11 environment, change to a
directory outside the checkout and run tests with
`--import-mode=importlib -o pythonpath= -p no:cacheprovider`. Verify import
locations and VCS metadata. The shallow-water runtime CI uses `fenicsx-env.yml`
and executes actual offline solves, repeated plotting and a moved-package
consumer with no worker override. The [hosted Ubuntu run](https://github.com/dtcc-platform/dtcc-sim/actions/runs/42599713941)
passed on commit `8cf49f359f5bfde476ca11d1877aeb803480a711`, using DOLFINx 0.11.0,
real PETSc 3.25.6 and MPICH 5.0.1. Its sole skip is the separately executed live
city test.

## Evidence scope

The archived [city](shallow-water-city.json),
[saved-mesh](shallow-water-revalidation.json) and
[wave](shallow-water-wave.json) audits record their original DOLFINx 0.10
runtimes. Their numerical measurements apply to those recorded sources and
runtimes.

Generated city terrain is flat; outer and building interfaces reflect, with
building interiors simulated as separate regions. There is no rainfall,
infiltration, drainage, roof runoff or open inflow/outflow. Smooth fully wet
wave refinement does not establish moving-shoreline accuracy, measured-terrain
accuracy, city wet/dry convergence or quantitative flood prediction. Exported
face means omit DG subcell structure, retained in raw results. No CRS
reprojection is performed. Interactive GUI behavior remains unverified. No
required acceptance check is open.
