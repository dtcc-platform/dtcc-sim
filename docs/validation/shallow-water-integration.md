# Shallow-water integration validation

## Source identity and runtime

The numerical source is identified by
[`reference.json`](../../dtcc_sim/shallow_water/reference.json): backend
`cecilia-reference`, upstream commit
`2cf10273eea2441137393b57040528b1c03e8725`, reference
`equilibrium-terrain-20261028+integration-controls+domain-tvb-zero-skip+active-backend-cleanup`.
The manifest records SHA256 for every retained worker file. Completed dataset
packages preserve that complete identity in processing provenance.

The caller requires Python 3.12+ and Matplotlib for maps. The dedicated
[worker recipe](../../shallow-water-worker.yml) supplies DOLFINx 0.10.x with
real Float64 PETSc, one MPI rank and one OMP/OpenBLAS thread. The standard
DOLFINx 0.11 environment is incompatible. Caller and worker run on the same
host; MPI may require local socket permissions even for a single rank.

The supplied scientific audits dated 2026-10-29, 2026-10-30 and 2026-11-07
used Core `6f983244b630a2ac820e9e9734a8d653aece9f38`. The 2026-11-07 Sim
source is PR head `c53dbe5d02ed5bf514b4eb6ee4d25975250a7af9`; exact worker,
verification-tool and wheel hashes are in the
[wave audit](shallow-water-wave.json). The supplied server worker used
DOLFINx 0.10.0 and real PETSc 3.24.5.

## Reproduction

From the repository root in the DTCC caller environment:

```sh
export DTCC_SWE_PYTHON=/path/to/real-worker/python
export MPLBACKEND=Agg
python -m pytest tests/test_shallow_water.py \
  tests/test_shallow_water_dataset.py tests/test_shallow_water_demos.py \
  tests/test_shallow_water_plotting.py tests/test_shallow_water_wave_reference.py \
  tests/test_demo_catalog.py tests/test_city_demo_domains.py \
  tests/test_dataset_qa.py tests/test_import.py
python -m pytest tests/test_core_pin_update.py tests/test_helmholtz_installer.py
uv build
```

For installed-wheel checks, install the built wheel with its declared Core
revision into an isolated environment. Confirm `dtcc_sim.__file__` is in that
environment's site-packages, then run the selection with
`--import-mode=importlib -o pythonpath= -p no:cacheprovider`. Do not put the
source tree or a local Core override on `PYTHONPATH`.

The offline demo test executes `demos/shallow_water_pulse.py` twice and checks
that an unrelated output file survives. The city demo test requires both
explicit opt-ins and live data:

```sh
DTCC_RUN_DEMOS=1 python -m pytest \
  tests/test_shallow_water_demos.py::test_city_dataset_demo_runs_twice \
  --run-expensive -q
```

The full independent wave study is separate:

```sh
python -m tools.verification.shallow_water.wave \
  --output output/verification/shallow-water-wave
```

It is required when the worker or its numerical behavior changes. Reference
PDE tests in the normal selection do not reproduce the full refinement study.

## Native maps and persistence

The demos request a native Core Mesh, call its `info()` method and plot that
same Mesh through `plot(projection="xy", field="water_depth", ...)`.
Core owns triangle filling, shared colour limits, equal XY scale, styling,
neutral context and perimeter outlines. Sim's transient plotting method selects
a snapshot and delegates to that public API.

The real-worker package consumer test exports an actual small tagged
simulation, moves its package to a consumer directory, deletes the numerical
run and input mesh, and reopens it through Core. Assertions cover face values,
associations, dimensions, units, exact vertices/connectivity/markers, CRS,
affine and numerical reference provenance. Building-interior values exceed
ground values so colour-range exclusion is observable. Polygon/value assertions
and a rendered PNG check the public native map without transient attributes.
Generic renderer tests belong to Core and cover transforms, display budgeting,
constant fields, invalid inputs, supplied axes and read-only behavior.

## Independently reproduced checks: 2026-11-09

Native-map integration used the published Core commit
`350a2faefdc2977588d67156a6a86062333fa2a5`. Both dependency declarations,
`uv.lock`, installer sources and checksum records agree. The caller was Python
3.12.12 on macOS arm64; the actual worker was Python 3.12.13, DOLFINx 0.10.0
and real Float64 PETSc 3.25.6. All 11 retained worker-file hashes matched the
manifest. Numerical settings and acceptance criteria are recorded in the
worker source and regression tests.

- Full Sim suite: **350 passed, 79 skipped**. Skips concern optional solver
  runtimes, live data and expensive checks.
- Required focused selection plus Core-pin/installer checks: **90 passed,
  5 skipped**, both in the source environment and against an isolated installed
  wheel. The five skips are the opt-in live city demo and four caller-side
  FEniCSx import/log checks; the separate real shallow-water worker was enabled.
- Wheel import assertions verified that both Sim and Core came from the isolated
  environment's site-packages and that Core's installed VCS identity was the
  declared published commit. `uv build` produced a wheel and source distribution.
  Installer checksums passed.
- The moved-package consumer test passed after deleting the original numerical
  run and input mesh. Its depth map was visually inspected: ground values set
  the range, larger interior values stayed neutral, and scale/CRS/time labels
  were readable. The actual offline demo ran twice, retained an unrelated file,
  and produced a visually inspected face map.
- Core consumer-baseline coverage: **3,329 passed, 51 skipped, 67 deselected**;
  strict public API verification covered **117/117** exported functions. Its
  unchanged baseline has Ruff errors in `dtcc_core/__init__.py` and
  `tests/common/test_progress.py`, plus formatting failures in the progress
  and structured-mesh files. All feature files pass.
- Core current-develop coverage: **3,233 passed, 49 skipped, 67 deselected**;
  **113/113** exported functions covered. Whole-repository Ruff checks and
  formatting passed. The offline calibration-grid demo completed on both
  Core baselines. Fresh-context code/security review reported no issues and
  independently passed 47 focused preview/style/import/map tests.

The approximately 31-minute two-run live city test was not repeated, and no
saved Johanneberg numerical result was available locally for a city-map check.
The demo API-wiring test verifies the native Mesh call and exact common domain,
pulse, mesh settings, duration, CFL and output schedule; its stubbed dataset
request supplies no scientific evidence. The full wave study was not rerun:
numerical source identity is recorded above. Interactive GUI behavior,
large context-union performance and cross-platform CI remain unverified locally.

## Supplied scientific evidence

These audits were supplied with the source; their full scientific runs are not
independently reproduced by the native-map test selection.

- [City audit, 2026-10-29](shallow-water-city.json): native output, regional
  conservation, fields and coordinate frame for a short flat-city case.
- [Saved-mesh audit, 2026-10-30](shallow-water-revalidation.json): a synthetic
  sloping-bed case with 829 vertices and 1,527 triangles, five accepted steps,
  no rejected attempts, and maximum regional volume drift `7.11e-14 m³`.
- [Wave audit, 2026-11-07](shallow-water-wave.json) and
  [benchmark procedure](../../tools/verification/shallow_water/README.md):
  an independent reflecting-basin linear-wave reference, integrated over the
  full DG1 state. Mesh refinement from 8 to 16 to 24 subdivisions reduced
  relative depth L2 errors from 26.17% to 12.98% to 8.37%, and momentum errors
  from 34.05% to 14.11% to 8.45%. Successive time-step differences decreased;
  amplitude halving changed normalized solutions by less than 0.011% of the
  discretization error. Reference-mode and integration-quadrature checks
  resolved the reported errors. This establishes neither formal DG1 order nor
  shoreline accuracy.

The supplied 2026-11-07 Johanneberg city execution used EPSG:3006, centre
`(319970, 6397910)`, `L = 250` m and bounds
`(319845, 6397785, 320095, 6398035)`. Geometry had 1,621 vertices and 3,108
triangles. A compact 0.2 m pulse of width 7 m started at `(320005, 6397890)` on
initially dry open ground. The 20-second scenario used CFL 0.08 and snapshots
at 0, 5, 10, 15 and 20 seconds. Both complete runs took 1,006 accepted steps
with no rejected attempts; maximum regional drift was `7.11e-14 m³` across
19 reflecting regions. At 20 seconds, 133 initially dry ground faces were wet,
including 32 beside buildings, using a `1e-4 m` face-depth threshold.
The two-run live test took approximately 31 minutes on sahakyan.

## Physical and numerical limits

Generated city terrain is flat. Outer walls and ground/building interfaces
reflect; building interiors are separately simulated regions. There is no
rainfall, infiltration, drainage, roof runoff, open inflow/outflow interface or
drainage-network coupling. Synthetic slopes and pulses are equation and
integration checks. No measured flood, measured terrain accuracy or moving
shoreline accuracy is validated.

Smooth fully wet wave refinement does not establish city wet/dry convergence
or quantitative flood prediction. Exported conservative face means omit DG
subcell structure, which remains in raw transient results. There is no automatic
CRS reprojection, and parallel scaling beyond one worker rank is unverified.
