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

The following selection passed all **64 tests** against the installed wheel,
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
Mac archive metadata initially polluted the staging demo inventory; removing
those transfer artifacts resolved the catalog failures without source changes.

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

## Limits

These are integration and short numerical health checks, not flood prediction
validation. The generated bed is flat; the pulse is synthetic; building interiors
are simulated regions. Rainfall, infiltration, drainage, observed flood data,
measured terrain accuracy and shoreline convergence remain outside this result.

The numerical source is byte-preserved from the retained experimental reference;
it is not an unmodified copy of the thesis commit. Its terrain, wetting/drying and
equilibrium extensions are identified in the workflow and source manifest.
Three inherited trailing-whitespace lines remain in the pinned backend; the
adapter, tests and documentation pass the whitespace check.
