# Contributing to DTCC Sim

Questions, bug reports and pull requests are welcome. Include a small reproducer
when reporting a bug and describe the user-visible behavior of a proposed change.

See [DESIGN.md](DESIGN.md) for architecture and contracts.

Name development branches `dev/<short-description>`, unless there is a specific
reason to use another convention.

## Development environment

Follow the [installation instructions](README.md#installation), then run commands
from the repository root. Sim is installed in editable mode, so Python edits
take effect immediately. See the [installation guide](docs/installation.md) for
the real-number and complex-number FEniCSx environments and local Core overrides.

| Task | Command |
| --- | --- |
| Set up or update the environment | `uv sync` |
| Run the test suite | `uv run pytest` |
| Build a source distribution and wheel | `uv build` |
| Add a dependency | `uv add <package>` |
| Upgrade a locked dependency | `uv lock --upgrade-package <package>` |

Commit `uv.lock` together with dependency changes in `pyproject.toml`.

## Demos

Follow the [demo guidelines](docs/demo-guidelines.md); the
[demo catalog](demos/README.md) lists purposes and requirements. Run from the
repository root in the appropriate environment, for example
`python demos/urban_heat_cooling.py`. Plots and viewers open where provided;
close them to continue. The Helmholtz box and noise dataset demos need no arguments.

Run `python -m pytest tests/test_helmholtz_demos.py` for wave-demo hygiene and
execution checks in the complex environment. The city run is opt-in with
`DTCC_RUN_DEMOS=1`; it needs live geodata. `tests/test_demo_catalog.py` checks
the demo inventory and structure. In the real environment,
`tests/test_flow_demos.py` exercises synthetic wind examples; its city checks
require `DTCC_WIND_CITY_MESH` pointing to a tagged city XDMF mesh and
`--run-expensive`. `DTCC_RUN_DEMOS=1 python -m pytest tests/test_dataset_demos.py`
exercises heat, reconstruction, air-quality and traffic examples with live data.
For solver verification, use the numerical tests below.

## Tests

For a quick development check:

```bash
uv run pytest tests/test_dataset_qa.py tests/test_import.py tests/test_traffic.py
```

Tests use these markers:

- `simulation`: simulation dataset or solver validation.
- `fenics`: requires FEniCSx/dolfinx.
- `slow`: small numerical checks slower than pure unit tests.
- `expensive`: manual-scale or city-scale simulations.

Run simulation checks with:

```bash
uv run pytest -m "simulation and not expensive"
```

FEniCSx tests skip when the runtime is absent. To exercise heat, wind and
reconstruction, activate the real-number
[FEniCSx environment](docs/installation.md#fenicsx-environment) and run:

```bash
conda activate fenicsx-env
python -m pytest tests/test_urban_heat_dataset_v2.py tests/test_urban_wind.py \
    tests/test_smooth_reconstruction.py
```

The Helmholtz solver requires complex PETSc. Run its numerical accuracy checks
in the [complex-number environment](docs/installation.md#complex-number-environment):

```bash
conda activate fenicsx-env-complex
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[test,helmholtz,helmholtz-mesh]"
python -m pytest tests/test_helmholtz_manufactured.py tests/test_helmholtz_numerical.py
```

Shallow-water tests launch a separate real-PETSc FEniCSx worker. Set
`DTCC_SWE_PYTHON` to that environment's Python and run
`python -m pytest tests/test_shallow_water.py tests/test_shallow_water_dataset.py tests/test_shallow_water_demos.py`.
The city demo additionally requires `DTCC_RUN_DEMOS=1` and `--run-expensive`.
See the [shallow-water runtime setup](docs/shallow-water.md).

Passing tests with mocked solvers does not validate the numerical solvers.
The [dataset QA matrix](docs/datasets/qa-matrix.md) describes validation scope.
Expensive tests require an explicit opt-in, even when selected:

```bash
python -m pytest -m expensive --run-expensive
```

Use `python -m pytest` in the activated Conda environment and `uv run pytest`
in the uv project environment.
