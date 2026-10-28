# Contributing to DTCC Sim

Questions, bug reports and pull requests are welcome. Include a small reproducer
when reporting a bug and describe the user-visible behavior of a proposed change.

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

Passing tests with mocked solvers does not validate the numerical solvers.
The [dataset QA matrix](docs/datasets/qa-matrix.md) describes validation scope.
Expensive tests require an explicit opt-in, even when selected:

```bash
python -m pytest -m expensive --run-expensive
```

Use `python -m pytest` in the activated Conda environment and `uv run pytest`
in the uv project environment.
