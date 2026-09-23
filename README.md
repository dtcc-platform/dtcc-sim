# DTCC Sim

Native simulation exports use `format="dtcc"` and `.dtcc` files, decoded by
Core's public `dtcc.proto` (`DTCC.ModelFile`). Vertex-valued simulation fields
carry `association="vertex"`. Legacy `.pb` model files are not supported by the
updated Core dependency. The Core contract workflow checks native result delivery
as well as dataset descriptions.


DTCC Sim is a simulation package for DTCC Platform. It provides simulation
utilities and example workflows built around FEniCSx.

This project is part of the
[Digital Twin Platform (DTCC Platform)](https://github.com/dtcc-platform/)
developed at the
[Digital Twin Cities Centre](https://dtcc.chalmers.se/)
supported by Sweden’s Innovation Agency Vinnova under Grant No. 2019-421 00041.

## Documentation

This project is documented as part of the
[DTCC Platform Documentation](https://platform.dtcc.chalmers.se/).

## Dataset Integration

`dtcc-sim` registers simulation datasets into the `dtcc_core.datasets`
registry when `dtcc_sim.datasets` is imported:

    import dtcc_core.datasets as datasets
    import dtcc_sim.datasets

    result = datasets.urban_wind_simulation(bounds=[xmin, ymin, xmax, ymax])

Simulation datasets follow the same descriptor contract as core datasets:

- `format=None` returns a Python simulation result.
- `format=<value>` returns serialized bytes for direct download paths.
- `describe()` exposes `data_category="simulation"`, `result_kind`,
  `python_return_type`, `supported_formats`, and `timeout_hint`.

Current simulation datasets:

| Dataset | Python result | Formats | Notes |
| --- | --- | --- | --- |
| `urban_heat_simulation` | `dtcc_core.model.VolumeMesh` | `xdmf`, `dtcc` | Steady-state heat equation with temperature attached as a `Field`. `dtcc` is single-file; `xdmf` is multi-file. |
| `air_quality_field` | `dtcc_core.model.VolumeMesh` | `xdmf`, `dtcc` | PDE-smoothed sensor field attached as a `Field`. `dtcc` is single-file; `xdmf` is multi-file. |
| `urban_wind_simulation` | `dtcc_core.model.VolumeMesh` | `dtcc` | CFD wind result with velocity, pressure, and speed fields. |
| `traffic_simulation` | `dtcc_core.model.RoadNetwork` | `dtcc` | Static user-equilibrium road assignment with synthetic DeSO demand. |

Dataset v2 review status: all registered simulation datasets return native DTCC
model objects when `format` is omitted. `urban_heat_simulation` and
`air_quality_field` return `VolumeMesh` objects with scalar fields attached;
`urban_wind_simulation` returns a `VolumeMesh` with velocity, pressure, and
speed fields; `traffic_simulation` returns a `RoadNetwork`.

For heat and air quality, request `format="dtcc"` to receive a single native
artifact containing the mesh and its vertex-associated scalar field. Save the
returned bytes to a `.dtcc` file and load it with `dtcc_core.io.load_model(path)`.
This preserves the nodal samples, not the full FEniCS function space. XDMF remains
available, and remains the service default when no format is supplied for these
two datasets.

The service wrapper strips `format` before running a dataset, then serializes
the returned Python object in `service.results.handle_result()`. This is
intentional: it preserves companion files for multi-file outputs such as
`xdmf`.

## Runtime support

FEniCSx is optional for importing DTCC Sim and running traffic simulations.
The numerical heat, wind, and field-reconstruction solvers require a working
FEniCSx/dolfinx runtime with MPI/PETSc. Conda is the documented way to provide
that runtime; Conda itself is not a requirement of the Python package.

| Capability | uv environment without FEniCSx |
| --- | --- |
| Import `dtcc_sim`; list datasets and inspect their arguments and descriptions | Supported |
| Run `TrafficAssignmentSimulator` or `traffic_simulation` | Supported; dataset execution still needs its upstream road and demand inputs |
| Develop and test service routes, result handling, and dataset contracts | Supported; queued jobs additionally need a broker and worker |
| Run `urban_heat_simulation` | Requires FEniCSx |
| Run `urban_wind_simulation` | Requires FEniCSx |
| Run `air_quality_field` or `SmoothReconstructionSimulator` | Requires FEniCSx |

All four simulation datasets are registered even without FEniCSx. Listing or
describing a solver dataset does not execute its solver; attempting to run it
without the runtime fails when the solver is imported. Tests using mocked
solvers can pass without FEniCSx, while actual FEniCSx tests skip. Passing the
non-FEniCSx suite therefore does not validate the numerical solvers.

## Installation

### Python development with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a C/C++17
compiler (for the Core and Mesher dependencies), then run:

```bash
git clone https://github.com/dtcc-platform/dtcc-sim.git
cd dtcc-sim
uv sync --python 3.12
uv run pytest tests/test_dataset_qa.py tests/test_import.py tests/test_traffic.py
```

`uv sync` creates `.venv`, installs the versions recorded in `uv.lock`, and
installs Sim in editable mode with test and service dependencies. It uses the
Core commit pinned in `pyproject.toml`. Python edits take effect immediately.
This environment supports traffic, dataset contracts, and service development;
FEniCSx solver work uses the Conda environment below.

| Task | Command |
| --- | --- |
| Set up or update the environment | `uv sync` |
| Run the test suite (solver tests skip without FEniCSx) | `uv run pytest` |
| Build a source distribution and wheel | `uv build` |
| Add a dependency | `uv add <package>` |
| Upgrade a locked dependency | `uv lock --upgrade-package <package>` |

Commit `uv.lock` with dependency changes in `pyproject.toml`. CI uses
`uv sync --locked`; the Core pin automation updates both files together.

To test with a sibling Core checkout:

```bash
uv pip install -e ../dtcc-core
uv run --no-sync pytest tests/test_dtcc_core_contract.py
```

Use `--no-sync` for commands using that override. Run `uv sync` to restore the
pinned Core version.

### FEniCSx developer environment

Install Miniconda or another Conda-compatible distribution first:

[Miniconda installation](https://www.anaconda.com/docs/getting-started/miniconda/).

Create the FEniCSx development environment from the checked-in recipe:

```bash
conda env create -f environment-fenicsx.yml
conda activate fenicsx-env
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[test,service]"
```

Conda supplies Python 3.12, FEniCSx/dolfinx, MPI/PETSc bindings, PyVista, and uv.
The `uv pip install` command adds Sim, its pinned Core dependency, and the test
and service extras to that environment. To use local Core changes, follow it
with `uv pip install --python "$CONDA_PREFIX/bin/python" -e ../dtcc-core`.

Keep this solver environment separate from the project `.venv`: run its commands
with `python` / `python -m pytest` after activating Conda. Do not use `uv sync`
or `uv run` for solver tests; those select the project environment. The Conda
workflow resolves against its installed scientific stack and does not use
`uv.lock`. Docker uses the same Conda plus `uv pip` installation approach.

Verify the environment imports before running solver tests:

```bash
python -c "import dolfinx, mpi4py, petsc4py, dtcc_core, dtcc_sim"
```

### Activating The Environment

For new shells:

```bash
conda activate fenicsx-env
```

If Conda has not initialized the shell yet, run the activation script first:

```bash
source ~/miniconda3/bin/activate
conda activate fenicsx-env
```

### Test Tiers

The commands below use the uv project environment. In the activated FEniCSx
Conda environment, replace `uv run pytest` with `python -m pytest`.

Static and cheap tests are the default development loop:

```bash
uv run pytest tests/test_dataset_qa.py tests/test_import.py tests/test_traffic.py
```

Simulation tests use pytest markers:

- `simulation`: simulation dataset or solver validation tests.
- `fenics`: tests that require a FEniCSx/dolfinx runtime.
- `slow`: small numerical tests that are slower than pure unit tests.
- `expensive`: manual-scale or city-scale simulations.

Run non-expensive simulation checks:

```bash
uv run pytest -m "simulation and not expensive"
```

Run FEniCSx-specific checks in the activated Conda environment:

```bash
conda activate fenicsx-env
python -m pytest -m "fenics and not expensive"
```

Run slower numerical smoke tests:

```bash
uv run pytest -m "slow and not expensive"
```

Expensive tests are skipped by default even when selected. They require an
explicit flag:

```bash
uv run pytest -m expensive --run-expensive
```

If FEniCSx is not installed, FEniCS-only tests skip with an explicit reason
instead of failing during import. Dataset contract tests and traffic tests do
not require FEniCSx.

## Authors (in order of appearance)

- [Anders Logg](http://anders.logg.org)

## License

This project is licensed under the
[MIT license](https://opensource.org/licenses/MIT).

Copyrights are held by the individual authors as listed at the top of each source file.

## Community guidelines

Comments, contributions, and questions are welcome.
Please engage with us through Issues, Pull Requests, and Discussions on our GitHub page.
