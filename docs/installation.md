# Installation guide

For the standard source checkout, follow the [README](../README.md#installation).
DTCC Sim requires Python 3.12 or later and a C++17 compiler for the Core and
Mesher dependencies. Run the commands below from the Sim repository root.

## Python environment with uv

`uv sync` creates `.venv`, installs the versions recorded in `uv.lock`, and
installs Sim in editable mode with test and service dependencies. This
environment supports traffic simulation, dataset inspection and service
development. Queued service jobs also need a broker and worker.

Heat, wind and air-quality reconstruction require FEniCSx/dolfinx with MPI/PETSc.
All simulation datasets can be listed and described in the uv environment;
solver execution requires the FEniCSx environment below.

## FEniCSx environment

Install a Conda-compatible distribution, then create the environment from the
checked-in recipe:

```bash
conda env create -f environment-fenicsx.yml
conda activate fenicsx-env
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[test,service]"
```

The recipe supplies Python 3.12, DOLFINx 0.11.0, MPI/PETSc bindings, PyVista and
uv. DOLFINx 0.11.0 is the supported solver baseline, also used by the Docker
image. Other versions are outside the tested support contract. The editable
installation adds Sim, Core from `develop`, and the test and service extras.

Verify the environment:

```bash
python -c "import dolfinx, mpi4py, petsc4py, dtcc_core, dtcc_sim"
```

For later shells, activate it with `conda activate fenicsx-env`. To update an
existing environment to the checked-in recipe:

```bash
conda env update -f environment-fenicsx.yml
```

Keep this environment separate from `.venv`. After activating Conda, use
`python` and `python -m pytest`; `uv sync` and `uv run` select the project
environment. The Conda workflow uses its installed scientific stack and does
not use `uv.lock`. Docker uses the same Conda plus `uv pip` approach.

## Core dependency and local development

`pyproject.toml` declares Core's `develop` branch; `uv.lock` records its resolved
commit. A plain `uv sync` keeps that snapshot. To adopt the latest Core revision:

```bash
uv sync --upgrade-package dtcc-core
```

Core and Sim develop together, with the latest Core `develop` as the
compatibility target. Commit `uv.lock` when adopting a new Core commit or
changing dependencies in `pyproject.toml`.

To test against a sibling Core checkout in the uv environment:

```bash
uv pip install -e ../dtcc-core
uv run --no-sync pytest tests/test_dtcc_core_contract.py
```

Use `--no-sync` while testing that override. Run `uv sync` to restore the locked
Core snapshot. In the activated FEniCSx environment, install the override with:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -e ../dtcc-core
```

For test commands and development conventions, see
[CONTRIBUTING.md](../CONTRIBUTING.md).
