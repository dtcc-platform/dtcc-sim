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
installation adds Sim, its pinned Core revision, and the test and service extras.

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

## Helmholtz environment

Helmholtz needs complex PETSc. Create a separate environment rather than
changing the environment used for heat and wind:

```bash
conda env create -f environment-helmholtz.yml
conda activate dtcc-helmholtz
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[helmholtz,helmholtz-mesh]"
python -m dtcc_sim.helmholtz.check_runtime --workers 16 --mesh-generation
```

The recipe supplies the complex numerical stack, a Conda C++ compiler,
CMake 3.24 or later, and Ninja for native dependencies. The TetGen wrapper
requires CMake 3.24 or later; an older system CMake is insufficient. Install
the Python extras after activating the environment. The
`helmholtz-mesh` extra obtains the volume-meshing dependencies through Core,
including its Git-pinned TetGen wrapper. This installation requires network
access and may compile native code. When only using existing XDMF meshes,
install `.[helmholtz]` and omit `--mesh-generation` from the check.

The readiness command imports libraries, verifies complex PETSc, the supported
DOLFINx version, one MPI rank, and whether the current process can start the
requested patch workers. It does not generate meshes, run the solver, or
contact a service. Passing it does not validate a particular mesh, available
memory, or a future Celery worker process. Launch Python once; do not launch
one MPI process per core. Select parallelism through `parameters.workers`.

Helmholtz execution in this contribution uses Python or the local command-line
launchers. It is excluded from the queued service registry; the general worker
and Docker/Compose deployment do not supply its dedicated complex runtime.
See the [Helmholtz quick start](helmholtz-quickstart.md) for a complete run.

## Core dependency and local development

`pyproject.toml` and the Helmholtz installer pin the same official Core revision.
`uv.lock` records that commit and its dependencies. To adopt another tested Core
revision, update both project dependency declarations and the installer source
pin, regenerate its `SHA256SUMS`, and resolve the lockfile:

```bash
python .github/scripts/update_core_pin.py --sha <full-core-commit>
uv lock --upgrade-package dtcc-core
```

Core and Sim develop together. Test a new Core revision before adopting it,
and align the installer's Mesher pin with Core's declared dependency. Commit
the source pins, checksums and `uv.lock` together with dependency changes.

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

For the optional recorded Linux x86-64 environment, see the
[pinned installation procedure](helmholtz-reproducible-install.md). It includes
the pinned Core revision and guarded packaging workaround, with an option to
run focused checks and small real solves.
