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

## Helmholtz environment

Helmholtz needs complex PETSc. Create a separate environment rather than
changing the environment used for heat and wind:

```bash
conda env create -f environment-helmholtz.yml
conda activate dtcc-helmholtz
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[helmholtz,helmholtz-mesh]"
python -m dtcc_sim.helmholtz.check_runtime --workers 16 --mesh-generation
```

The recipe supplies the complex numerical stack and a Conda C++ compiler for
native dependencies. Install the Python extras after activating it. The
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

The general Docker image and Compose worker do not select this Helmholtz
environment. Helmholtz service tasks route to `dtcc-sim-helmholtz`; other tasks
retain the `dtcc-sim` queue. A dedicated launcher uses a non-daemonic solo
pool, consumes only the Helmholtz queue, and allows one simulation job at a
time. Each job chooses its own patch worker count through `parameters.workers`.

To prepare its Python dependencies and print the launch plan:

```bash
conda activate dtcc-helmholtz
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[helmholtz,helmholtz-mesh,service]"
python -m service.helmholtz_worker
```

Printing the plan does not connect to Redis or start a worker. Deployment is a
separate action: configure `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, and
`SHARED_RESULTS_DIR` consistently with the API, then explicitly use
`python -m service.helmholtz_worker --start`. That command checks the numerical
runtime before starting Celery. The API and both workers must run code with
the same routing configuration; restart them together when adopting it.
An API using the old routing would still send Helmholtz jobs to the general
worker. With the new routing, Helmholtz jobs wait in their dedicated queue
until its worker is running. The supplied Compose file starts only the general
worker; it does not deploy the complex environment or dedicated worker.

Both workers and the API need access to the same result files. Helmholtz case
and mesh-cache paths must be accessible to its worker; each job needs a new
case directory. Use a process supervisor when deploying for restart and log
management. A small isolated localhost acceptance check verified API submission,
dedicated-queue delivery, job completion, the shared result package and cleanup.
Bounded checks also covered queued/started cancellation, worker loss during child
initialization, timeout enforcement and a successful fresh job after worker
restart. Each Helmholtz job now runs in a supervised process session; the worker
can stop it and its patch descendants. `DTCC_JOB_TIMEOUT` includes child startup.
See the [job-control details](helmholtz.md#service-cancellation-timeouts-and-interruption)
for status semantics, retained logs, and fresh-case recovery. Broker-outage
recovery and production deployment policy remain unverified. The API reports
the shared result filename; public HTTP file delivery was not tested. See the
[Helmholtz workflow](helmholtz.md) for local and detached solver launch commands.

## User-service deployment preview

The [Helmholtz deployment recipe](../deploy/helmholtz/README.md) renders a
reviewable Linux user-systemd bundle with private Redis, a localhost API and the
dedicated worker. It includes restart policy, proposed resource limits and
persistent broker storage. Rendering and offline unit validation do not install
or activate services. Use `DTCC_DATASET_ALLOWLIST=helmholtz_simulation` for a
dedicated Helmholtz API/worker; leaving it unset preserves general discovery.

Before activation, arrange user-manager persistence after logout where needed,
review the proposed budgets and state directory, and validate broker restart
and resource-controller behavior. The renderer sets inherited CPU affinity as
well as a cgroup quota; verify actual kernel delegation before relying on the
quota. No permanent service is enabled by this
recipe. Public access, filesystem policy and retention remain separate
deployment decisions.

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
