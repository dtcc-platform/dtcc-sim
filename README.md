# DTCC Sim

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
| `urban_heat_simulation` | `dtcc_core.model.VolumeMesh` | `xdmf` | Steady-state heat equation with temperature attached as a `Field`. `xdmf` is multi-file. |
| `air_quality_field` | `dtcc_core.model.VolumeMesh` | `xdmf` | PDE-smoothed sensor field attached as a `Field`. `xdmf` is multi-file. |
| `urban_wind_simulation` | `dtcc_core.model.VolumeMesh` | `pb` | CFD wind result with velocity, pressure, and speed fields. |
| `traffic_simulation` | `dtcc_core.model.RoadNetwork` | `pb` | Static user-equilibrium road assignment with synthetic DeSO demand. |

Dataset v2 review status: all registered simulation datasets return native DTCC
model objects when `format` is omitted. `urban_heat_simulation` and
`air_quality_field` return `VolumeMesh` objects with scalar fields attached;
`urban_wind_simulation` returns a `VolumeMesh` with velocity, pressure, and
speed fields; `traffic_simulation` returns a `RoadNetwork`.

The service wrapper strips `format` before running a dataset, then serializes
the returned Python object in `service.results.handle_result()`. This is
intentional: it preserves companion files for multi-file outputs such as
`xdmf`.

## Installation

### Recommended FEniCSx Developer Environment

Install Miniconda or another Conda-compatible distribution first:

    https://www.anaconda.com/docs/getting-started/miniconda/

Create the FEniCSx development environment from the checked-in recipe:

```bash
conda env create -f environment-fenicsx.yml
conda activate fenicsx-env
```

The recipe installs Python 3.12, FEniCSx/dolfinx, MPI/PETSc bindings, PyVista,
test dependencies, the sibling `../dtcc-core` checkout, and this package in
editable mode.

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

Static and cheap tests are the default development loop:

```bash
pytest tests/test_dataset_qa.py tests/test_import.py tests/test_traffic.py
```

Simulation tests use pytest markers:

- `simulation`: simulation dataset or solver validation tests.
- `fenics`: tests that require a FEniCSx/dolfinx runtime.
- `slow`: small numerical tests that are slower than pure unit tests.
- `expensive`: manual-scale or city-scale simulations.

Run non-expensive simulation checks:

```bash
pytest -m "simulation and not expensive"
```

Run FEniCSx-specific checks:

```bash
pytest -m "fenics and not expensive"
```

Run slower numerical smoke tests:

```bash
pytest -m "slow and not expensive"
```

Expensive tests are skipped by default even when selected. They require an
explicit flag:

```bash
pytest -m expensive --run-expensive
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
