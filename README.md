# DTCC Sim

DTCC Sim is a Python library for urban simulation. It provides heat and wind
solvers, Helmholtz wave simulation, air-quality field reconstruction, and static traffic assignment,
using DTCC Core models and datasets. The heat, wind and reconstruction solvers
are built on FEniCSx.

DTCC Sim is part of the [DTCC Platform](https://github.com/dtcc-platform/).

## Installation

Install from source using [uv](https://docs.astral.sh/uv/). You need Python 3.12
or later and a C++17 compiler for the Core and Mesher dependencies.

```bash
git clone https://github.com/dtcc-platform/dtcc-sim.git
cd dtcc-sim
uv sync
```

Run Python scripts in the installed environment with `uv run python`.
This environment supports traffic simulation and dataset inspection. For heat,
wind and air-quality reconstruction, follow the
[FEniCSx installation guide](docs/installation.md#fenicsx-environment).
The Helmholtz solver requires complex PETSc; use the
[complex-number environment](docs/installation.md#complex-number-environment).

## Quickstart

Inspect the traffic simulation dataset. From the repository root, start Python:

```bash
uv run python
```

Then run:

```python
import dtcc_core.datasets as datasets
import dtcc_sim.datasets

metadata = datasets.traffic_simulation.describe()
print(metadata["description"])
```

Importing `dtcc_sim.datasets` registers the simulation datasets with Core.
This example reads local metadata. To run a traffic simulation for Gothenburg,
see the [traffic demo](demos/dataset_traffic.py), which downloads road and population
and employment data and plots the resulting flows.

## Examples and documentation

- [Equation solvers](docs/solvers.md): DTCC mesh inputs and application dataset mapping.

- [Demo catalog](demos/README.md): dataset and solver examples with runtime requirements.
- [Installation guide](docs/installation.md): FEniCSx setup and local Core development.
- [Simulation datasets](docs/datasets/simulations.md): available simulations and result formats.
- [Helmholtz quickstart](docs/helmholtz-quickstart.md): setup and running a simulation.
- [Helmholtz workflow](docs/helmholtz.md): spectral preconditioning, city meshes and complex result fields.
- [Helmholtz visual demo](docs/helmholtz-visual-demo.md): self-contained Gaussian source and amplitude slice.
- [Dataset QA matrix](docs/datasets/qa-matrix.md): validation scope.
- [DTCC Platform documentation](https://platform.dtcc.chalmers.se/).
- [Architecture and design](DESIGN.md).

## Contributing

Questions, bug reports and contributions are welcome through the repository's
[issues](https://github.com/dtcc-platform/dtcc-sim/issues) and
[pull requests](https://github.com/dtcc-platform/dtcc-sim/pulls).
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests and coding
conventions.

## Credits

Developed at the [Digital Twin Cities Centre](https://dtcc.chalmers.se/),
supported by Sweden’s Innovation Agency Vinnova under Grant No. 2019-421 00041.

Authors (in order of appearance):

* [Anders Logg](http://anders.logg.org)
* [Vasilis Naserentin](https://www.chalmers.se/en/Staff/Pages/vasnas.aspx)
* [Marina Sahakyan](https://github.com/splinter1640)

## License

DTCC Sim is licensed under the [MIT license](LICENSE). Copyrights are held by
the individual authors as listed in the source files.
