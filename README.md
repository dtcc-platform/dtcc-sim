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

### Install Conda

The first step is to install (mini) Conda. Follow these instructions:

    https://www.anaconda.com/docs/getting-started/miniconda/

### Install FEniCSx

The next step is to create and activate a new environment fenicsx-env:

    source ~/miniconda3/bin/activate
    conda create -n fenicsx-env python=3.12
    conda activate fenicsx-env

Note the use of Python v3.12 which is required for installation of DTCC (below).

Next, we install FEniCSx and some other packages:

    conda install -c conda-forge fenics-dolfinx mpich pyvista

### Install DTCC Platform

Install DTCC Platform via PyPi:

    pip install dtcc dtcc-viewer

### Install DTCC TetGen wrapper

Install the DTCC TetGen wrapper. This needs to be installed from source
since we don't yet provide a PyPi package:

    git clone git@github.com:dtcc-platform/dtcc-tetgen-wrapper.git
    cd dtcc-tetgen-wrapper
    bash vendor_tetgen.sh
    pip install .

### Activating the environment (in new sessions)

If you have followed all the above instructions, you should have a
Conda environment that has both FEniCSx and DTCC Platform.

To activate the environment in new sessions (terminals), run the
commands

    source ~/miniconda3/bin/activate
    conda activate fenicsx-env

### Testing the environment

To test the environment, run the following commands from inside
the dtcc-repository:

    cd sandbox
    python build_volume_mesh_gbg.py
    python solve_poisson.py

## Authors (in order of appearance)

- [Anders Logg](http://anders.logg.org)

## License

This project is licensed under the
[MIT license](https://opensource.org/licenses/MIT).

Copyrights are held by the individual authors as listed at the top of each source file.

## Community guidelines

Comments, contributions, and questions are welcome.
Please engage with us through Issues, Pull Requests, and Discussions on our GitHub page.
