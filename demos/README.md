# Demos

Run from the repository root in the appropriate environment, for example:

```bash
conda activate fenicsx-env
python demos/urban_heat_cooling.py
```

Follow the [demo guidelines](../docs/demo-guidelines.md) when adding or revising
examples. See [CONTRIBUTING.md](../CONTRIBUTING.md#demos) for verification and
[installation](../docs/installation.md) for dependencies. Close plots and viewers
to continue where provided. Existing scripts are not all compliant yet.

## Dataset examples

| Demo | Purpose | Runtime |
| --- | --- | --- |
| [dataset_urban_heat.py](dataset_urban_heat.py) | Heat through the dataset API | Real FEniCSx; city data |
| [dataset_urban_wind_stokes_simulation.py](dataset_urban_wind_stokes_simulation.py) | Stokes wind through the dataset API | Real FEniCSx; city data |
| [dataset_air_quality_field.py](dataset_air_quality_field.py) | Air-quality field reconstruction through the dataset API | Real FEniCSx; live data |
| [dataset_helmholtz.py](dataset_helmholtz.py) | Helmholtz dataset launcher; requires arguments and `--run` | Complex FEniCSx; supplied mesh or city data |
| [traffic.py](traffic.py) | Traffic assignment using road and population/employment data | Standard Python environment; live data |

## Solver examples

| Demo | Purpose | Runtime |
| --- | --- | --- |
| [urban_heat_cooling.py](urban_heat_cooling.py) | Cooling with cold ambient conditions | Real FEniCSx; city data |
| [urban_heat_heatwave.py](urban_heat_heatwave.py) | Heatwave with hot surfaces | Real FEniCSx; city data |
| [urban_wind_simulation.py](urban_wind_simulation.py) | Urban Navier–Stokes flow | Real FEniCSx; city data |
| [urban_wind_stokes_simulation.py](urban_wind_stokes_simulation.py) | Stationary urban Stokes flow | Real FEniCSx; city data |
| [smooth_reconstruction.py](smooth_reconstruction.py) | Smooth fields from point observations | Real FEniCSx; city data |
| [driven_cavity.py](driven_cavity.py) | Lid-driven cavity | Real FEniCSx; synthetic |
| [channel_flow.py](channel_flow.py) | Poiseuille channel flow | Real FEniCSx; synthetic |
| [bluff_body.py](bluff_body.py) | Flow around an obstruction | Real FEniCSx; synthetic |
| [helmholtz_box.py](helmholtz_box.py) | Gaussian source and amplitude slice; requires `--output` | Complex FEniCSx; synthetic |
| [helmholtz_small.py](helmholtz_small.py) | Small city solve and export checks; requires `--output` and `--run` | Complex FEniCSx; city data |
| [helmholtz_result.py](helmholtz_result.py) | Configured workflow and exports; requires `--config` and `--run` | Complex FEniCSx; configured inputs |

## Requirements and limitations

Real solvers use `fenicsx-env`; Helmholtz uses `fenicsx-env-complex`. City and
station workflows may download data and need access to their upstream services.
Scripts currently use differing output paths; inspect their settings before
running. Heat solver examples write `output/cooling.xdmf` and
`output/heatwave.xdmf` relative to the repository root; open these in ParaView.

Helmholtz launchers currently require new result directories and overlap in
purpose. See the [box example](../docs/helmholtz-visual-demo.md),
[small city example](HELMHOLTZ_SMALL.md), and
[workflow guide](../docs/helmholtz-quickstart.md) for their arguments and inputs.
`helmholtz_plot.py` is a plotting helper, not a runnable demo; `configs/` and
`run_helmholtz_detached.sh` are operational assets awaiting separation from demos.
Helmholtz amplitude is not sound-pressure level or dB.
