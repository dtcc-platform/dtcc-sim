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
| [dataset_urban_noise.py](dataset_urban_noise.py) | City noise wave amplitude through the dataset API | Complex FEniCSx; city data |
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
| [helmholtz_box.py](helmholtz_box.py) | Offline Gaussian source, direct solver API and amplitude preview | Complex FEniCSx; synthetic |

## Requirements and limitations

Geographic Gothenburg demos use a 200 × 200 m box centred on Poseidon
(EPSG:3006: 319995.962899, 6399009.716755). Each script defines `center_x`,
`center_y` and `L`, with bounds at centre ± `L / 2`. The air-quality demo keeps
its Stockholm domain chosen for sensor coverage.

All Gothenburg volume-mesh demos use a target mesh edge length of 12.5 m.
Local geometry and quality constraints also affect actual element sizes.
Traffic uses a road network rather than a volume mesh.

Real solvers use `fenicsx-env`; Helmholtz uses `fenicsx-env-complex`. City and
station workflows may download data and need access to their upstream services.
Solver demos specify `output_dir="output/<demo-name>"`; the solver creates it.
Heat and reconstruction write `solution.xdmf` and its HDF5 companion. Wind
writes `solution_velocity.xdmf` and `solution_pressure.xdmf` with companions.
Open these files in ParaView. Traffic exports `solution.dtcc` when given an
output directory. Dataset requests also accept `output_dir`.

The two wave examples run without arguments:

```bash
conda activate fenicsx-env-complex
python demos/helmholtz_box.py
python demos/dataset_urban_noise.py
```

Both use Core’s `.plot(field="helmholtz_amplitude")` for a 3D field preview
and save `solution.vtu`
in `output/helmholtz_box` or `output/urban_noise`. Repeated runs replace solver outputs
in place and preserve unrelated files; the city example reuses `output/urban_noise-mesh-cache`.
The city example also saves `city.vtu`: building and ground triangles from the
solver mesh, excluding the domain top and sides. Open it alongside `solution.vtu`
in ParaView for spatial context. Amplitude is in solver units, not sound-pressure
level or dB.

See the [box example](../docs/helmholtz-visual-demo.md) and
[workflow guide](../docs/helmholtz-quickstart.md) for physical assumptions and
advanced launchers. Reference configs and detached launch tooling live under
`tools/`; numerical comparisons use the
[city validation tool](../docs/helmholtz-city-validation.md).
