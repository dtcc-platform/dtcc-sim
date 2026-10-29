# Demos

Run from the repository root in the appropriate environment, for example:

```bash
conda activate fenicsx-env
python demos/urban_heat_cooling.py
```

Follow the [demo guidelines](../docs/demo-guidelines.md) when adding or revising
examples. See [CONTRIBUTING.md](../CONTRIBUTING.md#demos) for verification and
[installation](../docs/installation.md) for dependencies. Close plots and viewers
to continue where provided.

## Dataset examples

| Demo | Purpose | Runtime |
| --- | --- | --- |
| [dataset_urban_heat.py](dataset_urban_heat.py) | Heat through the dataset API | Real FEniCSx; city data |
| [dataset_urban_wind.py](dataset_urban_wind.py) | Stationary Navier–Stokes wind through the dataset API | Real FEniCSx; city data |
| [dataset_urban_wind_stokes.py](dataset_urban_wind_stokes.py) | Stokes wind through the dataset API | Real FEniCSx; city data |
| [dataset_air_quality_field.py](dataset_air_quality_field.py) | Air-quality field reconstruction through the dataset API | Real FEniCSx; live data |
| [dataset_urban_noise.py](dataset_urban_noise.py) | City noise wave amplitude through the dataset API | Complex FEniCSx; city data |
| [dataset_traffic.py](dataset_traffic.py) | Traffic assignment using road and population/employment data | Standard Python environment; live data |

## Solver examples

| Demo | Purpose | Runtime |
| --- | --- | --- |
| [urban_heat_cooling.py](urban_heat_cooling.py) | Cooling with cold ambient conditions | Real FEniCSx; city data |
| [urban_heat_heatwave.py](urban_heat_heatwave.py) | Heatwave with hot surfaces | Real FEniCSx; city data |
| [urban_wind_simulation.py](urban_wind_simulation.py) | Urban Navier–Stokes flow | Real FEniCSx; city data |
| [urban_wind_stokes_simulation.py](urban_wind_stokes_simulation.py) | Stationary urban Stokes flow | Real FEniCSx; city data |
| [smooth_reconstruction.py](smooth_reconstruction.py) | Smooth fields from point observations | Real FEniCSx; city data |
| [driven_cavity.py](driven_cavity.py) | Smooth-lid stationary cavity | Real FEniCSx; synthetic |
| [channel_flow.py](channel_flow.py) | Exact planar Poiseuille flow | Real FEniCSx; synthetic |
| [bluff_body.py](bluff_body.py) | Stationary low-Reynolds-number obstruction flow | Real FEniCSx; synthetic |
| [helmholtz_box.py](helmholtz_box.py) | Offline Gaussian source, direct solver API and amplitude preview | Complex FEniCSx; synthetic |

## Requirements and limitations

For urban wind through the dataset API, run:

```bash
conda activate fenicsx-env
python demos/dataset_urban_wind.py
```

Dataset examples use the same sequence: request a native Core model, inspect it
with `.info()`, then preview its field with `.plot(...)`. The Stokes wind example
selects the linear equations explicitly. Direct solver examples expose parameter
objects and solver diagnostics for custom scenarios.

Geographic Gothenburg demos use a 200 × 200 m box centred on Poseidon
(EPSG:3006: 319995.962899, 6399009.716755). Each script defines `center_x`,
`center_y` and `L`, with bounds at centre ± `L / 2`. The air-quality demo keeps
its Stockholm domain and 100 m target size chosen for sensor coverage, using
LOD1 and the same cache infrastructure in a separate geometry entry.
The Stockholm LOD2 volume currently fails inside TetGen; the demo selects LOD1
explicitly. The Gothenburg LOD2 volume is supported.

All Gothenburg volume-mesh demos use the same terrain-following
200 × 200 × 80 m LOD2 geometry, 12.5 m target edge length and 0.5 m minimum
building detail. Roofs use Core's catalogue reconstruction and boundary corners
are repaired before solving. The demos share `output/city-mesh-cache` across
the real and complex solver environments. The first run generates the mesh;
subsequent demos verify and reuse the same files. Results retain map coordinates
in previews and exports, so geometry and fields overlay without translation.
Local geometry and quality constraints also affect actual element sizes.
Traffic uses a road network rather than a volume mesh. The direct reconstruction
example uses reproducible synthetic temperatures at approximate elevations;
observations outside the air volume are excluded. The air-quality dataset example
uses a 750 × 750 m Stockholm domain containing street stations; rooftop stations
with missing elevations cannot be placed above the ground mesh. It requires
finite NO2 observations and fails if none are available. The provider's latest
available values may be historical; inspect station timestamps before interpreting
the field as present conditions. It reconstructs concentrations, without modelling transport
or atmospheric chemistry. Heat boundary values and reconstructed temperatures are
in °C; wind speed is in m/s and wind pressure is kinematic pressure in m²/s².

Real solvers use `fenicsx-env`; Helmholtz uses `fenicsx-env-complex`. City and
station workflows may download data and need access to their upstream services.
All demos specify `output_dir="output/<demo-name>"`; the library creates it.
Heat and reconstruction write `solution.xdmf` and its HDF5 companion. Wind
writes `solution_velocity.xdmf` and `solution_pressure.xdmf` with companions.
Open these files in ParaView. Heat, wind and reconstruction demos with tagged
city meshes also save `city.vtu` and `city.coordinates.json`. Load the city as a
grey surface alongside a solution slice; the files share map coordinates.
Traffic exports `solution.dtcc` when given an
output directory. Dataset requests also accept `output_dir`.

The two wave examples run without arguments:

```bash
conda activate fenicsx-env-complex
python demos/helmholtz_box.py
python demos/dataset_urban_noise.py
```

Both use Core’s `.plot(field="helmholtz_amplitude")` for a 3D field preview
and save `solution.vtu`
in `output/helmholtz_box` or `output/dataset_urban_noise`. Repeated runs replace solver outputs
in place and preserve unrelated files; the city example shares `output/city-mesh-cache` with the other Gothenburg demos.
The city example also saves `city.vtu`: building and ground triangles from the
solver mesh, excluding the domain top and sides. Open it alongside `solution.vtu`
in ParaView for spatial context. Amplitude is in solver units, not sound-pressure
level or dB.

See the [box example](../docs/helmholtz-visual-demo.md) and
[workflow guide](../docs/helmholtz-quickstart.md) for physical assumptions and
advanced launchers. Reference configs and detached launch tooling live under
`tools/`; numerical comparisons use the
[city validation tool](../docs/helmholtz-city-validation.md).

The wind examples use the [stationary mixed solver](../docs/stationary-flow.md)
and preview `speed` through `result.plot(...)`. City wind examples request corner
repair during mesh generation and use the default iterative block solver.
See the [linear-solver measurements](../docs/flow-linear-solvers.md). Their 12.5 m target size and
200 × 200 × 80 m Poseidon domain remain explicit. These are stationary scenarios
with specified effective viscosity; they do not validate city turbulence. The
cavity uses a smooth tangential lid that vanishes at its edges. Channel and
bluff-body meshes use native Core geometry without Gmsh.
