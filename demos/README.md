# Demos

Run from the repository root in the appropriate environment, for example:

```bash
conda activate fenicsx-env
python demos/urban_heat_cooling.py
```

The wind demos also run unchanged in parallel, for example
`mpirun -n 4 python demos/dataset_urban_wind.py`; output comes from rank zero.
See [stationary flow](../docs/stationary-flow.md#mpi).

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
| [dataset_urban_flood.py](dataset_urban_flood.py) | Experimental shallow-water pulse on flat city geometry | Real FEniCSx 0.11; live city data |
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
| [shallow_water_pulse.py](shallow_water_pulse.py) | Experimental wetting/drying pulse on a closed unit square | Real FEniCSx 0.11; offline |

## Requirements and limitations

The shallow-water examples use the retained experimental reference in the
common real DOLFINx 0.11 environment. Activate `fenicsx-env`, then run
`python demos/shallow_water_pulse.py` or `python demos/dataset_urban_flood.py`.
They need no arguments or interpreter override. Complex-PETSc callers can
explicitly select a real 0.11 worker as described in the runtime guide. The
library replaces `solution.dtcc`, `solution_finaly.vtu` and coordinate sidecars on
repeated calls, preserves unrelated files, and retains raw runs
under `.shallow_water_runs`. The city example uses Core's `city_flat_mesh`,
not the tetrahedral volume cache. It has a flat bed, reflecting boundaries,
and an idealized pulse, not rainfall or a calibrated flood event. See
[shallow-water scope and usage](../docs/shallow-water.md).
Both demos call `.info()` and `.plot(projection="xy", field="water_depth", ...)`
on the returned native Mesh. Core fills conservative face values; tagged
building faces provide neutral context. Explicit titles supply simulation time,
and known metric coordinates support metre axes and a scale bar. A reopened
native dataset package supports the same plot without its numerical run files.
The generated city dataset also saves `city.vtu` and `city.coordinates.json`:
LOD1 buildings with estimated heights and a flat ground at z=0 matching the
water mesh. Load it beside `solution.pvd` in ParaView. The solution includes DG1
Point Data for direct Warp By Scalar with `water_depth`, alongside conservative
Cell Data; see the [visualization settings](../docs/shallow-water.md).
Supplied 2D mesh inputs
do not supply building heights, so they produce no inferred city surface.
Use `ShallowWaterResult.load(...)` separately for transient inspection/export or
its snapshot-plot convenience method.
Both demos automatically export `solution.pvd` at the output root, referencing
frames under `series/`. The city demo saves 101 physical time states; the offline
demo saves 51. Open that index with `city.vtu` for city context;
`solution_finaly.vtu` is a static final state. The finite pulse spreads into shallow
water; a warp scale above 1 exaggerates height without changing depth values.

The shallow-water city demo uses Chalmers Johanneberg in EPSG:3006, centred at
`(319970, 6397910)` with `L = 250` m and bounds
`(319845, 6397785, 320095, 6398035)`. Its compact 0.2 m pulse starts at
`(320005, 6397890)` on open ground, with a 7 m width. The 100-second scenario
uses CFL 0.24 and saves states every second to show propagation around
the building interfaces. Runtime depends on mesh size and hardware; the city
scenario takes longer than the offline unit-square example.

For urban wind through the dataset API, run:

```bash
conda activate fenicsx-env
python demos/dataset_urban_wind.py
```

Dataset examples use the same sequence: request a native Core model, inspect it
with `.info()`, then preview its field with `.plot(...)`. The Stokes wind example
selects the linear equations explicitly. Direct solver examples expose parameter
objects and solver diagnostics for custom scenarios.

Geographic Gothenburg demos use a 250 × 250 m box centred on Chalmers Johanneberg
(EPSG:3006: 319970, 6397910). Each script defines `center_x`,
`center_y` and `L`, with bounds at centre ± `L / 2`, giving
`(319845, 6397785, 320095, 6398035)`. The air-quality demo keeps
its Stockholm domain and 100 m target size chosen for sensor coverage, using
LOD1 and the same cache infrastructure in a separate geometry entry.
The Stockholm LOD2 volume currently fails inside TetGen; the demo selects LOD1
explicitly. The Gothenburg LOD2 volume is supported.

All Gothenburg volume-mesh demos use the same terrain-following
250 × 250 × 80 m LOD2 geometry, 12.5 m target edge length and 0.5 m minimum
building detail. Roofs use Core's catalogue reconstruction and request
boundary-corner repair when the Core meshing API exposes it. The demos share
`output/city-mesh-cache` across
the real and complex solver environments. The first run generates the mesh;
subsequent demos verify and reuse the same files. Results retain map coordinates
in previews and exports, so geometry and fields overlay without translation.
Local geometry and quality constraints also affect actual element sizes.
The noise source is at the domain centre, 2 m above sampled terrain.
Traffic uses a road network rather than a volume mesh. The direct reconstruction
example uses reproducible synthetic temperatures 5–15 m above sampled terrain;
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
Open these files in ParaView. City simulations also save `city.vtu` and
`city.coordinates.json` when context geometry is available. Load the city as a
grey surface alongside the solution field; the files share map coordinates.
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
250 × 250 × 80 m Chalmers Johanneberg domain remain explicit. The city wind examples use the
algebraic [atmospheric boundary-layer](../docs/atmospheric-boundary-layer.md)
closure with log-law walls: mixing length for Navier–Stokes, the equilibrium
profile for Stokes. They are stationary scenarios: the mixing length is compared with
wind-tunnel measurements around isolated buildings in the
[AIJ benchmarks](../benchmarks/aij/README.md), where it overpredicts wind in building
wakes, and is not validated for city areas. The
cavity uses a smooth tangential lid that vanishes at its edges. Channel and
bluff-body meshes use native Core geometry without Gmsh.
