# Simulation datasets

Importing `dtcc_sim.datasets` registers simulation datasets with
`dtcc_core.datasets`:

```python
import dtcc_core.datasets as datasets
import dtcc_sim.datasets

print(datasets.traffic_simulation.describe())
```

Call a dataset with `bounds=[xmin, ymin, xmax, ymax]` to run it.
Dataset execution fetches upstream geospatial data. Heat, wind and air-quality
reconstruction also require the
[FEniCSx environment](../installation.md#fenicsx-environment).

| Dataset | Python result | Formats | Output |
| --- | --- | --- | --- |
| `urban_heat_simulation` | `dtcc_core.model.VolumeMesh` | `xdmf`, `dtcc` | Temperature field from a steady-state heat equation. |
| `air_quality_field` | `dtcc_core.model.VolumeMesh` | `xdmf`, `dtcc` | Scalar concentration field reconstructed from station observations. |
| `urban_wind_simulation` | `dtcc_core.model.VolumeMesh` | `dtcc` | Velocity, pressure and speed fields from CFD. |
| `traffic_simulation` | `dtcc_core.model.RoadNetwork` | `dtcc` | Static user-equilibrium assignment with synthetic DeSO demand. |

Omitting `format` returns a Python model object. Supplying a supported `format`
returns serialized bytes. `describe()` exposes the argument schema,
`data_category`, `result_kind`, `python_return_type`, `supported_formats` and
`timeout_hint`.

## Exporting results

Use `format="dtcc"` for a single native artifact. Save the returned bytes to a
`.dtcc` file and load it with `dtcc_core.io.load_model(path)`. Core's public
`dtcc.proto` schema (`DTCC.ModelFile`) defines this format.

Vertex-valued fields use `association="vertex"`. Native heat and air-quality
exports preserve mesh vertices and nodal scalar samples, rather than the full
FEniCS function space. Legacy `.pb` model files are unsupported.

XDMF exports include companion HDF5 files. The service defaults to XDMF for heat
and air quality when no format is requested. It runs the dataset to obtain a
Python object, then serializes and packages the result with its companion files.

## Air-quality station elevations

Station metadata `elevation_source` distinguishes supplied elevations from
missing ones. Missing elevations are placed at the volume mesh's local ground
elevation plus `station_height_above_ground` (default 2 m), an estimated
instrument height. Supplied elevations, including zero, are preserved.
`z_offset` adds a further offset after elevation resolution.

A station with a finite observation outside the domain causes an error that
identifies the station and its resolved coordinates.

For validation scope, see the [dataset QA matrix](qa-matrix.md). Example scripts
are in [demos/](../../demos/).
