# Equation solvers and application datasets

Use equation names for numerical APIs and application names for datasets:

| Numerical API | Parameters | Application dataset |
| --- | --- | --- |
| `HelmholtzSolver` | `HelmholtzParameters` | `urban_noise_simulation` |
| `NavierStokesSolver` | `NavierStokesParameters` | `urban_wind_simulation` |
| `StokesSolver` | `StokesParameters` | `urban_wind_simulation(equations="stokes")` |
| `HeatSolver` | `HeatParameters` | `urban_heat_simulation` |

Equation packages keep solver implementations in `solver.py` and configuration
in `parameters.py`:

| Equation | Solver module | Parameter module |
| --- | --- | --- |
| Helmholtz | `dtcc_sim.helmholtz.solver` | `dtcc_sim.helmholtz.parameters` |
| Navier–Stokes | `dtcc_sim.navierstokes.solver` | `dtcc_sim.navierstokes.parameters` |
| Stokes | `dtcc_sim.stokes.solver` | `dtcc_sim.stokes.parameters` |
| Heat | `dtcc_sim.heat.solver` | `dtcc_sim.heat.parameters` |

Import public classes from `dtcc_sim` or the equation package. The private
`_flow` package supplies the common Stokes/Navier–Stokes formulation.

The flow solvers use the corresponding parameter type and cannot switch
equations behind their names. These names replace the former application-named
solver classes; there are no compatibility aliases.

```python
from dtcc_core import BoxMesh
from dtcc_sim import HelmholtzSolver, HelmholtzParameters

mesh = BoxMesh(8, 8, 8)
parameters = HelmholtzParameters(boundary_mode="impedance_only", n_subdomains=2)
solver = HelmholtzSolver(mesh=mesh, params=parameters, output_dir="output/waves")
result = solver.simulate()
result.info()
```

Pass a DTCC `VolumeMesh` to a 3D solver; conversion to FEniCSx stays inside Sim.
Heat and flow require tagged boundary faces, supplied by `BoxMesh`. Native DTCC
mesh conversion currently uses one MPI rank. Heat and flow use the real PETSc
environment; Helmholtz uses the complex environment.

All finite-element solvers and smooth reconstruction use Float64 local
coordinates for assembly, solving and numerical field sampling. Sources,
observation points and custom spatial callbacks use the input coordinate frame;
the solver transforms them internally. Built-in wind profiles use height above
the domain base.

Results and XDMF/VTU exports retain the input mesh coordinates. Internal
translations are restored before serialization, so geographic results can be
overlaid without manually moving them. Supplied meshes retain their coordinates
exactly even when a solve fails. A coordinate translation does not reproject a CRS.

`solver.diagnostics["coordinate_frame"]` records the local-to-input translation,
declared CRS and computation/output frames. Native mesh CRS is retained; city
generation declares EPSG:3006. For an XDMF or backend mesh with known CRS, pass
`mesh_crs="EPSG:3006"` to the solver. Omitted CRS stays unknown. Helmholtz stores
the frame in its preparation data, diagnostics and reloadable result.
Solver-managed exports also write `<stem>.coordinates.json` beside the mesh.
For the standard output name this is `solution.coordinates.json`; flow's
velocity and pressure files share that record. The sidecar is replaced together
with successful solver output.

Core also provides `RectangleMesh(nx, ny)`, a triangular `Mesh` on the unit
square. It is a surface mesh, not a tetrahedral input for these 3D solvers.
See the [installation guide](installation.md#solver-output) for outputs and
[demo catalog](../demos/README.md) for complete examples.

The [stationary flow guide](stationary-flow.md) covers Navier–Stokes and Stokes
configuration, pressure gauges, traction boundaries, numerical acceptance and
failure behavior.

Every solver prints a wall-clock breakdown at the end of a run. See
[run timings](timings.md) for phase meanings, diagnostics, saved reports and MPI
aggregation.

## City geometry output

Spatial `simulate()` calls and simulation datasets save a `city.vtu` context
surface when the mesh has
DTCC physical boundary tags: ground `-1`, building facets `0` and above.
Artificial top and side markers `-2` through `-6` are excluded. The surface
reuses the simulation boundary geometry; no additional city download or meshing
is needed. It contains compact triangle connectivity and boundary markers,
without solution fields, and stays in the same input coordinates as the solution.

This applies to Navier–Stokes, Stokes, heat, reconstruction and Helmholtz,
including their application datasets. Native Core meshes and tagged backend
meshes are supported. Reconstruction can recover tags from a supplied city
XDMF file; an untagged backend mesh has no city geometry to export. Distributed
meshes collect owned physical exterior facets once on rank zero.

Open `city.vtu` in ParaView and show it as a grey surface. Open the corresponding
`solution.vtu`, `solution.xdmf`, or `solution_velocity.xdmf` alongside it, and
apply a Slice filter to the solution. For wind, colour by velocity magnitude.
The city and field line up without translation.

`city.coordinates.json` records the coordinate frame and CRS. Repeated exports
replace the context files and remove stale context when an input has no physical
tags. An explicit solution named `city.vtu` uses `city_surface.vtu` for its
context instead, preserving the solution file. Context export is included in
`export / city_surface` timing, or `postprocessing / export / city_surface` for
the direct Helmholtz API. No context files are written for in-memory-only calls.

## Shared city meshing

Bounds-based heat, flow and reconstruction solvers share `CityMeshSettings`.
Their parameters and application datasets expose `mesh_lod`, `mesh_lod2_method`,
`mesh_flat_ground`, `mesh_min_building_detail`, `mesh_repair_boundary_corners`
and `mesh_cache_dir`, alongside target size, domain height and terrain raster
controls. The general API default is LOD1; the city demos explicitly use
`mesh_lod="lod2"`. The Stockholm sensor demo explicitly uses LOD1: its larger
LOD2 volume currently fails inside TetGen. This is a geometry limitation, not
an automatic runtime fallback. Catalogue reconstruction fits roof geometry to source points;
it does not guarantee that every roof has multiple planes.

Set `mesh_cache_dir` to reuse an intact generated mesh across solvers. Geometry
options and mesher dependency identities determine its key; thermal, wind and
wave parameters do not. Changed LOD, terrain, repair or resolution selects a new
entry. Cache hits validate the saved file hashes and avoid data acquisition and
meshing. Corrupt entries are rejected rather than silently replaced. Matching
mesher dependencies are required for real and complex environments to share an
entry. Mesh caches stay separate from solution outputs.

The public `dtcc_sim.meshing` module supplies `CityMeshParameters` and
`CityMeshCache`. Helmholtz accepts the same geometry through `mesh_options`
and `mesh_cache_dir`; it also re-exports both classes from `dtcc_sim.helmholtz`.
The demos select `output/city-mesh-cache` explicitly. Uncached scalar/flow calls
still work with `mesh_cache_dir=None`; Helmholtz uses a sibling cache
when no cache path is supplied.

Generated results retain mesh provenance in
`solver.diagnostics["mesh_provenance"]`, the returned model's diagnostics, and
`mesh_provenance.json` alongside saved fields. The record identifies the geometry
request, cache reuse and cached file hashes. Native or explicit file inputs do
not trigger generation or implicitly apply corner repair.

The [shared-mesh validation record](validation/shared-city-mesh.json) reports
live demo runs, verified mesh identity, numerical volume checks and measured
cache reuse. The tested shared Poseidon LOD2 mesh has 6,011 vertices and 27,474 positive
volume tetrahedra. Its first preparation took about 4 s; warm preparation across
solvers took 0.03–0.06 s in the recorded environments. These timings include
source-data cache state and are not general performance guarantees.
