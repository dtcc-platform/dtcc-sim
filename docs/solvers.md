# Equation solvers and application datasets

Use equation names for numerical APIs and application names for datasets:

| Numerical API | Parameters | Application dataset |
| --- | --- | --- |
| `HelmholtzSolver` | `HelmholtzParameters` | `urban_noise_simulation` |
| `NavierStokesSolver` | `NavierStokesParameters` | `urban_wind_simulation` |
| `StokesSolver` | `StokesParameters` | `urban_wind_simulation(equations="stokes")` |
| `HeatSolver` | `HeatParameters` | `urban_heat_simulation` |

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

Core also provides `RectangleMesh(nx, ny)`, a triangular `Mesh` on the unit
square. It is a surface mesh, not a tetrahedral input for these 3D solvers.
See the [installation guide](installation.md#solver-output) for outputs and
[demo catalog](../demos/README.md) for complete examples.
