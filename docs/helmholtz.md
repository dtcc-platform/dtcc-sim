# Helmholtz Python entry point

Start with the [installation and run quick start](helmholtz-quickstart.md).

`dtcc_sim.helmholtz` solves a complex scalar Helmholtz system on an existing
XDMF volume mesh using a hybrid overlapping preconditioner and spectral coarse
space. It requires a complex PETSc/DOLFINx environment and one MPI rank.
`workers` controls local eigenproblem processes, not MPI ranks. Numerical comparison passed for the saved 5 Hz, h=5 city cases with fixed
40 modes and rho-threshold selection; other cases require validation.

Install the optional Python dependencies in the activated FEniCSx environment:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[helmholtz]"
```

## Python usage

```python
from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzSimulator

def main():
    parameters = HelmholtzParameters(
        frequency_hz=5.0,
        rho=1.10,
        n_subdomains=16,
        overlap_layers=1,
        oversample_layers=3,
        workers=16,
        threads_per_worker=1,
        n_modes_per_patch=40,  # None selects modes by the rho threshold.
    )

    simulator = HelmholtzSimulator(
        mesh_path="/path/to/volume_mesh_gbg.xdmf",
        output_dir="/path/to/new-case",
        parameters=parameters,
    )

    print(simulator.describe())  # Configuration only; does not start a simulation.
    # result = simulator.run()  # Explicitly builds offline data, then solves.


if __name__ == "__main__":
    main()
```

`prepare()` builds the offline data. `solve()` requires matching prepared data
and returns `HelmholtzResult`: a complex solution array in FE DOF order,
iteration count, true relative residual, convergence flag, and GMRES history.
`run()` calls both stages. Output folders must be new or empty for preparation;
completed solve outputs cannot be overwritten through this interface.

Prepared data is checked against the XDMF and referenced HDF5 file contents,
parameters, and numerical-library versions. A different input, configuration,
or runtime requires a fresh case. The interface does not import environment
variables named `HELMHOLTZ_*`; use the parameter object instead.
Numeric controls and coefficient-map values must be finite numbers; boolean
controls require actual booleans. JSON coefficient-map keys may be integer
strings. Boundary tag selections require unique integer markers. These checks
run before numerical preparation and do not alter accepted coefficient values
or boundary selections.

Preparation manifests use schema 2 and include the SHA-256 checksum of the
saved preconditioner. Before and after the online solve, the interface checks
that file, the mesh inputs, settings, and runtime. Failed checks prevent a
completed `diagnostics.json` from being written; partial artifacts require a
fresh case directory. Checking the checksum reads the whole preconditioner
file but does not rebuild it. Older preparation manifests without a checksum
cannot be reused for a new solve; prepare a new case. Existing completed cases
can still be loaded and exported without rerunning their numerical stages.

BLAS/OpenMP thread limits apply during numerical calls and are restored afterward.
Fresh patch workers receive the limits and configuration explicitly. Python
scripts must guard their entry point with `if __name__ == "__main__":`, as above,
to prevent spawned workers from executing the launcher again. The parameter
context is isolated per call;
in-process numerical calls are serialized because MPI/PETSc and thread limits
are process resources.

## Command line

This prints the configuration without computing:

```bash
python -m dtcc_sim.helmholtz \
  --mesh /path/to/volume_mesh_gbg.xdmf \
  --output /path/to/new-case \
  --frequency 5 --rho 1.10 --subdomains 16 \
  --overlap 1 --oversampling 3 --workers 16 --modes 40
```

Add `--run` to explicitly perform both stages. Omit `--modes` for threshold
selection. The CLI supports common controls; Python exposes the full parameter
object.

## Artifacts and current scope

- `preconditioner_data.npz`: local operators, modes, decomposition, and coarse basis.
- `prepared.json`: input/configuration/runtime identity for reuse checks.
- `solution.npy`: full complex FE solution, when solution output is enabled.
- `solution.xdmf` and its HDF5 companion: visualization output if the backend writer succeeds.
- `gmres_history.txt`: history of the transformed, left-preconditioned system.
- `diagnostics.json`: final convergence flag and true relative residual.

The returned convergence flag reflects GMRES on the transformed system. Consult
the separately reported true residual when judging solution accuracy.
Fixed mode counts apply to the ARPACK path; dense eigenproblem fallback retains
threshold-based selection. The method also retains its existing cap behavior
and patch bulk assembly. These numerical limitations require separate validation.

This entry point accepts XDMF files; DTCC mesh generation and dataset
registration are available through the workflow and descriptor below.
Distributed MPI and PETSc outer-solver integration remain separate work. No existing heat, wind, traffic, or reconstruction
solver is changed by importing this module.

## Mesh selection and generation

Use `HelmholtzWorkflow` to coordinate DTCC mesh generation and the solver.
`CityMeshParameters` specifies EPSG:3006 bounds, target size `h`, domain height,
and minimum building detail. Its `options` accepts the other
`city_volume_mesh` dataset arguments, such as terrain raster resolution,
building conditioning, and TetGen volume limits. Boundary markers are enabled.

Install generation support in the activated complex FEniCSx environment:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[helmholtz,helmholtz-mesh]"
```

```python
from dtcc_sim.helmholtz import CityMeshParameters, HelmholtzParameters, HelmholtzWorkflow


def main():
    workflow = HelmholtzWorkflow(
        mesh_parameters=CityMeshParameters(
            bounds=(319745.962899, 6398759.716755, 320245.962899, 6399259.716755),
            h=5.0, domain_height=80.0, min_building_detail=1.0,
        ),
        mesh_cache_dir="/path/to/mesh-cache",
        output_dir="/path/to/new-case",
        parameters=HelmholtzParameters(
            frequency_hz=5.0, n_subdomains=16, overlap_layers=1,
            oversample_layers=3, workers=16, threads_per_worker=1,
            rho=1.10, n_modes_per_patch=40,
        ),
    )
    print(workflow.describe())  # No download, meshing, or solve.
    # workflow.prepare_mesh()  # Generate/reuse and check tags, without solving.
    # result = workflow.run()  # Full mesh → preconditioner → solve workflow.


if __name__ == "__main__":
    main()
```

The cache key includes all resolved dataset arguments and installed mesher
versions. An intact matching mesh is reused; a missing mesh is generated by
DTCC. File checksums detect changed cache contents. Failed generation is not
published, and concurrent generation of the same request is refused. A lock
left by a killed generator requires checking the old process before removing it.
Requested boundary markers absent from a mesh select no facets, matching the
numerical backend. The workflow records available, present, and absent markers
for impedance and Dirichlet conditions, and warns when a nonempty requested list
selects none. Mixed boundary mode still requires readable boundary markers.
Each prepared case records `mesh_provenance.json`, including `h` and the complete
meshing request. Changes to the mesh require new preconditioner data.
The dataset adapter labels generated city meshes as EPSG:3006. For a supplied
XDMF, the optional dataset argument `mesh_crs` records a caller-declared CRS;
omitting it leaves CRS metadata unspecified, as appropriate for a local box.
It does not transform coordinates. Dataset `bounds` must describe a positive
area in the input coordinate system.

Result export reads linear tetrahedral geometry and boundary/cell tags through
DOLFINx before attaching checked P1 fields to a DTCC volume mesh. This also supports
older multi-grid XDMF files whose HDF5 tag datasets use `Values` capitalization,
without modifying the original mesh or repeating the solve.
Export checks the saved preparation identity against the result's retained
identity, mesh name, and FE degree before loading geometry. Keep `prepared.json`
with the solved case; editing it requires a fresh case rather than relabeling
an existing solution. This consistency check does not authenticate artifacts
that were already modified before loading a saved result.

The archived `cube_2026_05_08_volume_family` generator explicitly marks every
exterior box face, including the bottom, as `-2` (`all_impedance`). Its box baseline
therefore has absorbing conditions on all six faces. This is distinct from the
sound-hard ground/building treatment described for urban meshes; reproducing the
box baseline preserves its original marking rather than changing the model.

`h` maps to DTCC's `max_mesh_size`, a target size for surface meshing; it is not
a guarantee that every tetrahedron edge has length `h`. DTCC derives the volume
cap from this size unless `max_volume` is specified. A newly generated mesh is
not guaranteed to reproduce a historical mesh with the same nominal `h`.
Cache reuse does not check whether upstream geographic data has been updated;
use a new mesh-cache directory when intentionally rebuilding from newer data.

For an existing mesh, use `mesh_path` instead of `mesh_parameters` and
`mesh_cache_dir`. Optional `mesh_h` records the user's declared resolution; it
is labeled as user supplied and is not inferred or verified from edge lengths.
A missing explicit file raises an error. To generate a missing mesh, provide
`mesh_parameters` with the geometry and generation settings.

## Configuration launcher

The example `demos/configs/helmholtz_city.json` exposes `h`, geometry, solver
parameters, and resources together. Relative paths resolve against the JSON
file's directory. Change the output directory for each new run.

The checked-in city example selects these settings:

| Setting | Value and meaning |
| --- | --- |
| Geometry | 500 m × 500 m EPSG:3006 area; domain height 80 m |
| Mesh | h=5 m target; minimum building detail 1 m; terrain raster cell 2 m and radius 3 m |
| Physics | Frequency 5 Hz; sound speed 343 m/s; constant coefficients A=V=beta=1 |
| Source | Gaussian, sigma=5 m, amplitude 1; mesh-bounds placement with the existing 0.9/0.1 height weights |
| Boundary conditions | Mixed mode; impedance markers -2 through -6; no Dirichlet markers selected |
| Preconditioner | 16 METIS subdomains; overlap 1 layer; oversampling 3 layers |
| Coarse space | Target 40 modes per patch on the ARPACK path; rho=1.10 controls dense fallback selection |
| Resources | One MPI rank; 16 patch workers; one thread per worker |
| Discretization and convergence | P1; transformed-system GMRES relative tolerance 1e-6, maximum 300 iterations, restart 100 |
| Output | New case directory `results/helmholtz/city-h5-f5-m16-modes40` relative to the repository |

Several entries above are parameter defaults; the planning command below shows
the complete effective configuration in the active environment. The source
centre can lie below terrain; its warning preserves the requested placement
rule. The true residual is reported separately from the GMRES tolerance.
The coarse-space size can differ from 16 × 40 if eigenproblem caps or dense
fallback apply. A matching cached mesh is reused; otherwise this configuration
generates a mesh before solving. That mesh need not be identical to the
historical h=5 mesh used for the saved solver comparisons.

```bash
# Inspect the resolved settings without computing:
python -m dtcc_sim.helmholtz.launch --config demos/configs/helmholtz_city.json

# Only generate/select the mesh:
python -m dtcc_sim.helmholtz.launch --config demos/configs/helmholtz_city.json --mesh-only

# Full workflow:
python -m dtcc_sim.helmholtz.launch --config demos/configs/helmholtz_city.json --run

# Detached Linux server job, after activating the environment:
bash demos/run_helmholtz_detached.sh demos/configs/helmholtz_city.json
```

The detached launcher survives SSH disconnection and stores `run.log`, `pid`,
and `exit-code.txt` under a fresh `run-logs` directory. Exit code zero means
convergence; examine `diagnostics.json` for the true residual. Geometry downloads
and meshing can fail independently of the solver; their errors appear in the log.

## DTCC result fields and visualization

A completed `HelmholtzResult` can return the input mesh as a DTCC `VolumeMesh`
with three scalar vertex fields: `helmholtz_real`, `helmholtz_imaginary`, and
`helmholtz_amplitude`. Amplitude is `abs(u)`; it is not a sound-pressure level.
Units are left unspecified unless supplied explicitly based on the physical
meaning of the modeled solution.

```python
mesh_with_fields = result.to_dtcc()
result.save_dtcc()  # <case>/helmholtz.vtu, with all three arrays for viewers.
result.save_dtcc(result.output_dir / "helmholtz.dtcc")  # Native DTCC model.
```

P1 DOF values are matched to the input mesh's vertices by absolute coordinates.
The mapping must be bijective within 1e-8 coordinate units; missing, duplicate,
or ambiguous coordinates and nonfinite values are rejected. Mesh contents and
numerical-library versions must still match the saved case. Export supports
FE degree 1; higher-order vertex evaluation is not currently implemented.

VTU stores the fields for visualization. The installed DTCC VTU loader currently
loads geometry only; use `.dtcc` to preserve fields when reopening in DTCC.
XDMF/HDF5 export is also supported. Existing exports are never overwritten.

The launcher can save fields after solving:

```bash
python -m dtcc_sim.helmholtz.launch --config demos/configs/helmholtz_city.json --run --export vtu
python demos/helmholtz_result.py --config demos/configs/helmholtz_city.json --run
```

The detached launcher includes VTU export automatically. For a completed case,
load the result and export without running the solver again:

```python
from dtcc_sim.helmholtz import HelmholtzResult

result = HelmholtzResult.load("/path/to/completed-case")
mesh_with_fields = result.to_dtcc()
result.save_dtcc()
```

Open the VTU in a mesh viewer and select `helmholtz_amplitude`, `helmholtz_real`,
or `helmholtz_imaginary`. A slice through the volume shows the interior field.

## Validation scope

The saved city comparisons used the same h=5 input mesh, 5 Hz, 16 METIS
subdomains, overlap 1, oversampling 3, and 16 patch workers. Fields were compared
by absolute mesh coordinates, not raw DOF index.

| Selection | Original / integrated iterations | Original / integrated coarse modes | Relative complex solution L2 difference | Integrated true residual |
| --- | --- | --- | --- | --- |
| Fixed 40 modes per patch | 15 / 15 | 640 / 640 | 9.19e-7 | 7.35e-6 |
| rho=1.10 threshold | 39 / 39 | 160 / 164 | 3.03e-6 | 9.56e-6 |

These comparisons passed the recorded criteria: solution difference <=1e-3,
GMRES convergence, and true residual <=max(2 times reference residual, 2e-5).
The integrated environment used DOLFINx 0.11.0 and complex PETSc 3.25.6; the
original used DOLFINx 0.9.0. The METIS partition changed, so coarse spaces are
not expected to be identical. Agreement is established for these two cases,
not for every mesh, frequency, boundary condition, or fallback path.

Actual DTCC generation produced a 331-vertex, 1,131-tetrahedron test mesh; a
full solve on it converged in 8 iterations with true residual 3.10e-6. Its
export mapping matched the saved FEniCS field. The full city export's 330,073
vertex values also matched its saved FEniCS field within 1e-12 tolerances.
Native DTCC field reload and VTU stored arrays were checked.

A small end-to-end acceptance check generated the same-sized DTCC test mesh
with h=20 m on a 100 m × 100 m flat-ground domain, height 40 m. With 5 Hz,
rho=1.10, two subdomains, two patch workers and one thread each, the registered
dataset converged in 8 iterations with true residual 3.10e-6. Canonical package
reload preserved geometry, fields and completed-run provenance exactly; saved
numerical-case reload reproduced the mapped fields. Schema-2 preconditioner
integrity was checked. This check took approximately 28 seconds including mesh
generation.

The bounding-box source rule is inherited from the solver. In the tested city
case, it places the Gaussian center at approximately (319995.97, 6399009.71,
13.53), outside the tetrahedral domain and below local terrain. The finite-width
Gaussian still contributes inside the domain. An exterior source center emits
a RuntimeWarning during RHS construction,
regardless of verbosity. The location and Gaussian forcing are left unchanged;
points on the mesh boundary are accepted. Numerical agreement does not
validate this physical source placement. Use `source_location_mode="manual"`
and an intended in-domain position when designing a new physical experiment;
changing the source requires a new case and separate numerical validation.

## Registered DTCC dataset

Importing `dtcc_sim.datasets` registers `datasets.helmholtz_simulation` alongside
the other simulations. Inspecting its descriptor or validating arguments does
not generate a mesh or run the solver:

```python
import dtcc_core.datasets as datasets
import dtcc_sim.datasets

print(datasets.helmholtz_simulation.describe())
```

Execution requires explicit `bounds` (four EPSG:3006 coordinates), `h`, and a
new/empty `output_dir`. For generation, also supply `mesh_cache_dir`; alternatively
supply `mesh_path` for an existing XDMF. With an explicit file, `h` is a user
provided label and does not modify or measure the mesh. `domain_height`,
`min_building_detail`, and `mesh_options` configure generated meshes.

All solver, physical, boundary, source, and worker settings use the same grouped
`parameters` object as the workflow. A Python dictionary is accepted and missing
settings receive the documented `HelmholtzParameters` defaults. For example:

```python
request = dict(
    bounds=[319745.962899, 6398759.716755, 320245.962899, 6399259.716755],
    h=5.0, output_dir="results/helmholtz/new-city-case",
    mesh_cache_dir="meshes/helmholtz", min_building_detail=1.0,
    parameters=dict(frequency_hz=5.0, n_subdomains=16, workers=16,
                    threads_per_worker=1, overlap_layers=1, oversample_layers=3,
                    rho=1.10, n_modes_per_patch=40),
)
# Calling the dataset performs the full workflow; uncomment only to execute:
# mesh = datasets.helmholtz_simulation(**request)
```

The Python result is a `VolumeMesh` with `helmholtz_real`,
`helmholtz_imaginary`, and `helmholtz_amplitude`. `field_unit` sets their units
when known. Request `format="dtccpkg"` for a portable model plus run provenance,
or `format="dtcc"` / `format="vtu"` for standalone mesh bytes.
XDMF bytes are deliberately unsupported here because an XML file alone omits
its HDF5 companion; the workflow/result API supports multi-file XDMF export.

The explicit case folder retains `diagnostics.json`, `gmres_history.txt`,
`prepared.json`, and `mesh_provenance.json`. A nonconverged GMRES result raises
an error rather than returning a successful field; diagnostics remain in that
folder. The true residual still needs interpretation separately from GMRES's
transformed-system convergence. `HelmholtzResult.load(output_dir)` reopens the
completed numerical result and diagnostics without rerunning the solver.

The [installation guide](installation.md#helmholtz-environment) provides the
separate complex-PETSc runtime. Local Python, foreground CLI and detached Linux
execution are supported. The general queued service excludes Helmholtz from
both discovery and task registration; dedicated supervision and deployment are
outside this contribution. Other simulation datasets keep their existing routes.
The dataset rejects daemonic execution with multiple patch workers before
starting a workflow.

A planning-only demo is available:

```bash
python demos/dataset_helmholtz.py --bounds 319745.96 6398759.71 320245.96 6399259.71 \
  --h 5 --output results/helmholtz/new-case --mesh-cache meshes/helmholtz \
  --frequency 5 --workers 16 --rho 1.10 --overlap 1 --oversampling 3 --modes 40
```

It prints validated settings. Add `--run` only when a new simulation is intended.

## Defaults and entry points

The Python API, dataset, and command-line solver options use the same
`HelmholtzParameters` defaults. Without overrides these include frequency 1 Hz,
sound speed 343, 16 subdomains, one worker, one thread per worker, overlap 1,
oversampling 1, and rho=0.5 threshold selection. A 16-core server does not
implicitly select 16 workers. The city JSON example explicitly selects its
5 Hz/16-worker/40-mode preset; it is not the general default.

| Entry point | Mesh information | Execution | Intended use |
| --- | --- | --- | --- |
| `HelmholtzSimulator` / `python -m dtcc_sim.helmholtz` | Existing XDMF file; no nominal h inference | Explicit `prepare`, `solve`, `run`, or CLI `--run` | Low-level numerical adapter |
| `HelmholtzWorkflow` / `python -m dtcc_sim.helmholtz.launch` | Complete generation request including h, or existing file with optional declared `mesh_h` | Explicit stage methods, `--mesh-only`, or `--run` | Reproducible configured workflow |
| `datasets.helmholtz_simulation` | Explicit bounds, h, output folder, and mesh cache or mesh file | Calling the dataset executes it; `describe()` and validation are offline | Standard DTCC dataset interface |

All command-line demos plan without executing unless `--run` or `--mesh-only`
is supplied. `--mesh-only` generates/loads the mesh but does not solve. Direct
Python stage methods and dataset calls are executable APIs, not planning calls.
Relative JSON paths resolve beside the JSON file; Python/dataset paths resolve
against the caller's working directory.

Offline and online stages take an exclusive `.helmholtz.lock` in the case
folder. A competing or interrupted writer is refused, and outputs are rechecked
after acquiring the lock. Normal exceptions release the lock. After a killed
process, check that its recorded PID is no longer running before removing the
lock; partial artifacts still require a fresh case directory.

`point_source` retains the backend's finite-width Gaussian behavior; it is not a
Dirac delta source. Alternative forcing and boundary configurations have not
been validated by the two saved city comparisons. CLI exports and the dataset
reject higher-order field export before starting a workflow. The low-level
result API also enforces degree 1 at conversion time.

## Portable run provenance

Completed dataset results carry convergence diagnostics in their
`dataset_context.provenance.processing_steps`, under the entry named
`Helmholtz completed run`. It records convergence, iterations, the true relative
residual, GMRES history, the prepared input/runtime identity, mesh provenance,
and the original case-folder path. The request context stores h, generation
settings, and the complete solver configuration. The case path is informational;
loading a package does not access that path or rerun the solver.

To save a completed dataset result without rerunning it, use Core's canonical
package exporter:

```python
from dtcc_core.datasets.package import export_model_package
from dtcc_core.datasets import load_model_package

# mesh is the VolumeMesh returned by a completed dataset call.
export_model_package(mesh, "helmholtz-run.dtccpkg", canonical=True)
restored = load_model_package("helmholtz-run.dtccpkg")
run = next(
    step for step in restored.dataset_context.provenance.processing_steps
    if isinstance(step, dict) and step.get("name") == "Helmholtz completed run"
)
print(run["iterations"], run["true_relative_residual"])
```

For dataset calls, `format="dtccpkg"` returns those package bytes directly.
The package uses Core's v3 manifest and native model artifact; Core
verifies declared file sizes and checksums when loading it.

Standalone `.dtcc` preserves geometry and fields but the installed Core writer
does not serialize `dataset_context`. VTU is also a standalone visualization
artifact. Use the package when moving settings and diagnostics with a dataset
result. The low-level `HelmholtzResult.save_dtcc()` remains a standalone mesh
export; its diagnostics and preparation identity remain in the case directory.
No numerical settings, preconditioner calculations, or source placement are
changed by packaging.
