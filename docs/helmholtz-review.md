# Helmholtz integration review

This contribution connects the research Helmholtz preconditioner to DTCC's local
mesh-to-solve-to-result workflow. It supports supplied meshes or city mesh
generation/cache, explicit solver controls, two partition-of-unity constructions,
a registered dataset and checked P1 result exports. Start with the
[quick start](helmholtz-quickstart.md) and [API reference](helmholtz.md).

## Read the implementation in this order

1. [Parameters](../dtcc_sim/helmholtz/parameters.py) and
   [simulator](../dtcc_sim/helmholtz/simulator.py): explicit controls, separate
   preparation/solve, case locks and input/preconditioner/runtime identities.
2. [Mesh workflow](../dtcc_sim/helmholtz/workflow.py) and
   [cache](../dtcc_sim/helmholtz/meshing.py): explicit h, full generation request,
   checksummed reuse and boundary-marker reports.
3. [Weights](../dtcc_sim/helmholtz/_backend/pou.py): `geometric_hats` and
   `core_overlap_ramp`; the recovered geometric function is added while the
   existing ramp construction is retained.
4. [Results](../dtcc_sim/helmholtz/results.py) and
   [dataset](../dtcc_sim/helmholtz/dataset.py): bijective coordinate mapping,
   real/imaginary/amplitude fields, boundaries, declared CRS and provenance.

Existing configurations that omit `pou_mode` retain ramps. The dedicated
[thesis city profile](../demos/configs/helmholtz_thesis_city_h5.json) selects
geometric hats and rho-threshold modes. The choice is recorded and separates
prepared caches. PyMetis/METIS native-build fingerprints also separate caches,
because the same version string can produce a different partition. Completed
older results remain loadable/exportable; older prepared cases need fresh
preparation for another solve.

The contribution excludes specialized Celery/supervisor/systemd support.
General service discovery excludes Helmholtz so that unsuitable workers cannot
accept it. Other datasets retain their existing service behavior. The personal
Chalmers launcher is separate from generic library code.

## Evidence and its limits

The controlled comparisons, historical reconstruction, production city run and
isolated installation checks establish distinct claims. Numerical agreement
with the historical field is separate from bitwise equality.

| Question | Verified outcome | Scope |
| --- | --- | --- |
| Does the integration preserve the later ramp solver? | Original/integrated city: 39 iterations, 160 coarse modes, bitwise equal assembled systems, all 61 preconditioner entries, residual histories and final fields. | Same current runtime, restored historical partition and identical test-only ARPACK starting vectors. |
| Was the thesis city result recovered? | Recovered geometric source: 32 iterations, 281 modes, true residual 8.002162926752011e-6; field relative L2 difference 4.038710368992294e-14. | Table 4.17 h=5 city row, preserved original environment. Field is numerically equal to the archive, not bitwise identical. |
| Are both weight choices integrated correctly? | All full-city weights match their reference weights bitwise. Small real solves match each weight-source oracle in all 61 entries, histories and fields; both DTCC export workflows pass. | Full-city reference replay plus small source-oracle solves and export workflows. |

### Final production city workflow

The [production city report](validation/helmholtz-final-integrated-city.json)
checks the final selectable `geometric_hats` implementation on the exact original
h=5 city mesh, 5 Hz, rho 1.10, 16 workers/subdomains, overlap 1 and oversampling 3.
The historical partitioner build is pinned; the current integration runtime is
used and ARPACK initialization is unforced. It passed every criterion fixed
before launch: 32 iterations, 281 coarse modes and all 16 local retained-mode
counts match the archive. The partition and geometric weights match bitwise.

The independently reassembled true residual is 8.002162925941243e-6. The relative
complex field L2 difference is 3.650061829763775e-14 (maximum absolute difference
3.4961647556298987e-13); the field is not bitwise identical. All four exports
passed, with exact boundary and package-context round trips and 138,444 boundary
facets. The original inputs and source remained unchanged. This validates this
case, not every historical experiment or runtime.

### Isolated installation validation

The [installation report](validation/helmholtz-clean-install.json) records a new
separate Conda prefix on the same Linux host, pinned Python dependencies, fresh
native builds of Mesher, TetGen and patched Core, and installation of the reviewed
Sim wheel. All project imports came from this prefix, without the production
overlay. Dependency consistency and all 120 focused tests passed. Real small
runs generated a mesh, reused it for the other weight option, and passed both
solves and all four export checks. Each took 8 iterations; independent residuals
were 3.022592052942598e-6 (geometric) and 3.245275017206979e-6 (ramp).

This needed explicit CMake 3.31.10 and a reversible VTK metadata workaround.
The Conda VTK build supplied both valid legacy egg-info and incomplete dist-info;
the latter was preserved outside site-packages. No package binaries were
changed. The failed attempts remain in the validation records. See the
[installation notes](helmholtz-installation-validation.md) for the prerequisite,
workaround and scope. This is a same-host installation with cached upstream
geodata, not a pristine-machine or other-OS test.

### Controlled integration equivalence

The [city comparison](validation/helmholtz-controlled-city.json) uses the exact
original h=5 m city mesh (330,073 DOFs, 1,841,296 tetrahedra), 5 Hz, rho 1.10,
16 subdomains/workers, overlap 1, oversampling 3 and one numerical thread.
Both independent true residuals are 7.604134416308496e-6. Test-only wrappers
control ARPACK initialization; production defaults do not force it. Object
arrays are compared through typed numerical leaves, not object-pointer bytes.
Independent postprocessing confirmed the equality. Original inputs were unchanged.

The earlier [controlled h=4 box comparison](validation/helmholtz-controlled-comparison.json)
also matched bitwise under controlled initialization: 29 iterations and true
residual 3.808188393821847e-6. These results apply to their recorded source
snapshots and cases, rather than every mesh or runtime.

### Historical thesis reconstruction

The [reconstruction report](validation/helmholtz-thesis-geometric-reconstruction.json)
is grounded in server launchers, archived logs, exact input mesh and saved
fields. An older surviving source copy supplied the geometric PoU revision;
the thesis jobs themselves ran on the server. All 16 patch dimension/facet
records, retained-mode counts and printed spectral signatures match. Numerical
stages exited successfully; an initial missing-h5py comparison-helper failure
is retained in the raw receipts, and separate final verification succeeded.

This establishes the recovered numerical behavior for this thesis row. There
are no historical per-run source hashes or preconditioner arrays proving every
past source byte. It does not identify the revision used for every thesis table.
The final integration exposes the geometric construction explicitly; the
recovered full source and the final integration are separate tested artifacts.

### Weight choice, cache and result integration

The [weight validation](validation/helmholtz-weight-options.json) records 120
focused checks on the final staged installation. On the full thesis mesh and
saved partition, all 16 x 330,073 weights match each construction's reference
bitwise. For small real solves, each original weight function was substituted
into the same backend as the oracle; all 61 cache entries, histories and fields
matched the selectable implementation bitwise. Both took 8 iterations; the
independent true residuals were 3.0225920529420222e-6 (geometric) and
3.245275017260653e-6 (ramp). These are correctness checks, not comparative
performance benchmarks.

Actual detached dataset runs with each choice passed native DTCC, XDMF, VTU
and portable package exports. Geometry, boundaries, fields and package context
round-tripped exactly. Cross-method cache reuse was refused before solving;
result reload was exact. The separate [partitioner identity check](validation/helmholtz-partitioner-identity.json)
verifies refusal across two real METIS builds with the same PyMetis version.

## Reproducible setup and automated checks

The [pinned Linux installer](helmholtz-reproducible-install.md) recreates the
recorded packages and official native source pins in a new prefix.
The [scripted trial](validation/helmholtz-scripted-install.json) passed
without manual repairs after launch: native builds, dependency consistency,
129 tests and two real mesh/solver/export cases. Both small solves took
8 iterations with independent residuals about 3.02e-6 and 3.25e-6. A failed first
trial exposed a spawn-import side effect in the verification helper; it is
preserved separately and a regression test now covers that failure.

The GitHub Actions workflow uses that installer and the focused test selection,
with the geodata-dependent check available separately. Workflow syntax and the
server commands have been checked. Hosted GitHub Actions results are separate
evidence and must be read from the corresponding commit checks; server
validation alone does not establish a hosted CI pass. The small
[runnable example](../demos/HELMHOLTZ_SMALL.md) provides commands and observed
expectations without thesis inputs. No full city run is required by CI.

## Core boundary persistence dependency

The pinned official Core revision preserves optional facet-vertex and marker
arrays in native VolumeMesh serialization. The installer builds that revision
directly, without source patches. Cell tags and facet boundary tags remain
distinct. Older strict Core readers reject tagged files with the extended
schema; readers supporting boundary arrays also accept untagged files.
XDMF/HDF5 remains an interchange option. Boundary persistence preserves labels;
it does not choose physical boundary conditions.

Earlier full-case field/export evidence is retained in
[the box report](validation/helmholtz-thesis-boundary-check.json) and
[the first city report](validation/helmholtz-thesis-city-check.json). The latter
used a different METIS build and produced 39 iterations/164 modes. Its report
remains an accurate record of that run; the controlled and recovered-source
checks above provide the stronger, distinct evidence now available.

## Supported limits

The checked runtime uses complex PETSc, DOLFINx 0.11.0 and one MPI rank. Patch
workers are processes, not MPI ranks. Geometry is linear tetrahedral and DTCC
vertex-field export is P1. Equal nominal h does not imply equal mesh files.
The retained source placement can warn when its center lies outside the mesh.
Neither weight variant is claimed universally faster. Existing cases cannot be
overwritten or numerically resumed; portable packages are result artifacts,
not solver restart checkpoints. The isolated installation validation above has
the stated host and packaging limits. No full project test suite or validation of every
physical configuration is claimed. Large meshes, fields, caches and personal
server paths do not belong in the generic contribution.
