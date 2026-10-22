# Helmholtz core contribution

This contribution adds local Helmholtz preparation/solving, DTCC mesh generation
and caching, a dataset adapter and checked result export. Start with the
[quick start](helmholtz-quickstart.md); parameter/API details are in
[the workflow reference](helmholtz.md).

## Review scope

1. [Simulator](../dtcc_sim/helmholtz/simulator.py): explicit parameters, separate
   preparation/solve stages, case locks and input/preconditioner identity.
2. [Mesh workflow](../dtcc_sim/helmholtz/workflow.py) and
   [cache](../dtcc_sim/helmholtz/meshing.py): explicit h, full meshing request,
   checksummed reuse and boundary-marker reports.
3. [Result adapter](../dtcc_sim/helmholtz/results.py) and
   [dataset](../dtcc_sim/helmholtz/dataset.py): bijective P1 mapping, geometry/tags,
   real/imaginary/amplitude fields, caller-declared external CRS and provenance.
4. [Numerical backend](../dtcc_sim/helmholtz/_backend/): preserved research
   implementation with explicit binding and spawned patch workers.

No dedicated Celery worker, process supervisor or systemd renderer is included.
The existing service discovery/task-registration loops exclude Helmholtz so
that the general worker cannot accidentally accept it. Other datasets retain
their existing service behavior. Service support is a separate follow-up.

## Numerical evidence

The [controlled comparison](validation/helmholtz-controlled-comparison.json)
used the original h=4 m box, 10 Hz, rho=1.10, 16 workers, one thread per worker,
identical runtime and complete ARPACK settings. Test-only wrappers supplied the
same explicit eigensolver starting vector; production runs do not force it.
The [reference hashes](validation/helmholtz-original-source-sha256.json)
identify the preserved original scripts.

Global matrices/RHS/DOF coordinates, all 61 saved preconditioner arrays, final
solutions and residual histories matched bit for bit. Both runs took 29
iterations; the integrated true relative residual was 3.808188393821847e-6.
All 11 numerical backend files are unchanged from that tested snapshot.
This is evidence for the recorded case, not universal numerical equivalence,
physical calibration or bitwise reproduction under historical libraries.

## Bounded software checks

The core-only checkout is validated separately from the larger private branch.
Its [validation report](validation/helmholtz-core-contribution.json) records the
selected tests and package checks. No full project suite or full numerical
simulation is required by this scope split. Existing saved box/city numerical
comparisons and the generated small-mesh workflow remain supporting evidence;
service acceptance results belong to the separate follow-up.

## Real small end-to-end run

A [real core-package check](validation/helmholtz-core-smoke.json) installed the
core wheel into an isolated target and verified all 38 Python source files.
DTCC generated a fresh h=25 m mesh (242 vertices, 762 tetrahedra) from cached
source data. The 5 Hz solve used two workers and one thread per worker,
converging in 8 iterations with true relative residual 3.245275017045522e-6.
VTU, native DTCC and portable package fields round-tripped exactly. The full
mesh-to-solve-to-export check took 29.46 seconds and returned exit code zero.
This exercised actual numerical and meshing code, with no mocks; it is a tiny
workflow check, not full-scale performance or physical validation. The recorded
core commit identifies its source snapshot; subsequent evidence-only changes
do not modify numerical or workflow code.

## Supported limits

Complex PETSc, DOLFINx 0.11.0, one MPI rank, linear tetrahedral mesh geometry and
P1 vertex-field export. Nominal h does not identify an exact historical mesh;
explicit mesh files and their hashes do. Cache identity does not detect upstream
data revisions. Cases cannot be overwritten or numerically resumed. The
unchanged source rule can warn for an exterior centre. The reference box has
impedance on all six faces, distinct from sound-hard urban boundaries.
Meshes, solutions, logs and environment installations stay outside Git.
