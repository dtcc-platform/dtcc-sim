# Helmholtz integration review

This private branch integrates Marina Sahakyan's two-level spectral Schwarz preconditioner into DTCC-Sim. `main` preserves upstream DTCC-Sim commit `41055d560c66939a93e23c51f4cc71210541cb26`; `integration/helmholtz` contains the proposed changes. Existing upstream authorship, history and license are retained.

Clone the review branch with an account that has access to this private repository:

```bash
git clone --branch integration/helmholtz https://github.com/splinter1640/dtcc-sim-helmholtz.git
cd dtcc-sim-helmholtz
```

## Review order

1. [Workflow guide](helmholtz.md): parameters, explicit/generated meshes, preparation, solving and result export.
2. [Public simulator](../dtcc_sim/helmholtz/simulator.py), [mesh workflow](../dtcc_sim/helmholtz/workflow.py) and [result adapter](../dtcc_sim/helmholtz/results.py).
3. [Numerical backend](../dtcc_sim/helmholtz/_backend/): preserved decomposition, local operators, partition-of-unity, spectral coarse construction and preconditioner action, with explicit parameter binding and spawned patch workers.
4. [Dataset adapter](../dtcc_sim/helmholtz/dataset.py) and [service integration](../service/helmholtz_worker.py), then [optional deployment renderer](../deploy/helmholtz/README.md).
5. [Machine-readable controlled comparison](validation/helmholtz-controlled-comparison.json) and the focused tests under `tests/test_helmholtz_*.py`.

The service/deployment portion is broader than the direct numerical workflow and can be reviewed as a separate scope decision. It routes Helmholtz jobs to a dedicated non-daemon worker, supervises numerical child processes, and handles cancellation/time limits. The renderer prepares private localhost-only service files; it does not install or start them.

## Exact comparison evidence

The preserved original solver and integration ran on the same original h=4 m box, at 10 Hz and rho=1.10, with the same runtime and complete ARPACK settings. A test-only wrapper supplied the same explicit ARPACK starting vector to both; original production files were hash-checked and unmodified. The [reference source hashes](validation/helmholtz-original-source-sha256.json) identify that preserved copy.

| Check | Result |
|---|---|
| Global matrix CSR arrays, RHS, DOF coordinates | Bit-for-bit identical, offline and online |
| All 61 saved preconditioner payload arrays | Exact matches |
| Decomposition, PoU, local operators, eigenmodes and coarse basis | Exact matches |
| Final complex solution | Bit-for-bit identical; relative difference 0 |
| GMRES residual history | Bit-for-bit identical |
| Iterations | 29 in both |
| Integrated true relative residual | 3.808188393821847e-6 |

This establishes equivalence for that controlled case, not universal equivalence or bitwise reproduction of historical results generated with older software. Ordinary runs do not force the test-only starting vector. Runtime, partitioning and eigensolver initialization can affect iteration counts and retained modes.

Supporting checks include numerical reproduction of both selected thesis box cases, two archived city cases, a newly generated small h=25 m mesh-to-package run, and 29 focused result/workflow regression tests after the boundary/export fixes. Full logs and large artifacts are retained privately on the compute server and in the handover, rather than committed here. The complete project test suite was not rerun for this private upload. Repository Actions are disabled for this initial review so the upload does not trigger a full suite; the saved bounded checks are reported here.

## Preview and run the box example

Use the [complex-PETSc Helmholtz environment](installation.md#helmholtz-environment), with DOLFINx 0.11.0. Mesh data are supplied separately. Place the original mesh and its HDF5 companion in `meshes/helmholtz/box-h4/`, retaining the filenames `volume_mesh_gbg.xdmf` and `volume_mesh_gbg.h5`.

The [box configuration](../demos/configs/helmholtz_box_10hz_h4.json) includes h, physical/source/boundary settings, preconditioner controls and worker counts. From this checkout in the activated environment:

```bash
python -m dtcc_sim.helmholtz.launch --config demos/configs/helmholtz_box_10hz_h4.json
```

That previews without solving. To run detached with VTU export:

```bash
bash demos/run_helmholtz_detached.sh demos/configs/helmholtz_box_10hz_h4.json
```

Choose a new `output_dir` before another run. Relative paths resolve from the JSON file. Record the launcher's printed log directory. Detached execution survives ordinary SSH disconnection; there is no numerical checkpoint/resume after a killed process or reboot. The completed box run took about seven minutes on the server, using 16 patch workers and one numerical thread per worker.

For generated city meshes, use the [city configuration](../demos/configs/helmholtz_city.json) and workflow guide. Missing matching meshes are generated and cached by full request and installed mesher identity. Changing h changes the cache request; the same nominal h does not guarantee an identical historical discretization.

## Boundary and export behavior

The original box generator labels all six exterior faces, including the bottom, -2 (`all_impedance`). That model is preserved. It is distinct from sound-hard ground/building treatment on urban meshes. Absent requested markers select no facets, matching the original solver; selections are recorded and a nonempty request selecting none warns. Mixed mode still requires readable boundary markers.

Result export supports linear tetrahedral geometry and P1 fields with checked coordinate mapping. It reads geometry and boundary/cell tags through DOLFINx, including legacy multi-grid XDMF/HDF5 `Values` naming. Standard VTU, DTCC and XDMF export were verified on saved thesis and generated-mesh results without solving again.

## Current boundaries of the work

- Numerical settings and sampled configurations are documented; mesh convergence and calibrated acoustic predictions are separate research questions.
- Temporary private service checks covered bounded cancellation/restart scenarios. Permanent activation and lingering require server administration; no permanent service is enabled by this branch.
- The host enforces tested memory/pids limits and inherited CPU affinity. Strict cgroup CPU quota delegation is unavailable on the tested account.
- Meshes, generated solutions/preconditioners, logs, environment installations, personal files and full scratch backups are excluded from Git.
