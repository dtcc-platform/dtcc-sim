# Run timings

All solver `simulate()` methods and simulation datasets print one timing table
at the end of a run. Helmholtz `prepare()`, `solve()` and `run()` also report
when called independently. Nested calls contribute to the enclosing report.

The table measures wall-clock seconds with a monotonic clock, including I/O and
waiting. Each phase has elapsed time, time excluding its children (`Self`), and
a call count. Parent elapsed times include their children, so do not add all
rows. `Total run` includes orchestration outside named phases. Reporting and
writing the timing report itself are excluded.

The relevant phases depend on the solver:

| Phase | Work measured |
| --- | --- |
| Data acquisition | Weather, air-quality observations, roads and census zones |
| Mesh preparation | Mesh loading, conversion, cache validation or generation, including upstream data acquisition |
| Solver | Form setup, assembly, preconditioner setup, numerical solves and validation, with child phases where available |
| Postprocessing | Mapping numerical fields into returned Core models |
| Export | Solution files, coordinate sidecars and dataset serialization |

Core progress phases provide a finer breakdown within mesh preparation,
including `download_pointcloud`, `download_footprints`, `build_terrain` and
`build_mesh`. This requires Core's `dtcc_core.common.progress.phase_timing`
hook, included in Sim's pinned Core revision. For local development against a
Core version without that hook, the report has `core_phases: false`, and mesh
preparation includes acquisition without a separate download row. Local Core
overrides are described in the [installation guide](installation.md).

The shared city demos use `output/city-mesh-cache`. Cache hits measure cache
lookup, integrity checking and loading, not the original download or mesh
construction. Repeated calls reset timings. Helmholtz offline preparation and
online solving have separate phase trees; exports made within either stage
appear as children of that stage. Heat's DOLFINx `LinearProblem.solve()`
assembles and solves internally, so that row is explicitly named
`assembly_and_linear_solve`. Timings are not PETSc CPU profiles or per-iteration
measurements.

Access the report after a run:

```python
result = solver.simulate()
report = solver.last_timings
assert report is solver.diagnostics["run_timings"]
```

Returned Core models carry the report in `simulation_diagnostics` and model
attributes. `HelmholtzResult.run_timings` retains it for direct `run()` and
`solve()` calls. Existing flow `diagnostics["timings"]` benchmark keys remain
available separately.

When an output directory or explicit output path is supplied, the run writes
`timings.json` alongside the output. It contains the complete report, including
export time, and replaces the previous timing report. Use this sidecar for
persisted final timings: solution files written during export cannot include
measurements that finish after the files are serialized. No output directory
is created solely for timings. A failed run retains partial timings and marks
failed phases; timing-publication errors are logged without changing the
solver's result or original exception.

Successful MPI runs gather phase dictionaries once at the end and report the
maximum time across ranks for each path. Only rank zero prints and writes.
Rows that occur only on rank zero, such as data acquisition, are retained.
Per-phase maxima can come from different ranks, so their sum need not equal
the maximum total. No timing barriers are inserted. Failed runs retain local
measurements without adding a collective during exception unwinding.
