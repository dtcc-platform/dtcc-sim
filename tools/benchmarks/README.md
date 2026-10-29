# Flow benchmarks

## Linear solvers

`flow_linear.py` compares explicit PETSc configurations on captured Stokes and
Newton systems, then complete stationary solves. It repairs the native city
mesh before finite-element conversion. `--runs 4` gives one startup and three
warm measurements; `--reference <fields.npz>` checks velocity and pressure
coefficients against LU. Keep physical inputs and acceptance fixed and run
timing cases sequentially. Use `--adaptive --ew-cap .1` for bounded
Eisenstat–Walker tolerances during Newton iteration; initialization stays tight.

Commands, numerical gates, measurements and limitations are in the
[linear-solver evaluation](../../docs/flow-linear-solvers.md).

## Elements

Run in the real PETSc FEniCSx environment from the repository root. The tools
compare P2/P1, P2 plus quartic cell bubbles / P1 (`p2b`) and MINI (`mini`).
They use the stationary solver's physical equations, boundaries, SNES driver,
full-field acceptance and native output handling. Element selection is scoped
to the benchmark; it is not a public solver parameter.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python tools/benchmarks/flow_bubbles.py --element mini --condense \
    --case channel --n 24 --runs 4 --output .agent/work/bubbles/channel-mini
```

`--condense` uses exact algebraic elimination of the three cell-interior
velocity variables, a retained MUMPS solve, and full Newton-correction recovery.
The full matrix is assembled and retained. This does not implement element-level
condensed assembly or reduce the stored nonlinear state.

For city measurements, supply a directory containing the frozen tagged
`city.xdmf` mesh and its HDF5 companion:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python tools/benchmarks/flow_bubbles.py --element p2 --case city --repaired \
    --source .agent/work/stationary --runs 4 \
    --output .agent/work/bubbles/city-p2-repaired
```

`--repaired` detects uncoupled P2/P1 pressures and locally refines their cell
stars before timing. Use the identical prepared mesh for element comparisons.
Omit it to test the original city mesh. `--uniform-refinements 1` uniformly
refines the city mesh once. Physical city settings are fixed in `case_mesh`.

The JSON manifest records parameters, mesh and production-source checksums,
versions, requested thread limits, wall and process CPU times, assembly times,
PETSc factorization/solve events, convergence and field checks. Field coefficients
and common city probes are saved alongside it. Channel errors are evaluated even
when the solver rejects the field on quality grounds; such runs do not qualify
for speed claims. The first of four runs is startup; compare the remaining three.
`--cold-cache PATH` selects a separate FFCx cache for compilation measurements.

```bash
python tools/benchmarks/summarize_flow_bubbles.py .agent/work/bubbles
python -m pytest tests/test_flow_bubbles.py
mpiexec -n 2 python -m pytest tests/test_flow_bubbles.py -k correction
```

Run timing cases sequentially. Numerical thread-limit environment variables do
not control MPI progress threads. Inspect the installed MPI runtime when process
CPU time exceeds wall time; this can occur even in an idle MPI process. PETSc's
`mat_mumps_use_omp_threads` option requires an OpenMP-enabled PETSc build and must
not be supplied to a build that lacks that support.

The measured results and limitations are in
[the element evaluation](../../docs/flow-element-evaluation.md).
