# Benchmarks

Validation of DTCC Sim solvers against measurements. Tests and
[`tools/verification`](../tools/verification/README.md) check that the
equations are solved correctly; the cases here compare the solutions with
experiments, which tests the physical model as well. Solver performance
benchmarks are in [`tools/benchmarks`](../tools/benchmarks/README.md).

| Directory | Cases | Solver |
| --- | --- | --- |
| [`aij`](aij/README.md) | AIJ pedestrian-wind cases A (high-rise building) and B (slab) | stationary Navier–Stokes with the atmospheric boundary-layer closures |

## Reference data

Measurement data are not redistributed with DTCC Sim.
[`data/README.md`](data/README.md) lists the files to download and where to
place them; Git ignores everything in `benchmarks/data/` except the README
files. Install the reader dependencies with
`uv pip install -e ".[benchmarks]"` or, in the Conda FEniCSx environment,
`conda install scipy xlrd`.

## Running

Run from the repository root in the real-number FEniCSx environment:

```bash
python -m benchmarks.aij.run --case A --resolution coarse
python -m benchmarks.aij.report output/benchmarks/aij
```

Results, figures and a metric summary are written below
`output/benchmarks/`. `tests/test_benchmarks_aij.py` checks geometry, data
reading and metrics without solving.
