# Small Helmholtz example

Use the validated complex runtime. The [pinned Linux installer](../docs/helmholtz-reproducible-install.md)
can create it automatically. Activate its new environment with
`conda activate /path/to/installation/env`, then run from this checkout:

```bash
python demos/helmholtz_small.py --output /path/to/new-geometric-case
python demos/helmholtz_small.py --output /path/to/new-geometric-case --run
python demos/helmholtz_small.py --output /path/to/new-ramp-case \
  --pou-mode core_overlap_ramp --run
```

The first command only prints the configuration. Actual runs use h=25 m,
5 Hz, rho=1.10, two patches/workers and one numerical thread per worker in a
100 x 100 x 40 m flat-ground city region. Set `--h` explicitly to change the
mesh resolution. The first actual run generates a mesh; the second reuses the
same cache beside the results. `--mesh-cache` selects another shared cache.
The full mesh request controls cache identity; a new h requests a different mesh.
Each result directory must be new. Execution may download geodata or reuse
upstream tiles; Internet connectivity and access to those public data services
are required if the data is not cached.

Expected for the recorded dependencies and data: 242 vertices, 762 tetrahedra,
8 iterations for each option, and true relative residual approximately 3.02e-6
for geometric hats or 3.25e-6 for ramps. Mesh counts and iteration counts may
change with input data, partitioner or numerical library builds; the example
requires convergence and a true residual <=1e-5 rather than those exact counts.
These are small correctness examples, not performance benchmarks or the thesis
city run.

Each case writes `example-summary.json`, diagnostics/provenance, and
`field.dtcc`, `field.xdmf` (with its HDF5 companion), `field.vtu` and
`field.dtccpkg`. The package includes run context. Native tagged persistence
uses the pinned Core revision. Open VTU in ParaView or reload the
DTCC/package result through Core. The installer verification additionally checks
independent residuals and exact export round trips for this setup.
