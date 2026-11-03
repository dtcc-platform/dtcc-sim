# Flow verification

Run from the repository root in the real-number FEniCSx environment.

`abl_homogeneity.py` checks horizontal homogeneity of the atmospheric
boundary layer: an equilibrium log-law inflow should cross a flat, empty
domain unchanged. It compares closures and wall treatments on the same mesh.

```bash
OMP_NUM_THREADS=1 python tools/verification/abl_homogeneity.py \
    --size 10 --output .agent/work/abl/homogeneity-h10.json
```

Results and interpretation are in the
[atmospheric boundary-layer guide](../../docs/atmospheric-boundary-layer.md).
