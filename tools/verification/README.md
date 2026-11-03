# Flow verification

Run from the repository root in the real-number FEniCSx environment.

`abl_homogeneity.py` checks horizontal homogeneity of the atmospheric
boundary layer: an equilibrium log-law inflow should cross a flat, empty
domain unchanged. It compares closures and wall treatments on the same mesh.

```bash
OMP_NUM_THREADS=1 python tools/verification/abl_homogeneity.py \
    --size 10 --output .agent/work/abl/homogeneity-h10.json
```

`wall_leakage.py` measures flow through weakly enforced curved terrain for
several cell sizes and Nitsche penalties, and optionally a city mesh:

```bash
OMP_NUM_THREADS=1 python tools/verification/wall_leakage.py --sizes 20 10 5 \
    --penalties 100 1000 --output .agent/work/abl/leakage.json
```

Results and interpretation are in the
[atmospheric boundary-layer guide](../../docs/atmospheric-boundary-layer.md).
