# Solver verification

Run from the repository root. The flow checks below and the shallow-water study
use the real-number FEniCSx environment.

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

`curved_wall_mms.py` solves a manufactured flow over a curved wall with
facet, vertex-averaged or exact normals in the Nitsche terms, refining either
the solution on fixed polyhedral geometry or the geometry with it:

```bash
OMP_NUM_THREADS=1 python tools/verification/curved_wall_mms.py --levels 4 8 16 \
    --amplitude .05 --normals facet vertex exact --output .agent/work/abl/curved.json
```

Results and interpretation are in the
[atmospheric boundary-layer guide](../../docs/atmospheric-boundary-layer.md).

[Shallow-water wave verification](shallow_water/README.md) checks an independent
reflecting-basin reference with mesh, time and amplitude refinement.
