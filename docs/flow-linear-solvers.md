# Stationary flow linear solvers

The wind solver uses FGMRES with a lower Schur factorization. The velocity
block uses additive Schwarz with one layer of overlap and local ILU(0). The
pressure preconditioner is BoomerAMG applied to PETSc's algebraic `selfp`
approximation, formed from the Jacobian blocks:

\[
S_p = A_{11} - A_{10}\operatorname{diag}(A_{00})^{-1}A_{01}.
\]

This changes preconditioning only. The mixed residual, Newton Jacobian,
finite elements, pressure gauge, boundary conditions and physical parameters
remain the numerical problem. Both blocks come from the Jacobian; an auxiliary
pressure-mass matrix is unnecessary for this configuration.

When the assembled operator has a verified constant-pressure nullspace, the
default pressure block uses Jacobi instead of applying AMG to a singular
Schur approximation. This keeps the arbitrary gauge mode bounded. The existing
left/right nullspace projection and zero-mean output pressure remain in force.
An explicit user pressure-PC override is respected.

Stokes initialization uses a relative linear tolerance of `1e-10`. During
Navier–Stokes Newton iteration, Eisenstat–Walker version 2 adapts the tolerance
with initial value and maximum `0.1`. The nonlinear tolerances and independent
assembled-residual, divergence, flux and finite-field acceptance checks remain
in force. A failed KSP or SNES solve raises without a fallback.

## Measurement protocol

The city case uses the frozen Poseidon mesh, repaired before finite-element
conversion: 29,107 tetrahedra, 136,059 velocity unknowns and 6,405 pressure
unknowns. Parameters are 5 m/s wind, effective viscosity `2.000015` m²/s,
grad-div coefficient 3, backflow coefficient 2, logarithmic inflow and slip
at the top and sides. No mesh refinement, viscosity adjustment or acceptance
relaxation is used to obtain a timing result.

Runs use one MPI rank on macOS arm64 with DOLFINx 0.11.0 and PETSc 3.25.6.
OpenMP, OpenBLAS and Accelerate thread limits are one. Candidates run
sequentially. Four full runs give one startup run and three warm measurements;
the reported full-solve time is the warm median, including assembly,
preconditioning, Newton iteration and numerical acceptance. It excludes city
data acquisition, mesh generation, plotting and export.

Matrix screens use actual Stokes and Newton Jacobians and right-hand sides.
A passing matrix solve requires a positive KSP convergence reason and an
explicitly computed relative residual no larger than `1e-8`. Full solves must
also pass all solver acceptance checks and agree with LU in velocity and
pressure coefficients to relative norm `1e-7` on the identical mesh. A screen
timeout is 60 seconds. Incomplete or rejected runs do not qualify for speed
claims. Timing monitors preserve PETSc's native adaptive-tolerance callbacks.

## Measured results

Full city solves, seconds:

| Equations | LU | Iterative, fixed tolerances | Iterative, adaptive tolerances | Speedup over LU |
| --- | ---: | ---: | ---: | ---: |
| Navier–Stokes | 52.34 | 22.37 | **15.99** | **3.27×** |
| Stokes | 10.91 | **4.98** | Not applicable | **2.19×** |

The three warm Navier–Stokes runs took 16.029, 15.992 and 15.947 seconds.
Peak process RSS was 1.96 GiB for iterative Navier–Stokes and 2.71 GiB for
LU, about 28% lower. Maximum relative field-coefficient differences from LU
were `2.60e-10` for Navier–Stokes and `1.19e-9` for Stokes, below the fixed
`1e-7` comparison gate. All physical residual, divergence and flux gates passed.

Twenty-eight configurations were screened on the Stokes matrix. Representative
single-system results at the tight `1e-10` relative KSP tolerance are:

| Velocity / pressure preconditioner | Time, s | Outcome |
| --- | ---: | --- |
| Full MUMPS LU | 7.96 | Converged |
| ILU(1) / pressure mass diagonal | 9.08 | Converged |
| ILU(0), restart 200 / pressure mass diagonal | 3.49 | Stokes converged; first Newton matrix failed |
| Vector GAMG / pressure mass diagonal | 3.42 | Stokes converged; first Newton matrix failed |
| ILU(0) / algebraic Schur with BoomerAMG | 1.79 | Stokes and sampled Newton matrices converged |
| Hypre velocity AMG / pressure mass diagonal | >60 | Timed out |
| Scaled LSC pressure with ILU(0) velocity | 8.75 | Reached the 500-iteration limit |

The screens also covered full/upper Schur variants, increased ILU fill, extra
AMG cycles, stronger coarsening, vector/nodal Hypre settings and LSC. Candidates
advanced according to screen time and convergence; complete solves selected the
final configuration. Diffusion-only velocity preconditioning failed a complete
Newton solve. A low Stokes-only time was insufficient to qualify a configuration. The
[measurement data](validation/flow-linear-solvers.json) records all 28 matrix
screens, explicit settings, warm runs, field differences and fixture checksums.

## Reproduction

The benchmark takes a directory containing the tagged `city.xdmf` and `city.h5`.
All generated matrices and measurements belong under `.agent/work/`.

```bash
conda activate fenicsx-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python tools/benchmarks/flow_linear.py --mode capture --profile lu --runs 1 \
    --source .agent/work/stationary --output .agent/work/linear/capture
python tools/benchmarks/flow_linear.py --profile lu --runs 4 \
    --source .agent/work/stationary --output .agent/work/linear/lu
python tools/benchmarks/flow_linear.py --profile asm_selfp --adaptive --ew-cap .1 \
    --runs 4 --source .agent/work/stationary \
    --reference .agent/work/linear/capture/fields-0.npz \
    --output .agent/work/linear/iterative
```

Use `--mode matrix --matrix <capture/matrix-N> --profile <name>` to screen
captured systems. `--case channel` runs the exact planar channel;
`--case cavity --n 16` runs the smooth-lid closed cavity. Benchmark profiles
contain explicit PETSc settings so comparisons do not depend on library defaults.
Use `--profile asm_selfp_gauge` for the closed-cavity default's diagonal
pressure preconditioner.

## Scope

These timings select a practical configuration for this mesh and machine.
One-level Schwarz/ILU is not a mesh-independent velocity preconditioner.
Higher Reynolds numbers, larger meshes and more MPI ranks require separate
measurements. LU remains an explicit reference and diagnostic option.
Stationary convergence at a supplied effective viscosity does not establish
physical validity for turbulent urban wind.

PETSc documents the [Schur approximations](https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/),
[block factorizations](https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurFactType/),
[BoomerAMG controls](https://petsc.org/release/manualpages/PC/PCHYPRE/) and
[Eisenstat–Walker method](https://petsc.org/release/manualpages/SNES/SNESKSPSetUseEW/).
