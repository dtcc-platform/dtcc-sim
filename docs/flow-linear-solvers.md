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

## Eddy-viscosity closures

The `abl` and `mixing_length` closures make the velocity block much harder:
ILU(0) reaches the 500-iteration limit in the Stokes initialization of city
meshes, so they use ILU(1). With the natural ordering of the mixed space, ILU(1)
factorizations and triangular solves took two thirds of a city solve. Reverse
Cuthill–McKee ordering of the local velocity blocks
(`fieldsplit_u_sub_pc_factor_mat_ordering_type=rcm`) keeps the fill low and is
the default for these closures.

Candidates were compared on the 27,474-tetrahedron Poseidon mesh with
`mixing_length`, log-law walls, grad-div 3 and backflow 2, in serial on a
two-core Linux container. Times are complete `simulate()` runs; Newton counts
of identical runs vary by one or two.

| Velocity block / Schur factorization | Time, s | Newton | Linear iterations, Stokes / Newton | Outcome |
| --- | ---: | ---: | ---: | --- |
| ASM ILU(1), natural ordering | 239 | 14 | 147 / 301 | Converged; 79 s factorization, 74 s solves |
| ASM ILU(1), reverse Cuthill–McKee (default) | **133** | 14 | 174 / 343 | Converged; 12 s factorization, 28 s solves |
| ASM ILU(1), nested dissection | 147 | 13 | 177 / 310 | Converged |
| ASM ILU(1) RCM, upper Schur factorization | 138 | 15 | 172 / 325 | Converged |
| ASM ILU(1) RCM, full Schur factorization | 170 | 13 | 206 / 384 | Converged |
| ASM ILU(1) RCM, pressure mass `1/(ν_eff + γ)` Schur | 329 | 14 | 111 / 402 | Converged; the extra preconditioning matrix costs 200 s of assembly |
| GAMG, unsmoothed aggregation, two SOR sweeps | 181 | 13 | 201 / 301 | Converged |
| GAMG, one SOR sweep, threshold 0.02 | 157 | 10 | 291 / 388 | Converged |
| Full MUMPS LU | 189 | 7 | – | Converged; 150 s of factorizations |
| ASM ILU(0) RCM | – | – | 500 in Stokes | Failed |
| ASM ILU(1) RCM, preconditioner rebuilt every second Newton step | – | – | 500 in Newton | Failed |
| BoomerAMG | – | – | 500 in Stokes | Failed |

The reordering gave the same speedup on a second realization of the Poseidon
mesh (231 to 126 s), and 76 s on two ranks against 181 s with the natural
ordering. Velocities agree with LU to a relative norm of `1e-9`, and those of
the reordered and natural-ordering solves to `2.5e-8`. With constant `nu_t=2`
the reordering changed the ILU(0) time from 63 to 60 s, and that default is
unchanged. GAMG, with a lower Newton count and no ILU fill, may scale better
on larger meshes and more ranks; it has not been measured there.

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
