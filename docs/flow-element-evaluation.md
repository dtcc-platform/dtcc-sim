# Stationary flow: bubbles, MINI and condensation

For the measured direct-LU city workflow, global quadratic bubble enrichment
adds substantial assembly cost. MINI gives a useful fixed-mesh speed reduction,
but its coarse city fields differ substantially from P2/P1. Explicit algebraic
condensation gives little additional benefit for MINI. The measurements support
retaining P2/P1 with local corner repair when preserving the current field
resolution matters. MINI is a candidate for an explicitly approximate preview.

## Formulations and controls

The velocity spaces are continuous P2, P2 plus one quartic cell bubble per
component, and MINI (continuous P1 plus the same bubbles). Pressure remains
continuous P1. The bubble is proportional to the product of the four barycentric
coordinates and vanishes on every face. Strong no-slip and the existing traction
and slip conditions are retained. Physical parameters, full convection, residual
limits and conservation gates are unchanged.

MINI has the standard inf-sup stability mechanism on shape-regular tetrahedral
meshes. Global P2-plus-bubble enrichment contains the MINI velocity space. MINI
has lower approximation order: under suitable regularity, velocity L2 order two
and pressure L2 order one, compared with orders three and two for P2/P1.
See the [3D MINI analysis](https://informatica.vu.lt/journal/INFORMATICA/article/1321/text).
These results do not make discretization error negligible on a coarse city mesh.

Measurements use DOLFINx/Basix 0.11, PETSc 3.25.6, Apple M5 Pro and one MPI rank.
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are set to
one. MPICH 5.0.1 ch4:ofi additionally consumes about two cores in an idle process:
6.016 process CPU seconds in 3.009 wall seconds. Process CPU time therefore does
not measure numerical work alone. This PETSc build does not support the
`mat_mumps_use_omp_threads` controller option.

Each warm comparison has three accepted runs after one excluded startup run.
Times include Stokes initialization, stationary Newton, setup and independent
full-field quality checks. Mesh preparation, Python imports, file output and
probe sampling are excluded. Assembly callbacks include boundary treatment and
communication. Factorization and triangular-solve times use PETSc events.

Absolute performance varied substantially during the session. A later cached
P2/P1 city check took 108.39 s, versus an earlier warm median of 56.21 s; the
corresponding MINI check took 32.43 s, versus 17.15 s. Both comparisons give about
3.3 times lower fixed-mesh runtime for MINI. This is not an equal-accuracy claim.
The quadratic condensed batch has especially large variation and does not
establish a reliable timing ratio.

The reference is the stationary P2/P1 solver. The frozen IPCS city run did not
converge and cannot supply a city speed ratio. Certified IPCS/channel results
remain in [wind verification](wind-verification.md).

## City runtime

The common comparison mesh has 30,251 tetrahedra. It uses the local repair of
seven corner pressure supports, with the same geometry as the 29,086-cell input.
All methods use the same common-mesh checksum. The inlet is 5 m/s from 270°,
`nu_t=2`, log-law profile, slip sides/top, grad-div coefficient 3 and backflow
coefficient 2. Each solve takes one Stokes step and five full-convection Newton
steps.

| Velocity / pressure | Full unknowns | Retained unknowns | Warm total, median [range], s | Jacobian assembly, s | Numerical LU, s | Triangular solves, s | Condensation, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 / P1 | 147,954 | 147,954 | 56.21 [55.17–56.41] | 5.18 | 46.20 | 0.338 | — |
| P2+B4 / P1 | 238,707 | 238,707 | 106.28 [106.14–106.43] | 24.30 | 52.79 | 0.392 | — |
| P2+B4 / P1, condensed | 238,707 | 147,954 | 250.34 [202.33–258.40] | 51.08 | 102.86 | 0.606 | 6.98 |
| MINI | 117,393 | 117,393 | 17.15 [16.79–17.20] | 6.48 | 2.27 | 0.092 | — |
| MINI, condensed | 117,393 | 26,640 | 16.56 [16.48–16.86] | 6.51 | 1.62 | 0.032 | 0.59 |

Assembly and LU columns sum the work over the entire solve. They exclude some
setup and independent checks, so they do not sum to total time. Component medians
also need not describe the same repetition. Average Jacobian assembly per call
was 0.863 s for P2/P1, 4.051 s for full P2+B4 and 1.081 s for full MINI in their
respective warm batches. Residual assembly totals were 1.62, 15.19 and 4.24 s.
The quadratic condensed startup run took 112.04 s; it is excluded from the warm
median and does not remove the later timing variation.

A more stable work comparison is the numerical factorization count. Quadratic
bubble elimination reduces reported factorization arithmetic from 2.654 to
2.395 billion operations, about 9.7%; MINI decreases from 199.3 to 148.4 million,
about 25.6%. The quadratic condensed flop count is identical across its four
runs despite widely differing wall times. Removing cell-local unknowns does not
remove the retained problem's factorization cost.

For MINI, the 0.59 s condensation cost consumes much of the 0.64 s numerical-LU
saving. The 3.4% difference in total medians has overlapping timing ranges and
falls far below a robust twofold improvement. Recorded process peak RSS was
about 0.91 GB full and 1.00 GB condensed for the common MINI case. This prototype
adds full-matrix subblocks and a Schur operator; it demonstrates no memory saving.

On the original unrefined city mesh, condensed MINI passes all quality gates:
112,850 full / 25,592 retained unknowns, warm median 15.88 s [15.87–16.13 s]. No
pressure-space refinement is needed. Quadratic enrichment with exact condensation
also converges on the original mesh: 229,610 full / 142,352 retained unknowns,
five Newton steps, residual 3.52e-6, dimensionless divergence 0.001171 and flux
imbalance 9.43e-15. That correctness run overlaps the regression suite and is
excluded from performance comparisons.

## City field differences

Comparisons use 3,823 common probes on horizontal slices at z=10 m and z=30 m
relative to the mesh origin. The reported relative norms are sampled Euclidean
norms, not volume-integrated errors. P2/P1 is a numerical reference, not ground
truth. Pressure is traction-determined; no arbitrary mean shift is removed.

| Method | Sampled velocity difference | Sampled pressure difference | Speed RMS difference |
| --- | ---: | ---: | ---: |
| P2+B4, common mesh | 1.72% | 3.67% | 0.076 m/s |
| MINI, common mesh | 19.77% | 131.95% | 0.991 m/s |
| MINI, original mesh | 19.83% | 132.39% | 0.994 m/s |
| MINI, uniformly refined original mesh | 9.28% | 30.44% | 0.418 m/s |

Coarse MINI's velocity difference is 36.7% on the z=10 m slice and 18.3% on the
z=30 m slice. It should not be described as having comparable local field
accuracy merely because the algebraic and conservation checks pass. Large
pressure differences also require spatial and mesh-quality investigation before
using pressure-sensitive quantities.

Uniformly refined MINI has 232,688 tetrahedra, 879,336 full / 181,272 retained
unknowns and takes six Newton steps after Stokes initialization. One accepted
cached solve took 306.32 s. It improves agreement but does not demonstrate a cost
advantage at closer field resolution. A single run does not establish a warm
performance ratio. The same analytic inlet is used throughout; its finite-element
trace and the no-slip constraints at inlet/solid intersections depend on the
velocity space. After solid constraints are applied at inlet intersections,
coarse linear velocity prescribes 96,907 m³/s inflow versus 100,071 m³/s for
quadratic velocity, a 3.16% difference. Refined linear velocity prescribes
99,148 m³/s. These boundary-approximation differences are part of the element
comparison; no wind rescaling is used to conceal them.

## Numerical acceptance and analytical accuracy

The fixed residual gate is `max(1e-9, 1e-8 * max(initial residual, 1))`, with
boundary mismatch and pressure compatibility included. City dimensionless
divergence must be at most 0.1, total flux imbalance at most 1e-4 and no-slip
wall-normal RMS at most 1e-8 m/s. Accepted common-mesh values include:

| Method | Full residual | Dimensionless divergence | Flux imbalance | Wall-normal RMS |
| --- | ---: | ---: | ---: | ---: |
| P2/P1 | 6.11e-6 | 1.098e-3 | 4.73e-16 | 0 |
| P2+B4 | 4.31e-6 | 1.170e-3 | 7.19e-15 | 0 |
| MINI | 1.22e-9 | 3.152e-3 | 1.25e-15 | 0 |

Manufactured solutions have nonzero convection. Finest observed L2 orders are
3.405 velocity / 2.102 pressure for P2+B4 and 1.939 / 1.625 for MINI. The quadratic
study uses n=2,4,8; MINI uses n=4,8,16 to reach its asymptotic regime. The same
finite-difference Jacobian tolerance and absolute benchmark error gates apply.

The planar channel requires velocity and pressure L2 errors below 0.002,
dimensionless divergence below 0.005 and flux imbalance below 1e-4.

| Method | Cells | Warm total, s | Velocity L2 error | Pressure L2 error | All fixed gates |
| --- | ---: | ---: | ---: | ---: | --- |
| P2/P1, n=4 | 768 | 0.238 | 1.58e-15 | 7.37e-15 | pass |
| P2+B4, n=4 | 768 | 0.823 | 3.13e-13 | 7.18e-13 | pass |
| P2+B4 condensed, n=4 | 768 | 0.817 | 3.13e-13 | 7.19e-13 | pass |
| MINI condensed, n=4 | 768 | rejected | 0.06435 | 0.01328 | fail |
| MINI condensed, n=8 | 6,144 | rejected | 0.01609 | 0.00391 | fail |
| MINI condensed, n=16 | 49,152 | rejected | 0.00402 | 0.00118 | fail |
| MINI condensed, n=24 | 165,888 | 83.80 [82.80–83.89] | 0.001787 | 0.000600 | pass |

The n=24 MINI run has dimensionless divergence 0.001138. It meets the gates
without weakening them, but needs 216 times as many cells as the quadratic
reference. The exact parabolic profile lies in P2, so this comparison strongly
favors that space and must not be generalized into a universal element speed
ratio. It establishes that lower order does not automatically mean lower cost
at a fixed error target.

## Condensation and implementation cost

The implementation partitions every full Newton matrix into retained variables
and three cell-interior velocity variables per tetrahedron. It forms the exact
Schur complement using local three-by-three inverses, corrects the RHS, solves
for retained increments and recovers bubble increments. PETSc SNES keeps the
full state, residual and backtracking line search. The induced pressure block
comes from elimination of velocity unknowns; no arbitrary pressure diagonal or
physical pressure-mass regularization is introduced.

Full and condensed city coefficient arrays agree at roundoff in the recorded
cases (relative norms below 3e-16). Independent tests compare nonzero Newton corrections
and check the defect in the full Jacobian system. Pressure gauges, corner-cell
linear-pressure recovery, failure propagation and native output behavior are
also tested. A two-rank comparison covers Newton corrections and final fields.

This is **algebraic condensation after full assembly**, implemented as an exact
Python PETSc PC. It retains the full matrix, state and assembly cost. The cell
blocks must remain invertible; that requirement needs attention for strongly
convective Newton iterates. The full-space LU treatment of a very coarse closed
manufactured case was not robust with the selected null-pivot defaults; the
manufactured convergence study uses the established block iterative solver.
Closed-domain condensed checks pass, but do not establish universal LU gauge
robustness.

Basix supports the enriched elements directly through
[`enriched_element`](https://docs.fenicsproject.org/basix/v0.11.0/python/_autosummary/basix.ufl.html).
Element-level condensation would require custom local assembly and coefficient
packing, residual corrections, bubble recovery, facet contributions and MPI/gauge
handling. The [DOLFINx static-condensation demo](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/demos/demo_static-condensation.html)
shows the lower-level kernel approach. The measured benefit does not justify
adding that machinery to the production nonlinear solver in this evaluation.

## Practical alternatives

- **P2/P1 with local corner repair:** established numerical behavior, about 4%
  more cells here, and the smallest implementation. It retains the current
  approximation space and improves the restricted corner velocity supports.
- **Selective corner bubbles:** only 21 additional velocity coefficients for
  these seven cells. A suitable selective space or mixed/local assembly strategy
  could avoid global bubble quadrature. General stability and actual cost still
  need validation; the global experiment does not establish them.
- **MINI:** suitable for evaluating a cheaper coarse approximation when the
  measured velocity differences are acceptable. Pressure accuracy and finer-mesh
  cost need an explicit target before choosing it as a default.
- **Pressure-space reduction or null-mode removal:** can resolve algebraic
  singularity with little assembly overhead, but leaves the original corner
  velocity restrictions. Corner pressure then needs an explicit reconstruction
  or approximation constraint; it is not physically determined by a solver pin.
- **Nitsche no-slip enforcement:** retains boundary velocity freedom, but changes
  boundary discretization and requires leakage, penalty and conditioning checks.
- **Consistent stabilized P1/P1:** a separate low-order formulation worth comparing
  if MINI accuracy is adequate. It needs residual-consistent stabilization and
  its own convergence/conservation verification.

These measurements do not support global P2 bubble enrichment or general static
condensation as a default optimization. The main fixed-mesh saving comes from
MINI's smaller velocity space. Retaining P2/P1 with local repair is the measured
choice for preserving the current fields; selective enrichment is the focused
candidate when avoiding refinement is a requirement.

## Reproduction and scope

See [benchmark commands](../tools/benchmarks/README.md),
[implementation](../tools/benchmarks/flow_bubbles.py),
[summary tooling](../tools/benchmarks/summarize_flow_bubbles.py) and
[numerical tests](../tests/test_flow_bubbles.py). Raw meshes, manifests, fields,
probes and logs are under ignored `.agent/work/bubbles/` and the frozen source
inputs under `.agent/work/stationary/`.

This is one moderate city domain and one machine, using direct LU. It does not
establish performance on million-cell cities, scalable iterative solvers or
more than two MPI ranks, nor validate turbulence or uncertain real-world inputs.
The production solver and public parameter/dataset APIs are unchanged.

## Verification results

The combined real-runtime stationary and enriched regression suite passes
**39 tests**. It covers manufactured convergence, Jacobian actions, pressure
gauges, exact full/condensed corrections, corner pressure recovery, quality
rejection, failure-safe outputs and native persistence/replacement. The two
MPI comparison tests pass on each of two ranks. MINI native export uses
BoxMesh(6,6,6), which passes the unchanged quality gate; the coarser BoxMesh(3,3,3) remains an explicit quality
rejection test. The production-source checksum and pre-existing tracked diff
are preserved.

The source distribution and wheel build successfully with `uv build`; Python
compilation and `git diff --check` pass. No production dependencies are added.
