# Reflecting-basin wave verification

Run in the common real-valued DOLFINx 0.11 environment:

```sh
conda activate fenicsx-env
unset DTCC_SWE_PYTHON
python -m tools.verification.shallow_water.wave --output output/verification/shallow-water-wave
```

The unit square has flat bed, reflecting walls, background depth H = 1 m,
zero initial momentum and a Gaussian depth perturbation of amplitude
`a = 1e-4 m`, sigma 0.1 m, centred at (0.5, 0.5). Final time is 0.04 s.
This is an accuracy benchmark, not an observed-flood validation.

## Independent reference

Linearizing about rest gives `eta_t + div(q) = 0` and
`q_t + g H grad(eta) = 0`. With zero normal momentum, cosine modes have
`kx = m pi`, `ky = n pi`, `omega = sqrt(gH) sqrt(kx² + ky²)`.
The Gaussian's separable cosine coefficients are integrated by Gauss–Legendre
quadrature. Each depth mode evolves as `cos(omega t)` and its momentum modes
as `gH kx sin(omega t)/omega` and `gH ky sin(omega t)/omega`.
The constant mode preserves mass. This solution is derived for the linearized
system; it is not an exact solution of the finite-amplitude nonlinear system.
The underlying shallow-water equations and gravity-wave characteristics are
described in [LeVeque's Riemann book](https://faculty.washington.edu/rjl/riemann_book/Shallow_water.html).

The implementation uses 48 modes per direction and 96-point coefficient
quadrature, checked against 64 modes and 128 points. Triangle integration uses
a six-by-six Duffy Gauss rule on the full DG1 depth and momentum polynomials.
Depth errors are divided by `a`, momentum errors by `a sqrt(gH)`; the unit-square
L2 norm therefore reports perturbation error without hiding it under H.

## Checks

- Mesh refinement: 8, 16 and 24 subdivisions per direction, CFL 0.02;
  all three component errors must decrease at each refinement, and the finest
  relative L2 error must be below 25% of each evolving reference component.
  The accuracy requirement prevents a nearly stationary solution
  from passing solely because its mesh sensitivity decreases. The report also
  records errors for a deliberately frozen Gaussian as a negative control.
- Time refinement: fixed 12 subdivisions, CFL 0.08, 0.04 and 0.02;
  successive full-DG solution differences must decrease in all components.
  This isolates temporal sensitivity even when spatial error dominates the
  reference error. It does not assert a formal time convergence order.
- Amplitude halving: repeat the 16-subdivision case with `a/2`. Differences
  between normalized solutions must be below 1% of the corresponding
  full-amplitude reference errors, bounding the linearization discrepancy.
- Reference uncertainty: mode/quadrature changes must be below `1e-6` in
  normalized point values and below 0.01% of the smallest measured L2 error.

The command exits unsuccessfully if any check fails and writes `summary.json`
with errors, step counts, reference uncertainty and raw-case locations. Existing
completed cases can be reanalysed; parameters, numerical reference and mesh size
are checked before reuse. Use a new output directory for a fresh solve.

The case tests smooth fully wet dynamics and reflecting walls. It does not
validate moving shorelines, building impact forces, rough terrain, rainfall or
observed urban flooding. No formal spatial order is assumed for the limited
scheme on these meshes.
