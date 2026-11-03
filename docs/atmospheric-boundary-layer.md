# Atmospheric boundary layer

The stationary flow solvers can replace the explicit constant `nu_t` with an
algebraic eddy viscosity for the neutral atmospheric surface layer. The closure,
the inflow, the ground treatment and the top boundary are chosen together so
that the logarithmic wind profile is an exact solution of the continuous
equations over flat, empty terrain. This is the first step towards RANS; it is
not a transport-equation turbulence model and has not been validated against
urban measurements.

```python
from dtcc_sim import NavierStokesParameters, NavierStokesSolver

parameters = NavierStokesParameters(
    wind_speed=5., wind_dir_deg=270., inlet_profile="log_law", z0=.5,
    turbulence_model="mixing_length", wall_model="log_law",
    side_top_boundary="slip", grad_div_gamma=3., backflow_beta=2.,
    mesh_max_mesh_size=12.5, mesh_domain_height=80., mesh_lod="lod2",
)
result = NavierStokesSolver(bounds=bounds, params=parameters).simulate()
```

For the solver APIs, `turbulence_model="constant"` remains the default and keeps
the previous formulation and results unchanged. The `urban_wind_simulation`
dataset defaults to the ABL model: `turbulence_model="mixing_length"` for
Navier–Stokes, `"abl"` for Stokes, with `wall_model="log_law"`,
`inlet_profile="log_law"` and `side_top_boundary="slip"`. Pass
`turbulence_model="constant"` with `nu_t` for the earlier constant-viscosity
scenario. The city wind demos use the same settings.

## Model

The equilibrium surface layer has constant shear stress `u*²`, mean speed
`U(z) = (u*/κ) ln((z + z_w)/z0)` and eddy viscosity `ν_t = κ u* (z + z_w)`,
with `κ = 0.41`, roughness length `z0` and log-law offset `z_w`. The friction
velocity follows from the reference speed:
`u* = κ wind_speed / ln((u_ref_height + z_w)/z0)`.

Both closures use the mixing length `l_m = κ (d + z_w)`, where `d` is the
distance to the nearest building, roof or ground facet. `mixing_length_limit`
applies the Blackadar form `l_m / (1 + l_m/λ)`; without it, the equilibrium is
exact.

| `turbulence_model` | Eddy viscosity | Nonlinearity | Equations |
| --- | --- | --- | --- |
| `constant` | `nu_t` | none | Navier–Stokes, Stokes |
| `abl` | `u* l_m` | none | Navier–Stokes, Stokes |
| `mixing_length` | `l_m² sqrt(2 ε(u):ε(u))` | velocity gradient | Navier–Stokes |

`abl` prescribes the equilibrium eddy viscosity from geometry alone. The Prandtl
mixing length responds to local shear and reduces to `u* l_m` in equilibrium.
It vanishes where the strain vanishes, so recirculation cores can approach
molecular viscosity. Newton differentiates the closure exactly. Initialization
uses `abl` viscosity and the linearized wall law; Stokes therefore rejects the
mixing-length closure.

With variable viscosity the momentum equation uses the symmetric stress
`2 ν_eff ε(u)`, which keeps `div(ν_eff ∇uᵀ)` when `ν_eff` varies. On traction
boundaries the term `ν_eff (∇u)ᵀ n` is subtracted, so the natural outflow
condition remains `ν_eff ∂u/∂n - p n = 0`, compatible with a developed
log-law profile. Constant viscosity keeps the gradient form.

## Boundaries

**Inflow.** The log-law profile uses terrain-relative height: `d_g`, the distance
to the ground facets. Over flat ground it equals the height above the domain
base. A supplied `inlet_expression` replaces the built-in profile.

**Top.** With slip sides and top, the top receives the equilibrium shear stress
`u*²` along the wind direction (Richards and Hoxey). Without it, the momentum
lost to the ground is not replaced and the profile decays downstream.
`abl_top_shear=False` disables it.

**Rough walls.** `wall_model="noslip"` with an ABL closure is a continuum
rough-wall model: `ν_t = κ u* z0` at the wall carries the full `u*²` stress at
zero velocity. On coarse meshes the logarithm is underresolved in the first
cell. Over 400 m the outflow profile changes by about 8 % with 10 m cells and
6 % with 5 m cells.

`wall_model="log_law"` places the computational wall at height
`wall_function_height` (`δ`, default 2 m) above the roughness surface, so
`z_w = z0 + δ`. Walls impose no penetration and the quadratic drag
`τ = C_d |u_t| u_t` with `C_d = (κ / ln(z_w/z0))²`. In equilibrium this gives
the wall speed `(u*/κ) ln(z_w/z0)` and exactly `u*²` stress. In a
one-dimensional P2 model of the surface layer, `δ = 2 m` gives about 2 % nodal
error above the first cell with 13 m cells and below 1 % with 6 m cells; `δ`
of a quarter of the wall-normal cell height keeps it below 1 %. The same `z0`
and `δ` apply to buildings, roofs and ground.

Axis-aligned wall facets receive a strong normal constraint; other facets use
Nitsche. The default weak-wall penalty is
`100 (ν_eff / h_F + wind_speed)` with `h_F = 3|K|/|F|`, the cell height normal to
the facet. The convective part limits leakage where the stagnation pressure is
large compared with `ν_eff/h_F`; the normal height keeps flat cells next to
small terrain facets coercive. `gamma_normal` overrides the penalty, as for
`wall_model="friction"`.

## Wall distance

`d` solves `-Δφ = 1` with `φ = 0` on the selected facets, natural conditions
elsewhere, and `d = 2φ / (sqrt(|∇φ|² + 2φ) + |∇φ|)` (Spalding, Tucker). It is
exact for one plane wall with parallel natural boundaries and approximate near
corners and far from walls. P2 `φ` is solved with CG/BoomerAMG. Forms containing
`d`, closures or wall laws use quadrature degree 5; automatic degree estimation
gives degrees above 30 for these non-polynomial coefficients.

## Linear solver

The ABL closures keep the default FGMRES/Schur solver but use ILU(1) instead of
ILU(0) in the overlapping Schwarz velocity blocks. On the Poseidon mesh with
`mixing_length` and log-law walls, Stokes initialization needs 142 instead of
376 iterations; ILU(0) could reach the 500-iteration limit after a different
mesh ordering. The full solve takes 14 instead of 17 Newton iterations and
288 instead of 338 s. Explicit `petsc_options` override this choice.

## Diagnostics

`solver.diagnostics["turbulence"]` records the closure, wall model, viscous
form, `κ`, `z0`, `u*`, `z_w`, applied top stress, wall drag coefficient and
cell-sampled minimum, maximum and mean of `d` and `ν_t`.

## Verification

The tests in `tests/test_atmospheric_boundary_layer.py` cover:

- exact Poisson wall distance over flat ground;
- the Newton Jacobian against centred finite differences, including closure,
  wall law, Nitsche terms on sloped ground, grad-div and backflow damping;
- a manufactured solution with both closures, imposed through the full weak
  residual. This includes symmetric stress, the open-boundary correction, wall
  law and top shear. On 2, 4 and 8 subdivisions per unit length the L2 errors
  are:

  | Closure | Velocity errors | Pressure errors | Finest-pair orders |
  | --- | --- | --- | --- |
  | `abl` | 1.73e-2, 2.52e-3, 3.47e-4 | 1.87e-2, 4.25e-3, 1.03e-3 | 2.86 / 2.04 |
  | `mixing_length` | 2.07e-2, 3.01e-3, 4.13e-4 | 2.02e-2, 4.61e-3, 1.09e-3 | 2.87 / 2.08 |

- horizontal homogeneity for Navier–Stokes and Stokes on a 200 × 20 × 80 m
  box with 10 m cells: outlet speeds within 1 % of equilibrium between 5 and
  75 m height.

### Horizontal homogeneity

`tools/verification/abl_homogeneity.py` solves a flat 400 × 20 × 80 m domain
with `wind_speed=5`, `z0=0.5`, slip sides and top. The table gives the largest
relative change between inlet and outlet profiles at 1–75 m height, and the
largest outlet deviation from equilibrium above the first cell.

| Configuration | 10 m cells: change / error | 5 m cells: change / error |
| --- | ---: | ---: |
| constant `nu_t=2`, no-slip | 58.8 % / 38.3 % | 69.7 % / 58.7 % |
| `abl`, no-slip | 8.1 % / 4.3 % | 5.8 % / 4.2 % |
| `abl`, log-law wall | 0.61 % / 0.61 % | 0.30 % / 0.30 % |
| `mixing_length`, no-slip | 8.5 % / 7.1 % | 6.5 % / 6.5 % |
| `mixing_length`, log-law wall | 1.11 % / 0.92 % | 0.43 % / 0.43 % |

The former demo setting, constant `nu_t=2` with no-slip ground, changes the
profile by more than half within 400 m. The constant viscosity cannot carry
the equilibrium stress profile, and the slip top supplies no momentum. With
the log-law wall both closures keep the inflow within about 1 %, improving
with refinement. No-slip rough walls converge slowly because the logarithm is
underresolved in the wall cells.

### City

On the Poseidon city mesh (200 × 200 × 80 m, 12.5 m target size,
27,474 tetrahedra, 270° wind at 5 m/s, slip sides and top, `grad_div_gamma=3`,
`backflow_beta=2`), all ABL configurations pass the standard acceptance gates
with the default iterative solver:

| Configuration | Newton iterations | Solve time | Dimensionless divergence | Flux imbalance | Mean / max `ν_t` [m²/s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant `nu_t=2`, no-slip (reference) | 6 | 86 s | 2.09e-3 | 4.5e-14 | 2 / 2 |
| `abl`, no-slip | 7 | 101 s | 2.01e-3 | 1.5e-16 | 8.75 / 19.3 |
| `abl`, log-law wall | 7 | 120 s | 2.24e-3 | 7.0e-5 | 8.80 / 18.7 |
| `mixing_length`, no-slip | 11 | 179 s | 1.69e-3 | 1.4e-14 | 10.3 / 22.7 |
| `mixing_length`, log-law wall | 17 | 338 s | 1.93e-3 | 4.7e-5 | 11.1 / 25.0 |

The log-law wall runs have 0.033 and 0.041 m/s RMS wall-normal velocity from
weak enforcement on terrain and non-axis-aligned walls; the flux-imbalance
gate is 1e-4.

Times are from a two-core Linux container and are not comparable with the
measurements in [wind verification](wind-verification.md). With `abl`, `ν_t`
ranges from `κ u* z_w` at walls to about 19 m²/s at the top; the domain mean is
about 9 m²/s, compared with the constant 2 m²/s of the earlier demos.

## Limitations

- Neutral stratification only; no buoyancy, Coriolis turning or Ekman layer.
  The surface-layer profile is assumed over the full domain height.
- Algebraic closures have no turbulence history or transport. Separated wakes,
  recirculation lengths and pedestrian-level gusts are not predicted reliably.
- One roughness length for terrain and buildings. Building surfaces are
  aerodynamically smoother than urban terrain.
- The inflow is a single lateral face. Oblique winds still prescribe a
  velocity component through slip side faces.
- The domain size, mesh sensitivity and wall-function height have not been
  studied for city cases. City runs do not establish physical validity.

The next step is a two-equation closure (k-ε with Richards–Hoxey inflow or
k-ω SST) on the same variable-viscosity formulation, wall distance and wall law.
