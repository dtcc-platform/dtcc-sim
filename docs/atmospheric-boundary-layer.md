# Atmospheric boundary layer

The stationary flow solvers can replace the explicit constant `nu_t` with an
algebraic eddy viscosity for the neutral atmospheric surface layer. The closure,
the inflow, the ground treatment and the top boundary are chosen together so
that the logarithmic wind profile is an exact solution of the continuous
equations over flat, empty terrain. This is an algebraic RANS closure without
turbulence transport equations. It is compared with wind-tunnel measurements
around isolated buildings in the [AIJ benchmarks](../benchmarks/aij/README.md)
and has not been validated against urban measurements.

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
the gradient-form constant-viscosity formulation. The `urban_wind_simulation`
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

**Wind direction.** Every lateral face the wind enters through receives the
inflow profile, every face it leaves through is a traction outlet, and faces
parallel to the wind stay side faces. A wind 20° off the x-axis enters through
the west and south faces and leaves through the east and north faces. In an
empty 160 × 160 × 80 m domain with 20 m cells the volume-mean flow direction is
20.01° (`abl`) and 20.02° (`mixing_length`), within 1.1 % RMS of the
equilibrium profile. With only one inflow and one outflow face, slip on the
other two removed most of the crosswind: the mean direction was 5.3° and the
RMS deviation 36 %. Explicit `inlet_marker`/`outlet_marker` and closed cavities
keep single faces.

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
`nitsche_penalty (ν_eff / h_F + wind_speed)` with `nitsche_penalty=100` and
`h_F = 3|K|/|F|`, the cell height normal to the facet. The convective part
limits leakage where the stagnation pressure is large compared with
`ν_eff/h_F`; the normal height keeps flat cells next to small terrain facets
coercive. `gamma_normal` overrides the penalty, as for `wall_model="friction"`.

No penetration and wall drag act along an averaged wall normal `m`. Each wall
vertex receives the area-weighted mean of the outward normals of its weak-wall
facets, summed across MPI ranks, and `m` is the normalized P1 interpolant of
these vertex normals. Where the facets at a vertex deviate more than 30° from
their mean, as at building edges and wall–ground corners, and wherever the
interpolant is short or more than 30° from the facet normal, the facet normal
is used. On planar walls `m` is the facet normal. The traction in the Nitsche
terms is `σ(u) n - p n` projected on `m`.

On inlet edges shared with these walls, the inflow is projected onto the wall
tangent planes. The log-law wall speed is nonzero, so the horizontal profile
would otherwise cross sloping terrain there. Facets within 30° of each other
share their mean normal. The incident facets of each node are collected from
all MPI ranks and ordered by position, so the projected inflow is the same for
every partition: on curved terrain serial, two-rank and four-rank values agree
to 1e-15 m/s, where averaging per-rank projections differed by up to
0.17 m/s. Nitsche continuity terms make the net weak-wall
leakage equal to the total boundary flux imbalance. On a second TetGen
realization of the Poseidon mesh (27,211 tetrahedra), the unprojected inflow
gave a net ground inflow of 25.8 m³/s, 20.7 m³/s of it within 15 m of the
inlet, and a flux imbalance of 1.23e-4, above the 1e-4 gate. With projection the
ground net is 7.5 m³/s (2.3 m³/s near the inlet) and the imbalance 2.4e-5.

## Impermeability

Weak enforcement lets some flow cross terrain and non-axis-aligned walls.
Acceptance bounds it separately for ground and for building walls and roofs:
`∫|u·m|` over each surface, along the normal `m` on which no penetration is
imposed, divided by the inflow volume flux, must not exceed
`ground_leakage_tolerance` and `building_leakage_tolerance` (default 2 %).
Diagnostics record these fractions, the absolute fluxes, the RMS normal
velocity per surface and the flux `∫|u·n|` through the facets themselves. The
net flux imbalance gate alone does not detect leakage, because inflow and
outflow through walls largely cancel.

The gates are an explicit error allowance. They bound the flow across the
walls; they do not establish accuracy, and 2 % per surface group is not a
physically derived tolerance. On curved terrain the facet flux contains
geometry: the exact flow is tangent to the true surface and itself crosses the
facets that approximate it, which is why the gates use `m`.

On planar terrain the leakage is negligible, below 2e-5 of the inflow with 20 m
and 10 m cells on a plane rising 10 % along the wind.
`tools/verification/wall_leakage.py` solves 200 × 40 × 80 m of terrain
`z = 4 sin(2πx/80) sin(2πy/80)` m with `mixing_length` and log-law walls.
Ground leakage fractions:

| Normal, `nitsche_penalty` | 20 m cells | 10 m cells | 5 m cells |
| --- | ---: | ---: | ---: |
| averaged `m`, 100 (default) | 1.05 % | 0.39 % | 0.12 % |
| facet, 100 | 1.10 % | 1.19 % | 1.19 % |
| facet, 1000 | 0.27 % | 0.27 % | 0.32 % |
| facet, 10000 | | 0.078 % | |

With averaged normals the leakage decreases at observed orders 1.5 and 1.7 and
the net flux imbalance from 2.2e-4 to 1.0e-4 and 2.7e-5. With facet normals
the leakage did not decrease under refinement and was controlled only by the
penalty. Raising the facet-normal penalty from 100 to 10000 reduced the mean
tangential ground speed by 2.0 % and the volume-mean speed by 0.01 %; a penalty
of 1000 also made the first `mixing_length` Newton system stall the default
iterative solver on some meshes and orderings. On the 27,474-tetrahedron Poseidon
mesh with `mixing_length`, averaged normals lower the ground leakage from
0.71 % to 0.17 % at the default penalty; building leakage stays at 0.23 %,
since building walls are planar between sharp edges.

Slip and no-penetration conditions on polyhedral approximations of curved
walls have known geometric consistency and locking problems whose outcome
depends on the formulation and the geometry approximation; strong imposition
has its own difficulties
([Urquiza et al., 2014](https://www.sciencedirect.com/science/article/abs/pii/S0021999113005895)).

### Curved-wall manufactured solution

`tools/verification/curved_wall_mms.py` separates solution refinement from
geometry refinement. The domain lies between `z = 0.05 sin(2πx)` and `z = 1`
on `[0,2] × [0,0.5]`; the exact velocity comes from a stream function constant
on both walls, so it is tangent to the true ground. Forcing, outflow and top
tractions and log-law wall drag are derived from the exact solution and the
enforced normal (`ν = 0.05`, `U = 1`, maximum slope 0.31,
`nitsche_penalty=100`, MUMPS LU). Geometry refinement places the ground nodes
of each mesh on the curve; fixed geometry maps the coarsest mesh and refines it
uniformly, keeping its polyhedral ground. Velocity L2 errors with `n` cells per
unit height:

| Normal in the Nitsche terms | Refinement | n = 4 | n = 8 | n = 16 |
| --- | --- | ---: | ---: | ---: |
| averaged `m` (solver) | geometry | 6.66e-2 | 2.09e-2 | 5.66e-3 |
| averaged `m` (solver) | fixed | 6.66e-2 | 3.87e-2 | 3.47e-2 |
| facet | geometry | 1.38e-1 | 1.17e-1 | 5.90e-2 |
| facet | fixed | 1.38e-1 | 9.79e-2 | 7.76e-2 |
| exact curve normal | geometry | 1.56e-2 | 1.26e-3 | 1.41e-4 |
| exact curve normal | fixed | 1.56e-2 | 3.80e-3 | 1.63e-3 |

On flat ground the same construction gives 6.8e-3 and 4.5e-4 for n = 4 and 8.
Under geometry refinement the averaged normals converge at observed orders 1.7
and 1.9, facet normals at 0.24 and 0.98 and the exact normal at 3.6 and 3.2.
At n = 16 the averaged-normal error is a tenth of the facet-normal error and
40 times the exact-normal error; the P1 normal reconstruction now limits the
accuracy. With fixed geometry every variant stalls at its geometric error.
The facet flux is not an error measure here: with the exact normal it equals
the exact solution's own facet flux (5.9 % of the inflow at n = 16), while the
flux along the true normal is 0.039 %.

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
  wall law, Nitsche terms on sloped ground, grad-div and backflow damping,
  serially and on two MPI ranks;
- manufactured solutions with both closures on `[0,2] × [0,1] × [0,1]`, with
  log-law ground, slip sides and top and a traction outlet, in two
  constructions. The weak construction imposes the exact solution through the
  implemented residual and checks its discretization. The independent
  construction derives the forcing `-div(2 ν_eff ε(u)) + (u·∇)u + ∇p`, the
  eddy viscosity, the outflow traction `ν_eff ∇u n - p n`, the top shear `u*²`
  and the wall drag `C_d |u_t| u_t` from the model equations; the implemented
  terms must reproduce them. Reversing the top shear, dropping the outflow
  correction or doubling `C_d` stalls velocity convergence at 5.5e-2, 3.6e-3
  and 7.0e-2. L2 errors on 2, 4 and 8 subdivisions per unit length:

  | Construction | Closure | Velocity errors | Pressure errors | Finest-pair orders |
  | --- | --- | --- | --- | --- |
  | weak | `abl` | 1.73e-2, 2.52e-3, 3.47e-4 | 1.87e-2, 4.25e-3, 1.03e-3 | 2.86 / 2.04 |
  | weak | `mixing_length` | 2.07e-2, 3.01e-3, 4.13e-4 | 2.02e-2, 4.61e-3, 1.09e-3 | 2.87 / 2.08 |
  | independent | `abl` | 7.38e-3, 9.07e-4, 1.01e-4 | 1.67e-2, 4.06e-3, 1.01e-3 | 3.17 / 2.01 |
  | independent | `mixing_length` | 6.62e-3, 9.84e-4, 1.29e-4 | 1.71e-2, 4.13e-3, 1.02e-3 | 2.93 / 2.02 |

- horizontal homogeneity on a 200 × 20 × 80 m box with 10 m cells for
  Navier–Stokes with `abl` and `mixing_length` and for Stokes with `abl`:
  outlet speeds within 1 % of equilibrium between 5 and 75 m height;
- lateral-face classification for several wind directions, and the oblique
  20° case above for both closures;
- the impermeability gates on curved terrain, including rejection with a
  tightened tolerance.

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

| Configuration | Newton iterations | Solve time | Dimensionless divergence | Flux imbalance | Ground / building leakage | Mean / max `ν_t` [m²/s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant `nu_t=2`, no-slip (reference) | 6 | 86 s | 2.09e-3 | 4.5e-14 | 0 / 0 | 2 / 2 |
| `abl`, no-slip | 7 | 101 s | 2.01e-3 | 1.5e-16 | 0 / 0 | 8.75 / 19.3 |
| `abl`, log-law wall | 7 | 171 s | 2.19e-3 | 5.9e-6 | 0.15 % / 0.13 % | 8.80 / 18.7 |
| `mixing_length`, no-slip | 11 | 179 s | 1.69e-3 | 1.4e-14 | 0 / 0 | 10.3 / 22.7 |
| `mixing_length`, log-law wall | 13 | 226 s | 1.87e-3 | 1.9e-5 | 0.17 % / 0.23 % | 11.0 / 24.8 |

Leakage is `∫|u·m|` over the surface divided by the inflow volume flux; the
gates are 2 %. The flux through the terrain facets themselves is 3.4 % and
3.1 % of the inflow; on the planar building walls it equals the leakage. The no-slip ABL rows were measured with ILU(0) velocity blocks.
All four city wind demos also pass on the 27,211-tetrahedron realization.

Times are from a two-core Linux container and are not comparable with the
measurements in [wind verification](wind-verification.md). With `abl`, `ν_t`
ranges from `κ u* z_w` at walls to about 19 m²/s at the top; the domain mean is
about 9 m²/s, compared with the constant 2 m²/s of the earlier demos.

## Validation

The [AIJ benchmarks](../benchmarks/aij/README.md) compare the mixing-length
closure with wind-tunnel measurements of the mean velocity around an isolated
high-rise building (case A) and a slab normal to the wind (case B). For case A
the VDI 3783-9 hit rate of the speed is 0.76 over 186 points, above the
required 0.66; the roof separation is too weak and the reverse flow near the
ground behind the building too strong. For case B it is 0.55 over 224 points:
the wake behind the slab recovers too fast. Mesh independence has not been
established. The `abl` closure does not reach a stationary solution for case
A; its eddy viscosity depends on the wall distance alone.

## Limitations

- Neutral stratification only; no buoyancy, Coriolis turning or Ekman layer.
  The surface-layer profile is assumed over the full domain height.
- Algebraic closures have no turbulence history or transport. Separated wakes,
  recirculation lengths and pedestrian-level gusts are not predicted reliably.
- One roughness length and wall-function height for terrain, façades and
  roofs. Building surfaces are aerodynamically smoother than urban terrain;
  both need separate justification and sensitivity studies.
- The mixing length assumes local equilibrium of turbulence production and
  dissipation. It does not transport turbulence generated upstream into wakes
  and recirculation zones, so building separation, wake recovery and
  pedestrian-level wind are not validated by the flat-domain checks.
- The domain stays aligned with the map axes. Oblique winds enter through two
  faces; the inflow at the corner between them has no upstream development.
- Wall normals are reconstructed to second order from a P1 polyhedral surface.
  On the curved-wall manufactured solution this limits the velocity to about
  second-order convergence, 40 times the exact-normal error at the finest
  level; higher-order geometry or normals would be needed beyond that.
- The domain size, mesh sensitivity and wall-function height have not been
  studied for city cases. City runs do not establish physical validity.

The next step is a two-equation closure (k-ε with Richards–Hoxey inflow or
k-ω SST) on the same variable-viscosity formulation, wall distance and wall law.
