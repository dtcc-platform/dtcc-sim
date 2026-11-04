# AIJ pedestrian-wind benchmarks

Cases A and B of the Architectural Institute of Japan (AIJ)
[CFD benchmarks for pedestrian wind environments](https://www.aij.or.jp/jpn/publish/cfdguide/index_e.htm):
wind-tunnel measurements of the mean velocity around isolated buildings in a
boundary-layer flow.

| Case | Building (width × depth × height) | Measurements | Source |
| --- | --- | --- | --- |
| A | 0.08 × 0.08 × 0.16 m (1:1:2) | 66 points in the vertical centre plane, 60 points each at z = 0.125b and 1.25b | Meng and Hibi (1998) |
| B | 0.2 × 0.05 × 0.2 m (4:4:1 slab, broad face to the wind) | 109 points in the vertical centre plane, 115 points at z = 12.5 mm | Miyazaki and Tominaga (2003) |

Measured are the mean velocity components and the turbulent kinetic energy.
The workbooks give case A coordinates in multiples of b; b = 0.08 m is the
model size of Meng and Hibi. Case B states b = 50 mm.

## Data

Download `CaseA(1_1_2).xls` and `CaseB(4_4_1).xls` from the AIJ page and place
them in `benchmarks/data/aij/`. The page states no licence for the data, so
they are not part of this repository.

## Setup

- **Geometry and domain.** The building stands on flat ground with its
  footprint centred at the origin and the wind along x. Both cases are
  symmetric about y = 0, where all measurements lie, so the half domain
  y ≤ 0 is solved with a slip plane at y = 0. Following the AIJ guidelines
  (Tominaga et al., 2008, J. Wind Eng. Ind. Aerodyn. 96, 1749–1761) the domain
  extends 5H upstream, laterally and above the building and 10H downstream,
  with H the building height.
- **Mesh.** A graded tensor grid split into tetrahedra
  ([`geometry.py`](geometry.py)). The spacing is `h` at the building and over
  the horizontal extent of the measurements, `h/3` at the ground, and grows by
  20 % per cell to at most H/2. Resolutions `coarse`, `medium` and `fine` use
  `h` = H/6, H/8 and H/12. Building faces are planar and axis aligned.
- **Model.** Stationary Navier–Stokes with Taylor–Hood elements, the
  mixing-length closure (`mixing_length`, default) or the equilibrium profile
  closure (`abl`), and log-law walls on the ground and the building; see the
  [atmospheric boundary layer guide](../../docs/atmospheric-boundary-layer.md).
  As in the city wind demos, grad-div stabilization (γ = 3) and outlet
  backflow damping (β = 2) are used.
- **Approach flow.** A log law u*/κ ln((z + z0)/z0) fitted to the measured
  inflow profile up to the building height gives u* and z0. The closures need
  this equilibrium profile; the measured profile follows a power law with
  exponent about 1/4 and is not a log law over the whole height (see
  `inflow.png`).
- **Wall offset.** The log-law wall places the computational wall δ = 2 mm
  outside every solid surface. The mesh building is the measured building
  grown by δ on its walls, and a probe at height Z is sampled at mesh height
  Z − δ.
- **Normalization.** Velocities are divided by the approach-flow speed at the
  building height: the fitted log law for the computation and the measured
  inflow profile for the experiment.

## Metrics

[`analysis.py`](analysis.py) compares computed and measured values point by
point. The hit rate follows VDI 3783 Part 9: the fraction of points with
|P − O| ≤ 0.25 |O| or |P − O| ≤ 0.05 (normalized), where VDI 3783-9 asks for at
least 0.66. FAC2 is the fraction with 0.5 ≤ P/O ≤ 2, FB the fractional bias
(positive when the computation is low) and R the correlation. Metrics are
reported for the speed of the mean velocity |U| and for its streamwise
component.

## Running

```bash
python -m benchmarks.aij.run --case A --resolution coarse
python -m benchmarks.aij.run --case B --resolution coarse --turbulence-model abl
python -m benchmarks.aij.report output/benchmarks/aij
```

`run` writes `output/benchmarks/aij/case-<case>/<model>-<resolution>/results.json`
with the inflow fit, mesh, solver diagnostics, sampled and measured values
and metrics. `report` writes `summary.md` and per-case figures: the approach
flow, the measured sections and a scatter plot of the speeds.

## Results

Mixing-length closure, serial runs on two cores with 7 GB of memory; metrics
for the speed |U| / U_H:

| Case | Resolution | Cells | Newton | Time | Section | Hit rate | FAC2 | FB | R |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| A | coarse (H/4) | 62,928 | 8 | 501 s | vertical | 0.65 | 0.83 | +0.08 | 0.90 |
| | | | | | z = 0.125b | 0.77 | 0.88 | +0.02 | 0.85 |
| | | | | | z = 1.25b | 0.82 | 0.93 | −0.02 | 0.89 |
| | | | | | all 186 points | 0.74 | 0.88 | +0.03 | 0.89 |
| A | medium (H/6) | 106,488 | 8 | 1013 s | vertical | 0.70 | 0.86 | +0.06 | 0.92 |
| | | | | | z = 0.125b | 0.77 | 0.90 | +0.02 | 0.87 |
| | | | | | z = 1.25b | 0.82 | 0.92 | −0.04 | 0.91 |
| | | | | | all 186 points | 0.76 | 0.89 | +0.02 | 0.91 |
| B | coarse (H/4) | 75,024 | 7 | 583 s | vertical | 0.54 | 0.74 | −0.12 | 0.85 |
| | | | | | z = 12.5 mm | 0.56 | 0.88 | −0.16 | 0.90 |
| | | | | | all 224 points | 0.55 | 0.81 | −0.14 | 0.87 |

- **Case A** meets the VDI 3783-9 hit rate of 0.66 overall and in every
  section at the medium resolution. The approach flow, the flow around the
  sides and the pedestrian level agree well. The separation over the roof
  is too weak: at z = 1.06H above x = −0.25b and 0 the measured speeds are
  0.38 and 0.28 U_H, the computed 1.16 and 0.52 U_H. Behind the building at
  z = 0.06H the computed reverse flow is stronger than measured: U = −0.44 to
  −0.47 U_H against −0.14 to −0.16 U_H at x = 0.75b and 1.25b. Above 1.25H the
  computed speeds are 6–16 % low, as the fitted log law is below the measured
  inflow there.
- **Case B** misses the hit rate. The wake behind the slab recovers too fast:
  from x = 4b to 8b the computed speeds in the wake are well above the
  measured ones, and the overall fractional bias of −0.14 is an
  overprediction.
- **Mesh sensitivity.** From coarse to medium the computed speeds at the case
  A measurement points change by 0.028 U_H on average and at most 0.23 U_H,
  near the roof edge; the metrics change by at most 0.05. Case B medium and
  the fine levels need more than 7 GB of memory (about 6 GB for 106,488 cells
  with the default solver settings), so mesh independence is not established.
- **Equilibrium closure.** With `--turbulence-model abl` Newton iteration
  stalls for case A at a residual of 2.7e-4, 2 % of its initial value. The
  closure sets the eddy viscosity from the wall distance alone, so it is small
  in the separated shear layers; a stationary solution may not exist at that
  viscosity. Use the mixing length for flow around buildings.

These results test isolated buildings in a wind-tunnel boundary layer. They do
not establish accuracy in dense urban areas.

## Limitations

- The approach flow is a log-law fit, not the measured power-law profile.
- The algebraic closures do not use the measured turbulence; the wind-tunnel
  turbulent kinetic energy near the ground is two to three times the
  equilibrium value u*²/√Cμ of the fitted profile.
- The same z0 and δ apply to the ground and the smooth model building.
- Turbulent kinetic energy is not computed, so only mean velocities are compared.
