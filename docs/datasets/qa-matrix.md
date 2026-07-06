# dtcc-sim Dataset QA Matrix

This matrix tracks the simulation Dataset v2 descriptors registered by
`dtcc_sim.datasets`. It is intentionally static and offline: solver-heavy,
provider-backed, and FEniCSx-specific validation remains explicitly marked by
scope instead of being hidden in default CI.

Runtime tiers:

- `static/default`: descriptor, service, traffic, and plumbing tests that do
  not require FEniCSx.
- `simulation`: simulation dataset or solver validation tests.
- `fenics`: tests that require a FEniCSx/dolfinx runtime and skip with an
  explicit reason when it is absent.
- `slow`: small numerical checks that are slower than pure unit tests.
- `expensive`: manual-scale simulations skipped unless `--run-expensive` is
  passed.

| Dataset | Family | Python result | Formats | Static contract | Lightweight validation | Solver/provider status | Notes |
|---|---|---|---|---|---|---|---|
| `urban_heat_simulation` | urban heat | `dtcc_core.model.VolumeMesh` | `xdmf` | contract-checked | implemented: descriptor context, boundary parameter plumbing, field attachment, XDMF serialization, and tiny bounded Dirichlet box validation | implemented tiny FEniCSx validation; broader city-scale validation planned as expensive/manual | Full physical validation needs calibrated boundary data and representative urban meshes. |
| `urban_wind_simulation` | urban wind | `dtcc_core.model.VolumeMesh` | `pb` | contract-checked | implemented: descriptor context, wind direction/profile math, field contract checks, diagnostics, and small IPCS/Stokes smoke tests | implemented tiny FEniCSx validation; broader city-scale CFD validation planned as expensive/manual | Default tests avoid large city meshes; pressure is kinematic p/rho in m^2/s^2. |
| `air_quality_field` | air-quality field | `dtcc_core.model.VolumeMesh` | `xdmf` | contract-checked | implemented: descriptor context, provider observation validation, unit propagation, scalar field checks, tiny synthetic reconstruction, diagnostics, and XDMF serialization contract | implemented tiny FEniCSx reconstruction validation; live provider drift planned outside default CI | Derived field is not a measured station product; SMHI/provider drift should be tested with explicit live gates. |
| `traffic_simulation` | traffic | `dtcc_core.model.RoadNetwork` | `pb` | contract-checked | implemented: descriptor context, deterministic static assignment on synthetic roads/zones, flow/capacity/travel-time checks, one-way behavior, required DeSO statistics, diagnostics, and protobuf serialization | broader live-data review and calibration planned | Synthetic OD demand is not calibrated traffic; outputs inherit roads/DeSO source and attribute limitations. |
