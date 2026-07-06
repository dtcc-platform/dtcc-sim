# dtcc-sim Dataset QA Matrix

This matrix tracks the simulation Dataset v2 descriptors registered by
`dtcc_sim.datasets`. It is intentionally static and offline: solver-heavy,
provider-backed, and FEniCSx-specific validation remains explicitly marked by
scope instead of being hidden in default CI.

| Dataset | Family | Python result | Formats | Static contract | Lightweight validation | Solver/provider status | Notes |
|---|---|---|---|---|---|---|---|
| `urban_heat_simulation` | urban heat | `dtcc_core.model.VolumeMesh` | `xdmf` | contract-checked | implemented: field attachment and XDMF serialization contract | planned slow solver validation | Full physical validation needs a FEniCSx runtime and representative urban meshes. |
| `urban_wind_simulation` | urban wind | `dtcc_core.model.VolumeMesh` | `pb` | contract-checked | implemented: wind direction/profile math and small IPCS smoke test | implemented small numerical smoke test; broader CFD validation planned | Default tests avoid large city meshes. |
| `air_quality_field` | air-quality field | `dtcc_core.model.VolumeMesh` | `xdmf` | contract-checked | implemented: sensor-to-field plumbing and XDMF serialization contract | provider/live validation planned outside default CI | SMHI/provider drift should be tested with explicit live gates. |
| `traffic_simulation` | traffic | `dtcc_core.model.RoadNetwork` | `pb` | contract-checked | implemented: deterministic static assignment on synthetic roads/zones | broader calibration planned | Existing tests validate flow, background flow, one-way behavior, and protobuf serialization. |
