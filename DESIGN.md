# DTCC Sim Design

## Purpose

DTCC Sim provides urban simulation and field reconstruction through direct
Python APIs and the shared DTCC Dataset contract.

## Place in the DTCC Platform

Sim owns specialized physical models, numerical solvers and their runtime
requirements. DTCC Core owns the semantic models, dataset registry, I/O and
Package contracts; reuse them rather than creating parallel representations.
Application orchestration and catalog operation belong in DTCC Twin.

## Design principles

- Return native Core models with explicit field associations, units and CRS.
  Keep numerical backend objects and diagnostics available through solver APIs.
- Separate scenario parameters, numerical solving and dataset integration.
  Dataset names describe domain products; solver names describe equations.
- Make the common call direct, with documented domain defaults and explicit
  advanced controls. Validate required inputs and fail with actionable errors.
- Keep heavy solver dependencies isolated from capabilities that do not need
  them. Document real versus complex PETSc and supported DOLFINx requirements.
- Distinguish numerical convergence from physical validity. Record assumptions,
  sources, boundary conditions and limitations; do not label amplitude as dB.

## Architecture

- **Parameters and solvers:** validate scenarios, solve equations and report
  convergence and numerical diagnostics.
- **Dataset adapters:** use Core's registry and Context to return semantic
  models with the request, actual provenance and limitations.
- **I/O and presentation:** use Core model persistence and visualization;
  operational launchers and service code remain separate from numerical APIs.

## Trust and quality

Preserve fields and their meaning across persistence. Use Core Packages when
run context and provenance must travel with the model. Protect existing solver
cases from accidental overwrite and keep mesh caches separate from results.

Use small offline contract tests and real numerical regression tests in the
appropriate environment. Mocked tests do not validate solvers. Live-data and
expensive runs are explicit opt-ins. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Evolution

Preserve public behavior unless a deliberate migration changes it. Reuse
existing Core contracts, keep changes focused, and avoid speculative layers.
Demos teach these APIs; follow [demo guidelines](docs/demo-guidelines.md).
