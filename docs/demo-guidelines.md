# Demo guidelines

Demos teach the public Python API. Each has one clear purpose that does not
repeat another demo; prefer improving an existing example over adding one.

- Run from the repository root with `python demos/<name>.py` in the documented
  environment, without arguments or an execution switch.
- Use a short docstring, editable settings near the top, and linear top-level
  statements. No `main()`, entry-point guard, helper framework or mode loops.
- Aim for at most 40 lines. Show public API calls directly; reusable setup,
  processing and plotting algorithms belong in the library.
- Name dataset examples `dataset_<name>.py`: request a model, call `.info()`,
  then plot or view the relevant result. Direct builder or solver examples
  should teach a distinct workflow. Use domain names for datasets.
- Prefer small, offline inputs where possible. State environments, live-data
  requirements, units and limitations in the [demo catalog](../demos/README.md).
  Missing required dependencies or data must fail clearly, without fake results.
- Use Core’s `object.plot(...)` as the main plotting entrypoint and show a useful
  visualization without a flag. Save only useful artifacts under
  `output/`; directory creation belongs in the library. Solvers handle
  directory creation internally. Repeated runs replace their own outputs in the
  specified directory and preserve unrelated files.
- Keep CLIs, deployment, detached launchers, benchmarks and reproduction tooling
  outside `demos/`. Numerical acceptance and export round-trip checks belong in
  tests. Keep headless setup, mocking and temporary output redirection there too.

Verify a new or revised demo's purpose, argument-free execution, visualization
and repeated-run behavior in its required environment. Use real numerical or
geometric execution for the claims it demonstrates.

## Sim specifics

Import the Core and solver APIs needed for the example. Linear Helmholtz demos
use `workers=1`: the parallel backend uses multiprocessing `spawn`, which
re-executes top-level scripts. Teach parallel execution through the launcher.
Helmholtz amplitude is not sound-pressure level or dB. Existing demos are not
all compliant yet; apply these guidelines when adding or revising them.
