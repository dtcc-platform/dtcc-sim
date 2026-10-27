# Helmholtz installation validation

Use `environment-helmholtz.yml` in the [installation guide](installation.md#helmholtz-environment).
The native TetGen dependency requires CMake >=3.24; the recipe states that minimum
explicitly. The installer uses CMake 3.31.10 and builds Mesher, TetGen and Core
from pinned official sources. Tagged native persistence requires the Core
revision declared by Sim and the installer.

## Recorded VTK packaging issue

Conda-forge `vtk-base` 9.7.0, build `py312h3887a20_1`, supplied both
`vtk-9.7.0.egg-info` and `vtk-9.7.0.dist-info`. Both described the same version
and belonged to the same Conda package. The dist-info contained METADATA and
INSTALLER but no WHEEL file. `uv pip check` rejected the duplicate records;
keeping only the incomplete dist-info also failed.

In the isolated validation prefix, the incomplete dist-info directory was moved
to a checksummed metadata backup and the original legacy egg-info retained.
No VTK binaries or Conda ownership records were changed. Dependency checking
then passed for all 203 installed distributions. The original records and
failed attempts are preserved. This is a documented workaround for that exact
build, not a general instruction to remove package metadata. A Conda reinstall
may restore it; inspect package ownership and record contents before applying
any similar workaround. Reversal restores the original dist-info directory
from the backup; it also restores the original duplicate-metadata condition.

## Verification scope

The [machine-readable report](validation/helmholtz-clean-install.json) records
120 passing focused tests and two real h=25 mesh/solver/export checks using
both weight choices. The first generated a new mesh and the second reused it.
Both solved in 8 iterations, with independently reassembled relative residuals
below 1e-5. Native DTCC, XDMF, VTU and model-package exports passed their checks.

This was a separate newly constructed prefix on the existing Linux host,
using exact recorded Conda builds, pinned Python dependencies, rebuilt native
projects and the reviewed Sim wheel. Downloads could use caches, and upstream
geodata was cached. No production Python overlay or user site packages were
used. This does not establish installation on another operating system or a
pristine machine, and it does not benchmark performance.
