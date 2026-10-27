# Pinned Linux Helmholtz installation

The optional installer reconstructs the recorded Linux x86-64 runtime in a new
prefix, builds the native dependencies and Sim, and runs dependency/readiness
checks. It uses exact Conda artifact URLs,
pinned Python/build packages and native Git commits. It does not modify the
active environment. Other operating systems should use the general installation
guide; this recorded recipe is Linux-specific.

Prerequisites: Python 3.9 or later to launch the installer, Git, a working Conda
executable, Internet access, native build resources and space for a new scientific
environment. The installed runtime uses Python 3.12. Core and Mesher are built
from the official revisions in `requirements/helmholtz-linux-64/sources.json`.
The pinned Core supports tagged native volume-mesh persistence. Its Mesher
dependency and the installer use the same revision. No source patches are applied.

From the checkout root:

```bash
python3 tools/install_helmholtz.py \
  --conda /path/to/conda \
  --work-dir /path/to/new-installation \
  --verify --detach
```

The output directory must not exist and must be outside the checkout. The
detached process survives normal terminal/SSH disconnects. It records a PID,
status.json, per-stage commands/logs/exit codes, input hashes, package inventory,
wheel hashes, comparison.json and a final exit-code.txt. It does not resume a
terminated installer: inspect the failure and use another new directory.

`--verify` adds focused tests and two small actual mesh-generation/cache,
solver and four-format export checks. It may download geodata or use upstream
caches. These checks do not require thesis inputs and do not run the full city
benchmark. They verify independent residuals <=1e-5 and exact export round trips;
small-case iteration counts are observations, not portable performance promises.

## Packaging safeguards

CMake 3.31.10 is installed explicitly. The VTK workaround applies only to the
recorded Conda vtk-base 9.7.0 build py312h3887a20_1: both records must describe
the same VTK version, have the same Conda owner, and match the known incomplete
metadata hashes and file list. It preserves incomplete dist-info outside
site-packages while retaining the valid legacy egg-info. Unexpected records
fail instead of being removed. The backup and metadata-fix.json make the move
reversible; package binaries and Conda ownership records remain untouched.
A reinstall can restore the upstream packaging defect. See also the
[installation validation notes](helmholtz-installation-validation.md).

Every lock input is checked against SHA256SUMS before installation. Conda
artifact URLs include their recorded package checksums. Python packages are
version-pinned from the public registry; this is not an air-gapped dependency
bundle or a claim that registry artifacts will remain available forever.

## Automated checks

The `Helmholtz runtime` GitHub Actions workflow builds the pinned installation
and runs the focused tests on changes to the solver, installer, dependencies,
tests or workflow. It has read-only repository permission, does not persist the
checkout credential, and retains diagnostic logs for seven days. It uses no
server credentials or thesis inputs. Every run includes a manufactured plane-wave
accuracy solve and the self-contained Gaussian box demo, preserving its summary
and amplitude screenshot as artifacts. The real geodata-dependent workflow can
be requested separately with the manual `real_workflow` option, so upstream
geodata availability does not gate ordinary focused checks.

A workflow file does not enable Actions in a repository whose settings disable
it. Enabling that capability is a separate repository-setting action. Local or
server execution of the same commands is separate evidence from a hosted
GitHub Actions result.

The recorded scripted installation completed in about 277 seconds on the
existing Linux server, including 129 focused/installer checks and both real
workflows, without manual repair after launch. This is a single same-host
measurement with package/geodata caches available; it is not a promised
installation time on another machine. See the
[scripted validation report](validation/helmholtz-scripted-install.json).
