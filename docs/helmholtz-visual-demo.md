# Helmholtz box demo

Activate the [complex environment](installation.md#complex-number-environment)
and run from the repository root:

```bash
python demos/helmholtz_box.py
```

The demo uses Core’s `BoxMesh(8, 8, 8)` and `HelmholtzSolver` with no
FEniCSx or MPI objects in the script. The offline box is 1 × 1 × 1 m.
It uses 729 vertices, 3,072 tetrahedra, 80 Hz, sound speed 343 m/s, two structured
subdomains and one worker. A unit-amplitude Gaussian source is centred at
(0.5, 0.5, 0.5) m with sigma 0.12 m. All exterior faces have homogeneous
impedance conditions; there are no Dirichlet faces or required boundary tags.

It uses `result.plot(field="helmholtz_amplitude")` for Core’s 3D field preview
and saves `solution.vtu` and numerical case data under `output/helmholtz_box`. The solver
creates the directory and replaces its previous outputs on repeated runs;
unrelated files are preserved.
Amplitude is in solver units, not calibrated sound pressure or dB.

For the high-level city dataset API, run `python demos/dataset_urban_noise.py`.
That example requires live city data and uses `datasets.urban_noise_simulation`.
It plots the amplitude field and reuses a separate mesh cache. Generated city
runs also save `city.vtu`, containing the solver mesh’s building and ground faces.
In ParaView, show `city.vtu` in grey and a slice of `solution.vtu` coloured by
the solution field; both use the same coordinates.
For configuration-driven runs, see the [workflow guide](helmholtz-quickstart.md).
Numerical acceptance and headless export checks live in
`tools/validate_helmholtz_box.py`, outside the demo. Its diagnostic slice
interpolates the complex P1 field before computing magnitude.
