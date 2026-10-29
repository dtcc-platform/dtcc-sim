"""Solve a Gaussian source in an offline box and plot its wave amplitude."""

from dtcc_core import BoxMesh
from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzSolver

output_dir = "output/helmholtz_box"
source = (0.5, 0.5, 0.5)
mesh = BoxMesh(8, 8, 8)
parameters = HelmholtzParameters(
    frequency_hz=80, sound_speed=343,
    boundary_mode="impedance_only", source_location_mode="manual",
    source_x=source[0], source_y=source[1], source_z=source[2],
    gaussian_source_sigma=0.12,
    n_subdomains=2, partitioner="structured", nx_sub=2, ny_sub=1, nz_sub=1,
    workers=1, overlap_layers=1, oversample_layers=1, n_modes_per_patch=4,
    outer_rtol=1e-8, verbose=False,
)
simulation = HelmholtzSolver(mesh=mesh, output_dir=output_dir, params=parameters)
result = simulation.simulate()
result.info()
result.plot(field="helmholtz_amplitude")
