"""Generate a small box, solve a Gaussian source, export and plot a volume slice."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New result directory"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory")
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, io
    from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzSimulator
    from helmholtz_plot import save_amplitude_slice

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run this example with one MPI rank")
    output.mkdir(parents=True)
    path = output / "input" / "box.xdmf"
    path.parent.mkdir()
    mesh = dmesh.create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)
    with io.XDMFFile(MPI.COMM_WORLD, str(path), "w") as stream:
        stream.write_mesh(mesh)
    source = (0.5, 0.5, 0.5)
    params = HelmholtzParameters(
        frequency_hz=80,
        sound_speed=343,
        boundary_mode="impedance_only",
        source_location_mode="manual",
        source_x=source[0],
        source_y=source[1],
        source_z=source[2],
        gaussian_source_sigma=0.12,
        n_subdomains=2,
        partitioner="structured",
        nx_sub=2,
        ny_sub=1,
        nz_sub=1,
        workers=1,
        overlap_layers=1,
        oversample_layers=1,
        n_modes_per_patch=4,
        outer_rtol=1e-8,
        verbose=False,
    )
    sim = HelmholtzSimulator(mesh_path=path, output_dir=output / "case", params=params)
    result_mesh = sim.simulate(output_path=output / "field.vtu")
    if sim.result.relative_residual > 1e-7:
        raise RuntimeError("True residual exceeds the example acceptance tolerance")
    save_amplitude_slice(
        result_mesh, output / "amplitude-slice.png", source=source, result=sim.result
    )
    summary = {
        "converged": sim.result.converged,
        "iterations": sim.result.iterations,
        "true_relative_residual": sim.result.relative_residual,
        "source": source,
        "frequency_hz": params.frequency_hz,
        "sigma_m": params.gaussian_source_sigma,
        "boundary": "Impedance (Robin) on all six exterior faces; no Dirichlet faces",
        "vertices": len(result_mesh.vertices),
        "tetrahedra": len(result_mesh.cells),
        "slice": "amplitude-slice.png",
        "volume": "field.vtu",
    }
    (output / "example-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
