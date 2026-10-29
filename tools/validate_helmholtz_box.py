"""Generate a small box, solve a Gaussian source, export and plot a volume slice."""

import argparse
import json
from pathlib import Path

import numpy as np


def horizontal_slice(vertices, cells, values, z):
    """Intersect tetrahedra and interpolate the complex P1 field on their edges."""
    vertices = np.asarray(vertices)
    values = np.asarray(values)
    points, triangles, samples = [], [], []
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    tolerance = max(np.ptp(vertices, axis=0).max(), 1.0) * 1e-12
    for cell in cells:
        xyz, field = vertices[cell], values[cell]
        if z < xyz[:, 2].min() - tolerance or z > xyz[:, 2].max() + tolerance:
            continue
        cut, data = [], []
        for i, j in edges:
            zi, zj = xyz[i, 2] - z, xyz[j, 2] - z
            for index in (i, j):
                if abs(xyz[index, 2] - z) <= tolerance:
                    cut.append(xyz[index, :2])
                    data.append(field[index])
            if zi * zj < 0:
                t = -zi / (zj - zi)
                cut.append((1 - t) * xyz[i, :2] + t * xyz[j, :2])
                data.append((1 - t) * field[i] + t * field[j])
        if len(cut) < 3:
            continue
        _, ids = np.unique(
            np.round(np.asarray(cut) / tolerance).astype(np.int64),
            axis=0,
            return_index=True,
        )
        if len(ids) < 3:
            continue
        xy, u = np.asarray(cut)[ids], np.asarray(data)[ids]
        delta = xy - xy.mean(axis=0)
        order = np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))
        xy, u = xy[order], u[order]
        offset = len(points)
        points.extend(xy)
        samples.extend(u)
        for i in range(1, len(xy) - 1):
            a, b = xy[i] - xy[0], xy[i + 1] - xy[0]
            if abs(a[0] * b[1] - a[1] * b[0]) > tolerance**2:
                triangles.append([offset, offset + i, offset + i + 1])
    if not triangles:
        raise ValueError("Slice does not intersect the volume with positive area")
    return np.asarray(points), np.asarray(triangles), np.asarray(samples)


def plot_amplitude_slice(mesh, *, z=None, source=None, path=None, show=True):
    """Plot |u| on a horizontal P1 cut; z defaults to the volume's mid-height.

    Amplitude is in solver units, not sound-pressure level. Return the axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from pathlib import Path

    fields = {f.name: np.asarray(f.values).ravel() for f in mesh.fields}
    values = fields["helmholtz_real"] + 1j * fields["helmholtz_imaginary"]
    if z is None:
        z = (mesh.vertices[:, 2].min() + mesh.vertices[:, 2].max()) / 2
    points, triangles, u = horizontal_slice(mesh.vertices, mesh.cells, values, z)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    artist = ax.tripcolor(tri, np.abs(u), shading="gouraud", cmap="viridis")
    fig.colorbar(artist, ax=ax, label="|u| (solver amplitude units)")
    if source is not None:
        ax.plot(source[0], source[1], "r*", markersize=13, label="Gaussian source centre")
        ax.legend(loc="upper right")
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal",
           title=f"Helmholtz amplitude at z={z:g} m")
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    if show:
        plt.show()
    return ax


def save_amplitude_slice(mesh, path, *, source, result, z=None):
    """Save a headless cut with convergence details for validation tools."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = plot_amplitude_slice(
        mesh, z=source[2] if z is None else z, source=source, show=False
    )
    ax.set_title(
        f"{ax.get_title()}\n{result.iterations} iterations; "
        f"true residual {result.relative_residual:.2e}"
    )
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)



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
    from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzSolver

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
    sim = HelmholtzSolver(mesh_path=path, output_dir=output / "case", params=params)
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
