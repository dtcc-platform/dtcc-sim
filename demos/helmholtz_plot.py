"""Horizontal cuts of linear tetrahedral complex fields, without extrapolation."""

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


def save_amplitude_slice(mesh, path, *, source, result, z=None):
    """Save a headless Matplotlib view of the complex solution's horizontal cut."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fields = {f.name: np.asarray(f.values).ravel() for f in mesh.fields}
    values = fields["helmholtz_real"] + 1j * fields["helmholtz_imaginary"]
    z = source[2] if z is None else z
    points, triangles, u = horizontal_slice(mesh.vertices, mesh.cells, values, z)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    artist = ax.tripcolor(tri, np.abs(u), shading="gouraud", cmap="viridis")
    fig.colorbar(artist, ax=ax, label="|u| (solver amplitude units)")
    ax.plot(source[0], source[1], "r*", markersize=13, label="Gaussian source centre")
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
        title=f"Helmholtz amplitude at z={z:g} m\n{result.iterations} iterations; true residual {result.relative_residual:.2e}",
    )
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=160)
    plt.close(fig)
