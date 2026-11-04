"""Inflow fitting, point sampling and validation metrics."""

import numpy as np

KAPPA = .41


def fit_log_law(z, speed, top):
    """Friction velocity u* and roughness z0 of u*/κ ln((z + z0)/z0) fitted to
    ``speed`` at heights ``z <= top``; returns (u*, z0, rms residual)."""
    from scipy.optimize import least_squares

    z, speed = np.asarray(z, dtype=float), np.asarray(speed, dtype=float)
    keep = z <= top * (1 + 1e-9)
    z, speed = z[keep], speed[keep]

    def residual(p):
        z0 = np.exp(p[1])
        return p[0] / KAPPA * np.log((z + z0) / z0) - speed

    start = [KAPPA * speed[-1] / np.log(z[-1] / (1e-3 * z[-1])), np.log(1e-3 * z[-1])]
    fit = least_squares(residual, start)
    return float(fit.x[0]), float(np.exp(fit.x[1])), float(np.sqrt(np.mean(fit.fun**2)))


def log_law(z, friction_velocity, z0):
    return friction_velocity / KAPPA * np.log((np.asarray(z) + z0) / z0)


def sample(function, points):
    """Values of a DOLFINx function at points, identical on every rank.

    Points outside the mesh give NaN.
    """
    import dolfinx.geometry as geometry

    mesh = function.function_space.mesh
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(mesh, candidates, points)
    size = int(np.prod(function.ufl_shape)) if function.ufl_shape else 1
    values = np.full((len(points), size), np.nan)
    owned = mesh.topology.index_map(mesh.topology.dim).size_local
    found, cells = [], []
    for i in range(len(points)):
        links = colliding.links(i)
        links = links[links < owned]
        if len(links):
            found.append(i)
            cells.append(links[0])
    if found:
        values[found] = function.eval(points[found], np.asarray(cells, dtype=np.int32)).reshape(len(found), size)
    gathered = mesh.comm.allgather(values)
    result = np.full_like(values, np.nan)
    for part in gathered:
        fill = np.isnan(result[:, 0]) & ~np.isnan(part[:, 0])
        result[fill] = part[fill]
    return result


def metrics(predicted, observed, relative=.25, absolute=.05):
    """Validation metrics of predicted against observed values.

    hit_rate is the VDI 3783-9 fraction of points with |P - O| <= relative |O|
    or |P - O| <= absolute; FAC2 the fraction with 0.5 <= P/O <= 2; FB the
    fractional bias, NMSE the normalized mean square error and R the
    correlation. Non-finite pairs are excluded.
    """
    p, o = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    keep = np.isfinite(p) & np.isfinite(o)
    p, o = p[keep], o[keep]
    if not len(p):
        return {"points": 0}
    error = np.abs(p - o)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p / o
    return {
        "points": int(len(p)),
        "hit_rate": float(np.mean((error <= relative * np.abs(o)) | (error <= absolute))),
        "fac2": float(np.mean((ratio >= .5) & (ratio <= 2.))),
        "fractional_bias": float(2 * (o.mean() - p.mean()) / (o.mean() + p.mean())),
        "nmse": float(np.mean((o - p)**2) / (o.mean() * p.mean())),
        "correlation": float(np.corrcoef(p, o)[0, 1]) if len(p) > 1 else float("nan"),
        "mean_absolute_error": float(error.mean()),
    }
