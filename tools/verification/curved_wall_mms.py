"""Manufactured flow over a curved wall: solution versus geometry refinement.

The domain lies between the curved ground z = g(x) = A sin(2πx/λ) and a flat
top z = H. The exact velocity follows from a stream function that is constant
on both, so it is tangent to the true ground. Forcing, outflow traction, top
traction and log-law wall drag are derived independently from the exact
solution and the discrete facet normals; the condition u·n = 0 on the
polyhedral ground is left as the solver imposes it.

Two refinements are compared:

* geometry: every mesh places its ground nodes on the curve;
* fixed: a coarse mesh is mapped once and then refined, so the polyhedral
  ground stays the same while the solution space is refined.

    python tools/verification/curved_wall_mms.py --levels 4 8 16 --output curved.json
"""

import argparse
import json

import numpy as np
import ufl
import dolfinx
import dolfinx.mesh as dm
from mpi4py import MPI

from dtcc_sim._flow.parameters import KAPPA
from dtcc_sim._flow.solver import (BndCat, DEFAULT_STATIONARY_PETSC, _MixedFlowSystem,
                                   categorize_boundary)
from dtcc_sim.navierstokes.parameters import NavierStokesParameters

L, W, H = 2., .5, 1.
U, NU, Z0, DELTA = 1., .05, .05, .1


def ground(x, amplitude, wavelength, module=np):
    return amplitude * module.sin(2 * np.pi * x / wavelength)


def mapped_box(n, amplitude, wavelength):
    mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [L, W, H]], [2 * n, max(1, n // 2), n])
    x = mesh.geometry.x
    x[:, 2] += ground(x[:, 0], amplitude, wavelength) * (1 - x[:, 2] / H)
    return mesh


def tags(mesh):
    """DTCC bbox markers from facet midpoints; remaining exterior facets are ground."""
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = dm.exterior_facet_indices(mesh.topology)
    mid = dm.compute_midpoints(mesh, 2, faces)
    values = np.full(len(faces), -1, dtype=np.int32)
    for axis, coordinate, marker in [(2, H, -2), (0, 0., -3), (0, L, -4), (1, 0., -5), (1, W, -6)]:
        values[np.isclose(mid[:, axis], coordinate)] = marker
    order = np.argsort(faces)
    return dm.meshtags(mesh, 2, faces[order], values[order])


def exact(x, amplitude, wavelength):
    g = ground(x[0], amplitude, wavelength, ufl)
    eta = (x[2] - g) / (H - g)
    stream = U * H * (1.5 * eta - .5 * eta**2)
    velocity = ufl.as_vector((stream.dx(2), 0., -stream.dx(0)))
    pressure = .5 * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[2])
    return velocity, pressure


def exact_inflow(amplitude, wavelength):
    """Numpy form of the exact velocity for the inflow boundary."""
    k = 2 * np.pi / wavelength

    def velocity(points):
        x, z = points[0], points[2]
        g, slope = amplitude * np.sin(k * x), amplitude * k * np.cos(k * x)
        eta = (z - g) / (H - g)
        factor = U * H * (1.5 - eta)
        return np.vstack([factor / (H - g), np.zeros_like(x), -factor * slope * (z - H) / (H - g)**2])

    return velocity


def vertex_normals(mesh, categories):
    """Continuous P1 ground normal from area-weighted facet normals at each vertex."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    field = dolfinx.fem.Function(V)
    facets = categories.find(int(BndCat.GROUND)).astype(np.int32)
    nodes = dm.entities_to_geometry(mesh, 2, facets)
    points = mesh.geometry.x[nodes]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    cross[cross[:, 2] < 0] *= -1  # upward; outward is downward and the sign cancels in u·m m
    vertex_dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
    coordinates = V.tabulate_dof_coordinates()
    lookup = {tuple(np.round(coordinates[d], 10)): d for d in vertex_dofs}
    values = field.x.array.reshape(-1, 3)
    for corner_points, normal in zip(points, cross):
        for point in corner_points:
            values[lookup[tuple(np.round(point, 10))]] += normal
    field.x.scatter_forward()
    return field / ufl.sqrt(ufl.dot(field, field))


class _NormalFlowSystem(_MixedFlowSystem):
    """Ground Nitsche terms with a supplied normal ``m`` instead of the facet normal."""

    normal = None

    def _normal_boundary(self, u, p, v, q, n, nu, tag, penalty):
        if self.normal is None or tag != int(BndCat.GROUND):
            return super()._normal_boundary(u, p, v, q, n, nu, tag, penalty)
        m = self.normal(self.mesh)
        um, vm = ufl.dot(u, m), ufl.dot(v, m)
        height = 3 * ufl.CellVolume(self.mesh) / ufl.FacetArea(self.mesh)
        gamma = self.params.nitsche_penalty * (nu / height + self.params.wind_speed)
        self.F += (-(nu * ufl.dot(ufl.grad(u) * n, m) - p * ufl.dot(n, m)) * vm
                   - (nu * ufl.dot(ufl.grad(v) * n, m) + q * ufl.dot(n, m)) * um + gamma * um * vm) * self.ds(tag)
        # The caller adds facet-tangential wall drag; replace it by drag tangential to m.
        facet_tangential, tangential = u - ufl.dot(u, n) * n, u - um * m
        self.F += (self._wall_friction(tangential) * ufl.inner(tangential, v)
                   - self._wall_friction(facet_tangential) * ufl.inner(facet_tangential, v)) * self.ds(tag)
        self._ground_normal = m


def solve(mesh, amplitude, wavelength, normal="facet"):
    markers = tags(mesh)
    params = NavierStokesParameters(
        nu=NU, wind_speed=U, z0=Z0, wall_function_height=DELTA, inlet_profile="log_law",
        wall_model="log_law", side_top_boundary="slip",
    )
    categories = categorize_boundary(mesh, markers, 0, -3, -4)
    x = ufl.SpatialCoordinate(mesh)
    u_exact, p_exact = exact(x, amplitude, wavelength)
    slope = ground(x[0], amplitude, wavelength, ufl).dx(0)
    true_normal = ufl.as_vector((-slope, 0., 1.)) / ufl.sqrt(1 + slope**2)
    system_type = _MixedFlowSystem if normal == "facet" else type("System", (_NormalFlowSystem,), {
        "normal": staticmethod((lambda mesh: true_normal) if normal == "exact"
                               else (lambda mesh: vertex_normals(mesh, categories)))})
    system = system_type(mesh, params, markers, categories, exact_inflow(amplitude, wavelength))
    v, _ = ufl.TestFunctions(system.W)
    n = ufl.FacetNormal(mesh)
    m = getattr(system, "_ground_normal", n)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 8})
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=categories, metadata={"quadrature_degree": 8})
    forcing = -NU * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    traction = NU * ufl.grad(u_exact) * n
    tangential = lambda w: w - ufl.dot(w, n) * n
    along = lambda w: w - ufl.dot(w, m) * m
    drag = (KAPPA / np.log((Z0 + DELTA) / Z0))**2 * ufl.sqrt(ufl.dot(along(u_exact), along(u_exact)))
    system.F -= (ufl.inner(forcing, v) * dx
                 + ufl.inner(traction - p_exact * n, v) * ds(int(BndCat.OUTLET))
                 + ufl.inner(tangential(traction), v) * ds(int(BndCat.TOP))
                 + ufl.inner(along(traction - p_exact * n) + drag * along(u_exact), v) * ds(int(BndCat.GROUND)))
    system.J = ufl.derivative(system.F, system.w)
    system.configure({**DEFAULT_STATIONARY_PETSC, "ksp_type": "preonly", "pc_type": "lu",
                      "pc_factor_mat_solver_type": "mumps", "snes_type": "newtonls",
                      "snes_linesearch_type": "bt", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 30})
    try:
        system.solve(0.)
        system.solve(1.)
        u, p = system.fields()
        integral = system.integral
        inflow_flux = integral(-ufl.dot(u_exact, n) * ds(int(BndCat.INLET)))
        ground_measure = ds(int(BndCat.GROUND))
        return {
            "cells": mesh.topology.index_map(3).size_global,
            "velocity_error": np.sqrt(integral(ufl.inner(u - u_exact, u - u_exact) * dx)),
            "pressure_error": np.sqrt(integral((p - p_exact)**2 * dx)),
            "leakage_fraction": integral(abs(ufl.dot(u, n)) * ground_measure) / inflow_flux,
            "exact_normal_flux_fraction": integral(abs(ufl.dot(u_exact, n)) * ground_measure) / inflow_flux,
            "wall_velocity_error": np.sqrt(integral(ufl.inner(u - u_exact, u - u_exact) * ground_measure)),
            "true_normal_flux_fraction": integral(abs(ufl.dot(u, true_normal)) * ground_measure) / inflow_flux,
        }
    finally:
        system.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--amplitude", type=float, default=.1)
    parser.add_argument("--wavelength", type=float, default=1.)
    parser.add_argument("--normals", nargs="+", default=["facet"], choices=["facet", "vertex", "exact"])
    parser.add_argument("--output")
    args = parser.parse_args()
    results = []
    for normal in args.normals:
        for n in args.levels:
            mesh = mapped_box(n, args.amplitude, args.wavelength)
            results.append({"refinement": "geometry", "normal": normal, "n": n,
                            **solve(mesh, args.amplitude, args.wavelength, normal)})
        mesh = mapped_box(args.levels[0], args.amplitude, args.wavelength)
        for n in args.levels:
            if n != args.levels[0]:
                mesh, _, _ = dm.refine(mesh)
            results.append({"refinement": "fixed", "normal": normal, "n": n,
                            **solve(mesh, args.amplitude, args.wavelength, normal)})
        if MPI.COMM_WORLD.rank == 0:
            for row in results[-2 * len(args.levels):]:
                print(f"{row['normal']:6s} {row['refinement']:8s} n={row['n']:3d} u={row['velocity_error']:.3e} "
                      f"p={row['pressure_error']:.3e} wall u={row['wall_velocity_error']:.3e} "
                      f"facet leak={100 * row['leakage_fraction']:.3f}% true-normal flux={100 * row['true_normal_flux_fraction']:.3f}% "
                      f"exact facet flux={100 * row['exact_normal_flux_fraction']:.3f}%", flush=True)
    if args.output and MPI.COMM_WORLD.rank == 0:
        with open(args.output, "w") as stream:
            json.dump({"arguments": vars(args), "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
