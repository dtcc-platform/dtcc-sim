"""Horizontal homogeneity of the atmospheric boundary layer in an empty domain.

An equilibrium inflow should leave an empty, flat domain unchanged. The script
solves stationary Navier–Stokes over a flat box for several closures and wall
treatments and compares velocity profiles near the outlet with the inflow.

    python tools/verification/abl_homogeneity.py --output homogeneity.json

Profiles are sampled on the centre line at the inlet, midpoint and outlet.
``max_outlet_change`` compares outlet and inlet finite-element profiles;
the equilibrium error excludes heights below one cell, where the inlet is
itself a P2 interpolant of the logarithm.
"""

import argparse
import json
import time

import numpy as np
import dolfinx.mesh as dm
from dolfinx import geometry
from mpi4py import MPI

from dtcc_sim import NavierStokesParameters, NavierStokesSolver
from dtcc_sim._flow.parameters import KAPPA

CASES = {
    "constant_noslip": dict(turbulence_model="constant", nu_t=2., wall_model="noslip"),
    "abl_noslip": dict(turbulence_model="abl", wall_model="noslip"),
    "abl_log_law": dict(turbulence_model="abl", wall_model="log_law"),
    "mixing_length_noslip": dict(turbulence_model="mixing_length", wall_model="noslip"),
    "mixing_length_log_law": dict(turbulence_model="mixing_length", wall_model="log_law"),
}


def flat_box(length, width, height, size):
    """Flat empty domain with canonical DTCC bounding-box markers."""
    cells = [max(1, round(extent / size)) for extent in (length, width, height)]
    mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [length, width, height]], cells)
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = dm.exterior_facet_indices(mesh.topology)
    mid = dm.compute_midpoints(mesh, 2, faces)
    values = np.zeros(len(faces), dtype=np.int32)
    for axis, coordinate, marker in [(2, 0., -1), (2, height, -2), (0, 0., -3), (0, length, -4), (1, 0., -5), (1, width, -6)]:
        values[np.isclose(mid[:, axis], coordinate)] = marker
    order = np.argsort(faces)
    return mesh, dm.meshtags(mesh, 2, faces[order], values[order])


def sample(mesh, function, points):
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(mesh, candidates, points)
    cells = np.array([colliding.links(i)[0] for i in range(len(points))], dtype=np.int32)
    return function.eval(points, cells)


def run(case, length, width, height, size, wind_speed, z0, wall_function_height):
    mesh, tags = flat_box(length, width, height, size)
    params = NavierStokesParameters(
        wind_speed=wind_speed, wind_dir_deg=270., inlet_profile="log_law", z0=z0,
        side_top_boundary="slip", wall_function_height=wall_function_height, **CASES[case],
    )
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=params)
    start = time.perf_counter()
    u, _ = solver.simulate()
    elapsed = time.perf_counter() - start
    heights = np.array([z for z in [1., 2., 5., 10., 20., 40., 60., 75.] if z < height])
    zw = params.wall_offset
    exact = params.friction_velocity / KAPPA * np.log((heights + zw) / z0)
    profiles = {}
    for name, x in [("inlet", 1e-6 * length), ("mid", .5 * length), ("outlet", length * (1 - 1e-6))]:
        points = np.column_stack([np.full_like(heights, x), np.full_like(heights, .5 * width), heights])
        profiles[name] = sample(mesh, u, points)[:, 0].tolist()
    outlet = np.array(profiles["outlet"])
    inlet = np.array(profiles["inlet"])
    return {
        "case": case, "parameters": CASES[case], "elapsed_s": elapsed,
        "heights_m": heights.tolist(), "equilibrium_speed": exact.tolist(), "profiles": profiles,
        "max_outlet_change": float(np.max(np.abs(outlet - inlet) / exact)),
        "max_outlet_error_above_first_cell": float(np.max((np.abs(outlet - exact) / exact)[heights >= size])),
        "friction_velocity": params.friction_velocity,
        "nonlinear_iterations": solver.diagnostics["nonlinear_iterations"],
        "turbulence": solver.diagnostics.get("turbulence"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--length", type=float, default=400.)
    parser.add_argument("--width", type=float, default=20.)
    parser.add_argument("--height", type=float, default=80.)
    parser.add_argument("--size", type=float, default=10.)
    parser.add_argument("--wind-speed", type=float, default=5.)
    parser.add_argument("--z0", type=float, default=.5)
    parser.add_argument("--wall-function-height", type=float, default=2.)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--output")
    args = parser.parse_args()
    results = []
    for case in args.cases:
        result = run(case, args.length, args.width, args.height, args.size,
                     args.wind_speed, args.z0, args.wall_function_height)
        results.append(result)
        if MPI.COMM_WORLD.rank == 0:
            print(f"{case:24s} outlet change {100 * result['max_outlet_change']:6.2f}%  "
                  f"error above first cell {100 * result['max_outlet_error_above_first_cell']:6.2f}%  {result['elapsed_s']:.1f} s")
    if args.output and MPI.COMM_WORLD.rank == 0:
        with open(args.output, "w") as stream:
            json.dump({"arguments": vars(args), "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
