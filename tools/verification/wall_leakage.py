"""Weak-wall leakage under mesh refinement and penalty strength.

Weak (Nitsche) no-penetration on terrain and non-axis-aligned walls lets some
flow cross solid surfaces. The script solves stationary Navier–Stokes over
undulating terrain with log-law walls and reports the absolute ground flux as
a fraction of the inflow, for several cell sizes and penalty factors. A city
mesh can be added with ``--city-mesh``.

    python tools/verification/wall_leakage.py --sizes 20 10 5 --penalties 100 1000 \\
        --output leakage.json
"""

import argparse
import json
import time

import numpy as np
import dolfinx.mesh as dm
from mpi4py import MPI

from dtcc_sim import NavierStokesParameters, NavierStokesSolver


def hilly_box(length, width, height, size, amplitude, wavelength):
    """Terrain-following box over z = A sin(2πx/λ) sin(2πy/λ) with DTCC bbox markers."""
    cells = [max(1, round(extent / size)) for extent in (length, width, height)]
    mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [length, width, height]], cells)
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = dm.exterior_facet_indices(mesh.topology)
    mid = dm.compute_midpoints(mesh, 2, faces)
    values = np.zeros(len(faces), dtype=np.int32)
    for axis, coordinate, marker in [(2, 0., -1), (2, height, -2), (0, 0., -3), (0, length, -4), (1, 0., -5), (1, width, -6)]:
        values[np.isclose(mid[:, axis], coordinate)] = marker
    x = mesh.geometry.x
    ground = amplitude * np.sin(2 * np.pi * x[:, 0] / wavelength) * np.sin(2 * np.pi * x[:, 1] / wavelength)
    x[:, 2] += ground * (1 - x[:, 2] / height)
    order = np.argsort(faces)
    return mesh, dm.meshtags(mesh, 2, faces[order], values[order])


def run(params, **mesh_arguments):
    solver = NavierStokesSolver(params=params, **mesh_arguments)
    start = time.perf_counter()
    solver.simulate()
    convergence = solver.diagnostics["convergence"]
    keys = ["inflow_flux", "ground_leakage_fraction", "building_leakage_fraction",
            "ground_normal_velocity_rms", "building_normal_velocity_rms", "flux_imbalance"]
    return {"elapsed_s": time.perf_counter() - start,
            "nonlinear_iterations": solver.diagnostics["nonlinear_iterations"],
            **{key: convergence[key] for key in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=float, nargs="+", default=[20., 10., 5.])
    parser.add_argument("--penalties", type=float, nargs="+", default=[100., 1000.])
    parser.add_argument("--length", type=float, default=200.)
    parser.add_argument("--width", type=float, default=40.)
    parser.add_argument("--height", type=float, default=80.)
    parser.add_argument("--amplitude", type=float, default=4.)
    parser.add_argument("--wavelength", type=float, default=80.)
    parser.add_argument("--turbulence-model", default="mixing_length")
    parser.add_argument("--city-mesh", help="Tagged city XDMF mesh solved with each penalty")
    parser.add_argument("--output")
    args = parser.parse_args()
    loose = dict(ground_leakage_tolerance=.99, building_leakage_tolerance=.99, flux_imbalance_tolerance=.99,
                 divergence_tolerance=10.)
    common = dict(wind_speed=5., wind_dir_deg=270., inlet_profile="log_law", side_top_boundary="slip",
                  turbulence_model=args.turbulence_model, wall_model="log_law", **loose)
    results = []
    for penalty in args.penalties:
        for size in args.sizes:
            mesh, tags = hilly_box(args.length, args.width, args.height, size, args.amplitude, args.wavelength)
            result = run(NavierStokesParameters(nitsche_penalty=penalty, **common), mesh=mesh, markers=tags)
            results.append({"case": "hills", "size_m": size, "nitsche_penalty": penalty, **result})
            if MPI.COMM_WORLD.rank == 0:
                print(f"hills h={size:5.1f} C={penalty:6.0f}: ground {100 * result['ground_leakage_fraction']:.3f} %, "
                      f"imbalance {result['flux_imbalance']:.2e}, {result['nonlinear_iterations']} Newton, "
                      f"{result['elapsed_s']:.0f} s", flush=True)
        if args.city_mesh:
            city = dict(common, grad_div_gamma=3., backflow_beta=2., nitsche_penalty=penalty)
            result = run(NavierStokesParameters(**city), mesh_path=args.city_mesh, mesh_crs="EPSG:3006")
            results.append({"case": "city", "mesh": args.city_mesh, "nitsche_penalty": penalty, **result})
            if MPI.COMM_WORLD.rank == 0:
                print(f"city C={penalty:6.0f}: ground {100 * result['ground_leakage_fraction']:.3f} %, "
                      f"buildings {100 * result['building_leakage_fraction']:.3f} %, "
                      f"imbalance {result['flux_imbalance']:.2e}, {result['nonlinear_iterations']} Newton, "
                      f"{result['elapsed_s']:.0f} s", flush=True)
    if args.output and MPI.COMM_WORLD.rank == 0:
        with open(args.output, "w") as stream:
            json.dump({"arguments": vars(args), "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
