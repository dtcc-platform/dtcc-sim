"""Solve an AIJ benchmark case and compare it with the wind-tunnel data.

    python -m benchmarks.aij.run --case A --resolution coarse

The approach flow is the log law fitted to the measured inflow profile up to
the building height. The log-law wall places the computational wall
``wall_function_height`` (δ) outside every solid surface, so the mesh building
is the measured building grown by δ on its walls, and a probe at physical
height Z is sampled at mesh height Z - δ. Velocities are normalized by the
approach-flow speed at the building height: the fitted log law for the
simulation and the measured profile for the experiment.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .analysis import fit_log_law, log_law, metrics, sample
from .cases import CASES
from .data import read_workbook
from .geometry import Spacing, block_mesh

RESOLUTIONS = {"coarse": 4, "medium": 6, "fine": 8}
"""Cells per building height next to the building."""

GRAD_DIV = .05
"""Grad-div stabilization γ relative to U_H h, the ratio of the city wind demos."""

LOW_MEMORY_PETSC = {"fieldsplit_u_pc_type": "bjacobi", "fieldsplit_u_sub_pc_factor_levels": 0,
                    "ksp_gmres_restart": 60, "ksp_max_it": 3000}
"""Velocity block Jacobi with ILU(0) and no overlap copies: about two thirds of
the default memory, with more linear iterations."""


def mesh_point(points, delta):
    """Mesh coordinates of physical points; heights below δ become NaN."""
    mapped = np.array(points, dtype=float)
    mapped[:, 2] -= delta
    mapped[mapped[:, 2] < 0, 2] = np.nan
    return mapped


def run(case_name, resolution="coarse", turbulence_model="mixing_length", wall_function_height=2e-3,
        ratio=1.2, output=Path("output/benchmarks/aij"), low_memory=False):
    from mpi4py import MPI

    from dtcc_sim import NavierStokesParameters, NavierStokesSolver

    comm = MPI.COMM_WORLD
    case = CASES[case_name]
    if not case.path.exists():
        raise FileNotFoundError(
            f"{case.path} not found; download {case.workbook} from "
            "https://www.aij.or.jp/jpn/publish/cfdguide/index_e.htm (see benchmarks/data/aij/README.md)")
    data = read_workbook(case.path, case.length)
    H, delta = case.height, wall_function_height
    friction_velocity, z0, fit_rms = fit_log_law(data.inflow_z, data.inflow_speed, H)
    observed_reference = float(np.interp(H, data.inflow_z, data.inflow_speed))
    simulated_reference = float(log_law(H, friction_velocity, z0))

    h = H / RESOLUTIONS[resolution]
    spacing = Spacing(h=h, ground=h / 3, ratio=ratio, maximum=H / 2)
    domain = case.domain()
    probes = np.concatenate([s.points for s in data.sections.values()])
    region = ((probes[:, 0].min(), probes[:, 0].max()), (probes[:, 1].min(), probes[:, 1].max()))
    started = time.perf_counter()
    mesh, markers, axes = block_mesh([case.building(delta)], domain, spacing, region)
    mesh_seconds = time.perf_counter() - started

    params = NavierStokesParameters(
        wind_speed=float(log_law(H + delta, friction_velocity, z0)), u_ref_height=H, z0=z0,
        wind_dir_deg=270., inlet_profile="log_law", wall_model="log_law",
        wall_function_height=delta, turbulence_model=turbulence_model, side_top_boundary="slip",
        grad_div_gamma=GRAD_DIV * simulated_reference * h, backflow_beta=2.,
        petsc_options=LOW_MEMORY_PETSC if low_memory else None,
    )
    directory = output / f"case-{case_name}" / f"{turbulence_model}-{resolution}"
    solver = NavierStokesSolver(mesh=mesh, markers=markers, params=params, output_dir=str(directory))
    started = time.perf_counter()
    u, _ = solver.simulate()
    solve_seconds = time.perf_counter() - started

    sections = {}
    for name, section in data.sections.items():
        # probes below the computational ground or inside the grown building are outside the mesh
        predicted = sample(u, np.nan_to_num(mesh_point(section.points, delta), nan=-1.))
        observed = section.velocity / observed_reference
        predicted = predicted / simulated_reference
        speed_observed = np.linalg.norm(observed, axis=1)
        speed_predicted = np.linalg.norm(predicted, axis=1)
        sections[name] = {
            "number": section.number.tolist(),
            "points_m": section.points.tolist(),
            "observed": observed.tolist(),
            "predicted": predicted.tolist(),
            "metrics": {
                "speed": metrics(speed_predicted, speed_observed),
                "u": metrics(predicted[:, 0], observed[:, 0]),
            },
        }
    every = {key: np.concatenate([np.asarray(s[key]) for s in sections.values()]) for key in ("observed", "predicted")}
    convergence = solver.diagnostics.get("convergence", {})
    result = {
        "case": case_name, "title": case.title, "reference": case.reference,
        "resolution": resolution, "turbulence_model": turbulence_model,
        "building_m": {"width": case.width, "depth": case.depth, "height": H},
        "domain_m": domain, "wall_function_height_m": delta,
        "inflow": {
            "friction_velocity": friction_velocity, "z0_m": z0, "fit_rms_m_per_s": fit_rms,
            "observed_reference_speed": observed_reference, "simulated_reference_speed": simulated_reference,
            "observed": {"z_m": data.inflow_z.tolist(), "speed": data.inflow_speed.tolist(), "k": data.inflow_k.tolist()},
            "equilibrium_k": friction_velocity**2 / np.sqrt(.09),
        },
        "mesh": {
            "cells": mesh.topology.index_map(3).size_global, "grid": [len(a) - 1 for a in axes],
            "h_building_m": h, "h_ground_m": spacing.ground, "ratio": ratio, "h_max_m": spacing.maximum,
            "seconds": mesh_seconds,
        },
        "solver": {
            "seconds": solve_seconds, "ranks": comm.size, "low_memory": low_memory,
            "nonlinear_iterations": solver.diagnostics.get("nonlinear_iterations"),
            "convergence": {k: v for k, v in convergence.items() if isinstance(v, (int, float, bool))},
        },
        "sections": sections,
        "metrics": {
            "speed": metrics(np.linalg.norm(every["predicted"], axis=1), np.linalg.norm(every["observed"], axis=1)),
            "u": metrics(every["predicted"][:, 0], every["observed"][:, 0]),
        },
    }
    if comm.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=sorted(CASES), default="A")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="coarse")
    parser.add_argument("--turbulence-model", choices=["mixing_length", "abl"], default="mixing_length")
    parser.add_argument("--wall-function-height", type=float, default=2e-3, help="δ [m]")
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/aij"))
    parser.add_argument("--low-memory", action="store_true", help="Use the lower-memory linear solver settings")
    args = parser.parse_args()
    result = run(args.case, args.resolution, args.turbulence_model, args.wall_function_height,
                 output=args.output, low_memory=args.low_memory)
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        print(f"case {args.case} {args.turbulence_model} {args.resolution}: {result['mesh']['cells']} cells, "
              f"{result['solver']['nonlinear_iterations']} Newton, {result['solver']['seconds']:.0f} s")
        for name, section in result["sections"].items():
            m = section["metrics"]["speed"]
            print(f"  {name:22s} speed hit rate {m['hit_rate']:.2f} FAC2 {m['fac2']:.2f} "
                  f"FB {m['fractional_bias']:+.2f} R {m['correlation']:.2f} ({m['points']} points)")
        m = result["metrics"]["speed"]
        print(f"  {'all':22s} speed hit rate {m['hit_rate']:.2f} FAC2 {m['fac2']:.2f} "
              f"FB {m['fractional_bias']:+.2f} R {m['correlation']:.2f} ({m['points']} points)")


if __name__ == "__main__":
    main()
