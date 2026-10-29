"""Benchmark stationary flow linear solvers on identical systems and fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import ufl
import dolfinx
from dolfinx import fem, mesh as dm
from dolfinx.fem import petsc as fp
from mpi4py import MPI
from petsc4py import PETSc

from dtcc_core import BoxMesh, io
from dtcc_sim import NavierStokesParameters, NavierStokesSolver, StokesParameters, StokesSolver
from dtcc_sim._flow import solver as flow
from dtcc_sim.fenics import _mesh_from_dtcc


BENCHMARK_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOLVER_SHA256 = hashlib.sha256((ROOT / "dtcc_sim/_flow/solver.py").read_bytes()).hexdigest()


BLOCK_BASE = {
    "ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_atol": 1e-13,
    "ksp_max_it": 500, "ksp_gmres_restart": 100,
    "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower", "pc_fieldsplit_schur_precondition": "a11",
    "fieldsplit_u_ksp_type": "preonly", "fieldsplit_u_pc_type": "hypre",
    "fieldsplit_u_pc_hypre_type": "boomeramg",
    "fieldsplit_p_ksp_type": "preonly", "fieldsplit_p_pc_type": "jacobi",
}

LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
PROFILES = {
    "lu": LU,
    "default": {},
    "mass_ilu": {"fieldsplit_p_pc_type": "ilu"},
    "velocity_ilu0": {"fieldsplit_u_pc_type": "ilu"},
    "ilu0_restart200": {"fieldsplit_u_pc_type": "ilu", "ksp_gmres_restart": 200},
    "ilu0_full": {"fieldsplit_u_pc_type": "ilu", "pc_fieldsplit_schur_fact_type": "full"},
    "ilu0_upper": {"fieldsplit_u_pc_type": "ilu", "pc_fieldsplit_schur_fact_type": "upper"},
    "ilu0_full200": {"fieldsplit_u_pc_type": "ilu", "pc_fieldsplit_schur_fact_type": "full",
                     "ksp_gmres_restart": 200},
    "ilu0_mass": {"fieldsplit_u_pc_type": "ilu", "fieldsplit_p_pc_type": "ilu"},
    "asm_ilu0": {"fieldsplit_u_pc_type": "asm", "fieldsplit_u_pc_asm_overlap": 1,
                 "fieldsplit_u_sub_ksp_type": "preonly", "fieldsplit_u_sub_pc_type": "ilu"},
    "ilu0_selfp": {"fieldsplit_u_pc_type": "ilu", "pc_fieldsplit_diag_use_amat": True,
                   "pc_fieldsplit_schur_precondition": "selfp", "fieldsplit_p_pc_type": "hypre",
                   "fieldsplit_p_pc_hypre_type": "boomeramg"},
    "ilu0_selfp_full": {"fieldsplit_u_pc_type": "ilu", "pc_fieldsplit_diag_use_amat": True, "pc_use_amat": True,
                        "pc_fieldsplit_schur_precondition": "selfp", "pc_fieldsplit_schur_fact_type": "full",
                        "fieldsplit_p_pc_type": "hypre", "fieldsplit_p_pc_hypre_type": "boomeramg"},
    "asm_selfp": {"fieldsplit_u_pc_type": "asm", "fieldsplit_u_pc_asm_overlap": 1,
                  "fieldsplit_u_sub_ksp_type": "preonly", "fieldsplit_u_sub_pc_type": "ilu",
                  "pc_fieldsplit_diag_use_amat": True, "pc_use_amat": True, "pc_fieldsplit_schur_precondition": "selfp",
                  "fieldsplit_p_pc_type": "hypre", "fieldsplit_p_pc_hypre_type": "boomeramg"},
    "gamg_vector_selfp": {"fieldsplit_u_pc_type": "gamg", "pc_fieldsplit_diag_use_amat": True, "pc_use_amat": True,
                          "pc_fieldsplit_schur_precondition": "selfp", "fieldsplit_p_pc_type": "hypre",
                          "fieldsplit_p_pc_hypre_type": "boomeramg"},
    "gamg_vector_selfp_full": {"fieldsplit_u_pc_type": "gamg", "pc_fieldsplit_diag_use_amat": True, "pc_use_amat": True,
                               "pc_fieldsplit_schur_precondition": "selfp", "pc_fieldsplit_schur_fact_type": "full",
                               "fieldsplit_p_pc_type": "hypre", "fieldsplit_p_pc_hypre_type": "boomeramg"},
    "gamg_stokes_selfp": {"fieldsplit_u_pc_type": "gamg", "pc_fieldsplit_schur_precondition": "selfp",
                          "fieldsplit_p_pc_type": "hypre", "fieldsplit_p_pc_hypre_type": "boomeramg"},
    "ilu0_lsc": {"fieldsplit_u_pc_type": "ilu", "pc_use_amat": True,
                 "pc_fieldsplit_schur_precondition": "self", "fieldsplit_p_pc_type": "lsc",
                 "fieldsplit_p_lsc_pc_type": "hypre", "fieldsplit_p_lsc_pc_hypre_type": "boomeramg"},
    "ilu0_lsc_scaled": {"fieldsplit_u_pc_type": "ilu", "pc_use_amat": True,
                        "pc_fieldsplit_schur_precondition": "self", "fieldsplit_p_pc_type": "lsc",
                        "fieldsplit_p_pc_lsc_scale_diag": True,
                        "fieldsplit_p_lsc_pc_type": "hypre", "fieldsplit_p_lsc_pc_hypre_type": "boomeramg"},
    "velocity_ilu1": {"fieldsplit_u_pc_type": "ilu", "fieldsplit_u_pc_factor_levels": 1},
    "velocity_ilu2": {"fieldsplit_u_pc_type": "ilu", "fieldsplit_u_pc_factor_levels": 2},
    "velocity_lu": {"fieldsplit_u_pc_type": "lu", "fieldsplit_u_pc_factor_mat_solver_type": "mumps",
                    "fieldsplit_p_pc_type": "ilu"},
    "both_ilu1": {"fieldsplit_u_pc_type": "ilu", "fieldsplit_u_pc_factor_levels": 1,
                  "fieldsplit_p_pc_type": "ilu"},
    "gamg": {"fieldsplit_u_pc_type": "gamg"},
    "gamg_vector": {"fieldsplit_u_pc_type": "gamg"},
    "amg_vector": {},
    "amg_nodal": {"fieldsplit_u_pc_hypre_boomeramg_nodal_coarsen": 1,
                  "fieldsplit_u_pc_hypre_boomeramg_vec_interp_variant": 3},
    "amg_vector_strong": {"fieldsplit_u_pc_hypre_boomeramg_strong_threshold": 0.7},
    "amg_two_cycles": {"fieldsplit_u_pc_hypre_boomeramg_max_iter": 2},
    "amg_strong": {"fieldsplit_u_pc_hypre_boomeramg_strong_threshold": 0.7},
    "amg_full": {"pc_fieldsplit_schur_fact_type": "full"},
    "amg_upper": {"pc_fieldsplit_schur_fact_type": "upper"},
    "amg_additive": {"pc_fieldsplit_type": "additive"},
    "amg_lsc": {"pc_fieldsplit_schur_precondition": "self", "pc_use_amat": True,
                "fieldsplit_p_pc_type": "lsc", "fieldsplit_p_lsc_pc_type": "hypre",
                "fieldsplit_p_lsc_pc_hypre_type": "boomeramg"},
    "monolithic_ilu1": {"ksp_type": "fgmres", "pc_type": "ilu",
                        "pc_factor_levels": 1, "pc_factor_shift_type": "nonzero"},
}

PROFILES["asm_selfp_gauge"] = {**PROFILES["asm_selfp"], "fieldsplit_p_pc_type": "jacobi"}

VECTOR_PROFILES = {"gamg_vector", "amg_vector", "amg_nodal", "amg_vector_strong",
                   "gamg_vector_selfp", "gamg_vector_selfp_full", "gamg_stokes_selfp"}


def vector_metadata(ksp):
    """Attach velocity component layout and translations to the extracted block."""
    ksp.getPC().setUp()
    velocity = ksp.getPC().getFieldSplitSubKSP()[0]
    A, P = velocity.getOperators()
    for matrix in (A, P):
        matrix.setBlockSize(3)
    vectors = [P.createVecRight() for _ in range(3)]
    start, end = vectors[0].getOwnershipRange()
    for component, vector in enumerate(vectors):
        vector.array[:] = (np.arange(start, end) % 3 == component)
        vector.normalize()
    null = PETSc.NullSpace().create(vectors=vectors, comm=P.comm)
    P.setNearNullSpace(null)
    velocity.getPC().reset()
    velocity.setOperators(A, P)
    null.destroy()
    for vector in vectors:
        vector.destroy()


def prepare(case, source, n):
    if case == "city":
        native = io.load_volume_mesh(source / "city.xdmf").repair_boundary_corners()
        mesh, tags = _mesh_from_dtcc(native)
        physical = dict(wind_speed=5., wind_dir_deg=270., nu_t=2., side_top_boundary="slip",
                        inlet_profile="log_law", grad_div_gamma=3., backflow_beta=2.)
        inlet = None
    elif case == "cavity":
        native = BoxMesh(n, n, n)
        native.boundary_markers[native.boundary_markers != -2] = -1
        mesh, tags = _mesh_from_dtcc(native)
        physical = dict(nu=.1, wind_speed=.1, inlet_marker=-2, closed_cavity=True,
                        grad_div_gamma=1., side_top_boundary="open")
        def inlet(x):
            return np.array([1.6*x[0]*(1-x[0])*x[1]*(1-x[1]),
                             np.zeros(x.shape[1]), np.zeros(x.shape[1])])
    else:
        mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [2., 1., 1.]], [2*n, n, n])
        mesh.topology.create_entities(2)
        mesh.topology.create_connectivity(2, 3)
        faces = np.sort(dm.exterior_facet_indices(mesh.topology))
        mid = dm.compute_midpoints(mesh, 2, faces)
        values = np.full(len(faces), -1, dtype=np.int32)
        for axis, coordinate, marker in [(0, 0, -3), (0, 2, -4), (1, 0, -5), (1, 1, -6)]:
            values[np.isclose(mid[:, axis], coordinate)] = marker
        tags = dm.meshtags(mesh, 2, faces, values)
        physical = dict(wind_speed=1., nu=.1, side_top_boundary="slip", divergence_tolerance=.005)
        def inlet(x):
            return np.array([4*x[2]*(1-x[2]), np.zeros(x.shape[1]), np.zeros(x.shape[1])])
    return mesh, tags, physical, inlet


def write_petsc(obj, path):
    viewer = PETSc.Viewer().createBinary(str(path), "w", comm=MPI.COMM_WORLD)
    obj.view(viewer)
    viewer.destroy()


def read_petsc(kind, path):
    viewer = PETSc.Viewer().createBinary(str(path), "r", comm=MPI.COMM_WORLD)
    obj = kind().load(viewer)
    viewer.destroy()
    return obj


def profile_options(name, adaptive=False, ew_cap=None):
    overrides = PROFILES[name]
    options = {} if overrides.get("pc_type", "fieldsplit") != "fieldsplit" else dict(BLOCK_BASE)
    options.update(overrides)
    if adaptive:
        options.update(snes_ksp_ew=True, snes_ksp_ew_version=2)
        if ew_cap is not None:
            options.update(snes_ksp_ew_rtol0=ew_cap, snes_ksp_ew_rtolmax=ew_cap)
    return options


def full(args):
    mesh, tags, physical, inlet = prepare(args.case, args.source, args.n)
    if args.equations == "stokes":
        physical.pop("backflow_beta", None)
    options = profile_options(args.profile, args.adaptive, args.ew_cap)
    options.setdefault("snes_ksp_ew", False)
    if args.case == "cavity" and args.profile == "lu":
        options.update(mat_mumps_icntl_24=1, mat_mumps_icntl_25=0)
    solver_type, parameter_type, key = (
        (NavierStokesSolver, NavierStokesParameters, "petsc_options")
        if args.equations == "navier_stokes" else (StokesSolver, StokesParameters, "petsc_stokes")
    )
    records = []
    original = flow._MixedFlowSystem

    class MeasuredSystem(original):
        def configure(self, opts):
            if args.profile == "gamg_stokes_selfp":
                self.P = ufl.replace(self.J, {self.convection: ufl.as_ufl(0.)})
            if opts.get("pc_use_amat"):
                self.P = None
            super().configure(opts)
            self.linear_records = []
            if args.adaptive:
                self.problem.solver.getKSP().setMonitor(self.adaptive_linear_monitor)
                self.problem.solver.setMonitor(self.completed_newton)
            else:
                self.problem.solver.getKSP().setPreSolve(self.before_linear)
                self.problem.solver.getKSP().setPostSolve(self.after_linear)

        def solve(self, convection):
            if args.adaptive:
                self.problem.solver.setUseEW(bool(convection))
            return super().solve(convection)

        def adaptive_linear_monitor(self, ksp, iteration, norm):
            if iteration == 0:
                self.linear_started = time.perf_counter()
                if args.profile in VECTOR_PROFILES:
                    vector_metadata(ksp)
                self.linear_records.append(dict(rtol=ksp.getTolerances()[0]))
            self.linear_records[-1].update(
                iterations=iteration, reported_residual=float(norm),
                wall_s=time.perf_counter()-self.linear_started,
            )

        def completed_newton(self, snes, iteration, norm):
            if iteration and self.linear_records:
                self.linear_records[-1]["reason"] = int(snes.getKSP().getConvergedReason())

        def before_linear(self, ksp, b, x):
            self.linear_started = time.perf_counter()
            if args.profile in VECTOR_PROFILES:
                vector_metadata(ksp)
            if args.mode == "capture":
                index = len(self.linear_records)
                directory = args.output / f"matrix-{index}"
                directory.mkdir(exist_ok=True)
                A = ksp.getOperators()[0]
                P = fp.assemble_matrix(fem.form(self.P), bcs=self.bcs)
                P.assemble()
                write_petsc(A, directory / "A.dat")
                write_petsc(P, directory / "P.dat")
                write_petsc(b, directory / "b.dat")
                for i, split in enumerate(self.splits):
                    write_petsc(split, directory / f"is-{i}.dat")
                P.destroy()

        def after_linear(self, ksp, b, x):
            residual = b.duplicate()
            ksp.getOperators()[0].mult(x, residual)
            residual.axpy(-1, b)
            self.linear_records.append(dict(
                wall_s=time.perf_counter()-self.linear_started, iterations=ksp.getIterationNumber(),
                reason=int(ksp.getConvergedReason()), true_relative_residual=residual.norm()/max(b.norm(), 1e-30),
            ))
            residual.destroy()

        def close(self):
            records[-1]["linear"] = list(getattr(self, "linear_records", []))
            if self.problem is not None and not args.adaptive:
                self.problem.solver.getKSP().setPreSolve(None)
                self.problem.solver.getKSP().setPostSolve(None)
            super().close()

    flow._MixedFlowSystem = MeasuredSystem
    try:
        for index in range(args.runs):
            record = {"run": index, "warm": index > 0}
            records.append(record)
            params = parameter_type(**physical, **{key: options})
            solver = solver_type(mesh=mesh, markers=tags, params=params, inlet_expression=inlet)
            started = time.perf_counter()
            try:
                u, p = solver.simulate()
                record["accepted"] = True
                fields = {"u": u.x.array.copy(), "p": p.x.array.copy()}
                if args.reference:
                    reference = np.load(args.reference)
                    record["relative_difference"] = {
                        name: float(np.linalg.norm(values-reference[name])/max(np.linalg.norm(reference[name]), 1e-30))
                        for name, values in fields.items()
                    }
                    record["reference_pass"] = max(record["relative_difference"].values()) <= 1e-7
                np.savez(args.output / f"fields-{index}.npz", **fields)
            except Exception as exc:
                record.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
            record.update(wall_s=time.perf_counter()-started, diagnostics=solver.diagnostics,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
            save(args, records)
            print(json.dumps(record), flush=True)
            if not record["accepted"]:
                break
    finally:
        flow._MixedFlowSystem = original


def matrix(args):
    directory = args.matrix
    A, P = [read_petsc(PETSc.Mat, directory / name) for name in ("A.dat", "P.dat")]
    b = read_petsc(PETSc.Vec, directory / "b.dat")
    splits = [read_petsc(PETSc.IS, directory / f"is-{i}.dat") for i in range(2)]
    records = []
    for index in range(args.runs):
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        opts = {"ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_atol": 1e-13,
                "ksp_max_it": 500, "ksp_gmres_restart": 100, **profile_options(args.profile)}
        prefix = f"benchmark_{index}_"
        ksp.setOptionsPrefix(prefix)
        database = PETSc.Options(prefix)
        for name, value in opts.items():
            database[name] = value
        split_pc = opts.get("pc_type") == "fieldsplit"
        ksp.setOperators(A, P if split_pc else A)
        ksp.setFromOptions()
        ksp.setMonitor(lambda ksp, iteration, norm: print(
            f"iteration={iteration} residual={norm:.6e}", flush=True,
        ) if iteration % 25 == 0 else None)
        if split_pc:
            ksp.getPC().setFieldSplitIS(("u", splits[0]), ("p", splits[1]))
        x = A.createVecRight()
        x.set(0)
        started = time.perf_counter()
        record = {"run": index, "warm": index > 0}
        try:
            ksp.setUp()
            if args.profile in VECTOR_PROFILES:
                vector_metadata(ksp)
            setup = time.perf_counter()-started
            ksp.solve(b, x)
            r = b.duplicate()
            A.mult(x, r)
            r.axpy(-1, b)
            residual = r.norm()/max(b.norm(), 1e-30)
            r.destroy()
            record.update(setup_s=setup, iterations=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()),
                          true_relative_residual=residual, accepted=ksp.getConvergedReason() > 0 and residual <= 1e-8)
        except Exception as exc:
            record.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
        record["wall_s"] = time.perf_counter()-started
        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        records.append(record)
        save(args, records)
        print(json.dumps(record), flush=True)
        ksp.destroy()
        x.destroy()
        for name in opts:
            del database[name]
    for obj in [A, P, b, *splits]:
        obj.destroy()


def save(args, records):
    manifest = dict(
        profile=args.profile, mode=args.mode, case=args.case, equations=args.equations,
        adaptive=args.adaptive, ew_cap=args.ew_cap,
        options=profile_options(args.profile, args.adaptive, args.ew_cap), records=records,
        dolfinx=dolfinx.__version__, petsc=PETSc.Sys.getVersion(), hardware=platform.uname()._asdict(),
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        source_sha256=SOLVER_SHA256,
        benchmark_sha256=BENCHMARK_SHA256,
    )
    (args.output / "measurements.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["capture", "matrix", "full"], default="full")
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument("--case", choices=["city", "channel", "cavity"], default="city")
    parser.add_argument("--equations", choices=["navier_stokes", "stokes"], default="navier_stokes")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--source", type=Path, default=ROOT / ".agent/work/stationary")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--ew-cap", type=float)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error("This timing harness requires one MPI rank")
    if args.mode == "capture" and args.runs != 1:
        parser.error("Capture exactly one reference run")
    if args.mode == "matrix" and args.matrix is None:
        parser.error("Matrix mode requires --matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    (matrix if args.mode == "matrix" else full)(args)
    records = json.loads((args.output / "measurements.json").read_text())["records"]
    return 0 if all(record.get("accepted") and record.get("reference_pass", True) for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
