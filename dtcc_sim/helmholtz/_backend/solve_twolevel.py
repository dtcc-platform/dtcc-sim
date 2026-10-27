from __future__ import annotations

from pathlib import Path
import numpy as np
from petsc4py import PETSc
from scipy import sparse as sps

from .params import PARAMS
from .mesh import build_system
from .preconditioner import (
    build_two_level_preconditioner,
    apply_hybrid_preconditioner,
)
from .coarse_space import deserialize_E0


def tprint(msg: str) -> None:
    if PARAMS.verbose:
        print(msg, flush=True)


def _ensure_parent(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _load_npz_payload(path_str: str) -> dict:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Offline preconditioner data not found: {path}. "
            f"Call simulator.prepare() first."
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _extract_prefixed(payload: dict, prefix: str) -> dict:
    out = {}
    needle = f"{prefix}__"
    for k, v in payload.items():
        if k.startswith(needle):
            out[k[len(needle):]] = v
    return out


def _rebuild_pou_data(payload: dict):
    class _POU:
        pass

    pou = _POU()
    pou.chi_full = [np.asarray(x, dtype=np.float64) for x in payload["chi_full"].tolist()]
    pou.multiplicity = np.asarray(payload["multiplicity"], dtype=np.int32)
    pou.ndofs_in_union = int(np.asarray(payload["ndofs_in_union"]).reshape(-1)[0])
    return pou


def _petsc_vec_to_numpy(x: PETSc.Vec) -> np.ndarray:
    return np.asarray(x.array, dtype=np.complex128).copy()


def _numpy_to_petsc_vec(x: np.ndarray) -> PETSc.Vec:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    v = PETSc.Vec().createSeq(len(x))
    v.setUp()
    v.array[:] = x
    return v


class HybridPreconditionerPC:
    def __init__(self, precond, A_global: PETSc.Mat):
        self.precond = precond
        self.A_global = A_global
        self.count = 0

    def apply(self, x: np.ndarray) -> np.ndarray:
        self.count += 1
        if PARAMS.verbose and self.count <= 5:
            tprint(f"[PC] apply {self.count}")
        return apply_hybrid_preconditioner(
            precond=self.precond,
            A_global=self.A_global,
            x=x,
        )


def _build_linear_operator(A_global: PETSc.Mat):
    n, m = A_global.getSize()
    if n != m:
        raise RuntimeError(f"Global matrix must be square, got {n}x{m}")

    def _matvec(x: np.ndarray) -> np.ndarray:
        x_p = _numpy_to_petsc_vec(x)
        y_p = PETSc.Vec().createSeq(n)
        y_p.setUp()
        A_global.mult(x_p, y_p)
        return _petsc_vec_to_numpy(y_p)

    return _matvec


def _relative_residual(A_global: PETSc.Mat, x: np.ndarray, b: np.ndarray) -> float:
    x_p = _numpy_to_petsc_vec(x)
    Ax_p = PETSc.Vec().createSeq(len(b))
    Ax_p.setUp()
    A_global.mult(x_p, Ax_p)
    r = np.asarray(b, dtype=np.complex128) - _petsc_vec_to_numpy(Ax_p)
    denom = np.linalg.norm(np.asarray(b, dtype=np.complex128))
    if denom == 0.0:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r) / denom)


def _write_gmres_history(history: list[float]) -> None:
    if not PARAMS.write_gmres_history:
        return
    _ensure_parent(PARAMS.gmres_history_path)
    with open(PARAMS.gmres_history_path, "w") as f:
        for r in history:
            f.write(f"{r:.16e}\n")


def _write_solution(mesh, V, x: np.ndarray) -> None:
    if not PARAMS.write_solution:
        return

    _ensure_parent(PARAMS.solution_npy_path)
    np.save(PARAMS.solution_npy_path, np.asarray(x, dtype=np.complex128))

    try:
        from dolfinx import io, fem
        uh = fem.Function(V, name="u_h")
        uh.x.array[:] = np.asarray(x, dtype=uh.x.array.dtype)

        _ensure_parent(PARAMS.solution_xdmf_path)
        with io.XDMFFile(mesh.comm, str(Path(PARAMS.solution_xdmf_path)), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(uh)
    except Exception as e:
        tprint(f"[solve] WARNING: failed to write XDMF solution: {type(e).__name__}: {e}")


def solve() -> dict:
    tprint("=" * 72)
    tprint("[solve] 3D heterogeneous two-level Helmholtz solve")
    tprint("=" * 72)

    payload = _load_npz_payload(PARAMS.preconditioner_npz_path)

    N_saved = int(np.asarray(payload["N_global"]).reshape(-1)[0])

    decomp_payload = _extract_prefixed(payload, "decomp")
    pou_payload = _extract_prefixed(payload, "pou")
    local_ops_payload = _extract_prefixed(payload, "local_ops")
    coarse_payload = _extract_prefixed(payload, "coarse")

    pou_data = _rebuild_pou_data(pou_payload)
    E0 = deserialize_E0(coarse_payload)

    if PARAMS.one_level_only:
        tprint("[solve] one_level_only=True -> disabling coarse level")
        E0 = np.zeros((N_saved, 0), dtype=np.complex128)

    mesh, facet_tags, cell_tags, V, Acoef, Vcoef, beta, A_global, b_global = build_system()
    N_now = int(V.dofmap.index_map.size_local)

    if N_saved != N_now:
        raise RuntimeError(
            f"Offline/online mismatch: saved N={N_saved}, current FE dofs N={N_now}"
        )

    tprint(f"[load_offline_data] N_saved={N_saved}, E0_loaded={E0.shape}")

    precond = build_two_level_preconditioner(
        A_global=A_global,
        decomposition=decomp_payload,
        pou_data=pou_data,
        E0=E0,
        local_patch_ops_payload=local_ops_payload,
    )
    pc = HybridPreconditionerPC(precond=precond, A_global=A_global)

    n = N_now
    A_mv = _build_linear_operator(A_global)

    def Amatvec(x: np.ndarray) -> np.ndarray:
        y = A_mv(np.asarray(x, dtype=np.complex128))
        return pc.apply(y)

    A_lin = sps.linalg.LinearOperator((n, n), matvec=Amatvec, dtype=np.complex128)

    b = _petsc_vec_to_numpy(b_global)
    b_norm = float(np.linalg.norm(b))
    b_max = float(np.max(np.abs(b))) if b.size else 0.0

    b_prec = pc.apply(b)
    b_prec_norm = float(np.linalg.norm(b_prec))
    b_prec_max = float(np.max(np.abs(b_prec))) if b_prec.size else 0.0

    history: list[float] = []
    x0 = np.zeros(n, dtype=np.complex128)

    tprint(f"[solve] N               = {n}")
    tprint(f"[solve] k               = {float(PARAMS.k()):.6e}")
    tprint(f"[solve] rhs mode        = {PARAMS.rhs_mode}")
    tprint(f"[solve] outer solver    = {PARAMS.outer_solver}")
    tprint(f"[solve] local solver    = {PARAMS.local_solver}")
    tprint(f"[solve] coarse solver   = {PARAMS.coarse_solver}")
    tprint(f"[solve] precond data    = {PARAMS.preconditioner_npz_path}")
    tprint(f"[solve] ||b||_2         = {b_norm:.6e}")
    tprint(f"[solve] max|b|          = {b_max:.6e}")
    tprint(f"[solve] ||M^-1 b||_2    = {b_prec_norm:.6e}")
    tprint(f"[solve] max|M^-1 b|     = {b_prec_max:.6e}")

    if b_norm == 0.0:
        tprint("[solve] WARNING: assembled RHS is exactly zero")
    if b_prec_norm == 0.0:
        tprint("[solve] WARNING: preconditioned RHS is exactly zero")

    if PARAMS.outer_solver == "gmres":
        x, info = sps.linalg.gmres(
            A_lin,
            b_prec,
            x0=x0,
            restart=int(PARAMS.outer_restart),
            maxiter=int(PARAMS.outer_max_it),
            rtol=float(PARAMS.outer_rtol),
            atol=float(PARAMS.outer_atol),
            callback=lambda rk: history.append(float(rk)),
            callback_type="pr_norm",
        )
    elif PARAMS.outer_solver == "fgmres":
        tprint("[solve] WARNING: outer_solver='fgmres' requested, using scipy gmres fallback")
        x, info = sps.linalg.gmres(
            A_lin,
            b_prec,
            x0=x0,
            restart=int(PARAMS.outer_restart),
            maxiter=int(PARAMS.outer_max_it),
            rtol=float(PARAMS.outer_rtol),
            atol=float(PARAMS.outer_atol),
            callback=lambda rk: history.append(float(rk)),
            callback_type="pr_norm",
        )
    else:
        raise ValueError(f"Unsupported outer_solver={PARAMS.outer_solver!r}")

    relres = _relative_residual(A_global, x, b)
    reason = 2 if info == 0 else -1
    its = len(history) if len(history) > 0 else (0 if info == 0 else int(PARAMS.outer_max_it))

    tprint("=" * 72)
    tprint("[solve] finished")
    tprint(f"[solve] reason      = {reason}")
    tprint(f"[solve] iterations  = {its}")
    tprint(f"[solve] residual    = {relres:.6e}")
    tprint("=" * 72)

    _write_gmres_history(history)
    _write_solution(mesh, V, x)

    if not PARAMS.write_solution:
        tprint("[solve] skipping XDMF/HDF5 output")

    return {"solution": x, "iterations": its, "relative_residual": relres,
            "converged": reason == 2, "gmres_history": tuple(history)}
