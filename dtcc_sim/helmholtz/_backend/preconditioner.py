from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from petsc4py import PETSc
from scipy import sparse as sps
from scipy.sparse import linalg as spsla

from .params import PARAMS


def _csr_to_scipy(indptr, indices, data, shape):
    return sps.csr_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=shape,
    )


def _mat_to_scipy_csr(A: PETSc.Mat) -> sps.csr_matrix:
    A.assemble()
    indptr, indices, data = A.getValuesCSR()
    m, n = A.getSize()
    return _csr_to_scipy(indptr, indices, data, (m, n))


@dataclass
class LocalPatchSolver:
    patch_index: int
    dofs_olap: np.ndarray
    dofs_star: np.ndarray
    chi_olap: np.ndarray
    solve_mode: str
    A_star_solver: object


@dataclass
class TwoLevelPreconditioner:
    n_global: int
    local_patch_solvers: list[LocalPatchSolver]
    E0: np.ndarray
    coarse_solver: object

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128).reshape(-1)
        if x.size != self.n_global:
            raise ValueError(
                f"Preconditioner input has size {x.size}, expected {self.n_global}"
            )

        y = np.zeros_like(x)

        for patch in self.local_patch_solvers:
            rhs_star = x[patch.dofs_star]

            if patch.solve_mode == "lu":
                z_star = patch.A_star_solver.solve(rhs_star)
            elif patch.solve_mode == "gmres_ilu":
                z_star, info = spsla.gmres(
                    patch.A_star_solver["A"],
                    rhs_star,
                    M=patch.A_star_solver["M"],
                    restart=50,
                    rtol=1.0e-8,
                    atol=0.0,
                    maxiter=200,
                )
                if info != 0:
                    raise RuntimeError(
                        f"Local GMRES-ILU failed on patch {patch.patch_index} with info={info}"
                    )
            else:
                raise ValueError(f"Unsupported local solve_mode={patch.solve_mode!r}")

            star_g2l = {int(g): j for j, g in enumerate(patch.dofs_star.tolist())}
            local_vals = np.zeros(len(patch.dofs_olap), dtype=np.complex128)
            for j, g in enumerate(patch.dofs_olap.tolist()):
                local_vals[j] = z_star[star_g2l[int(g)]]

            y[patch.dofs_olap] += patch.chi_olap * local_vals

        return y

    def apply_coarse(self, x: np.ndarray) -> np.ndarray:
        if self.E0.size == 0:
            return np.zeros(self.n_global, dtype=np.complex128)

        x = np.asarray(x, dtype=np.complex128).reshape(-1)
        xc = self.E0.conj().T @ x

        if isinstance(self.coarse_solver, dict) and self.coarse_solver.get("mode") == "gmres_ilu":
            yc, info = spsla.gmres(
                self.coarse_solver["A"],
                xc,
                M=self.coarse_solver["M"],
                restart=50,
                rtol=1.0e-10,
                atol=0.0,
                maxiter=500,
            )
            if info != 0:
                raise RuntimeError(f"Coarse GMRES-ILU failed with info={info}")
        else:
            yc = self.coarse_solver.solve(xc)

        return self.E0 @ yc


def _build_local_solver(A_star: sps.csr_matrix):
    mode = PARAMS.local_solver
    if mode == "lu":
        return "lu", spsla.splu(A_star.tocsc())

    if mode == "gmres_ilu":
        ilu = spsla.spilu(A_star.tocsc())
        M = spsla.LinearOperator(A_star.shape, matvec=ilu.solve, dtype=np.complex128)
        return "gmres_ilu", {"A": A_star, "M": M}

    raise ValueError(f"Unsupported PARAMS.local_solver={mode!r}")


def _build_coarse_solver(B0: sps.csr_matrix):
    mode = PARAMS.coarse_solver
    if B0.shape[0] == 0:
        return None

    if mode == "lu":
        try:
            return spsla.splu(B0.tocsc())
        except Exception:
            if not PARAMS.allow_dense_coarse_fallback:
                raise
            Bd = B0.toarray()
            return type(
                "DenseSolveWrapper",
                (),
                {"solve": staticmethod(lambda x: np.linalg.solve(Bd, x))},
            )()

    if mode == "gmres_ilu":
        ilu = spsla.spilu(B0.tocsc())
        M = spsla.LinearOperator(B0.shape, matvec=ilu.solve, dtype=np.complex128)
        return {"mode": "gmres_ilu", "A": B0, "M": M}

    raise ValueError(f"Unsupported PARAMS.coarse_solver={mode!r}")


def build_two_level_preconditioner(
    *,
    A_global: PETSc.Mat,
    decomposition: dict,
    pou_data,
    E0: np.ndarray,
    local_patch_ops_payload: dict,
) -> TwoLevelPreconditioner:
    B = _mat_to_scipy_csr(A_global)
    n_global = B.shape[0]

    dofs_olap_list = [np.asarray(x, dtype=np.int32) for x in local_patch_ops_payload["dofs_olap"].tolist()]
    dofs_star_list = [np.asarray(x, dtype=np.int32) for x in local_patch_ops_payload["dofs_star"].tolist()]

    # Impedance matrices on the oversampled patches.
    B_star_indptr_list = local_patch_ops_payload["B_star_imp_indptr"].tolist()
    B_star_indices_list = local_patch_ops_payload["B_star_imp_indices"].tolist()
    B_star_data_list = local_patch_ops_payload["B_star_imp_data"].tolist()

    chi_full = list(pou_data.chi_full)

    local_patch_solvers: list[LocalPatchSolver] = []

    for i, (dofs_olap, dofs_star) in enumerate(zip(dofs_olap_list, dofs_star_list)):
        n_star = len(dofs_star)

        A_star = _csr_to_scipy(
            np.asarray(B_star_indptr_list[i], dtype=np.int32),
            np.asarray(B_star_indices_list[i], dtype=np.int32),
            np.asarray(B_star_data_list[i], dtype=np.complex128),
            (n_star, n_star),
        ).tocsr()

        solve_mode, solver = _build_local_solver(A_star)
        chi_olap = np.asarray(chi_full[i], dtype=np.float64)[dofs_olap]

        local_patch_solvers.append(
            LocalPatchSolver(
                patch_index=i,
                dofs_olap=dofs_olap,
                dofs_star=dofs_star,
                chi_olap=chi_olap,
                solve_mode=solve_mode,
                A_star_solver=solver,
            )
        )

    if E0.size == 0:
        coarse_solver = None
    else:
        B0 = sps.csr_matrix(E0.conj().T @ (B @ E0))
        shift = float(PARAMS.coarse_regularization_shift) * max(
            1.0, np.max(np.abs(B0.diagonal()))
        )
        if shift > 0.0:
            B0 = B0 + shift * sps.identity(B0.shape[0], dtype=np.complex128, format="csr")
        coarse_solver = _build_coarse_solver(B0)

    return TwoLevelPreconditioner(
        n_global=n_global,
        local_patch_solvers=local_patch_solvers,
        E0=np.asarray(E0, dtype=np.complex128),
        coarse_solver=coarse_solver,
    )


def apply_hybrid_preconditioner(
    *,
    precond: TwoLevelPreconditioner,
    A_global: PETSc.Mat,
    x: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y1 = precond.apply(x)

    if precond.E0.size == 0:
        return y1

    Bx = A_global.createVecRight()
    Bx.array[:] = y1
    tmp = A_global.createVecLeft()
    A_global.mult(Bx, tmp)
    r = x - np.asarray(tmp.array, dtype=np.complex128)

    yc = precond.apply_coarse(r)
    return y1 + yc
