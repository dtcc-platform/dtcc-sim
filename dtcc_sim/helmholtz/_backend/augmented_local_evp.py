from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse as sps
from scipy.sparse import linalg as spsla
from scipy import linalg as sla

from .params import PARAMS


def _csr_to_scipy(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
) -> sps.csr_matrix:
    return sps.csr_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=shape,
    )


def _selection_matrix_from_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    ncols: int,
) -> sps.csr_matrix:
    nrows = int(len(indptr) - 1)
    return _csr_to_scipy(indptr, indices, data, (nrows, ncols))


def _sorted_real_eigs(vals: np.ndarray, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.real(vals))[::-1]
    return vals[order], vecs[:, order]


def _filter_modes(vals: np.ndarray, vecs: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.real(vals) > float(rho) ** 2
    if not np.any(mask):
        return (
            np.zeros(0, dtype=np.complex128),
            np.zeros((vecs.shape[0], 0), dtype=np.complex128),
        )
    vals_f = vals[mask]
    vecs_f = vecs[:, mask]
    vals_f, vecs_f = _sorted_real_eigs(vals_f, vecs_f)
    return vals_f, vecs_f


def _normalize_A(vecs: np.ndarray, A: sps.csr_matrix) -> np.ndarray:
    if vecs.size == 0:
        return vecs
    out = np.array(vecs, dtype=np.complex128, copy=True)
    for j in range(out.shape[1]):
        v = out[:, j]
        nrm2 = np.real(np.vdot(v, A @ v))
        if nrm2 > 0.0:
            out[:, j] /= np.sqrt(nrm2)
    return out


@dataclass
class AugmentedLocalEVPData:
    patch_index: int
    n_star: int
    n_int: int
    A_star: sps.csr_matrix
    A_star_k: sps.csr_matrix
    B_star_imp: sps.csr_matrix
    C: sps.csr_matrix


def build_augmented_evp_matrices(
    *,
    patch_index: int,
    n_star: int,
    n_int: int,
    A_star_indptr: np.ndarray,
    A_star_indices: np.ndarray,
    A_star_data: np.ndarray,
    A_star_k_indptr: np.ndarray,
    A_star_k_indices: np.ndarray,
    A_star_k_data: np.ndarray,
    B_star_imp_indptr: np.ndarray,
    B_star_imp_indices: np.ndarray,
    B_star_imp_data: np.ndarray,
    C_indptr: np.ndarray,
    C_indices: np.ndarray,
    C_data: np.ndarray,
    chi_star: np.ndarray | None,
) -> AugmentedLocalEVPData:
    A_star = _csr_to_scipy(A_star_indptr, A_star_indices, A_star_data, (n_star, n_star)).tocsr()
    A_star_k = _csr_to_scipy(A_star_k_indptr, A_star_k_indices, A_star_k_data, (n_star, n_star)).tocsr()
    B_star_imp = _csr_to_scipy(B_star_imp_indptr, B_star_imp_indices, B_star_imp_data, (n_star, n_star)).tocsr()
    C = _selection_matrix_from_csr(C_indptr, C_indices, C_data, n_star).tocsr()

    if chi_star is not None:
        chi = np.asarray(chi_star, dtype=np.float64).reshape(-1)
        if chi.size != n_star:
            raise RuntimeError(
                f"Patch {patch_index}: chi_star has size {chi.size}, expected {n_star}"
            )
        # Paper equation (2.28) uses A_{omega_i*,k}(I_h(chi_i phi), I_h(chi_i v)).
        # For nodal Lagrange spaces this is represented by Chi^H A_k Chi.
        Chi = sps.diags(chi.astype(np.complex128), format="csr")
        A_star_k = (Chi.getH() @ A_star_k @ Chi).tocsr()

    if C.shape != (n_int, n_star):
        raise RuntimeError(
            f"Patch {patch_index}: inconsistent C shape {C.shape}, expected {(n_int, n_star)}"
        )

    return AugmentedLocalEVPData(
        patch_index=int(patch_index),
        n_star=int(n_star),
        n_int=int(n_int),
        A_star=A_star,
        A_star_k=A_star_k,
        B_star_imp=B_star_imp,
        C=C,
    )


def augmented_matrices(data: AugmentedLocalEVPData) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    A = data.A_star.tocsr()
    Ak = data.A_star_k.tocsr()
    B = data.B_star_imp.tocsr()
    C = data.C.tocsr()

    n = data.n_star
    m = data.n_int

    Znm = sps.csr_matrix((n, m), dtype=np.complex128)
    Zmn = sps.csr_matrix((m, n), dtype=np.complex128)
    Zmm = sps.csr_matrix((m, m), dtype=np.complex128)

    K_top = sps.hstack([A, B.getH() @ C.getH()], format="csr")
    K_bot = sps.hstack([C @ B, Zmm], format="csr")
    K = sps.vstack([K_top, K_bot], format="csr")

    M_top = sps.hstack([Ak, Znm], format="csr")
    M_bot = sps.hstack([Zmn, Zmm], format="csr")
    M = sps.vstack([M_top, M_bot], format="csr")

    return K, M


def transformed_operator(
    data: AugmentedLocalEVPData,
    *,
    reg: float,
) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    K, M = augmented_matrices(data)
    N = K.shape[0]
    I = sps.identity(N, dtype=np.complex128, format="csr")
    Kreg = K + complex(reg) * I
    return Kreg, M


def _solve_dense(
    data: AugmentedLocalEVPData,
    *,
    rho: float,
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    Kreg, M = transformed_operator(data, reg=reg)
    Kd = Kreg.toarray()
    Md = M.toarray()
    T = sla.solve(Kd, Md, assume_a="gen")
    vals, vecs = sla.eig(T)

    finite = np.isfinite(vals)
    vals = vals[finite]
    vecs = vecs[:, finite]

    vals, vecs = _sorted_real_eigs(vals, vecs)
    phi = vecs[: data.n_star, :]
    vals, phi = _filter_modes(vals, phi, rho=float(rho))
    phi = _normalize_A(phi, data.A_star)
    return vals, phi


def _solve_arpack_once_raw(
    data: AugmentedLocalEVPData,
    *,
    reg: float,
    nev: int,
    ncv: int,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray]:
    Kreg, M = transformed_operator(data, reg=reg)
    lu = spsla.splu(Kreg.tocsc())

    def mv(x):
        return lu.solve(M @ x)

    N = Kreg.shape[0]
    T = spsla.LinearOperator((N, N), matvec=mv, dtype=np.complex128)

    vals, vecs = spsla.eigs(
        T,
        k=int(nev),
        which="LR",
        ncv=int(ncv),
        tol=float(tol),
        maxiter=int(maxiter),
    )

    finite = np.isfinite(vals)
    vals = vals[finite]
    vecs = vecs[:, finite]
    vals, vecs = _sorted_real_eigs(vals, vecs)
    return vals, vecs


def solve_augmented_local_modes(
    *,
    patch_index: int,
    n_star: int,
    n_int: int,
    A_star_indptr: np.ndarray,
    A_star_indices: np.ndarray,
    A_star_data: np.ndarray,
    A_star_k_indptr: np.ndarray,
    A_star_k_indices: np.ndarray,
    A_star_k_data: np.ndarray,
    B_star_imp_indptr: np.ndarray,
    B_star_imp_indices: np.ndarray,
    B_star_imp_data: np.ndarray,
    C_indptr: np.ndarray,
    C_indices: np.ndarray,
    C_data: np.ndarray,
    rho: float,
    chi_star: np.ndarray | None = None,
    verbose: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if verbose is None:
        verbose = bool(PARAMS.verbose)

    data = build_augmented_evp_matrices(
        patch_index=patch_index,
        n_star=n_star,
        n_int=n_int,
        A_star_indptr=A_star_indptr,
        A_star_indices=A_star_indices,
        A_star_data=A_star_data,
        A_star_k_indptr=A_star_k_indptr,
        A_star_k_indices=A_star_k_indices,
        A_star_k_data=A_star_k_data,
        B_star_imp_indptr=B_star_imp_indptr,
        B_star_imp_indices=B_star_imp_indices,
        B_star_imp_data=B_star_imp_data,
        C_indptr=C_indptr,
        C_indices=C_indices,
        C_data=C_data,
        chi_star=chi_star,
    )

    backend = PARAMS.local_evp_backend.strip().lower()
    nev_initial = int(PARAMS.eps_initial_dim)
    ncv_extra = int(PARAMS.arpack_ncv_min_extra)
    maxiter = int(PARAMS.eps_max_it)
    tol = float(PARAMS.eps_tol)
    growth = float(PARAMS.eps_growth_factor)
    max_nev_frac = float(PARAMS.arpack_max_nev_frac)
    reg_env = float(PARAMS.aug_shift)
    dense_fallback_max_dim = int(PARAMS.dense_fallback_max_dim)

    total_dim = int(data.n_star + data.n_int)
    max_nev = max(2, min(total_dim - 2, int(round(max_nev_frac * total_dim))))
    nev = max(2, min(max_nev, int(nev_initial)))

    if verbose:
        print(
            f"[aug-evp] patch={patch_index:03d} backend={backend} "
            f"total_dim={total_dim} n_star={data.n_star} n_int={data.n_int} "
            f"rho2={float(rho)**2:.6e} nev_initial={nev_initial} max_nev={max_nev}",
            flush=True,
        )

    if backend == "dense":
        vals, phi = _solve_dense(data, rho=float(rho), reg=float(reg_env))
        if verbose:
            min_re = np.real(vals).min() if vals.size else float("nan")
            max_re = np.real(vals).max() if vals.size else float("nan")
            print(
                f"[aug-evp:dense] patch={patch_index:03d} kept={phi.shape[1]} "
                f"min_re={min_re:.6e} max_re={max_re:.6e}",
                flush=True,
            )
        return vals, phi

    if backend not in ("arpack", "arpack_then_dense"):
        raise ValueError(f"Unsupported local_evp_backend={backend!r}")

    fixed_nloc = int(PARAMS.n_modes_per_patch)
    if verbose and fixed_nloc > 0:
        print(
            f"[aug-evp] patch={patch_index:03d} fixed mode selection: "
            f"n_modes_per_patch={fixed_nloc}",
            flush=True,
        )

    last_error: Exception | None = None
    best_vals = np.zeros(0, dtype=np.complex128)
    best_phi = np.zeros((n_star, 0), dtype=np.complex128)
    attempt = 0
    rho2 = float(rho) ** 2

    while True:
        attempt += 1
        ncv = min(total_dim - 1, max(nev + 2, nev + ncv_extra))
        try:
            vals_raw, vecs_raw = _solve_arpack_once_raw(
                data,
                reg=float(reg_env),
                nev=int(nev),
                ncv=int(ncv),
                tol=float(tol),
                maxiter=int(maxiter),
            )

            phi_raw = vecs_raw[: data.n_star, :]

            if fixed_nloc > 0:
                keep = min(int(fixed_nloc), int(vals_raw.size))
                vals_kept = vals_raw[:keep]
                phi_kept = phi_raw[:, :keep]
                phi_kept = _normalize_A(phi_kept, data.A_star)
            else:
                vals_kept, phi_kept = _filter_modes(vals_raw, phi_raw, rho=float(rho))
                phi_kept = _normalize_A(phi_kept, data.A_star)

            best_vals = vals_kept
            best_phi = phi_kept

            raw_min_re = np.real(vals_raw).min() if vals_raw.size else float("nan")
            raw_max_re = np.real(vals_raw).max() if vals_raw.size else float("nan")
            frontier_crossed = bool(vals_raw.size > 0 and raw_min_re <= rho2)

            if fixed_nloc > 0:
                frontier_crossed = bool(vals_raw.size >= fixed_nloc)

            if verbose:
                print(
                    f"[aug-evp] patch={patch_index:03d} attempt={attempt} "
                    f"nev={nev} ncv={ncv} kept={phi_kept.shape[1]} "
                    f"raw_min_re={raw_min_re:.6e} raw_max_re={raw_max_re:.6e} "
                    f"frontier_crossed={'yes' if frontier_crossed else 'no'}",
                    flush=True,
                )

            if frontier_crossed:
                return best_vals, best_phi

            if nev >= max_nev:
                break

            nev_next = min(max_nev, max(nev + 1, int(np.ceil(growth * nev))))
            if nev_next == nev:
                break
            nev = nev_next

        except Exception as e:
            last_error = e
            if verbose:
                print(
                    f"[aug-evp] patch={patch_index:03d} attempt={attempt} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            if nev >= max_nev:
                break
            nev_next = min(max_nev, max(nev + 1, int(np.ceil(growth * nev))))
            if nev_next == nev:
                break
            nev = nev_next

    if backend == "arpack_then_dense" and total_dim <= dense_fallback_max_dim:
        if verbose:
            print(
                f"[aug-evp] patch={patch_index:03d} falling back to dense solve "
                f"(total_dim={total_dim})",
                flush=True,
            )
        vals, phi = _solve_dense(data, rho=float(rho), reg=float(reg_env))
        if verbose:
            min_re = np.real(vals).min() if vals.size else float("nan")
            max_re = np.real(vals).max() if vals.size else float("nan")
            print(
                f"[aug-evp:dense] patch={patch_index:03d} kept={phi.shape[1]} "
                f"min_re={min_re:.6e} max_re={max_re:.6e}",
                flush=True,
            )
        return vals, phi

    if verbose:
        print(
            f"[aug-evp] patch={patch_index:03d} WARNING: threshold frontier not crossed "
            f"before max_nev={max_nev}; returning best available kept={best_phi.shape[1]}",
            flush=True,
        )

    if best_phi.shape[1] > 0:
        return best_vals, best_phi
    if last_error is not None:
        raise last_error
    return np.zeros(0, dtype=np.complex128), np.zeros((n_star, 0), dtype=np.complex128)
