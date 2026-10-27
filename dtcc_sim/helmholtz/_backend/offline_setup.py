from __future__ import annotations

from pathlib import Path
import multiprocessing as mp
import numpy as np

from .params import PARAMS, current_parameters, set_worker_parameters
from .mesh import build_system
from .decomposition import build_decomposition
from .pou import build_partition_of_unity
from .local_operators import (
    build_all_local_patch_operators,
    serialize_local_patch_operators,
)
from .augmented_local_evp import solve_augmented_local_modes
from .coarse_space import (
    build_E0_from_local_modes,
    serialize_coarse_space,
    summarize_coarse_space,
)


def _obj_array_to_list(arr) -> list[np.ndarray]:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(x) for x in arr.tolist()]
    if isinstance(arr, list):
        return [np.asarray(x) for x in arr]
    raise TypeError("Expected object ndarray or list")


def tprint(msg: str) -> None:
    if PARAMS.verbose:
        print(msg, flush=True)


def _ensure_parent(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


_worker_thread_limits = None


def _initialize_patch_worker(params):
    """Configure a fresh worker after numerical libraries have been imported."""
    global _worker_thread_limits
    from threadpoolctl import threadpool_limits

    set_worker_parameters(params)
    _worker_thread_limits = threadpool_limits(limits=params.threads_per_worker)


def _solve_patch_task(args):
    op, rho, chi_star, verbose = args
    lambdas_i, Phi_i = solve_augmented_local_modes(
        patch_index=int(op.patch_index),
        n_star=int(op.n_star),
        n_int=int(op.n_int),
        A_star_indptr=op.A_star_indptr,
        A_star_indices=op.A_star_indices,
        A_star_data=op.A_star_data,
        A_star_k_indptr=op.A_star_k_indptr,
        A_star_k_indices=op.A_star_k_indices,
        A_star_k_data=op.A_star_k_data,
        B_star_imp_indptr=op.B_star_imp_indptr,
        B_star_imp_indices=op.B_star_imp_indices,
        B_star_imp_data=op.B_star_imp_data,
        C_indptr=op.C_indptr,
        C_indices=op.C_indices,
        C_data=op.C_data,
        rho=float(rho),
        chi_star=np.asarray(chi_star, dtype=np.float64),
        verbose=bool(verbose),
    )
    return (
        int(op.patch_index),
        np.asarray(Phi_i, dtype=np.complex128),
        np.asarray(lambdas_i, dtype=np.complex128),
        np.asarray(op.dofs_star, dtype=np.int32),
        int(op.n_star),
        int(Phi_i.shape[1]),
    )


def build_and_save_preconditioner_data() -> dict:
    """
    Offline stage for the final 3D heterogeneous code path.

    Builds and saves:
      - decomposition
      - partition of unity
      - local patch operators
      - local spectral modes
      - global coarse basis E0
    """
    tprint("=" * 72)
    tprint("[offline] 3D heterogeneous two-level Helmholtz preconditioner setup")
    tprint("=" * 72)
    tprint(f"[offline] mesh source      : {PARAMS.mesh_source}")
    tprint(f"[offline] coefficient mode : {PARAMS.coefficient_mode}")
    tprint(f"[offline] partitioner      : {PARAMS.partitioner}")
    tprint(f"[offline] rho             : {float(PARAMS.rho):.6e}")
    tprint(f"[offline] k               : {float(PARAMS.k()):.6e}")
    tprint(f"[offline] output NPZ      : {PARAMS.preconditioner_npz_path}")

    _ensure_parent(PARAMS.preconditioner_npz_path)

    mesh, facet_tags, cell_tags, V, Acoef, Vcoef, beta, A_global, b_global = build_system()

    n_global = int(V.dofmap.index_map.size_local)
    tprint(f"[offline] fine-space dofs N = {n_global}")

    decomposition = build_decomposition(mesh, V, verbose=PARAMS.verbose)
    pou = build_partition_of_unity(mesh, V, decomposition=decomposition)

    local_ops = build_all_local_patch_operators(
        mesh,
        V,
        Acoef,
        Vcoef,
        beta,
        facet_tags=facet_tags,
        decomposition=decomposition,
    )

    local_ops_payload = serialize_local_patch_operators(local_ops)

    local_modes_by_patch: list[np.ndarray] = [None] * len(local_ops)
    lambdas_by_patch: list[np.ndarray] = [None] * len(local_ops)
    dofs_star_by_patch: list[np.ndarray] = [None] * len(local_ops)

    n_workers = int(PARAMS.patch_workers)

    if n_workers <= 1:
        tasks = [
            (
                op,
                float(PARAMS.rho),
                np.asarray(pou.chi_full[int(op.patch_index)], dtype=np.float64)[
                    np.asarray(op.dofs_star, dtype=np.int32)
                ],
                bool(PARAMS.verbose),
            )
            for op in local_ops
        ]
        results = map(_solve_patch_task, tasks)
    else:
        tprint(f"[offline] parallel local EVP workers = {n_workers}")
        tasks = [
            (
                op,
                float(PARAMS.rho),
                np.asarray(pou.chi_full[int(op.patch_index)], dtype=np.float64)[
                    np.asarray(op.dofs_star, dtype=np.int32)
                ],
                bool(PARAMS.verbose),
            )
            for op in local_ops
        ]
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=n_workers, initializer=_initialize_patch_worker,
                        initargs=(current_parameters(),))
        results = pool.imap_unordered(_solve_patch_task, tasks)

    try:
        for idx, Phi_i, lambdas_i, dofs_star_i, n_star_i, kept_i in results:
            local_modes_by_patch[idx] = Phi_i
            lambdas_by_patch[idx] = lambdas_i
            dofs_star_by_patch[idx] = dofs_star_i

            tprint(
                f"[coarse] patch={idx:03d} "
                f"star_dofs={n_star_i} kept={kept_i}"
            )
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    coarse = build_E0_from_local_modes(
        local_modes_by_patch=local_modes_by_patch,
        lambdas_by_patch=lambdas_by_patch,
        dofs_star_by_patch=dofs_star_by_patch,
        chi_full_by_patch=list(pou.chi_full),
        n_global=n_global,
    )

    coarse_summary = summarize_coarse_space(coarse)
    tprint(f"[offline] E0 size = {coarse.E0.shape}")
    tprint(
        f"[offline] coarse summary: nc={coarse_summary['nc']} "
        f"nloc_min={coarse_summary['nloc_min']} "
        f"nloc_max={coarse_summary['nloc_max']} "
        f"nloc_mean={coarse_summary['nloc_mean']:.2f}"
    )

    # Save in a plain npz payload, keeping arrays/object-arrays explicit.
    payload: dict = {}

    # Basic metadata
    payload["N_global"] = np.array([n_global], dtype=np.int32)
    payload["k_value"] = np.array([float(PARAMS.k())], dtype=np.float64)
    payload["rho"] = np.array([float(PARAMS.rho)], dtype=np.float64)
    payload["mesh_source"] = np.array([PARAMS.mesh_source], dtype=object)
    payload["coefficient_mode"] = np.array([PARAMS.coefficient_mode], dtype=object)
    payload["partitioner"] = np.array([PARAMS.partitioner], dtype=object)
    payload["fe_degree"] = np.array([int(PARAMS.fe_degree)], dtype=np.int32)

    # Decomposition payload
    for k, v in decomposition.items():
        payload[f"decomp__{k}"] = v

    # PoU payload
    payload["pou__chi_full"] = np.array(
        [np.asarray(x, dtype=np.float64) for x in pou.chi_full],
        dtype=object,
    )
    payload["pou__multiplicity"] = np.asarray(pou.multiplicity, dtype=np.int32)
    payload["pou__ndofs_in_union"] = np.array([int(pou.ndofs_in_union)], dtype=np.int32)

    # Local operators payload
    for k, v in local_ops_payload.items():
        payload[f"local_ops__{k}"] = v

    # Local spectral modes payload
    payload["local_modes__Phi_by_patch"] = np.array(
        [np.asarray(x, dtype=np.complex128) for x in local_modes_by_patch],
        dtype=object,
    )
    payload["local_modes__lambda_by_patch"] = np.array(
        [np.asarray(x, dtype=np.complex128) for x in lambdas_by_patch],
        dtype=object,
    )
    payload["local_modes__dofs_star"] = np.array(
        [np.asarray(x, dtype=np.int32) for x in dofs_star_by_patch],
        dtype=object,
    )

    # Coarse-space payload
    coarse_payload = serialize_coarse_space(
        coarse=coarse,
        dofs_star_by_patch=dofs_star_by_patch,
    )
    for k, v in coarse_payload.items():
        payload[f"coarse__{k}"] = v

    np.savez(PARAMS.preconditioner_npz_path, **payload)
    tprint(f"[offline] saved NPZ to {PARAMS.preconditioner_npz_path}")
    tprint("[offline] done.")
    tprint("=" * 72)

    return payload


def load_saved_preconditioner_data(npz_path: str | None = None) -> dict:
    """
    Convenience loader returning a plain dict.
    """
    path = npz_path or PARAMS.preconditioner_npz_path
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
