from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import PARAMS
from .pou import apply_partition_of_unity_operator_to_array


@dataclass
class CoarseSpaceData:
    E0: np.ndarray
    lambdas_by_patch: list[np.ndarray]
    nloc_by_patch: np.ndarray


def _obj_array_to_list(arr) -> list[np.ndarray]:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(x) for x in arr.tolist()]
    if isinstance(arr, list):
        return [np.asarray(x) for x in arr]
    raise TypeError("Expected object ndarray or list")


def build_E0_from_local_modes(
    *,
    local_modes_by_patch: list[np.ndarray],
    lambdas_by_patch: list[np.ndarray],
    dofs_star_by_patch: list[np.ndarray],
    chi_full_by_patch: list[np.ndarray],
    n_global: int,
) -> CoarseSpaceData:
    """
    Build the global coarse basis E0.

    For each local spectral basis vector phi_i^j on patch omega_i^*:
      1. embed it into the global FE vector on dofs_star_i
      2. apply the PoU operator P_{h,i} v = I_h(chi_i v)
      3. append to the global coarse basis matrix
    """
    cols: list[np.ndarray] = []
    kept_lambdas: list[np.ndarray] = []
    nloc: list[int] = []

    M = len(local_modes_by_patch)
    if not (len(lambdas_by_patch) == len(dofs_star_by_patch) == len(chi_full_by_patch) == M):
        raise RuntimeError("Patch-list lengths do not match while building E0")

    for i in range(M):
        Phi_i = np.asarray(local_modes_by_patch[i], dtype=np.complex128)
        lam_i = np.asarray(lambdas_by_patch[i], dtype=np.complex128).reshape(-1)
        dofs_star_i = np.asarray(dofs_star_by_patch[i], dtype=np.int32)
        chi_i = np.asarray(chi_full_by_patch[i], dtype=np.float64)

        if Phi_i.ndim != 2:
            raise RuntimeError(f"Patch {i}: local mode matrix must be 2D")
        if Phi_i.shape[0] != len(dofs_star_i):
            raise RuntimeError(
                f"Patch {i}: local mode rows {Phi_i.shape[0]} do not match "
                f"number of star dofs {len(dofs_star_i)}"
            )
        if Phi_i.shape[1] != len(lam_i):
            raise RuntimeError(
                f"Patch {i}: number of local eigenvalues {len(lam_i)} does not match "
                f"number of local mode columns {Phi_i.shape[1]}"
            )

        kept_lambdas.append(lam_i)
        nloc.append(Phi_i.shape[1])

        for j in range(Phi_i.shape[1]):
            global_vec = np.zeros(int(n_global), dtype=np.complex128)
            global_vec[dofs_star_i] = Phi_i[:, j]
            global_vec = apply_partition_of_unity_operator_to_array(chi_i, global_vec, interpolate=True)
            cols.append(global_vec)

    if len(cols) == 0:
        E0 = np.zeros((int(n_global), 0), dtype=np.complex128)
    else:
        E0 = np.column_stack(cols).astype(np.complex128, copy=False)

    return CoarseSpaceData(
        E0=E0,
        lambdas_by_patch=kept_lambdas,
        nloc_by_patch=np.asarray(nloc, dtype=np.int32),
    )


def build_E0_from_serialized_payload(
    *,
    local_modes_payload: dict,
    pou_data,
    n_global: int,
) -> CoarseSpaceData:
    """
    Convenience wrapper if modes/dofs are already stored in object-array form.
    Expected keys:
      - "Phi_by_patch"
      - "lambda_by_patch"
      - "dofs_star"
    """
    Phi_by_patch = _obj_array_to_list(local_modes_payload["Phi_by_patch"])
    lambda_by_patch = _obj_array_to_list(local_modes_payload["lambda_by_patch"])
    dofs_star_by_patch = _obj_array_to_list(local_modes_payload["dofs_star"])

    return build_E0_from_local_modes(
        local_modes_by_patch=Phi_by_patch,
        lambdas_by_patch=lambda_by_patch,
        dofs_star_by_patch=dofs_star_by_patch,
        chi_full_by_patch=list(pou_data.chi_full),
        n_global=int(n_global),
    )


def summarize_coarse_space(coarse: CoarseSpaceData) -> dict[str, object]:
    nloc = np.asarray(coarse.nloc_by_patch, dtype=np.int32)
    nc = int(coarse.E0.shape[1])

    if nloc.size > 0:
        nloc_min = int(nloc.min())
        nloc_max = int(nloc.max())
        nloc_mean = float(nloc.mean())
    else:
        nloc_min = 0
        nloc_max = 0
        nloc_mean = 0.0

    return {
        "nc": nc,
        "nloc_by_patch": nloc,
        "nloc_min": nloc_min,
        "nloc_max": nloc_max,
        "nloc_mean": nloc_mean,
    }


def serialize_coarse_space(
    *,
    coarse: CoarseSpaceData,
    dofs_star_by_patch: list[np.ndarray],
) -> dict:
    """
    Build an np.savez-friendly payload.
    """
    return {
        "E0_real": np.asarray(np.real(coarse.E0), dtype=np.float64),
        "E0_imag": np.asarray(np.imag(coarse.E0), dtype=np.float64),
        "lambda_by_patch": np.array(
            [np.asarray(x, dtype=np.complex128) for x in coarse.lambdas_by_patch],
            dtype=object,
        ),
        "nloc_by_patch": np.asarray(coarse.nloc_by_patch, dtype=np.int32),
        "dofs_star": np.array(
            [np.asarray(x, dtype=np.int32) for x in dofs_star_by_patch],
            dtype=object,
        ),
    }


def deserialize_E0(payload: dict) -> np.ndarray:
    return np.asarray(payload["E0_real"], dtype=np.float64) + 1j * np.asarray(
        payload["E0_imag"], dtype=np.float64
    )
