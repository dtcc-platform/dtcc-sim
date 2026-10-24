from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh

from .params import PARAMS


def tprint(msg: str) -> None:
    if MPI.COMM_WORLD.rank == 0:
        print(msg, flush=True)


def _to_np_int_list(obj_array) -> list[np.ndarray]:
    if isinstance(obj_array, np.ndarray) and obj_array.dtype == object:
        return [np.asarray(x, dtype=np.int32).reshape(-1) for x in obj_array.tolist()]
    if isinstance(obj_array, list):
        return [np.asarray(x, dtype=np.int32).reshape(-1) for x in obj_array]
    raise TypeError("Expected object ndarray or list")


@dataclass
class PartitionOfUnityData:
    chi_full: list[np.ndarray]
    multiplicity: np.ndarray
    ndofs_in_union: int


def _cell_barycenters(mesh: dmesh.Mesh) -> np.ndarray:
    gdim = mesh.geometry.dim
    x = np.asarray(mesh.geometry.x[:, :gdim], dtype=np.float64)
    dofmap = mesh.geometry.dofmap
    n_cells = int(mesh.topology.index_map(mesh.topology.dim).size_local)

    out = np.zeros((n_cells, gdim), dtype=np.float64)
    for c in range(n_cells):
        verts = dofmap[c]
        out[c, :] = np.mean(x[verts], axis=0)
    return out


def _structured_hat_value_1d(t: np.ndarray, center: float, half_width: float) -> np.ndarray:
    hw = max(float(half_width), 1.0e-15)
    return np.maximum(1.0 - np.abs(t - float(center)) / hw, 0.0)


def _build_geometric_hats_pou(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    *,
    decomposition: dict,
) -> PartitionOfUnityData:
    cells_olap = _to_np_int_list(decomposition["cells_olap"])
    dofs_olap = _to_np_int_list(decomposition["dofs_olap"])

    X = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    if X.ndim != 2:
        raise RuntimeError(f"Unexpected dof coordinate shape: {X.shape}")

    n_global = X.shape[0]
    gdim = mesh.geometry.dim
    X = X[:, :gdim]

    cell_bc = _cell_barycenters(mesh)

    raw_weights: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []

    for i, cell_ids in enumerate(cells_olap):
        cell_ids = np.asarray(cell_ids, dtype=np.int32).reshape(-1)

        if cell_ids.size == 0:
            raw_weights.append(np.zeros(n_global, dtype=np.float64))
            support_masks.append(np.zeros(n_global, dtype=bool))
            continue

        bc_i = np.asarray(cell_bc[cell_ids, :], dtype=np.float64)
        if bc_i.ndim == 1:
            bc_i = bc_i.reshape(1, -1)

        mins = np.asarray(np.min(bc_i, axis=0), dtype=np.float64).reshape(-1)
        maxs = np.asarray(np.max(bc_i, axis=0), dtype=np.float64).reshape(-1)

        if mins.size != gdim or maxs.size != gdim:
            raise RuntimeError(
                f"Patch {i}: expected mins/maxs of size {gdim}, got {mins.size}/{maxs.size}"
            )

        centers = 0.5 * (mins + maxs)
        half_widths = 0.5 * np.maximum(maxs - mins, 1.0e-15)

        w = np.ones(n_global, dtype=np.float64)
        for d in range(gdim):
            w *= _structured_hat_value_1d(X[:, d], centers[d], half_widths[d])

        support = np.zeros(n_global, dtype=bool)
        dofs_i = np.asarray(dofs_olap[i], dtype=np.int32).reshape(-1)
        support[dofs_i] = True
        w[~support] = 0.0

        raw_weights.append(w)
        support_masks.append(support)

    total = np.zeros(n_global, dtype=np.float64)
    for w in raw_weights:
        total += w

    multiplicity = np.zeros(n_global, dtype=np.int32)
    for sup in support_masks:
        multiplicity += sup.astype(np.int32)

    needs_fallback = (multiplicity > 0) & (total <= 1.0e-15)
    if np.any(needs_fallback):
        idx = np.where(needs_fallback)[0]
        for w, sup in zip(raw_weights, support_masks):
            take = idx[sup[idx]]
            if take.size > 0:
                w[take] = 1.0
        total[:] = 0.0
        for w in raw_weights:
            total += w

    chi_full: list[np.ndarray] = []
    for w in raw_weights:
        chi = np.zeros(n_global, dtype=np.float64)
        nz = total > 1.0e-15
        chi[nz] = w[nz] / total[nz]
        chi_full.append(chi)

    ndofs_in_union = int(np.count_nonzero(multiplicity > 0))

    if ndofs_in_union > 0:
        stacked = np.sum(np.vstack(chi_full), axis=0)
        max_dev = float(np.max(np.abs(stacked[multiplicity > 0] - 1.0)))
    else:
        max_dev = 0.0

    if PARAMS.verbose:
        tprint(f"[pou] geometric hats sum-to-one: max|sum_i chi_i - 1| = {max_dev:.3e}")

    return PartitionOfUnityData(
        chi_full=chi_full,
        multiplicity=multiplicity,
        ndofs_in_union=ndofs_in_union,
    )



def _cell_layer_distances(
    core_cells: np.ndarray,
    olap_cells: np.ndarray,
    cell_adjacency: list[np.ndarray],
) -> dict[int, int]:
    """Graph distance from the non-overlapping core inside the overlap patch."""
    olap_set = set(np.asarray(olap_cells, dtype=np.int32).tolist())
    current = set(np.asarray(core_cells, dtype=np.int32).tolist()) & olap_set
    distances = {int(c): 0 for c in current}

    layer = 0
    while current:
        layer += 1
        next_front: set[int] = set()
        for c in current:
            for nb in cell_adjacency[int(c)]:
                nb_i = int(nb)
                if nb_i in olap_set and nb_i not in distances:
                    distances[nb_i] = layer
                    next_front.add(nb_i)
        current = next_front

    return distances


def _build_core_overlap_ramp_pou(
    mesh,
    V: fem.FunctionSpace,
    *,
    decomposition: dict,
) -> PartitionOfUnityData:
    cells_core = _to_np_int_list(decomposition["cells_core"])
    cells_olap = _to_np_int_list(decomposition["cells_olap"])
    cell_adjacency = _to_np_int_list(decomposition["cell_adjacency"])
    dofs_core = _to_np_int_list(decomposition["dofs_core"])
    dofs_olap = _to_np_int_list(decomposition["dofs_olap"])

    X = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    if X.ndim != 2:
        raise RuntimeError(f"Unexpected dof coordinate shape: {X.shape}")

    n_global = X.shape[0]
    overlap_layers = max(0, int(np.asarray(decomposition["overlap_layers"]).reshape(-1)[0]))

    raw_weights: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []

    for i, olap_i in enumerate(cells_olap):
        core_i = np.asarray(cells_core[i], dtype=np.int32).reshape(-1)
        olap_i = np.asarray(olap_i, dtype=np.int32).reshape(-1)

        if olap_i.size == 0:
            raw_weights.append(np.zeros(n_global, dtype=np.float64))
            support_masks.append(np.zeros(n_global, dtype=bool))
            continue

        distances = _cell_layer_distances(core_i, olap_i, cell_adjacency)
        cell_weights = np.zeros(olap_i.size, dtype=np.float64)
        denom = float(overlap_layers + 1)
        for j, c in enumerate(olap_i.tolist()):
            d = distances.get(int(c))
            if d is None:
                continue
            cell_weights[j] = max(0.0, 1.0 - float(d) / denom)

        w = np.zeros(n_global, dtype=np.float64)
        for c, cw in zip(olap_i.tolist(), cell_weights.tolist()):
            if cw <= 0.0:
                continue
            cell_dofs = np.asarray(V.dofmap.cell_dofs(int(c)), dtype=np.int32)
            w[cell_dofs] = np.maximum(w[cell_dofs], float(cw))

        core_dofs_i = np.asarray(dofs_core[i], dtype=np.int32).reshape(-1)
        w[core_dofs_i] = 1.0

        support = np.zeros(n_global, dtype=bool)
        dofs_i = np.asarray(dofs_olap[i], dtype=np.int32).reshape(-1)
        support[dofs_i] = True
        w[~support] = 0.0

        raw_weights.append(w)
        support_masks.append(support)

    total = np.zeros(n_global, dtype=np.float64)
    for w in raw_weights:
        total += w

    multiplicity = np.zeros(n_global, dtype=np.int32)
    for sup in support_masks:
        multiplicity += sup.astype(np.int32)

    needs_fallback = (multiplicity > 0) & (total <= 1.0e-15)
    if np.any(needs_fallback):
        idx = np.where(needs_fallback)[0]
        for w, sup in zip(raw_weights, support_masks):
            take = idx[sup[idx]]
            if take.size > 0:
                w[take] = 1.0
        total[:] = 0.0
        for w in raw_weights:
            total += w

    chi_full: list[np.ndarray] = []
    for w in raw_weights:
        chi = np.zeros(n_global, dtype=np.float64)
        nz = total > 1.0e-15
        chi[nz] = w[nz] / total[nz]
        chi_full.append(chi)

    ndofs_in_union = int(np.count_nonzero(multiplicity > 0))

    if ndofs_in_union > 0:
        stacked = np.sum(np.vstack(chi_full), axis=0)
        max_dev = float(np.max(np.abs(stacked[multiplicity > 0] - 1.0)))
    else:
        max_dev = 0.0

    if PARAMS.verbose:
        tprint(f"[pou] core-overlap ramp sum-to-one: max|sum_i chi_i - 1| = {max_dev:.3e}")

    return PartitionOfUnityData(
        chi_full=chi_full,
        multiplicity=multiplicity,
        ndofs_in_union=ndofs_in_union,
    )


def _build_from_partition_data(
    mesh,
    V: fem.FunctionSpace,
    *,
    decomposition: dict,
) -> PartitionOfUnityData:
    return _build_core_overlap_ramp_pou(mesh, V, decomposition=decomposition)


def build_partition_of_unity(
    mesh,
    V: fem.FunctionSpace,
    *,
    decomposition: dict,
) -> PartitionOfUnityData:
    mode = PARAMS.pou_mode

    if mode == "geometric_hats":
        out = _build_geometric_hats_pou(mesh, V, decomposition=decomposition)
    elif mode in ("core_overlap_ramp", "structured_hats", "from_partition_data"):
        out = _build_core_overlap_ramp_pou(mesh, V, decomposition=decomposition)
    else:
        raise ValueError(f"Unsupported pou_mode: {mode!r}")

    if PARAMS.verbose:
        if out.ndofs_in_union > 0:
            tprint(
                f"[pou] built: ndofs_in_union={out.ndofs_in_union}, "
                f"multiplicity min={int(out.multiplicity[out.multiplicity > 0].min())}, "
                f"max={int(out.multiplicity.max())}"
            )
        else:
            tprint("[pou] built: empty union")

    return out


def apply_partition_of_unity_operator_to_array(
    chi_i_full: np.ndarray,
    vec_global: np.ndarray,
    *,
    interpolate: bool = True,
) -> np.ndarray:
    chi_i_full = np.asarray(chi_i_full, dtype=np.float64)
    vec_global = np.asarray(vec_global)

    if chi_i_full.shape != vec_global.shape:
        raise ValueError(
            f"Shape mismatch in PoU application: chi shape={chi_i_full.shape}, "
            f"vec shape={vec_global.shape}"
        )

    out = chi_i_full * vec_global

    # For nodal Lagrange spaces, multiplication at dofs already corresponds to
    # nodal interpolation of chi_i * v, but we keep the flag to make the intent explicit
    # and to match the paper-style operator P_{h,i} = I_h(chi_i v).
    if interpolate:
        return np.array(out, copy=True)

    return out
