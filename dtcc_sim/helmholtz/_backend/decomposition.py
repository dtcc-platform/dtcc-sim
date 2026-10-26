from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
import ufl

from .params import PARAMS


def tprint(msg: str) -> None:
    if MPI.COMM_WORLD.rank == 0:
        print(msg, flush=True)


def _tdim(mesh: dmesh.Mesh) -> int:
    return mesh.topology.dim


def _fdim(mesh: dmesh.Mesh) -> int:
    return mesh.topology.dim - 1


def _num_local_cells(mesh: dmesh.Mesh) -> int:
    return int(mesh.topology.index_map(_tdim(mesh)).size_local)


def _unique_sorted_int(a) -> np.ndarray:
    out = np.unique(np.asarray(a, dtype=np.int32))
    out.sort()
    return out.astype(np.int32, copy=False)


def _cell_dofs_unique(V: fem.FunctionSpace, cells: np.ndarray) -> np.ndarray:
    dm = V.dofmap
    dofs: list[int] = []
    for c in np.asarray(cells, dtype=np.int32).tolist():
        dofs.extend(dm.cell_dofs(int(c)))
    if not dofs:
        return np.zeros(0, dtype=np.int32)
    return _unique_sorted_int(dofs)


def build_cell_adjacency(mesh: dmesh.Mesh) -> list[np.ndarray]:
    """
    Build the cell-neighbor graph:
    two cells are neighbors if they share a facet.
    """
    tdim = _tdim(mesh)
    fdim = _fdim(mesh)

    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)

    c2f = mesh.topology.connectivity(tdim, fdim)
    f2c = mesh.topology.connectivity(fdim, tdim)

    n_cells = _num_local_cells(mesh)
    neighbors: list[set[int]] = [set() for _ in range(n_cells)]

    for c in range(n_cells):
        for f in c2f.links(c):
            for cc in f2c.links(f):
                cc = int(cc)
                if cc != c:
                    neighbors[c].add(cc)

    return [np.fromiter(sorted(s), dtype=np.int32) for s in neighbors]


def expand_by_layers(
    seeds: np.ndarray,
    neighbors: list[np.ndarray],
    layers: int,
) -> np.ndarray:
    """
    Expand a seed set of cells by a given number of adjacency layers.
    """
    seeds = _unique_sorted_int(seeds)
    if seeds.size == 0:
        return seeds.copy()
    if int(layers) <= 0:
        return seeds.copy()

    current = set(int(c) for c in seeds.tolist())
    frontier = set(current)

    for _ in range(int(layers)):
        new_frontier: set[int] = set()
        for c in frontier:
            for nb in neighbors[c]:
                nb = int(nb)
                if nb not in current:
                    current.add(nb)
                    new_frontier.add(nb)
        frontier = new_frontier
        if not frontier:
            break

    return np.array(sorted(current), dtype=np.int32)


def _cell_barycenters(mesh: dmesh.Mesh) -> np.ndarray:
    """
    Compute cell barycenters from geometry coordinates.

    Important for dolfinx:
    mesh.geometry.x may have 3 columns even for 2D meshes, so we explicitly
    slice to the geometric dimension.
    """
    gdim = mesh.geometry.dim
    x = np.asarray(mesh.geometry.x[:, :gdim], dtype=np.float64)
    dofmap = mesh.geometry.dofmap
    n_cells = _num_local_cells(mesh)

    out = np.zeros((n_cells, gdim), dtype=np.float64)
    for c in range(n_cells):
        verts = dofmap[c]
        out[c, :] = np.mean(x[verts], axis=0)
    return out


def structured_partition_cells(
    mesh: dmesh.Mesh,
    *,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, int]:
    """
    Partition cells by barycenter into a structured nx x ny x nz layout.

    This is useful for simple benchmark geometries only.
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must all be positive")

    bc = _cell_barycenters(mesh)
    gdim = bc.shape[1]

    mins = bc.min(axis=0)
    maxs = bc.max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-15)

    xscaled = (bc[:, 0] - mins[0]) / span[0]
    yscaled = (bc[:, 1] - mins[1]) / span[1] if gdim >= 2 else np.zeros(len(bc))
    zscaled = (bc[:, 2] - mins[2]) / span[2] if gdim >= 3 else np.zeros(len(bc))

    ix = np.minimum((xscaled * nx).astype(np.int32), nx - 1)
    iy = np.minimum((yscaled * ny).astype(np.int32), ny - 1)
    iz = np.minimum((zscaled * nz).astype(np.int32), nz - 1)

    part = ix + nx * (iy + ny * iz)
    M = int(nx * ny * nz)

    if part.min() < 0 or part.max() >= M:
        raise RuntimeError("Structured partition labels out of range")

    return part.astype(np.int32), M


def metis_partition_cells(
    neighbors: list[np.ndarray],
    nparts: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Partition the cell adjacency graph with Metis.
    """
    try:
        import pymetis
    except ImportError as e:
        raise ImportError(
            "pymetis is required for partitioner='metis'. "
            "Install it with: pip install pymetis"
        ) from e

    if nparts <= 0:
        raise ValueError("nparts must be positive")

    adjacency = [nb.tolist() for nb in neighbors]
    options = pymetis.Options(seed=int(seed))
    edgecuts, parts = pymetis.part_graph(
        int(nparts), adjacency=adjacency, options=options
    )
    part = np.asarray(parts, dtype=np.int32)

    if part.ndim != 1 or len(part) != len(neighbors):
        raise RuntimeError("METIS returned invalid partition array shape")
    if part.min() < 0 or part.max() >= nparts:
        raise RuntimeError("METIS returned labels out of range")

    tprint(
        f"[decomposition] Metis partitioned local cells into M={nparts} "
        f"parts (edgecuts={edgecuts})"
    )
    return part, int(nparts)


def artificial_facet_sides(
    mesh: dmesh.Mesh,
    star_cells: np.ndarray,
    artificial_facets: np.ndarray,
) -> np.ndarray:
    """
    Determine whether the '+' or '-' side of an interior facet lies in omega_i^*.

    With a complex PETSc backend, assemble_scalar can return a complex scalar even
    for a real-valued indicator. Compare the real parts.
    """
    artificial_facets = np.asarray(artificial_facets, dtype=np.int32)
    if artificial_facets.size == 0:
        return np.zeros(0, dtype=np.int8)

    star_set = set(np.asarray(star_cells, dtype=np.int32).tolist())

    tdim = _tdim(mesh)
    fdim = _fdim(mesh)

    mesh.topology.create_connectivity(fdim, tdim)
    f2c = mesh.topology.connectivity(fdim, tdim)

    Q = fem.functionspace(mesh, ("DG", 0))
    q = fem.Function(Q)
    q.x.array[:] = 0.0

    sides = np.empty(artificial_facets.size, dtype=np.int8)
    tol = 1.0e-12

    for j, f in enumerate(artificial_facets.tolist()):
        cells = f2c.links(int(f))
        if len(cells) != 2:
            raise RuntimeError(
                "Artificial facet is expected to be interior and have exactly two incident cells"
            )

        c0 = int(cells[0])
        c1 = int(cells[1])

        in0 = c0 in star_set
        in1 = c1 in star_set

        if in0 == in1:
            raise RuntimeError("Artificial boundary facet classification is inconsistent")

        c_in = c0 if in0 else c1

        q.x.array[:] = 0.0
        dof_in = Q.dofmap.cell_dofs(c_in)[0]
        q.x.array[dof_in] = 1.0

        mt = dmesh.meshtags(
            mesh,
            fdim,
            np.array([int(f)], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=mt)

        Iplus = fem.assemble_scalar(fem.form(q("+") * dS(1)))
        Iminus = fem.assemble_scalar(fem.form(q("-") * dS(1)))

        rp = float(np.real(Iplus))
        rm = float(np.real(Iminus))

        if rp > rm + tol:
            sides[j] = 1
        elif rm > rp + tol:
            sides[j] = -1
        else:
            raise RuntimeError(
                f"Could not determine artificial facet side orientation on facet {f}: "
                f"Iplus={Iplus!r}, Iminus={Iminus!r}"
            )

    return sides


def star_boundary_facets(
    mesh: dmesh.Mesh,
    star_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boundary facets of omega_i^*:
        - physical boundary facets
        - artificial boundary facets (interior interfaces)
        - side orientation for artificial facets
    """
    tdim = _tdim(mesh)
    fdim = _fdim(mesh)

    mesh.topology.create_connectivity(tdim, fdim)
    c2f = mesh.topology.connectivity(tdim, fdim)

    counts: dict[int, int] = {}
    for c in np.asarray(star_cells, dtype=np.int32).tolist():
        for f in c2f.links(int(c)):
            ff = int(f)
            counts[ff] = counts.get(ff, 0) + 1

    boundary_facets = np.array(
        sorted([f for f, ct in counts.items() if ct == 1]),
        dtype=np.int32,
    )

    mesh.topology.create_connectivity(fdim, tdim)
    ext = np.asarray(dmesh.exterior_facet_indices(mesh.topology), dtype=np.int32)
    ext_set = set(ext.tolist())

    is_physical = np.array([int(f) in ext_set for f in boundary_facets], dtype=bool)
    phys = boundary_facets[is_physical]
    artificial = boundary_facets[~is_physical]

    art_side = artificial_facet_sides(
        mesh,
        np.asarray(star_cells, dtype=np.int32),
        artificial,
    )

    return phys, artificial, art_side


@dataclass
class SubdomainData:
    index: int
    cells_core: np.ndarray
    cells_olap: np.ndarray
    cells_star: np.ndarray
    dofs_core: np.ndarray
    dofs_olap: np.ndarray
    dofs_star: np.ndarray
    star_bdry_facets_physical: np.ndarray
    star_bdry_facets_artificial: np.ndarray
    star_bdry_facets_artificial_side: np.ndarray


def build_decomposition(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    *,
    verbose: bool = True,
) -> dict:
    """
    Build decomposition data for the two-level method.
    """
    neighbors = build_cell_adjacency(mesh)

    if PARAMS.partitioner == "structured":
        part, M = structured_partition_cells(
            mesh,
            nx=int(PARAMS.nx_sub),
            ny=int(PARAMS.ny_sub),
            nz=int(PARAMS.nz_sub),
        )
        if verbose:
            tprint(
                f"[decomposition] structured partition: "
                f"{PARAMS.nx_sub} x {PARAMS.ny_sub} x {PARAMS.nz_sub} -> M={M}"
            )

    elif PARAMS.partitioner == "metis":
        if PARAMS.n_subdomains is None or int(PARAMS.n_subdomains) <= 0:
            raise ValueError(
                "For partitioner='metis', PARAMS.n_subdomains must be a positive integer"
            )
        part, M = metis_partition_cells(
            neighbors,
            int(PARAMS.n_subdomains),
            seed=int(PARAMS.metis_seed),
        )
        if verbose:
            tprint(f"[decomposition] metis partition -> M={M}")

    else:
        raise ValueError(f"Unsupported partitioner: {PARAMS.partitioner!r}")

    subdomains: list[SubdomainData] = []

    for i in range(M):
        core_i = np.where(part == i)[0].astype(np.int32)
        if core_i.size == 0:
            raise RuntimeError(f"Partition produced empty non-overlapping subdomain {i}")

        olap_i = expand_by_layers(core_i, neighbors, int(PARAMS.overlap_layers))
        star_i = expand_by_layers(olap_i, neighbors, int(PARAMS.oversample_layers))

        dofs_core_i = _cell_dofs_unique(V, core_i)
        dofs_olap_i = _cell_dofs_unique(V, olap_i)
        dofs_star_i = _cell_dofs_unique(V, star_i)

        phys_i, art_i, art_side_i = star_boundary_facets(mesh, star_i)

        subdomains.append(
            SubdomainData(
                index=i,
                cells_core=core_i,
                cells_olap=olap_i,
                cells_star=star_i,
                dofs_core=dofs_core_i,
                dofs_olap=dofs_olap_i,
                dofs_star=dofs_star_i,
                star_bdry_facets_physical=phys_i,
                star_bdry_facets_artificial=art_i,
                star_bdry_facets_artificial_side=art_side_i,
            )
        )

        if verbose:
            tprint(
                f"[decomposition] sub={i:03d}  "
                f"core={len(core_i)}  olap={len(olap_i)}  star={len(star_i)}  "
                f"dofs(core/olap/star)=({len(dofs_core_i)}/{len(dofs_olap_i)}/{len(dofs_star_i)})  "
                f"star-bdry(phys/art)=({len(phys_i)}/{len(art_i)})"
            )

    payload = {
        "partitioner": np.array([PARAMS.partitioner], dtype=object),
        "M": np.array([int(M)], dtype=np.int32),
        "overlap_layers": np.array([int(PARAMS.overlap_layers)], dtype=np.int32),
        "oversample_layers": np.array([int(PARAMS.oversample_layers)], dtype=np.int32),
        "cell_partition": np.asarray(part, dtype=np.int32),
        "cell_adjacency": np.array(
            [np.asarray(nb, dtype=np.int32) for nb in neighbors],
            dtype=object,
        ),
        "cells_core": np.array([sd.cells_core for sd in subdomains], dtype=object),
        "cells_olap": np.array([sd.cells_olap for sd in subdomains], dtype=object),
        "cells_star": np.array([sd.cells_star for sd in subdomains], dtype=object),
        "dofs_core": np.array([sd.dofs_core for sd in subdomains], dtype=object),
        "dofs_olap": np.array([sd.dofs_olap for sd in subdomains], dtype=object),
        "dofs_star": np.array([sd.dofs_star for sd in subdomains], dtype=object),
        "star_bdry_facets_physical": np.array(
            [sd.star_bdry_facets_physical for sd in subdomains],
            dtype=object,
        ),
        "star_bdry_facets_artificial": np.array(
            [sd.star_bdry_facets_artificial for sd in subdomains],
            dtype=object,
        ),
        "star_bdry_facets_artificial_side": np.array(
            [sd.star_bdry_facets_artificial_side for sd in subdomains],
            dtype=object,
        ),
    }

    return payload
