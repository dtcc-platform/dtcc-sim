from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh as dmesh
from dolfinx.fem import petsc as fem_petsc
import ufl

from .params import PARAMS


def tprint(msg: str) -> None:
    if MPI.COMM_WORLD.rank == 0:
        print(msg, flush=True)


def _to_csr(A: PETSc.Mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A.assemble()
    indptr, indices, data = A.getValuesCSR()
    return (
        np.asarray(indptr, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(data, dtype=np.complex128),
    )


def _submatrix_csr(A: PETSc.Mat, rows: np.ndarray, cols: np.ndarray):
    isrow = PETSc.IS().createGeneral(np.asarray(rows, dtype=np.int32), comm=PETSc.COMM_SELF)
    iscol = PETSc.IS().createGeneral(np.asarray(cols, dtype=np.int32), comm=PETSc.COMM_SELF)
    S = A.createSubMatrix(isrow, iscol)
    S.assemble()
    return _to_csr(S)


def _make_facet_meshtags(mesh: dmesh.Mesh, facets: np.ndarray, value: int = 1):
    fdim = mesh.topology.dim - 1
    facets = np.asarray(facets, dtype=np.int32)
    vals = np.full(len(facets), int(value), dtype=np.int32)
    return dmesh.meshtags(mesh, fdim, facets, vals)


def _constant_from_function_or_constant(value, mesh: dmesh.Mesh):
    if isinstance(value, fem.Function) or isinstance(value, fem.Constant):
        return value
    return fem.Constant(mesh, PETSc.ScalarType(value))


@dataclass
class LocalPatchOperators:
    patch_index: int

    dofs_core: np.ndarray
    dofs_olap: np.ndarray
    dofs_star: np.ndarray

    star_bdry_facets_physical: np.ndarray
    star_bdry_facets_artificial: np.ndarray
    star_bdry_facets_artificial_side: np.ndarray

    n_core: int
    n_olap: int
    n_star: int
    n_int: int

    star_local_interior: np.ndarray
    star_local_boundary: np.ndarray

    A_star_indptr: np.ndarray
    A_star_indices: np.ndarray
    A_star_data: np.ndarray

    A_star_k_indptr: np.ndarray
    A_star_k_indices: np.ndarray
    A_star_k_data: np.ndarray

    B_star_imp_indptr: np.ndarray
    B_star_imp_indices: np.ndarray
    B_star_imp_data: np.ndarray

    C_indptr: np.ndarray
    C_indices: np.ndarray
    C_data: np.ndarray

    star_to_global: np.ndarray
    olap_to_global: np.ndarray
    core_to_global: np.ndarray


def _assemble_form_matrix(form_expr) -> PETSc.Mat:
    A = fem_petsc.assemble_matrix(fem.form(form_expr))
    A.assemble()
    return A


def _facets_with_tags(facet_tags, tags: tuple[int, ...]) -> set[int]:
    if facet_tags is None or len(tags) == 0:
        return set()
    out: set[int] = set()
    for tag in tags:
        out.update(np.asarray(facet_tags.find(int(tag)), dtype=np.int32).tolist())
    return out


def _interior_local_indices(
    star_local_size: int,
    star_local_boundary: np.ndarray,
) -> np.ndarray:
    mark = np.ones(star_local_size, dtype=bool)
    mark[np.asarray(star_local_boundary, dtype=np.int32)] = False
    return np.where(mark)[0].astype(np.int32)


def _selection_constraint_matrix(
    n_star: int,
    star_local_interior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interior = np.asarray(star_local_interior, dtype=np.int32)
    m = int(interior.size)

    indptr = np.arange(m + 1, dtype=np.int32)
    indices = interior.copy()
    data = np.ones(m, dtype=np.complex128)
    return indptr, indices, data


def _assemble_global_patch_matrices(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    Acoef,
    Vcoef,
    beta,
    facet_tags=None,
    *,
    star_facets_physical: np.ndarray,
    star_facets_artificial: np.ndarray,
    star_facets_artificial_side: np.ndarray,
) -> tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat]:
    """
    Assemble:
      A      = ∫ A grad u · grad v
      A_k    = A + k^2 ∫ V^2 u v
      B_imp  = A - k^2 ∫ V^2 u v - i k * impedance terms on patch boundary

    Boundary contributions are assembled separately and then added with
    DIFFERENT_NONZERO_PATTERN, because they may introduce entries not present
    in the bulk matrix sparsity.
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh)
    k = float(PARAMS.k())

    Acoef = _constant_from_function_or_constant(Acoef, mesh)
    Vcoef = _constant_from_function_or_constant(Vcoef, mesh)
    beta = _constant_from_function_or_constant(beta, mesh)

    star_facets_physical = np.asarray(star_facets_physical, dtype=np.int32)
    if PARAMS.boundary_mode == "mixed":
        impedance_facets = _facets_with_tags(
            facet_tags, tuple(int(t) for t in PARAMS.impedance_facet_tags)
        )
        star_facets_physical_imp = np.array(
            [f for f in star_facets_physical.tolist() if int(f) in impedance_facets],
            dtype=np.int32,
        )
    elif PARAMS.boundary_mode == "impedance_only":
        if facet_tags is None:
            star_facets_physical_imp = star_facets_physical
        else:
            impedance_facets = _facets_with_tags(
                facet_tags, tuple(int(t) for t in PARAMS.impedance_facet_tags)
            )
            star_facets_physical_imp = np.array(
                [f for f in star_facets_physical.tolist() if int(f) in impedance_facets],
                dtype=np.int32,
            )
            if star_facets_physical_imp.size == 0 and star_facets_physical.size > 0:
                star_facets_physical_imp = star_facets_physical
    else:
        raise ValueError(f"Unsupported boundary_mode: {PARAMS.boundary_mode!r}")

    # Bulk matrices
    a_A = ufl.inner(Acoef * ufl.grad(u), ufl.grad(v)) * dx
    a_Ak = (
        ufl.inner(Acoef * ufl.grad(u), ufl.grad(v)) * dx
        + (k ** 2) * ufl.inner((Vcoef ** 2) * u, v) * dx
    )
    a_B_bulk = (
        ufl.inner(Acoef * ufl.grad(u), ufl.grad(v)) * dx
        - (k ** 2) * ufl.inner((Vcoef ** 2) * u, v) * dx
    )

    A = _assemble_form_matrix(a_A)
    Ak = _assemble_form_matrix(a_Ak)
    Bimp = _assemble_form_matrix(a_B_bulk)

    # Physical impedance contribution
    if len(star_facets_physical_imp) > 0:
        mt_phys = _make_facet_meshtags(mesh, star_facets_physical_imp, value=1)
        ds_phys = ufl.Measure("ds", domain=mesh, subdomain_data=mt_phys)
        a_phys = (-1j * k) * ufl.inner(beta * u, v) * ds_phys(1)
        Bphys = _assemble_form_matrix(a_phys)
        Bimp.axpy(
            1.0,
            Bphys,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )

    # Artificial impedance contribution, plus-side
    plus_facets = np.asarray(
        star_facets_artificial[np.asarray(star_facets_artificial_side) == 1],
        dtype=np.int32,
    )
    if len(plus_facets) > 0:
        mt_plus = _make_facet_meshtags(mesh, plus_facets, value=1)
        dS_plus = ufl.Measure("dS", domain=mesh, subdomain_data=mt_plus)
        a_plus = (-1j * k) * ufl.inner(Vcoef("+") * u("+"), v("+")) * dS_plus(1)
        Bplus = _assemble_form_matrix(a_plus)
        Bimp.axpy(
            1.0,
            Bplus,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )

    # Artificial impedance contribution, minus-side
    minus_facets = np.asarray(
        star_facets_artificial[np.asarray(star_facets_artificial_side) == -1],
        dtype=np.int32,
    )
    if len(minus_facets) > 0:
        mt_minus = _make_facet_meshtags(mesh, minus_facets, value=1)
        dS_minus = ufl.Measure("dS", domain=mesh, subdomain_data=mt_minus)
        a_minus = (-1j * k) * ufl.inner(Vcoef("-") * u("-"), v("-")) * dS_minus(1)
        Bminus = _assemble_form_matrix(a_minus)
        Bimp.axpy(
            1.0,
            Bminus,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )

    Bimp.assemble()
    return A, Ak, Bimp


def build_local_patch_operators(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    Acoef,
    Vcoef,
    beta,
    facet_tags=None,
    *,
    decomposition: dict,
    patch_index: int,
) -> LocalPatchOperators:
    dofs_core = np.asarray(decomposition["dofs_core"][patch_index], dtype=np.int32)
    dofs_olap = np.asarray(decomposition["dofs_olap"][patch_index], dtype=np.int32)
    dofs_star = np.asarray(decomposition["dofs_star"][patch_index], dtype=np.int32)

    star_bdry_facets_physical = np.asarray(
        decomposition["star_bdry_facets_physical"][patch_index], dtype=np.int32
    )
    star_bdry_facets_artificial = np.asarray(
        decomposition["star_bdry_facets_artificial"][patch_index], dtype=np.int32
    )
    star_bdry_facets_artificial_side = np.asarray(
        decomposition["star_bdry_facets_artificial_side"][patch_index], dtype=np.int32
    )

    n_core = int(len(dofs_core))
    n_olap = int(len(dofs_olap))
    n_star = int(len(dofs_star))

    if n_star == 0:
        raise RuntimeError(f"Patch {patch_index} has no star dofs")

    A_glob, Ak_glob, Bimp_glob = _assemble_global_patch_matrices(
        mesh,
        V,
        Acoef,
        Vcoef,
        beta,
        facet_tags=facet_tags,
        star_facets_physical=star_bdry_facets_physical,
        star_facets_artificial=star_bdry_facets_artificial,
        star_facets_artificial_side=star_bdry_facets_artificial_side,
    )

    A_star_indptr, A_star_indices, A_star_data = _submatrix_csr(A_glob, dofs_star, dofs_star)
    A_star_k_indptr, A_star_k_indices, A_star_k_data = _submatrix_csr(Ak_glob, dofs_star, dofs_star)
    B_star_imp_indptr, B_star_imp_indices, B_star_imp_data = _submatrix_csr(Bimp_glob, dofs_star, dofs_star)

    fdim = mesh.topology.dim - 1
    boundary_dofs_art = np.zeros(0, dtype=np.int32)

    if len(star_bdry_facets_artificial) > 0:
        boundary_dofs_art = np.asarray(
            fem.locate_dofs_topological(V, fdim, star_bdry_facets_artificial),
            dtype=np.int32,
        )

    boundary_global = np.unique(
        boundary_dofs_art
        if boundary_dofs_art.size > 0
        else np.zeros(0, dtype=np.int32)
    )

    star_global_to_local = {int(g): j for j, g in enumerate(dofs_star.tolist())}
    star_local_boundary = np.array(
        [star_global_to_local[int(g)] for g in boundary_global.tolist() if int(g) in star_global_to_local],
        dtype=np.int32,
    )
    star_local_boundary = np.unique(star_local_boundary)

    star_local_interior = _interior_local_indices(n_star, star_local_boundary)
    n_int = int(star_local_interior.size)

    if n_int <= 0:
        raise RuntimeError(f"Patch {patch_index} has no interior star dofs")

    C_indptr, C_indices, C_data = _selection_constraint_matrix(n_star, star_local_interior)

    if PARAMS.verbose:
        tprint(
            f"[local_operators] patch={patch_index:03d} "
            f"n_core={n_core} n_olap={n_olap} n_star={n_star} n_int={n_int} "
            f"star_bdry(phys/art)=({len(star_bdry_facets_physical)}/{len(star_bdry_facets_artificial)})"
        )

    return LocalPatchOperators(
        patch_index=int(patch_index),
        dofs_core=dofs_core,
        dofs_olap=dofs_olap,
        dofs_star=dofs_star,
        star_bdry_facets_physical=star_bdry_facets_physical,
        star_bdry_facets_artificial=star_bdry_facets_artificial,
        star_bdry_facets_artificial_side=star_bdry_facets_artificial_side,
        n_core=n_core,
        n_olap=n_olap,
        n_star=n_star,
        n_int=n_int,
        star_local_interior=star_local_interior,
        star_local_boundary=star_local_boundary,
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
        star_to_global=dofs_star,
        olap_to_global=dofs_olap,
        core_to_global=dofs_core,
    )


def build_all_local_patch_operators(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    Acoef,
    Vcoef,
    beta,
    facet_tags=None,
    *,
    decomposition: dict,
) -> list[LocalPatchOperators]:
    M = int(np.asarray(decomposition["M"]).reshape(-1)[0])
    out: list[LocalPatchOperators] = []

    for i in range(M):
        out.append(
            build_local_patch_operators(
                mesh,
                V,
                Acoef,
                Vcoef,
                beta,
                facet_tags=facet_tags,
                decomposition=decomposition,
                patch_index=i,
            )
        )
    return out


def serialize_local_patch_operators(ops: list[LocalPatchOperators]) -> dict:
    def pack(attr: str):
        return np.array([getattr(op, attr) for op in ops], dtype=object)

    payload = {
        "patch_index": np.array([op.patch_index for op in ops], dtype=np.int32),
        "dofs_core": pack("dofs_core"),
        "dofs_olap": pack("dofs_olap"),
        "dofs_star": pack("dofs_star"),
        "star_bdry_facets_physical": pack("star_bdry_facets_physical"),
        "star_bdry_facets_artificial": pack("star_bdry_facets_artificial"),
        "star_bdry_facets_artificial_side": pack("star_bdry_facets_artificial_side"),
        "n_core": np.array([op.n_core for op in ops], dtype=np.int32),
        "n_olap": np.array([op.n_olap for op in ops], dtype=np.int32),
        "n_star": np.array([op.n_star for op in ops], dtype=np.int32),
        "n_int": np.array([op.n_int for op in ops], dtype=np.int32),
        "star_local_interior": pack("star_local_interior"),
        "star_local_boundary": pack("star_local_boundary"),
        "A_star_indptr": pack("A_star_indptr"),
        "A_star_indices": pack("A_star_indices"),
        "A_star_data": pack("A_star_data"),
        "A_star_k_indptr": pack("A_star_k_indptr"),
        "A_star_k_indices": pack("A_star_k_indices"),
        "A_star_k_data": pack("A_star_k_data"),
        "B_star_imp_indptr": pack("B_star_imp_indptr"),
        "B_star_imp_indices": pack("B_star_imp_indices"),
        "B_star_imp_data": pack("B_star_imp_data"),
        "C_indptr": pack("C_indptr"),
        "C_indices": pack("C_indices"),
        "C_data": pack("C_data"),
        "star_to_global": pack("star_to_global"),
        "olap_to_global": pack("olap_to_global"),
        "core_to_global": pack("core_to_global"),
    }
    return payload
