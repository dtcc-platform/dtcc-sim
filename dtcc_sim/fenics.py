"""
fenics.py — A minimal, legacy-style wrapper for FEniCSx (dolfinx)

This module provides a "classic FEniCS" convenience layer on top of FEniCSx,
to enable classic (pretty) FEniCS programming, in contrast to the verbose
(ugly) FEniCSx style.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy import isclose as near

from mpi4py import MPI

import basix
import dolfinx
from dolfinx.io import XDMFFile

# FEM / PETSc
from dolfinx.fem import Constant, Expression, Function
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType

# UFL convenience imports (flat namespace)
from ufl import (  # noqa: F401
    TrialFunction,
    TestFunction,
    TrialFunctions,
    TestFunctions,
    SpatialCoordinate,
    CellDiameter,
    FacetNormal,
    Measure,
    as_vector,
    dot,
    inner,
    grad,
    curl,
    div,
    exp,
    sin,
    cos,
    sqrt,
    dx,
    ds,
)

# Boundary condition helpers
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import dirichletbc, locate_dofs_topological

# Assembly helpers
from dolfinx.fem import form as _fem_form
from dolfinx.fem import petsc as _fem_petsc

# Logging
from dolfinx.log import LogLevel, log, set_log_level


# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------

__all__: List[str] = []

__all__.extend(
    [
        # MPI / numpy
        "MPI",
        "np",
        "near",
        # logging
        "LogLevel",
        "set_log_level",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "info",
        "warning",
        "error",
        # IO
        "XDMFFile",
        # FEM core
        "ScalarType",
        "Constant",
        "Function",
        "Expression",
        "LinearProblem",
        "FunctionSpace",
        # UFL flat namespace
        "TrialFunction",
        "TestFunction",
        "TrialFunctions",
        "TestFunctions",
        "SpatialCoordinate",
        "CellDiameter",
        "FacetNormal",
        "Measure",
        "dx",
        "ds",
        "dot",
        "inner",
        "grad",
        "curl",
        "div",
        "exp",
        "sin",
        "cos",
        "sqrt",
        "as_vector",
        # BCs / measures
        "DirichletBC",
        "NeumannBC",
        # Assembly & solve
        "assemble_matrix",
        "assemble_vector",
        "assemble_scalar",
        "assemble",
        "interpolate",
        "project",
        "solve",
        # Mesh I/O / utilities
        "load_mesh",
        "load_mesh_with_markers",
        "cell_markers_to_facet_markers",
        "BoxMesh",
        "bounds",
        "offset_to_origin",
        # Optional viz
        "plot",
    ]
)

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

DEBUG = LogLevel.DEBUG
INFO = LogLevel.INFO
WARNING = LogLevel.WARNING
ERROR = LogLevel.ERROR


def info(message: str) -> None:
    """Log an informational message via dolfinx.log."""
    log(INFO, message)


def warning(message: str) -> None:
    """Log a warning message via dolfinx.log."""
    log(WARNING, message)


def error(message: str) -> None:
    """Log an error message via dolfinx.log."""
    log(ERROR, message)


# -----------------------------------------------------------------------------
# Function spaces
# -----------------------------------------------------------------------------

ElementSpec = Union[str, Tuple[Tuple[str, int], ...]]


def FunctionSpace(
    mesh: dolfinx.mesh.Mesh,
    element: ElementSpec,
    degree: Optional[int] = None,
    dim: Optional[int] = None,
) -> dolfinx.fem.FunctionSpace:
    """
    Create a function space, mimicking legacy `FunctionSpace(mesh, "Lagrange", 1)`.

    Parameters
    ----------
    mesh:
        The dolfinx mesh.
    element:
        Either:
        - a string element family name, e.g. "Lagrange"
        - a tuple of (family, degree) pairs for mixed elements, e.g.
          (("Lagrange", 1), ("Lagrange", 1)).
    degree:
        Polynomial degree (required if `element` is a string).
    dim:
        Optional vector dimension. If provided, creates a vector element with shape=(dim,).

    Returns
    -------
    dolfinx.fem.FunctionSpace
        The created function space.
    """
    if mesh is None:
        raise ValueError("FunctionSpace: mesh must not be None.")

    # Mixed elements
    if isinstance(element, tuple):
        if degree is not None:
            raise ValueError("FunctionSpace: degree must be None for mixed elements.")
        elements = [
            basix.ufl.element(family, mesh.basix_cell(), deg)
            for (family, deg) in element
        ]
        basix_element = basix.ufl.mixed_element(elements)
        return dolfinx.fem.functionspace(mesh, basix_element)

    # Single element
    if not isinstance(element, str):
        raise TypeError(
            "FunctionSpace: element must be a string (family) or a tuple for mixed elements."
        )
    if degree is None:
        raise ValueError("FunctionSpace: degree must be specified for string elements.")

    if dim is None:
        basix_element = basix.ufl.element(element, mesh.basix_cell(), degree)
    else:
        basix_element = basix.ufl.element(
            element, mesh.basix_cell(), degree, shape=(dim,)
        )

    return dolfinx.fem.functionspace(mesh, basix_element)


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------


def _as_int32_array(x: Sequence[int]) -> np.ndarray:
    return np.asarray(x, dtype=np.int32)


def DirichletBC(
    V: dolfinx.fem.FunctionSpace,
    value: Union[float, int, ScalarType],
    condition: Optional[Callable[..., np.ndarray]] = None,
    *,
    markers: Optional[dolfinx.mesh.MeshTags] = None,
    marker_value: Optional[Union[int, Sequence[int]]] = None,
    dofs: Optional[Sequence[int]] = None,
) -> dolfinx.fem.bcs.DirichletBC:
    """
    Create a Dirichlet boundary condition.

    This is a convenience wrapper to support legacy patterns:
    - `DirichletBC(V, value, condition=...)`
    - `DirichletBC(V, value, markers=facet_tags, marker_value=3)`
    - `DirichletBC(V, value, dofs=[0, 1, 2])`

    Parameters
    ----------
    V:
        Function space (or subspace via `W.sub(i)`).
    value:
        Boundary value (scalar).
    condition:
        A predicate used to locate boundary facets. In classic FEniCS this is often
        `lambda x, on_boundary: on_boundary and ...`. In dolfinx, the predicate is
        called with coordinates only. This wrapper will attempt both calling conventions.
    markers:
        MeshTags for facets.
    marker_value:
        Marker value(s) to select from `markers`.
    dofs:
        Explicit dof indices.

    Returns
    -------
    dolfinx.fem.bcs.DirichletBC
        A dolfinx DirichletBC.
    """
    mesh = V.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Determine dofs
    dof_arr: Optional[np.ndarray] = None

    if condition is not None:
        # Try both predicate conventions: cond(x) and cond(x, on_boundary)
        def _wrapped_marker(x: np.ndarray) -> np.ndarray:
            try:
                return np.asarray(condition(x), dtype=bool)  # type: ignore[misc]
            except TypeError:
                # Legacy signature: condition(x, on_boundary)
                return np.asarray(condition(x, True), dtype=bool)  # type: ignore[misc]

        facets = locate_entities_boundary(mesh, dim=fdim, marker=_wrapped_marker)
        dof_arr = locate_dofs_topological(V, fdim, facets)

    elif markers is not None:
        if marker_value is None:
            raise ValueError("DirichletBC: marker_value must be provided with markers.")

        if isinstance(marker_value, (list, tuple, np.ndarray)):
            # Union of multiple marker values
            vals = np.asarray(marker_value, dtype=markers.values.dtype)
            facets = markers.indices[np.isin(markers.values, vals)]
        else:
            facets = markers.find(int(marker_value))
        dof_arr = locate_dofs_topological(V, fdim, facets)

    elif dofs is not None:
        dof_arr = _as_int32_array(dofs)

    else:
        raise ValueError(
            "DirichletBC: must provide one of condition=..., markers+marker_value=..., or dofs=..."
        )

    if dof_arr is None:
        raise RuntimeError("DirichletBC: failed to determine constrained dofs.")

    info(f"Creating DirichletBC with value {value} on {len(dof_arr)} dofs.")
    return dirichletbc(value=ScalarType(value), dofs=dof_arr, V=V)


def NeumannBC(
    mesh: dolfinx.mesh.Mesh,
    condition: Optional[Callable[..., np.ndarray]] = None,
    *,
    markers: Optional[dolfinx.mesh.MeshTags] = None,
    marker_value: Optional[Union[int, Sequence[int]]] = None,
) -> Measure:
    """
    Create a Neumann boundary "condition" as a UFL measure `ds(1)` over selected facets.

    This mirrors a common legacy usage pattern where `ds = NeumannBC(...)` and then
    forms use `... * ds`.

    Parameters
    ----------
    mesh:
        The mesh.
    condition:
        Boundary predicate (see DirichletBC for calling convention compatibility).
    markers:
        MeshTags for facets.
    marker_value:
        Marker value(s) to select from `markers`.

    Returns
    -------
    ufl.Measure
        A measure restricted to the chosen facets.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1

    if condition is not None:

        def _wrapped_marker(x: np.ndarray) -> np.ndarray:
            try:
                return np.asarray(condition(x), dtype=bool)  # type: ignore[misc]
            except TypeError:
                return np.asarray(condition(x, True), dtype=bool)  # type: ignore[misc]

        facets = locate_entities_boundary(mesh, dim=fdim, marker=_wrapped_marker)
        tag_val = 1
        tag_vals = np.full(facets.size, tag_val, dtype=np.int32)
        facet_tags = meshtags(mesh, fdim, facets, tag_vals)

    elif markers is not None:
        if marker_value is None:
            raise ValueError("NeumannBC: marker_value must be provided with markers.")

        if isinstance(marker_value, (list, tuple, np.ndarray)):
            vals = np.asarray(marker_value, dtype=markers.values.dtype)
            facets = markers.indices[np.isin(markers.values, vals)]
        else:
            facets = markers.find(int(marker_value))

        tag_val = 1
        tag_vals = np.full(facets.size, tag_val, dtype=np.int32)
        facet_tags = meshtags(mesh, fdim, facets, tag_vals)

    else:
        raise ValueError(
            "NeumannBC: must provide condition=... or markers+marker_value=..."
        )

    ds_local = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    return ds_local(tag_val)


# -----------------------------------------------------------------------------
# Assembly helpers
# -----------------------------------------------------------------------------


def assemble_matrix(a: Any, bcs: Optional[Sequence[Any]] = None) -> Any:
    """
    Assemble a PETSc matrix from a bilinear form.

    Parameters
    ----------
    a:
        UFL bilinear form.
    bcs:
        Optional boundary conditions.

    Returns
    -------
    PETSc.Mat
        Assembled matrix (not assembled unless you call A.assemble()).
    """
    bc_list = list(bcs) if bcs is not None else []
    return _fem_petsc.assemble_matrix(_fem_form(a), bcs=bc_list)


def assemble_vector(L: Any) -> Any:
    """
    Assemble a PETSc vector from a linear form.

    Parameters
    ----------
    L:
        UFL linear form.

    Returns
    -------
    PETSc.Vec
        Assembled vector.
    """
    return _fem_petsc.assemble_vector(_fem_form(L))


def assemble_scalar(M: Any) -> ScalarType:
    """
    Assemble a scalar (rank-0) form.

    Parameters
    ----------
    M:
        UFL rank-0 form.

    Returns
    -------
    ScalarType
        Assembled scalar.
    """
    return _fem_petsc.assemble_scalar(_fem_form(M))


def assemble(form: Any) -> Any:
    """
    Legacy-style `assemble(...)` dispatcher.

    - rank-0 form -> scalar
    - rank-1 form -> vector
    - rank-2 form -> matrix

    Parameters
    ----------
    form:
        UFL form.

    Returns
    -------
    scalar / PETSc.Vec / PETSc.Mat
    """
    try:
        rank = len(form.arguments())
    except Exception:
        raise TypeError("assemble: expected a UFL form with .arguments().")

    if rank == 0:
        return assemble_scalar(form)
    if rank == 1:
        return assemble_vector(form)
    if rank == 2:
        return assemble_matrix(form)
    raise ValueError(f"assemble: unsupported form rank {rank}.")


# -----------------------------------------------------------------------------
# Interpolation / projection
# -----------------------------------------------------------------------------


def _all_cell_indices(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    """Return all (local + ghost) cell indices as int32."""
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n = imap.size_local + imap.num_ghosts
    return np.arange(n, dtype=np.int32)


def _is_ufl_expression(obj: Any) -> bool:
    """Return True if `obj` is a UFL expression (or looks like one)."""
    try:
        from ufl.core.expr import Expr  # type: ignore

        return isinstance(obj, Expr)
    except Exception:
        return hasattr(obj, "ufl_shape") or hasattr(obj, "ufl_operands")


def interpolate(f: Any, V: dolfinx.fem.FunctionSpace) -> Function:
    """Interpolate *f* into the function space *V*.

    This function returns a new ``dolfinx.fem.Function`` in *V*.

    Parameters
    ----------
    f:
        Input to interpolate. Supported inputs:
        - ``dolfinx.fem.Expression``
        - ``dolfinx.fem.Function``
        - UFL expression (wrapped internally as a ``dolfinx.fem.Expression``)
        - Python callable ``f(x)`` suitable for DOLFINx interpolation
        - scalar numbers (treated as a constant)
    V:
        Target function space.

    Returns
    -------
    dolfinx.fem.Function
        Interpolated function.
    """
    if V is None:
        raise ValueError("interpolate: V must not be None.")

    info(f"Interpolating into {V}")
    u = Function(V)

    # Cells must be int32 for the underlying C++ interpolation bindings
    cells = _all_cell_indices(V.mesh)

    # Promote Python scalars to a Constant on this mesh
    if isinstance(f, (int, float, complex, np.number)):
        f = Constant(V.mesh, ScalarType(f))

    # Expression interpolation
    if isinstance(f, Expression):
        u.interpolate(f, cells0=cells, cells1=cells)
        return u

    # Function-to-function interpolation
    if isinstance(f, Function):
        try:
            u.interpolate(f)
        except TypeError:
            u.interpolate(f, cells0=cells, cells1=cells)
        return u

    # UFL expression interpolation (UFL expressions are also callable; handle them first)
    if _is_ufl_expression(f):
        expr = Expression(f, V.element.interpolation_points())
        u.interpolate(expr, cells0=cells, cells1=cells)
        return u

    # Callable interpolation
    if callable(f):
        u.interpolate(f)
        return u

    raise TypeError(
        "interpolate: unsupported input type. Expected Function, Expression, callable, UFL expression, or scalar."
    )


def project(
    f: Any,
    V: dolfinx.fem.FunctionSpace,
    *,
    bcs: Optional[Sequence[Any]] = None,
    petsc_options: Optional[Mapping[str, Any]] = None,
) -> Function:
    """
    L2 projection of `f` into `V` using a default Krylov/AMG solver.

    Parameters
    ----------
    f:
        UFL expression or dolfinx Function.
    V:
        Target space.
    bcs:
        Optional Dirichlet BCs.
    petsc_options:
        Optional PETSc options dict.

    Returns
    -------
    dolfinx.fem.Function
        Projected function in V.
    """
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-8,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
        }

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx
    L = inner(f, v) * dx

    uh = solve(a == L, bcs=bcs, petsc_options=petsc_options)
    return uh


# -----------------------------------------------------------------------------
# Solve helper
# -----------------------------------------------------------------------------


def _extract_lhs_rhs(eq: Any) -> Tuple[Any, Any]:
    """
    Extract (lhs, rhs) from a UFL equation in a version-tolerant way.
    """
    if isinstance(eq, bool):
        # This is a common pitfall if `==` did a Python equality comparison.
        raise TypeError(
            "solve: received a Python bool from 'a == L'. "
            "This indicates '==' did not build a UFL Equation (e.g. you compared "
            "compiled dolfinx forms or non-UFL objects). Pass UFL forms directly "
            "or call solve(a, L, ...) explicitly."
        )

    # Preferred: use UFL formoperators (works across UFL versions)
    try:
        from ufl.formoperators import lhs as _lhs, rhs as _rhs  # type: ignore

        return _lhs(eq), _rhs(eq)
    except Exception:
        pass

    # Fallback: attributes or callables
    if hasattr(eq, "lhs") and hasattr(eq, "rhs"):
        l = eq.lhs() if callable(eq.lhs) else eq.lhs
        r = eq.rhs() if callable(eq.rhs) else eq.rhs
        return l, r

    raise TypeError("solve: could not interpret first argument as a UFL equation.")


def _infer_solution_space_from_bilinear_form(a: Any) -> dolfinx.fem.FunctionSpace:
    """
    Infer the trial space from a bilinear UFL form `a`.
    """
    try:
        args = a.arguments()
    except Exception as e:
        raise TypeError(
            "solve: could not access form arguments to infer function space."
        ) from e

    if len(args) == 0:
        raise ValueError("solve: bilinear form has no arguments.")
    if len(args) == 1:
        # Some forms may be linear; fall back to that space
        return args[0].ufl_function_space()
    # For bilinear forms, convention is (test, trial)
    return args[-1].ufl_function_space()


def solve(
    equation_or_a: Any,
    L: Any = None,
    u: Optional[Function] = None,
    *,
    bcs: Optional[Sequence[Any]] = None,
    petsc_options: Optional[Mapping[str, Any]] = None,
    form_compiler_options: Optional[Mapping[str, Any]] = None,
    jit_options: Optional[Mapping[str, Any]] = None,
) -> Function:
    """
    Solve a linear variational problem (legacy-style convenience).

    Supported call patterns
    -----------------------
    1) `uh = solve(a == L, bcs=..., petsc_options=...)`
    2) `uh = solve(a, L, bcs=..., petsc_options=...)`
    3) `solve(a == L, u, bcs=...)` is also supported by passing `u=` explicitly.

    Parameters
    ----------
    equation_or_a:
        Either a UFL Equation (`a == L`) or the bilinear form `a`.
    L:
        Linear form if `equation_or_a` is the bilinear form.
        If omitted/None, `equation_or_a` must be a UFL Equation.
    u:
        Optional existing Function to write the solution into. If omitted,
        a new Function is created in the inferred solution space.
    bcs:
        Optional boundary conditions.
    petsc_options:
        PETSc options dictionary passed to `dolfinx.fem.petsc.LinearProblem`.
    form_compiler_options:
        Optional form compiler options.
    jit_options:
        Optional JIT options.

    Returns
    -------
    dolfinx.fem.Function
        The solution function.
    """
    bc_list = list(bcs) if bcs is not None else []

    if L is None:
        a, L_form = _extract_lhs_rhs(equation_or_a)
    else:
        a = equation_or_a
        L_form = L

    if u is None:
        V = _infer_solution_space_from_bilinear_form(a)
        u = Function(V)

    problem = LinearProblem(
        a,
        L_form,
        u=u,
        bcs=bc_list,
        petsc_options=dict(petsc_options) if petsc_options is not None else {},
        form_compiler_options=(
            dict(form_compiler_options) if form_compiler_options is not None else {}
        ),
        jit_options=dict(jit_options) if jit_options is not None else {},
    )
    uh = problem.solve()
    return uh


# -----------------------------------------------------------------------------
# Mesh IO and utilities
# -----------------------------------------------------------------------------


def load_mesh(filename: str) -> dolfinx.mesh.Mesh:
    """Load a mesh from an XDMF file (expects mesh stored under name='mesh')."""
    if not filename.endswith(".xdmf"):
        raise ValueError("load_mesh: filename must end with .xdmf")

    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
    return mesh


def load_mesh_with_markers(
    filename: str,
) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
    """
    Load a mesh and facet markers from an XDMF file.

    Expects:
    - mesh stored under name='mesh'
    - facet MeshTags stored under name='boundary_markers'
    """
    if not filename.endswith(".xdmf"):
        raise ValueError("load_mesh_with_markers: filename must end with .xdmf")

    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        mesh.topology.create_entities(tdim - 1)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        markers = xdmf.read_meshtags(mesh, name="boundary_markers")
    return mesh, markers


def cell_markers_to_facet_markers(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags,
) -> dolfinx.mesh.MeshTags:
    """
    Convert cell MeshTags to facet MeshTags.

    A facet receives a marker when:
    - It is an interior facet shared by two cells with *different* markers
      (the larger marker value is used, so building markers 0, 1, 2, … take
      precedence over ground -2 and halo -1).
    - It is a boundary facet (one adjacent cell only); the owning cell's
      marker is used.

    Parameters
    ----------
    mesh :
        The dolfinx mesh.
    cell_tags :
        MeshTags of dimension ``mesh.topology.dim`` (cell markers).

    Returns
    -------
    dolfinx.mesh.MeshTags
        MeshTags of dimension ``mesh.topology.dim - 1`` (facet markers).
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1

    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)

    f_to_c = mesh.topology.connectivity(fdim, tdim)
    num_facets = mesh.topology.index_map(fdim).size_local

    # Build cell index → marker lookup (default: no marker)
    cell_marker = np.full(
        mesh.topology.index_map(tdim).size_local, np.iinfo(np.int32).min, dtype=np.int32
    )
    cell_marker[cell_tags.indices] = cell_tags.values

    facet_indices: list[int] = []
    facet_values: list[int] = []

    for f in range(num_facets):
        cells = f_to_c.links(f)
        if len(cells) == 1:
            # Boundary facet → use the owning cell's marker
            m = int(cell_marker[cells[0]])
            if m != np.iinfo(np.int32).min:
                facet_indices.append(f)
                facet_values.append(m)
        elif len(cells) == 2:
            m0 = int(cell_marker[cells[0]])
            m1 = int(cell_marker[cells[1]])
            if m0 != m1:
                # Interface facet → use the larger marker
                facet_indices.append(f)
                facet_values.append(max(m0, m1))

    fi = np.array(facet_indices, dtype=np.int32)
    fv = np.array(facet_values, dtype=np.int32)
    perm = np.argsort(fi)

    info(f"Converted {len(cell_tags.indices)} cell markers to {len(fi)} facet markers.")
    return dolfinx.mesh.meshtags(mesh, fdim, fi[perm], fv[perm])


def _save_mesh(self: dolfinx.mesh.Mesh, filename: str) -> None:
    """Monkeypatched convenience: `mesh.save("file.xdmf")`."""
    info(f"Saving mesh to file {filename}")
    if not filename.endswith(".xdmf"):
        raise ValueError("Mesh.save: filename must end with .xdmf")
    with XDMFFile(self.comm, filename, "w") as xdmf:
        xdmf.write_mesh(self)


def _save_function(self: Function, filename: str, t: Optional[float] = None) -> None:
    """Monkeypatched convenience: `u.save("file.xdmf", t=...)`."""
    info(f"Saving function to file {filename}")
    if not filename.endswith(".xdmf"):
        raise ValueError("Function.save: filename must end with .xdmf")

    mesh = self.function_space.mesh
    mode = "a" if (t is not None and t > 0) else "w"

    with XDMFFile(mesh.comm, filename, mode) as xdmf:
        if t is None or t == 0:
            xdmf.write_mesh(mesh)
        if t is None:
            xdmf.write_function(self)
        else:
            xdmf.write_function(self, t)


# Attach convenience methods
dolfinx.mesh.Mesh.save = _save_mesh  # type: ignore[attr-defined]
dolfinx.fem.Function.save = _save_function  # type: ignore[attr-defined]


def BoxMesh(
    xmin: float,
    ymin: float,
    zmin: float,
    xmax: float,
    ymax: float,
    zmax: float,
    nx: int,
    ny: int,
    nz: int,
) -> dolfinx.mesh.Mesh:
    """Create a tetrahedral box mesh (legacy-style)."""
    domain = [(xmin, ymin, zmin), (xmax, ymax, zmax)]
    return dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=domain,
        n=(nx, ny, nz),
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )


def bounds(mesh: dolfinx.mesh.Mesh) -> Tuple[float, float, float, float, float, float]:
    """
    Compute global mesh bounds in parallel (3D).

    Returns
    -------
    (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    coords = mesh.geometry.x
    local_min = np.min(coords, axis=0)
    local_max = np.max(coords, axis=0)

    global_min = np.zeros(3, dtype=coords.dtype)
    global_max = np.zeros(3, dtype=coords.dtype)

    mesh.comm.Allreduce(local_min, global_min, op=MPI.MIN)
    mesh.comm.Allreduce(local_max, global_max, op=MPI.MAX)

    xmin, ymin, zmin = map(float, global_min)
    xmax, ymax, zmax = map(float, global_max)
    return xmin, ymin, zmin, xmax, ymax, zmax


def offset_to_origin(
    mesh: dolfinx.mesh.Mesh,
) -> Tuple[float, float, float, float, float, float]:
    """
    Shift mesh coordinates so that (xmin, ymin, zmin) becomes (0, 0, 0).

    Returns the new bounds after shifting.
    """
    info("Shifting mesh to origin")
    xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)
    info(f"Original bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]")

    mesh.geometry.x[:] -= np.array([xmin, ymin, zmin], dtype=mesh.geometry.x.dtype)

    xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)
    info(f"New bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]")
    return xmin, ymin, zmin, xmax, ymax, zmax


def _hmin(self: dolfinx.mesh.Mesh) -> float:
    """Compute global minimum cell diameter (mesh size)."""
    tdim = self.topology.dim
    imap = self.topology.index_map(tdim)
    cells = np.arange(imap.size_local, dtype=np.int32)  # local cells only
    cell_diameters = self.h(tdim, cells)
    local_min = float(np.min(cell_diameters))
    return float(self.comm.allreduce(local_min, op=MPI.MIN))


# Attach `mesh.hmin()` convenience method (legacy-like)
dolfinx.mesh.Mesh.hmin = _hmin  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Optional plotting (lazy import)
# -----------------------------------------------------------------------------


def plot(u: Function, show: bool = True) -> Any:
    """
    Quick visualization helper using pyvista (optional dependency).

    Parameters
    ----------
    u:
        dolfinx Function to plot.
    show:
        If True, shows the plotter window.

    Returns
    -------
    pyvista.Plotter
        The created plotter.
    """
    try:
        import pyvista  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "plot: pyvista is not available. Install pyvista or avoid calling plot()."
        ) from e

    V = u.function_space
    cells, types, x = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if show:
        plotter.show()
    return plotter
