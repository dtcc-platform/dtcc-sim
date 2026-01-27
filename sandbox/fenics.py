"Simple FEniCS style FEniCSx wrapper"

__all__ = []

import dolfinx
import basix

# --- MPI imports --
from mpi4py import MPI

__all__.extend(["MPI"])

# --- Numpy imports
import numpy as np

__all__.extend(["np"])

# --- UFL imports ---
from ufl import (
    TrialFunction,
    TestFunction,
    TrialFunctions,
    TestFunctions,
    SpatialCoordinate,
    CellDiameter,
)
from ufl import dx, ds, dot, inner, grad, curl, div
from ufl import exp, sin, cos, sqrt
from ufl import as_vector

__all__.extend(
    [
        "TrialFunction",
        "TestFunction",
        "TrialFunctions",
        "TestFunctions",
        "SpatialCoordinate",
        "CellDiameter",
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
    ]
)

# --- DOLFINx io imports ---
from dolfinx.io import XDMFFile

__all__.extend(["XDMFFile"])

# --- DOLFINx fem imports ---
from dolfinx.fem import Constant, Function, Expression
from dolfinx.fem.petsc import LinearProblem


def FunctionSpace(mesh, element, degree=None, dim=None):
    """
    Create a function space for the given mesh and element.
    This is a wrapper around dolfinx.fem.FunctionSpace.
    """

    # FIXME: Might need to rethink this wrapper

    # Handle mixed elements
    if type(element) is tuple:
        # If element is a tuple, it is a mixed element
        if degree is not None:
            raise ValueError("Degree must not be specified for mixed elements.")
        # FIXME: Does not handle dim
        elements = [
            basix.ufl.element(el, mesh.basix_cell(), degree) for el, degree in element
        ]
        element = basix.ufl.mixed_element(elements)
    elif isinstance(element, str):
        # If element is a string, create a Basix element with the specified degree
        if degree is None:
            raise ValueError("Degree must be specified for string elements.")
        if dim is None:
            element = basix.ufl.element(element, mesh.basix_cell(), degree)
        else:
            element = basix.ufl.element(
                element, mesh.basix_cell(), degree, shape=(dim,)
            )
    else:
        error("Element must be a tuple of elements or a string with degree.")

    return dolfinx.fem.functionspace(mesh, element)


def assemble_matrix(a, **kwargs):
    return dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), **kwargs)


def interpolate(f, V):
    info(f"Interpolating function into {V}")
    u = Function(V)

    if isinstance(f, Expression):
        tdim = V.mesh.topology.dim
        imap = V.mesh.topology.index_map(tdim)
        cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
        u.interpolate(f, cells0=cells, cells1=cells)
    else:
        u.interpolate(f)

    return u


def project(f, V):
    "Interpolate a function f into a function space V."

    info(f"Projecting function into {V}")

    opts = {
        "ksp_monitor_short": None,
        "ksp_converged_reason": None,
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-6,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    }

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx
    L = inner(f, v) * dx

    problem = LinearProblem(a, L, petsc_options=opts)
    f_h = problem.solve()

    return f_h


__all__.extend(
    [
        "Constant",
        "Function",
        "Expression",
        "FunctionSpace",
        "LinearProblem",
        "assemble_matrix",
        "interpolate",
        "project",
    ]
)

# --- DOLFINx log imports ---
from dolfinx.log import log, set_log_level, LogLevel

DEBUG = LogLevel.DEBUG
INFO = LogLevel.INFO
WARNING = LogLevel.WARNING
ERROR = LogLevel.ERROR

info = lambda message: log(INFO, message)
error = lambda message: log(ERROR, message)

__all__.extend(
    ["info", "error", "set_log_level", "LogLevel", "DEBUG", "INFO", "WARNING", "ERROR"]
)

# --- DOLFINx boundary conditions ---
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import locate_dofs_topological, dirichletbc
from ufl import Measure
from petsc4py.PETSc import ScalarType


def DirichletBC(V, value, condition=None, markers=None, marker_value=None, dofs=None):
    """
    Create a Dirichlet boundary condition (fixed degrees of freedom).
    Markers can be set in three different ways:

    1. `condition`: a callable function that takes a point
       and returns True if it is on the boundary.

    2. `markers`: facet tags with integer values, where each value
       corresponds to a different boundary condition. If you pass in
       `markers`, you must also specify `marker_value` to select the
       specific tag.

    3. `dofs`: a list of dof indices that should be constrained.
    """

    # Get mesh and topological dimension
    mesh = V.mesh
    tdim = V.mesh.topology.dim

    # Extract dofs if we have a condition
    if condition is not None:
        facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=condition)
        dofs = locate_dofs_topological(V, tdim - 1, facets)

    # Extract dofs if we have markers
    elif markers is not None:
        if marker_value is None:
            raise ValueError(
                "If markers are provided, marker_value must also be specified."
            )
        dofs = locate_dofs_topological(V, tdim - 1, markers.find(marker_value))

    # Extract dofs if we have dofs
    elif dofs is not None:
        dofs = np.array(dofs, dtype="int32")

    # Raise error if missing data
    else:
        raise ValueError(
            "Either condition, markers, or dofs must be provided to create a DirichletBC."
        )

    info(f"Creating DirichletBC with value {value} on {len(dofs)} dofs.")

    # Create boundary condition
    bc = dirichletbc(value=ScalarType(value), dofs=dofs, V=V)

    return bc


def NeumannBC(mesh, condition=None, markers=None, marker_value=None):
    """
    Create a Neumann boundary condition (measure ds on facets).
    Markers can be set in two different ways:

    1. `condition`: a callable function that takes a point
       and returns True if it is on the boundary.

    2. `markers': facet tags with integer values, where each value
       corresponds to a different boundary condition. If you pass in
       `markers`, you must also specify `marker_value` to select the
       specific tag. Note that `marker_value` may be a tuple of list.
    """

    # Get topological dimension
    tdim = mesh.topology.dim

    # Extract markers (tags) if we have a condition
    if condition is not None:
        facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=condition)
        marker_value = 1
        new_vals = np.full(facets.size, marker_value, dtype=np.int32)
        markers = meshtags(mesh, tdim - 1, facets, new_vals)

    # Extract marker (tags) if we have markers
    elif markers is not None:
        if marker_value is None:
            raise ValueError(
                "If markers are provided, marker_value must also be specified."
            )
        if isinstance(marker_value, tuple) or isinstance(marker_value, list):
            facets = markers.indices[np.isin(markers.values, marker_value)]
        else:
            facets = markers.find(marker_value)
        marker_value = 1
        new_vals = np.full(facets.size, marker_value, dtype=np.int32)
        markers = meshtags(mesh, tdim - 1, facets, new_vals)

    # Raise error if missing data
    else:
        raise ValueError(
            "Either condition or markers must be provided to create a NeumannBC."
        )

    # Create surface measure from markers
    ds = Measure("ds", domain=mesh, subdomain_data=markers)
    ds = ds(marker_value)

    return ds


__all__.extend(["DirichletBC", "NeumannBC", "ScalarType"])

# --- Numpy imports ---

from numpy import isclose as near

__all__.extend(["near"])

# --- Plotting ---
import pyvista
import dolfinx


def plot(u, show=True):

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


__all__.extend(["plot"])


# --- Saving and loading ---


def load_mesh(filename):
    """Load a mesh from an XDMF file."""
    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    comm = MPI.COMM_WORLD
    with XDMFFile(comm, filename, "r") as xdmf_file:
        mesh = xdmf_file.read_mesh(name="mesh")

    return mesh


def load_mesh_with_markers(filename):
    """Load a mesh with facets from an XDMF file."""
    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    comm = MPI.COMM_WORLD
    with XDMFFile(comm, filename, "r") as xdmf_file:
        mesh = xdmf_file.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        mesh.topology.create_entities(tdim - 1)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        boundary_markers = xdmf_file.read_meshtags(mesh, name="boundary_markers")

    return mesh, boundary_markers


__all__.extend(["load_mesh", "load_mesh_with_markers"])


def _save_mesh(mesh, filename):
    """Save dolfinx Mesh to an XDMF file.

    Args:
        mesh (dolfinx.mesh.Mesh): The mesh to save.
        filename (str): Output XDMF filename (should end with .xdmf).
    """

    info(f"Saving mesh to file {filename}")

    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    with XDMFFile(mesh.comm, filename, "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)


def _save_function(self, filename, t=None):
    """Save Function to an XDMF file (supports time series).

    Args:
        filename (str): Output XDMF filename (should end with .xdmf).
        t (float, optional): Current simulation time. If given, writes a time series.
                             Mesh is written only once when t == 0.
    """

    info(f"Saving function to file {filename}")

    if not filename.endswith(".xdmf"):
        raise ValueError("Filename must end with .xdmf")

    mesh = self.function_space.mesh
    mode = "a" if (t is not None and t > 0) else "w"

    with XDMFFile(mesh.comm, filename, mode) as xdmf_file:
        if t is None or t == 0:
            xdmf_file.write_mesh(mesh)
        if t is None:
            xdmf_file.write_function(self)
        else:
            xdmf_file.write_function(self, t)


dolfinx.mesh.Mesh.save = _save_mesh
dolfinx.fem.Function.save = _save_function


# --- Meshes ---


def BoxMesh(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz):
    """
    Create a box mesh."
    """

    domain = [(xmin, ymin, zmin), (xmax, ymax, zmax)]
    mesh = dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=domain,
        n=(nx, ny, nz),
        cell_type=dolfinx.mesh.CellType.tetrahedron,
    )

    return mesh


__all__.extend(["BoxMesh"])


# --- Mesh bounds and shifting ---


def bounds(mesh):
    """
    Compute global mesh bounds in parallel (3D).

    Returns:
        xmin, ymin, zmin, xmax, ymax, zmax
    """
    comm = mesh.comm
    coords = mesh.geometry.x

    local_min = coords.min(axis=0)
    local_max = coords.max(axis=0)

    global_min = np.zeros(3)
    global_max = np.zeros(3)

    comm.Allreduce(local_min, global_min, op=MPI.MIN)
    comm.Allreduce(local_max, global_max, op=MPI.MAX)

    xmin, ymin, zmin = global_min
    xmax, ymax, zmax = global_max

    return xmin, ymin, zmin, xmax, ymax, zmax


def offset_to_origin(mesh):
    """
    Offset mesh coordinates so the global minimum (xmin, ymin, zmin)
    moves to the origin (0, 0, 0).

    Returns:
        xmin, ymin, zmin, xmax, ymax, zmax for the shifted mesh
    """

    info(f"Shifting mesh to origin")

    # Get bounds
    xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)

    # Print old bounds
    info(f"Original bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]")

    # Shift the mesh coordinates
    global_min = np.array([xmin, ymin, zmin])
    mesh.geometry.x[:] -= global_min

    # Print new bounds
    xmin, ymin, zmin, xmax, ymax, zmax = bounds(mesh)
    info(f"New bounds: [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}]")

    return xmin, ymin, zmin, xmax, ymax, zmax


def _hmin(self):
    "Compute global minimum cell diameter (mesh size)."

    # Get local cell indices
    cells = np.arange(
        self.topology.index_map(self.topology.dim).size_local, dtype=np.int32
    )

    # Compute cell diameters
    cell_diameters = self.h(self.topology.dim, cells)

    # Local minimum
    local_min = np.min(cell_diameters)

    # Global minimum reduction across MPI ranks
    global_min = self.comm.allreduce(local_min, op=MPI.MIN)

    return global_min


dolfinx.mesh.Mesh.hmin = _hmin

__all__.extend(["bounds", "offset_to_origin"])
