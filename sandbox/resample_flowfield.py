#!/usr/bin/env python3
#
# Resample flow field from IBOFlow XMF/HDF5 files onto a DOLFiNx tetrahedral mesh.

import numpy as np
import dolfinx
import dolfinx.io
from dolfinx import mesh, fem
import basix.ufl
from mpi4py import MPI
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# File paths
SOURCE_XDMF = "flowfield/CitySimulationFluid000000000Iteration000002000.xmf"
TARGET_XDMF = "../dtcc/gbg_volume_mesh.xdmf"
OUTPUT_XDMF = "flowfield/velocity.xdmf"


def read_xdmf_with_vtk(xdmf_file):
    """Read XDMF file using VTK reader."""
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(xdmf_file)
    reader.Update()
    return reader.GetOutput()


def create_vtk_locator(vtk_ugrid):
    """Create VTK cell locator for efficient point queries."""
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(vtk_ugrid)
    locator.BuildLocator()
    return locator


def interpolate_velocity_at_points(vtk_ugrid, locator, query_points):
    """
    Interpolate velocity field at query points using VTK.

    Args:
        vtk_ugrid: VTK unstructured grid with velocity data
        locator: VTK cell locator
        query_points: numpy array of shape (N, 3)

    Returns:
        interpolated_velocities: numpy array of shape (N, 3)
    """
    n_points = query_points.shape[0]
    velocities = np.zeros((n_points, 3))

    # Get velocity array from VTK data
    velocity_array = vtk_ugrid.GetCellData().GetArray("VelocityFluid")

    for i, point in enumerate(query_points):
        # Find cell containing the point
        cell_id = locator.FindCell(point)

        if cell_id >= 0:  # Point is inside the mesh
            # Get velocity at cell center (simple approach)
            velocity = velocity_array.GetTuple3(cell_id)
            velocities[i] = velocity
        else:
            # Point outside mesh - could use nearest neighbor or extrapolation
            # For now, set to zero
            velocities[i] = [0.0, 0.0, 0.0]

    return velocities


if __name__ == "__main__":

    print("Starting flow field resampling...")

    # 1. Read source flow field using VTK
    print("Reading source flow field...")
    source_grid = read_xdmf_with_vtk(SOURCE_XDMF)
    print(
        f"Source grid: {source_grid.GetNumberOfCells()} cells, "
        f"{source_grid.GetNumberOfPoints()} points"
    )

    # Create locator for efficient queries
    locator = create_vtk_locator(source_grid)

    # 2. Read target mesh using DOLFiNx
    print("Reading target mesh...")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, TARGET_XDMF, "r") as xdmf:
        target_mesh = xdmf.read_mesh(name="mesh")

    print(
        f"Target mesh: {target_mesh.topology.index_map(3).size_global} cells, "
        f"{target_mesh.topology.index_map(0).size_global} vertices"
    )

    # 3. Create function space on target mesh
    element = basix.ufl.element("Lagrange", target_mesh.basix_cell(), 1, shape=(3,))
    V = fem.functionspace(target_mesh, element)
    velocity_function = fem.Function(V)

    # 4. Get coordinates where we need to evaluate the velocity
    # For P1 elements, we evaluate at vertices
    dof_coordinates = V.tabulate_dof_coordinates()

    print(f"Interpolating velocity at {dof_coordinates.shape[0]} points...")

    # 5. Interpolate velocity from source to target points
    # Process in chunks to handle memory efficiently
    chunk_size = 10000
    n_points = dof_coordinates.shape[0]
    interpolated_velocities = np.zeros((n_points, 3))

    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = dof_coordinates[start_idx:end_idx]

        # Interpolate chunk
        chunk_velocities = interpolate_velocity_at_points(
            source_grid, locator, chunk_points
        )
        interpolated_velocities[start_idx:end_idx] = chunk_velocities

        if (start_idx // chunk_size) % 10 == 0:
            progress = (start_idx / n_points) * 100
            print(f"Progress: {progress:.1f}%")

    # 6. Set the interpolated values in the DOLFiNx function
    velocity_function.x.array[:] = interpolated_velocities.flatten()

    # 7. Save to output file
    print(f"Saving resampled velocity to {OUTPUT_XDMF}...")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, OUTPUT_XDMF, "w") as xdmf:
        xdmf.write_mesh(target_mesh)
        xdmf.write_function(velocity_function, 0.0)

    print("Resampling completed successfully!")

    # Print some statistics
    vel_magnitude = np.linalg.norm(interpolated_velocities, axis=1)
    print(f"Velocity statistics:")
    print(f"  Min magnitude: {vel_magnitude.min():.3f}")
    print(f"  Max magnitude: {vel_magnitude.max():.3f}")
    print(f"  Mean magnitude: {vel_magnitude.mean():.3f}")
