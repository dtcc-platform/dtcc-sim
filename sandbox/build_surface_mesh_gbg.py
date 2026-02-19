# This script builds a surface mesh for an area in Gothenburg.

from dtcc import datasets, Bounds

# Center coordinates (Poseidon statue in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755

# Meshing parameters
h = 25.0  # max mesh size
L = 400.0  # domain size

# Define bounds
bounds = Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Get surface mesh dataset
surface_mesh = datasets.city_surface_mesh(bounds=bounds, max_mesh_size=h)

# Offset to origin
surface_mesh.offset_to_origin()

# Save to file
surface_mesh.save("output/surface_mesh_gbg.xdmf")

# View mesh
surface_mesh.view()
