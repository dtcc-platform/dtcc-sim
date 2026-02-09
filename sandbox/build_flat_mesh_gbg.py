# This script builds a flat 2D mesh for an area in Gothenburg.

from dtcc import datasets, Bounds

# Center coordinates (Poseidon statue in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755

# Meshing parameters
h = 25.0  # max mesh size
L = 400.0  # domain size

# Define bounds
bounds = Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Get flat mesh dataset
flat_mesh = datasets.city_flat_mesh(bounds=bounds, max_mesh_size=h)

# Save to file
flat_mesh.save("output/flat_mesh_gbg.xdmf")

# View mesh
# flat_mesh.view()
