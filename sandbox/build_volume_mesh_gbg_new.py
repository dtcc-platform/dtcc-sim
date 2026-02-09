# This script builds a tetrahedral volume mesh for an area in Gothenburg.

from dtcc import datasets, Bounds

# Center coordinates (Poseidon statue in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755

# Meshing parameters
h = 25.0  # max mesh size
H = 80.0  # domain height
L = 400.0  # domain size

# Define bounds
bounds = Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Get volume mesh dataset
volume_mesh = datasets.city_volume_mesh(bounds=bounds, max_mesh_size=h, domain_height=H)

# Save to file
volume_mesh.save("output/volume_mesh_gbg.xdmf")

# View mesh
# volume_mesh.view()
