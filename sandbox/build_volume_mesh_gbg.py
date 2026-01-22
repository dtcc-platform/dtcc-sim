# This script builds a tetrahedral volume mesh for an area in Gothenburg.

import dtcc

# Center coordinates (Poseidon statue in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755

# Meshing parameters
h = 25.0  # max mesh size
H = 80.0  # domain height
L = 400.0  # domain size

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

# Remove global outliers
pointcloud = pointcloud.remove_global_outliers(3.0)

# Build terrain raster
raster = dtcc.build_terrain_raster(pointcloud, cell_size=2, radius=3, ground_only=True)

# Extract roof points and compute building heights
buildings = dtcc.extract_roof_points(buildings, pointcloud)
buildings = dtcc.compute_building_heights(buildings, raster, overwrite=True)

# Create city and add geometries
city = dtcc.City()
city.add_terrain(raster)
city.add_buildings(buildings, remove_outside_terrain=True)

# Build city volume mesh
volume_mesh = dtcc.build_city_volume_mesh(
    city,
    max_mesh_size=h,
    domain_height=H,
    boundary_face_markers=True,
    tetgen_switches={
        "max_volume": h,  # Max tet volume
        "extra": " -VV",
    },
)

# Save to file
volume_mesh.save("volume_mesh_gbg.xdmf")

# View mesh
volume_mesh.view()
