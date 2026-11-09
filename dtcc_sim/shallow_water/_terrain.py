"""Shared measured terrain for shallow-water meshes and city context."""

from copy import deepcopy
import hashlib

import numpy as np

from ..timing import phase


def _sample_dem(raster, xy):
    """Sample pixel-center elevations bilinearly, clamping half-pixel margins."""
    data = np.asarray(raster.data)
    transform = raster.georef
    if (data.ndim != 2 or not all(data.shape) or data.dtype.kind not in "fiu"
            or not np.isfinite(data).all()
            or (np.ma.isMaskedArray(raster.data) and np.ma.getmaskarray(raster.data).any())
            or (raster.nodata is not None and np.any(data == raster.nodata))):
        raise ValueError("Flood terrain requires a finite single-band DEM without missing elevations")
    if (not np.isfinite(tuple(transform)).all() or transform.a == 0 or transform.e == 0
            or transform.b != 0 or transform.d != 0):
        raise ValueError("Flood terrain requires an axis-aligned DEM with nonzero pixel spacing")
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Terrain sampling requires finite XY coordinates")
    inverse = ~transform
    col = inverse.a*xy[:, 0] + inverse.c
    row = inverse.e*xy[:, 1] + inverse.f
    height, width = data.shape
    if (np.any(col < -1e-7) or np.any(col > width+1e-7)
            or np.any(row < -1e-7) or np.any(row > height+1e-7)):
        raise ValueError("Flood mesh extends outside the terrain raster coverage")
    col = np.clip(col-.5, 0., width-1)
    row = np.clip(row-.5, 0., height-1)
    i = np.floor(col).astype(int); j = np.floor(row).astype(int)
    right = np.minimum(i+1, width-1); bottom = np.minimum(j+1, height-1)
    u = col-i; v = row-j
    return ((1-v)*((1-u)*data[j, i] + u*data[j, right])
            + v*((1-u)*data[bottom, i] + u*data[bottom, right]))


def _city_with_solver_ground(mesh, surface):
    """Combine the solver's exact ground triangles with Core building surfaces."""
    from dtcc_core.model import Mesh

    ground = mesh.markers < 0
    buildings = surface.markers >= 0
    context = Mesh(
        vertices=np.vstack([mesh.vertices, surface.vertices]),
        faces=np.vstack([np.asarray(mesh.faces, dtype=np.int64)[ground],
                         np.asarray(surface.faces, dtype=np.int64)[buildings]+len(mesh.vertices)]),
        markers=np.concatenate([mesh.markers[ground], surface.markers[buildings]]),
    )
    used, remap = np.unique(context.faces, return_inverse=True)
    context.vertices = context.vertices[used].copy()
    context.faces = remap.reshape(-1, 3)
    context.transform = deepcopy(mesh.transform)
    return context


def _build_terrain_city(args):
    from dtcc_core import builder, datasets, io
    from dtcc_core.model import Bounds, City

    bounds = Bounds(*args.bounds)
    with phase("download_pointcloud"):
        pointcloud = io.data.download_pointcloud(bounds=bounds, epsg="3006")
        pointcloud = pointcloud.remove_global_outliers(3.)
    with phase("download_footprints"):
        buildings = io.data.download_footprints(bounds=bounds, epsg="3006")
    with phase("terrain_raster"):
        raster = builder.build_terrain_raster(
            pointcloud, cell_size=args.raster_cell_size, radius=args.raster_radius,
            bounds=bounds, ground_only=True,
        )
        if raster.crs not in ("", "EPSG:3006"):
            raise ValueError("Generated flood terrain must use EPSG:3006")
        raster.crs = "EPSG:3006"
    with phase("building_heights"):
        buildings = builder.extract_roof_points(buildings, pointcloud)
        buildings = builder.compute_building_heights(buildings, raster, overwrite=True)
    city = City()
    city.add_terrain(raster)
    city.add_buildings(buildings, remove_outside_terrain=True)
    arguments = dict(bounds=list(args.bounds), max_mesh_size=args.h,
                     min_building_detail=args.min_building_detail,
                     center_on_origin=False, strict_live=args.strict_live)
    with phase("water_mesh"):
        mesh = datasets.city_flat_mesh.build_from_city(city, **arguments)
        mesh.vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
        mesh.vertices[:, 2] = _sample_dem(raster, mesh.vertices[:, :2])
        mesh.transform.srs = "EPSG:3006"
    with phase("city_surface"):
        surface = datasets.city_surface_mesh.build_from_city(
            city, **arguments, lod="lod1", flat_ground=False, smoothing=0,
        )
        context = _city_with_solver_ground(mesh, surface)
    terrain = {
        "source": "Core ground-classified point cloud", "ground_only": True,
        "raster_cell_size": args.raster_cell_size, "raster_radius": args.raster_radius,
        "hole_fill": "nearest", "sampling": "bilinear pixel centers; clamped half-pixel margins",
        "raster_shape": list(raster.data.shape), "raster_georef": list(raster.georef),
        "raster_sha256": hashlib.sha256(np.asarray(raster.data, dtype="<f8").tobytes()).hexdigest(),
        "horizontal_crs": "EPSG:3006", "unit": "m",
        "vertical_reference": "source point-cloud Z; no vertical transformation",
        "bed_min": float(mesh.vertices[:, 2].min()), "bed_max": float(mesh.vertices[:, 2].max()),
    }
    source = {"dataset": "city_flat_mesh.build_from_city", **arguments,
              "bed": "point-cloud terrain", "terrain": terrain,
              "city_surface": {"dataset": "city_surface_mesh.build_from_city", **arguments,
                               "lod": "lod1", "flat_ground": False, "smoothing": 0,
                               "ground": "exact simulation mesh"}}
    return mesh, context, source
