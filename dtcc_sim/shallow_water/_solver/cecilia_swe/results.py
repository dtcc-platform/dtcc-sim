"""Attach conservative face values to DTCC meshes without loading FEniCSx."""

from pathlib import Path
import json
import numpy as np

from .mesh_io import load_mesh


def to_dtcc(output_dir, snapshot=-1):
    from dtcc_core.model import Mesh, Field

    output_dir = Path(output_dir)
    if json.loads((output_dir / "status.json").read_text())["status"] != "completed":
        raise ValueError("Only completed runs can be exported")
    geometry = load_mesh(output_dir / "mesh.npz")
    index = json.loads((output_dir / "snapshots.json").read_text())
    with np.load(output_dir / index[snapshot]["file"], allow_pickle=False) as data:
        depth = data["depth"].copy()
        momentum = data["momentum"].copy()
        bed = data["bed_elevation"].copy()
    count = len(geometry["faces"])
    if depth.shape != (count,) or momentum.shape != (count, 2):
        raise ValueError("Result arrays do not match mesh faces")
    if not np.isfinite(depth).all() or not np.isfinite(momentum).all() or np.any(depth < 0):
        raise ValueError("Invalid result values")
    if np.any(momentum[depth == 0.] != 0.):
        raise ValueError("Zero-depth faces must have zero momentum")
    horizontal = np.zeros_like(momentum)
    np.divide(momentum, depth[:, None], out=horizontal, where=depth[:, None] > 0.)
    velocity = np.column_stack([horizontal, np.zeros(count)])
    result = Mesh(vertices=geometry["vertices"], faces=geometry["faces"], markers=geometry["markers"])
    if bed.shape != depth.shape or not np.isfinite(bed).all():
        raise ValueError("Invalid bed elevation field")
    result.fields = [
        Field(name="bed_elevation", unit="m", association="face", dim=1, values=bed[:,None], description="DG1 mean bed elevation"),
        Field(name="water_surface_elevation", unit="m", association="face", dim=1, values=(bed+depth)[:,None], description="Mean bed plus mean depth; dry faces show ground elevation"),
        Field(name="water_depth", unit="m", association="face", dim=1,
              values=depth[:, None], description="DG1 triangle mean water depth"),
        Field(name="water_momentum", unit="m2/s", association="face", dim=2,
              values=momentum, description="DG1 triangle mean depth-integrated horizontal momentum"),
        Field(name="water_velocity", unit="m/s", association="face", dim=3,
              values=velocity, description="Depth-weighted velocity: mean momentum divided by mean depth"),
    ]
    return result


def export_series(output_dir, destination):
    import dtcc_core.io

    output_dir = Path(output_dir)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    snapshots = json.loads((output_dir / "snapshots.json").read_text())
    index = []
    for i, entry in enumerate(snapshots):
        mesh = to_dtcc(output_dir, i)
        filename = f"water_{i:05d}"
        mesh.save(destination / (filename + ".vtu"))
        mesh.save(destination / (filename + ".dtcc"))
        index.append({"time": entry["time"], "vtu": filename + ".vtu", "dtcc": filename + ".dtcc"})
    (destination / "times.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
