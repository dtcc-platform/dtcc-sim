"""Reload transient shallow-water results as native Core meshes."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _water_fields(h, q, bed, association):
    from dtcc_core.model import Field

    if (h.ndim != 1 or q.shape != (len(h), 2) or bed.shape != h.shape
            or np.any(h < 0) or not all(np.isfinite(a).all() for a in (h, q, bed))
            or np.any(q[h == 0] != 0)):
        raise ValueError("Invalid shallow-water fields")
    velocity = np.zeros((len(h), 3))
    np.divide(q, h[:, None], out=velocity[:, :2], where=h[:, None] > 0)
    return [Field(name=name, unit=unit, association=association, dim=dim, values=values)
            for name, unit, dim, values in (
                ("water_depth", "m", 1, h[:, None]),
                ("water_momentum", "m2/s", 2, q),
                ("water_velocity", "m/s", 3, velocity),
                ("bed_elevation", "m", 1, bed[:, None]),
                ("water_surface_elevation", "m", 1, (bed+h)[:, None]),
            )]


@dataclass(frozen=True)
class ShallowWaterResult:
    """A completed worker run and its original-coordinate input mesh."""

    case_dir: Path

    @classmethod
    def load(cls, output_dir):
        directory = Path(output_dir).expanduser().resolve()
        latest = directory / "shallow_water.json"
        if latest.is_file():
            case = (directory / json.loads(latest.read_text())["run_directory"]).resolve()
            if not case.is_relative_to(directory / ".shallow_water_runs"):
                raise ValueError("Run directory must be inside the solver output directory")
            directory = case
        manifest = json.loads((directory / "run.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError("Only completed shallow-water runs can be loaded")
        status = json.loads((directory / "solver/status.json").read_text())
        diagnostics = json.loads((directory / "solver/diagnostics.json").read_text())
        if status["status"] != "completed" or not diagnostics["passed"]:
            raise ValueError("Only passing numerical results can be loaded")
        return cls(directory)

    @property
    def diagnostics(self):
        diagnostics = json.loads((self.case_dir / "solver/diagnostics.json").read_text())
        manifest = json.loads((self.case_dir / "run.json").read_text())
        diagnostics.update(coordinate_frame=manifest["coordinate_frame"],
                           backend="cecilia-reference", experimental=True,
                           reference=manifest["reference"],
                           assumptions=["Reflective outer boundaries and building walls",
                                        "Building interiors are separate simulated regions",
                                        "Terrain/wetting-drying extensions remain experimental"])
        return diagnostics

    @property
    def snapshot_times(self):
        entries = json.loads((self.case_dir / "solver/snapshots.json").read_text())
        return tuple(entry["time"] for entry in entries)

    def plot(self, snapshot=-1, *, ax=None, show=True, vmax=None, theme="light"):
        """Plot ground depth as filled triangles, with time, scale and buildings.

        Coordinates stay in the input frame. Building interiors are hidden by
        footprint context. Set a common positive vmax to compare snapshots.
        Return the Matplotlib 2D axes; show=False supports headless rendering.
        """
        from .plotting import _plot_depth

        return _plot_depth(self.to_dtcc(snapshot), self.snapshot_times[snapshot],
                          ax=ax, show=show, vmax=vmax, theme=theme)

    def to_dtcc(self, snapshot=-1):
        """Return conservative face means on the exact original input geometry."""
        from dtcc_core.io import load_mesh

        self.load(self.case_dir)
        mesh = load_mesh(self.case_dir / "input.dtcc")
        entries = json.loads((self.case_dir / "solver/snapshots.json").read_text())
        entry = entries[snapshot]
        with np.load(self.case_dir / "solver" / entry["file"]) as raw:
            h = raw["depth"].copy(); q = raw["momentum"].copy(); bed = raw["bed_elevation"].copy()
        if (h.shape != (len(mesh.faces),) or q.shape != (len(h), 2)
                or bed.shape != h.shape or np.any(h < 0)
                or not all(np.isfinite(a).all() for a in (h, q, bed))
                or np.any(q[h == 0] != 0)):
            raise ValueError("Invalid conservative face fields")
        mesh.fields = _water_fields(h, q, bed, "face")
        frame = self.diagnostics["coordinate_frame"]
        if frame["input_crs"]:
            mesh.transform.srs = frame["input_crs"]
        mesh.simulation_diagnostics = {**self.diagnostics, "snapshot_time": entry["time"]}
        return mesh

    def _dg_snapshot(self, snapshot):
        entries = json.loads((self.case_dir / "solver/snapshots.json").read_text())
        entry = entries[snapshot]
        names = ("dg_depth", "dg_momentum", "dg_bed_elevation")
        with np.load(self.case_dir / "solver" / entry["file"]) as raw:
            present = [name in raw for name in names]
            if all(present):
                return tuple(raw[name].copy() for name in names)
            if any(present):
                raise ValueError("Incomplete DG1 snapshot")

        # Initial and final polynomials are also retained in the worker state.
        index = range(len(entries))[snapshot]
        if index not in (0, len(entries)-1):
            raise ValueError("DG1 data are unavailable for this snapshot; rerun the simulation")
        from scipy.spatial import cKDTree
        from ._solver.cecilia_swe.layout import triangle_corner_order

        with np.load(self.case_dir / "solver/state.npz") as raw, np.load(
                self.case_dir / "solver/mesh.npz") as geometry:
            cells = raw["cell_dofs"]
            source = raw["coordinates"][cells]
            target = geometry["vertices"][geometry["faces"]]
            distances, face_order = cKDTree(source[:, :, :2].mean(axis=1)).query(
                target[:, :, :2].mean(axis=1))
            if np.any(distances > 1e-8) or len(np.unique(face_order)) != len(target):
                raise ValueError("DG1 cells must match the input triangles")
            corners = triangle_corner_order(target, source[face_order])

            def input_nodes(values):
                return np.take_along_axis(values[cells][face_order], corners, axis=1)

            state = raw["initial_state" if index == 0 else "state"]
            h = input_nodes(state[raw["h_dofs"]])
            q = np.stack([input_nodes(state[raw[name]]) for name in ("hu_dofs", "hv_dofs")], axis=-1)
            return h, q, input_nodes(raw["bed"])

    def to_visualization_mesh(self, snapshot=-1):
        """Return exact DG1 Point Data on triangles with unshared vertices.

        Original conservative means remain available as Cell Data. Coincident
        vertices on adjacent triangles carry independent finite-element values.
        """
        from copy import deepcopy
        from dtcc_core.model import Mesh

        native = self.to_dtcc(snapshot)
        h, q, bed = self._dg_snapshot(snapshot)
        count = len(native.faces)
        if h.shape != (count, 3) or q.shape != (count, 3, 2) or bed.shape != h.shape:
            raise ValueError("Invalid DG1 snapshot shapes")
        for values, field in zip((h, q, bed), (native.fields[0], native.fields[1], native.fields[3])):
            if not np.allclose(values.mean(axis=1).reshape(field.values.shape),
                               field.values, rtol=1e-12, atol=1e-14):
                raise ValueError("DG1 fields do not match conservative face means")
        mesh = Mesh(vertices=native.vertices[native.faces].reshape(-1, 3).copy(),
                    faces=np.arange(3*count).reshape(-1, 3), markers=native.markers.copy())
        mesh.transform = deepcopy(native.transform)
        mesh.fields = _water_fields(h.reshape(-1), q.reshape(-1, 2), bed.reshape(-1), "vertex") + native.fields
        mesh.simulation_diagnostics = native.simulation_diagnostics
        return mesh

    def export_vtu(self, path, snapshot=-1):
        """Save DG1 Point Data, conservative Cell Data and coordinate metadata."""
        from ..coordinates import write_coordinate_frame

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_visualization_mesh(snapshot).save(path)
        write_coordinate_frame(path, self.diagnostics["coordinate_frame"])

    def export_series(self, destination, *, index_path=None):
        """Replace owned native/VTU snapshots while preserving other files."""
        from ..coordinates import write_coordinate_frame

        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        animation = (destination / "water.pvd" if index_path is None
                     else Path(index_path).expanduser().resolve())
        animation.parent.mkdir(parents=True, exist_ok=True)
        index = destination / "times.json"
        previous = json.loads(index.read_text()) if index.exists() else []
        entries = []
        for i, time in enumerate(self.snapshot_times):
            mesh = self.to_dtcc(i); stem = f"water_{i:05d}"
            for suffix in ("dtcc", "vtu"):
                path = destination / f"{stem}.{suffix}"
                if suffix == "vtu":
                    self.export_vtu(path, i)
                else:
                    mesh.save(path)
                    write_coordinate_frame(path, self.diagnostics["coordinate_frame"])
            entries.append({"time": time, "dtcc": stem+".dtcc", "vtu": stem+".vtu"})
        owned = {entry[key] for entry in entries for key in ("dtcc", "vtu")}
        for entry in previous:
            for key in ("dtcc", "vtu"):
                name = entry[key]
                if name not in owned and Path(name).name == name and name.startswith("water_"):
                    (destination / name).unlink(missing_ok=True)
                    (destination / name).with_suffix(".coordinates.json").unlink(missing_ok=True)
        index.write_text(json.dumps(entries, indent=2) + "\n")
        from xml.etree import ElementTree as ET
        from os.path import relpath

        root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        collection = ET.SubElement(root, "Collection")
        for entry in entries:
            ET.SubElement(collection, "DataSet", timestep=str(entry["time"]),
                          group="", part="0",
                          file=Path(relpath(destination / entry["vtu"], animation.parent)).as_posix())
        ET.ElementTree(root).write(animation, encoding="utf-8", xml_declaration=True)
        if animation != destination / "water.pvd":
            (destination / "water.pvd").unlink(missing_ok=True)
        return entries
