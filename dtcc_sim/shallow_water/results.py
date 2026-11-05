"""Reload transient shallow-water results as native Core meshes."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


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
                           assumptions=["Reflective outer boundaries and building walls",
                                        "Building interiors are separate simulated regions",
                                        "Terrain/wetting-drying extensions remain experimental"])
        return diagnostics

    @property
    def snapshot_times(self):
        entries = json.loads((self.case_dir / "solver/snapshots.json").read_text())
        return tuple(entry["time"] for entry in entries)

    def to_dtcc(self, snapshot=-1):
        """Return conservative face means on the exact original input geometry."""
        from dtcc_core.io import load_mesh
        from dtcc_core.model import Field

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
        velocity = np.zeros((len(h), 3))
        np.divide(q, h[:, None], out=velocity[:, :2], where=h[:, None] > 0)
        mesh.fields = [
            Field(name="water_depth", unit="m", association="face", dim=1, values=h[:, None]),
            Field(name="water_momentum", unit="m2/s", association="face", dim=2, values=q),
            Field(name="water_velocity", unit="m/s", association="face", dim=3, values=velocity),
            Field(name="bed_elevation", unit="m", association="face", dim=1, values=bed[:, None]),
            Field(name="water_surface_elevation", unit="m", association="face", dim=1, values=(bed+h)[:, None]),
        ]
        frame = self.diagnostics["coordinate_frame"]
        if frame["input_crs"]:
            mesh.transform.srs = frame["input_crs"]
        mesh.simulation_diagnostics = {**self.diagnostics, "snapshot_time": entry["time"]}
        return mesh

    def export_series(self, destination):
        """Replace owned native/VTU snapshots while preserving other files."""
        from ..coordinates import write_coordinate_frame

        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        index = destination / "times.json"
        previous = json.loads(index.read_text()) if index.exists() else []
        entries = []
        for i, time in enumerate(self.snapshot_times):
            mesh = self.to_dtcc(i); stem = f"water_{i:05d}"
            for suffix in ("dtcc", "vtu"):
                path = destination / f"{stem}.{suffix}"
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
        return entries
