"""Shared output-directory handling for simulation APIs."""

from pathlib import Path
import xml.etree.ElementTree as ET


def _output_directory(directory):
    """Validate a supplied directory without creating it during configuration."""
    if directory is None:
        return None
    if not str(directory).strip():
        raise ValueError("output_dir must be a nonempty directory path")
    path = Path(directory).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {path}")
    return path


def _solution_path(directory, explicit_path, filename, *, input_mesh=None, stems=("",)):
    """Resolve an optional export, creating parents and replacing prior outputs."""
    if directory is not None and explicit_path is not None:
        raise ValueError("Specify output_dir or output_path, not both")
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
    elif directory is not None:
        path = directory / filename
    else:
        return None
    if input_mesh is not None:
        mesh_path = Path(input_mesh).expanduser().resolve()
        inputs = {mesh_path}
        for item in ET.parse(mesh_path).iter():
            if item.tag.split("}")[-1] == "DataItem" and item.get("Format") == "HDF":
                reference = (item.text or "").strip().split(":", 1)[0]
                if not reference:
                    raise ValueError(f"Empty HDF5 reference in input mesh: {mesh_path}")
                inputs.add((mesh_path.parent / reference).resolve())
        outputs = set()
        for stem in stems:
            target = path.with_name(path.stem + stem + path.suffix)
            outputs.add(target)
            if target.suffix.lower() in (".xdmf", ".xmf"):
                outputs.add(target.with_suffix(".h5"))
        if inputs.intersection(outputs):
            raise ValueError("Solution output would overwrite an input mesh file")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
