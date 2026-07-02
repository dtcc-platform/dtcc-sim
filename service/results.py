import os
import shutil
import tarfile
import tempfile
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

from service.config import SHARED_RESULTS_DIR

logger = logging.getLogger(__name__)


_XDMF_FIELD_GROUP = "Mesh/mesh/fields"


def _fsync_file_and_dir(path):
    """Fsync both the file and its parent directory for full durability."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dirfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _serialize_with_save(result, format_ext, tmpdir):
    """Serialize a dataset result into a temporary directory."""
    if not hasattr(result, "save"):
        raise TypeError(
            f"Result object of type {type(result).__name__} has no save() method"
        )

    # Pass str, not Path -- dolfinx monkeypatched save() methods
    # call filename.endswith() which is a str method.
    tmpfile = str(Path(tmpdir) / f"data.{format_ext}")
    result.save(tmpfile)

    files = sorted(Path(tmpdir).iterdir())
    if not files:
        raise RuntimeError(
            f"Result serialization for format '{format_ext}' produced no files"
        )
    return files


def _xml_elements(root, local_name):
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == local_name:
            yield element


def _parse_xdmf(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Could not parse XDMF output {path.name}: {exc}") from exc


def _hdf5_references_from_xdmf(path):
    root = _parse_xdmf(path)
    references = []
    for data_item in _xml_elements(root, "DataItem"):
        if data_item.attrib.get("Format", "").upper() != "HDF":
            continue
        reference = "".join(data_item.itertext()).strip()
        if not reference:
            continue
        if ":" not in reference:
            raise RuntimeError(
                f"XDMF output {path.name} has malformed HDF5 reference {reference!r}"
            )
        h5_filename, _dataset_path = reference.split(":", 1)
        references.append(h5_filename.strip())
    return references


def _validate_xdmf_companion_files(files, tmpdir):
    xdmf_files = [path for path in files if path.suffix.lower() == ".xdmf"]
    for xdmf_path in xdmf_files:
        for h5_filename in _hdf5_references_from_xdmf(xdmf_path):
            h5_path = Path(h5_filename)
            if h5_path.name != h5_filename:
                raise RuntimeError(
                    f"XDMF output {xdmf_path.name} references HDF5 companion "
                    f"{h5_filename!r} outside the output directory"
                )
            if not (Path(tmpdir) / h5_filename).exists():
                raise RuntimeError(
                    f"XDMF output {xdmf_path.name} references missing HDF5 "
                    f"companion file {h5_filename!r}"
                )


def _expected_field_names(result):
    expected = []
    for index, field in enumerate(getattr(result, "fields", []) or []):
        values = np.asarray(getattr(field, "values", np.empty(0)))
        if values.size == 0:
            continue
        expected.append(str(getattr(field, "name", "") or f"field_{index}"))
    return expected


def _xdmf_attribute_names(path):
    root = _parse_xdmf(path)
    names = set()
    for attribute in _xml_elements(root, "Attribute"):
        name = attribute.attrib.get("Name")
        if name:
            names.add(name)
    return names


def _decode_hdf5_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _hdf5_field_names(path):
    names = set()
    with h5py.File(path, "r") as h5_file:
        fields_group = h5_file.get(_XDMF_FIELD_GROUP)
        if fields_group is None:
            return names
        for dataset_name, dataset in fields_group.items():
            field_name = _decode_hdf5_attr(dataset.attrs.get("name"))
            names.add(str(field_name or dataset_name))
    return names


def _validate_xdmf_fields(result, files):
    expected_names = _expected_field_names(result)
    if not expected_names:
        return

    xdmf_files = [path for path in files if path.suffix.lower() == ".xdmf"]
    if not xdmf_files:
        raise RuntimeError(
            "XDMF serialization for a field-carrying result produced no .xdmf file"
        )

    h5_files = [path for path in files if path.suffix.lower() in {".h5", ".hdf5"}]
    if not h5_files:
        raise RuntimeError(
            "XDMF serialization for a field-carrying result produced no HDF5 "
            "companion file"
        )

    xdmf_names = set()
    for xdmf_path in xdmf_files:
        xdmf_names.update(_xdmf_attribute_names(xdmf_path))

    hdf5_names = set()
    for h5_path in h5_files:
        hdf5_names.update(_hdf5_field_names(h5_path))

    missing_from_xdmf = sorted(name for name in expected_names if name not in xdmf_names)
    missing_from_hdf5 = sorted(name for name in expected_names if name not in hdf5_names)
    if missing_from_xdmf or missing_from_hdf5:
        raise RuntimeError(
            "XDMF serialization dropped expected field(s): "
            f"missing from XDMF={missing_from_xdmf}, "
            f"missing from HDF5={missing_from_hdf5}; "
            f"expected fields={sorted(expected_names)}, "
            f"inspected XDMF files={[path.name for path in xdmf_files]}, "
            f"inspected HDF5 files={[path.name for path in h5_files]}."
        )


def handle_result(result, format_ext, task_id):
    """Write result to shared volume. Archive multi-file outputs as .tar.gz."""
    os.makedirs(SHARED_RESULTS_DIR, exist_ok=True)

    if isinstance(result, bytes):
        # Already serialized (dataset returned bytes when format was set)
        filename = f"{task_id}.{format_ext}" if format_ext else task_id
        path = os.path.join(SHARED_RESULTS_DIR, filename)
        with open(path, "wb") as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())
        _fsync_file_and_dir(path)
        return {"result_file": filename, "size_bytes": len(result)}

    # Result is a Python object -- write via save() then check for multi-file output
    if not format_ext:
        raise ValueError(
            f"Cannot serialize result for task {task_id}: "
            f"format_ext is required for non-bytes results"
        )
    tmpdir = tempfile.mkdtemp()
    try:
        files = _serialize_with_save(result, format_ext, tmpdir)
        if format_ext.lower() == "xdmf":
            _validate_xdmf_companion_files(files, tmpdir)
            _validate_xdmf_fields(result, files)

        if len(files) == 1:
            filename = f"{task_id}.{format_ext}"
            dest = os.path.join(SHARED_RESULTS_DIR, filename)
            shutil.move(str(files[0]), dest)
        else:
            filename = f"{task_id}.tar.gz"
            dest = os.path.join(SHARED_RESULTS_DIR, filename)
            with tarfile.open(dest, "w:gz") as tar:
                for f in files:
                    tar.add(str(f), arcname=f.name)

        # Ensure durability before reporting completion
        dest = os.path.join(SHARED_RESULTS_DIR, filename)
        _fsync_file_and_dir(dest)

        size = os.path.getsize(dest)
        return {"result_file": filename, "size_bytes": size}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
