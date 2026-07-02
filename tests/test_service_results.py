import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import dtcc_core.io
from dtcc_core.model import Field, VolumeMesh

from service import results


class _SavableResult:
    def __init__(self, writer):
        self._writer = writer

    def save(self, target):
        self._writer(Path(target))


def _field_volume_mesh(field_name="temperature"):
    mesh = VolumeMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    mesh.add_field(
        Field(name=field_name, values=np.array([18.0, 19.0, 20.0, 21.0]), dim=1)
    )
    return mesh


@pytest.fixture(autouse=True)
def _shared_results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "SHARED_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(results, "_fsync_file_and_dir", lambda path: None)
    return tmp_path


def test_handle_result_writes_bytes_result(tmp_path):
    meta = results.handle_result(b"mesh-bytes", "pb", "job-1")

    assert meta == {"result_file": "job-1.pb", "size_bytes": 10}
    assert (tmp_path / "job-1.pb").read_bytes() == b"mesh-bytes"


def test_handle_result_moves_single_output_file(tmp_path):
    def write_single_file(target):
        target.write_bytes(b"pb-data")

    meta = results.handle_result(_SavableResult(write_single_file), "pb", "job-2")

    assert meta == {"result_file": "job-2.pb", "size_bytes": 7}
    assert (tmp_path / "job-2.pb").read_bytes() == b"pb-data"


def test_handle_result_archives_multi_file_output(tmp_path):
    def write_xdmf_bundle(target):
        target.write_text("<xdmf />")
        target.with_suffix(".h5").write_bytes(b"hdf5-bytes")

    meta = results.handle_result(_SavableResult(write_xdmf_bundle), "xdmf", "job-3")

    assert meta["result_file"] == "job-3.tar.gz"
    archive = tmp_path / "job-3.tar.gz"
    assert archive.exists()

    with tarfile.open(archive, "r:gz") as tar:
        assert sorted(tar.getnames()) == ["data.h5", "data.xdmf"]


@pytest.mark.parametrize("field_name", ["temperature", "NO2"])
def test_handle_result_preserves_volume_mesh_fields_in_xdmf_archive(
    tmp_path, field_name
):
    meta = results.handle_result(
        _field_volume_mesh(field_name), "xdmf", f"job-{field_name}"
    )

    assert meta["result_file"] == f"job-{field_name}.tar.gz"
    archive = tmp_path / f"job-{field_name}.tar.gz"

    with tarfile.open(archive, "r:gz") as tar:
        assert sorted(tar.getnames()) == ["data.h5", "data.xdmf"]
        xdmf = tar.extractfile("data.xdmf").read().decode()
        h5_bytes = tar.extractfile("data.h5").read()

    assert f'Attribute Name="{field_name}"' in xdmf
    h5_path = tmp_path / "extracted-data.h5"
    h5_path.write_bytes(h5_bytes)
    with h5py.File(h5_path, "r") as h5_file:
        fields_group = h5_file["Mesh/mesh/fields"]
        datasets = {
            dataset.attrs["name"]: dataset[()] for dataset in fields_group.values()
        }
        np.testing.assert_allclose(
            datasets[field_name], [18.0, 19.0, 20.0, 21.0]
        )


def test_handle_result_rejects_xdmf_that_drops_expected_fields():
    def write_fieldless_xdmf_bundle(target):
        target.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Tetrahedron" NumberOfElements="1" NodesPerElement="4">
        <DataItem Format="HDF" NumberType="Int" Precision="8" Dimensions="1 4">
          data.h5:/Mesh/mesh/topology
        </DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="4 3">
          data.h5:/Mesh/mesh/geometry
        </DataItem>
      </Geometry>
    </Grid>
  </Domain>
</Xdmf>
"""
        )
        with h5py.File(target.with_suffix(".h5"), "w") as h5_file:
            mesh_group = h5_file.require_group("Mesh/mesh")
            mesh_group.create_dataset(
                "geometry", data=_field_volume_mesh().vertices, dtype="float64"
            )
            mesh_group.create_dataset(
                "topology", data=_field_volume_mesh().cells, dtype="int64"
            )

    result = _SavableResult(write_fieldless_xdmf_bundle)
    result.fields = _field_volume_mesh("temperature").fields

    with pytest.raises(RuntimeError, match="temperature"):
        results.handle_result(result, "xdmf", "job-fieldless")


def test_handle_result_rejects_xdmf_with_missing_hdf5_companion():
    def write_xdmf_without_hdf5(target):
        target.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Tetrahedron" NumberOfElements="1" NodesPerElement="4">
        <DataItem Format="HDF" NumberType="Int" Precision="8" Dimensions="1 4">
          data.h5:/Mesh/mesh/topology
        </DataItem>
      </Topology>
    </Grid>
  </Domain>
</Xdmf>
"""
        )

    with pytest.raises(RuntimeError, match="missing HDF5 companion"):
        results.handle_result(_SavableResult(write_xdmf_without_hdf5), "xdmf", "job-7")


def test_handle_result_requires_format_for_non_bytes():
    with pytest.raises(ValueError, match="format_ext is required"):
        results.handle_result(
            _SavableResult(lambda target: target.write_bytes(b"x")), None, "job-4"
        )


def test_handle_result_rejects_object_without_save():
    with pytest.raises(TypeError, match="has no save"):
        results.handle_result(object(), "pb", "job-5")


def test_handle_result_rejects_empty_serialization_output():
    with pytest.raises(RuntimeError, match="produced no files"):
        results.handle_result(_SavableResult(lambda target: None), "pb", "job-6")
