"""Tests for smooth reconstruction Dataset v2 output helpers."""

from __future__ import annotations

import numpy as np

from dtcc_core.model import Field
from dtcc_sim.smooth_reconstruction import _attach_scalar_field_to_volume_mesh


class _FakeGeometry:
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


class _FakeDolfinxMesh:
    geometry = _FakeGeometry()


class _FakeVolumeMesh:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def __init__(self):
        self.fields = [Field(name="existing", values=np.array([[1.0], [2.0]]), dim=1)]


def test_attach_scalar_field_to_volume_mesh_replaces_named_field(monkeypatch):
    import dtcc_sim.urban_wind as urban_wind

    monkeypatch.setattr(
        urban_wind,
        "_evaluate_function_at_vertices",
        lambda _mesh, _solution: np.array([10.0, 20.0]),
    )
    monkeypatch.setattr(
        urban_wind,
        "_reorder_by_coords",
        lambda _source_keys, values, _target_keys: values,
    )
    volume_mesh = _FakeVolumeMesh()

    result = _attach_scalar_field_to_volume_mesh(
        _FakeDolfinxMesh(),
        object(),
        volume_mesh,
        name="NO2",
        unit="ug/m3",
    )

    assert result is volume_mesh
    assert [field.name for field in result.fields] == ["existing", "NO2"]
    assert result.fields[-1].unit == "ug/m3"
    assert result.fields[-1].dim == 1
    assert result.fields[-1].values.tolist() == [[10.0], [20.0]]
