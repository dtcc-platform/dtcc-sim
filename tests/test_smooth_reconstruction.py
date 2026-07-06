"""Tests for smooth reconstruction Dataset v2 output helpers."""

from __future__ import annotations

import numpy as np
import pytest

from dtcc_core.model import Field

pytestmark = [pytest.mark.simulation, pytest.mark.fenics]

pytest.importorskip("dolfinx", reason="smooth reconstruction tests require dolfinx")

from dtcc_sim.smooth_reconstruction import (
    SmoothReconstructionParameters,
    SmoothReconstructionSimulator,
    _attach_scalar_field_to_volume_mesh,
    _validate_attached_scalar_field,
)


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


def test_validate_attached_scalar_field_rejects_wrong_length():
    volume_mesh = _FakeVolumeMesh()
    volume_mesh.fields = [
        Field(name="NO2", values=np.array([[10.0], [20.0], [30.0]]), dim=1)
    ]

    with pytest.raises(RuntimeError, match="expected 2"):
        _validate_attached_scalar_field(volume_mesh, name="NO2", unit="")


def test_prepare_point_data_filters_nonfinite_and_applies_z_offset():
    sim = SmoothReconstructionSimulator(
        mesh=object(),
        point_coords=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]),
        point_values=np.array([10.0, np.nan]),
        params=SmoothReconstructionParameters(z_offset=0.5),
    )

    points, values = sim._prepare_point_data()

    assert points.tolist() == [[0.0, 0.0, 1.5]]
    assert values.tolist() == [10.0]
    assert sim._raw_observation_count == 2
    assert sim._dropped_nonfinite_observation_count == 1


def test_tiny_reconstruction_records_diagnostics():
    import dolfinx.mesh
    from mpi4py import MPI

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [1, 1, 1],
        dolfinx.mesh.CellType.tetrahedron,
    )
    sim = SmoothReconstructionSimulator(
        mesh=mesh,
        point_coords=np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]),
        point_values=np.array([10.0, 20.0]),
        field_name="NO2",
        field_unit="ug/m3",
        params=SmoothReconstructionParameters(
            lambda_smooth=0.1,
            alpha=1.0,
            data_weight=10.0,
        ),
        petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-10, "pc_type": "jacobi"},
    )

    solution = sim.simulate()
    values = np.real(np.asarray(solution.x.array))

    assert np.all(np.isfinite(values))
    diagnostics = sim.diagnostics
    assert diagnostics["field_name"] == "NO2"
    assert diagnostics["field_unit"] == "ug/m3"
    assert diagnostics["background_value"] == 15.0
    assert diagnostics["raw_observation_count"] == 2
    assert diagnostics["valid_observation_count"] == 2
    assert diagnostics["located_observation_count"] == 2
    assert diagnostics["points_outside_mesh"] == 0
    assert diagnostics["constraints_added"] == 2
    assert diagnostics["field"]["finite"] is True
    assert diagnostics["field"]["unit"] == "ug/m3"
    assert diagnostics["linear_solve"]["converged_reason"] > 0
