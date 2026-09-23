"""Scalar dataset serialization without a FEniCSx runtime or live providers."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dtcc_core import datasets, io
from dtcc_core.model import Field, VolumeMesh
import dtcc_sim.datasets  # Register simulation datasets.


pytestmark = pytest.mark.simulation


@pytest.fixture(params=["urban_heat_simulation", "air_quality_field"])
def scalar_dataset(request, monkeypatch):
    if request.param == "urban_heat_simulation":
        module_name = "dtcc_sim.urban_heat"
        simulator_name = "UrbanHeatSimulator"
        parameters_name = "UrbanHeatParameters"
        field_name, unit = "temperature", "degC"
    else:
        module_name = "dtcc_sim.smooth_reconstruction"
        simulator_name = "SmoothReconstructionSimulator"
        parameters_name = "SmoothReconstructionParameters"
        field_name, unit = "NO2", "ug/m3"
        sensors = SimpleNamespace(
            to_arrays=lambda **kwargs: (np.array([[0., 0., 0.]]), np.array([20.])),
            stations=lambda: [SimpleNamespace(attributes={"unit": unit})],
        )
        monkeypatch.setattr(datasets, "air_quality", lambda **kwargs: sensors)

    mesh = VolumeMesh(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.],
                           [0., 1., 0.], [0., 0., 1.]]),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        fields=[Field(name=field_name, unit=unit, dim=1, association="vertex",
                      values=np.array([[18.], [19.], [20.], [21.]]))],
    )

    class FakeSimulator:
        solution = None

        def __init__(self, **kwargs):
            pass

        def simulate(self):
            return mesh

    module = ModuleType(module_name)
    setattr(module, simulator_name, FakeSimulator)
    setattr(module, parameters_name, SimpleNamespace)
    monkeypatch.setitem(sys.modules, module_name, module)
    return getattr(datasets, request.param), mesh, FakeSimulator


def test_native_scalar_dataset_roundtrip(scalar_dataset, tmp_path):
    dataset, mesh, _ = scalar_dataset

    payload = dataset(bounds=(0., 0., 1., 1.), format="dtcc")

    assert isinstance(payload, bytes)
    path = tmp_path / "result.dtcc"
    path.write_bytes(payload)
    restored = io.load_model(path)
    assert isinstance(restored, VolumeMesh)
    np.testing.assert_array_equal(restored.vertices, mesh.vertices)
    np.testing.assert_array_equal(restored.cells, mesh.cells)
    assert len(restored.fields) == 1
    field = restored.fields[0]
    assert field.name == mesh.fields[0].name
    assert field.unit == mesh.fields[0].unit
    assert field.association == "vertex"
    assert field.dim == 1
    np.testing.assert_array_equal(field.values, mesh.fields[0].values)
    assert dataset.list_supported_formats() == ["xdmf", "dtcc"]
    assert dataset.multi_file_formats == ("xdmf",)
    assert dataset(bounds=(0., 0., 1., 1.)) is mesh


def test_native_scalar_dataset_rejects_ambiguous_field(scalar_dataset):
    dataset, mesh, _ = scalar_dataset
    mesh.fields[0].association = None

    with pytest.raises(ValueError, match="explicit.*association"):
        dataset(bounds=(0., 0., 1., 1.), format="dtcc")


def test_scalar_dataset_xdmf_keeps_solver_solution_path(scalar_dataset, monkeypatch):
    dataset, _, simulator = scalar_dataset
    solution = simulator.solution = object()

    def export(obj, fmt):
        assert obj is solution
        assert fmt == "xdmf"
        return b"solver-xdmf"

    monkeypatch.setattr(dataset, "export_to_bytes", export)
    assert dataset(bounds=(0., 0., 1., 1.), format="xdmf") == b"solver-xdmf"

    simulator.solution = None
    with pytest.raises(RuntimeError, match="did not produce a FEniCS solution"):
        dataset(bounds=(0., 0., 1., 1.), format="xdmf")
