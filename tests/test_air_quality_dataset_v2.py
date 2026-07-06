import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dtcc_sim.datasets import AirQualityFieldArgs, AirQualityFieldDataset


pytestmark = pytest.mark.simulation


class _FakeSensors:
    def to_arrays(self, *, field_name):
        assert field_name == "NO2"
        return (
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
            np.array([20.0, 25.0], dtype=float),
        )

    def stations(self):
        return [SimpleNamespace(attributes={"unit": "ug/m3"})]


def _patch_air_quality_fetch(monkeypatch):
    import dtcc_core.datasets as datasets

    monkeypatch.setattr(datasets, "air_quality", lambda **kwargs: _FakeSensors())


def _patch_air_quality_fetch_with(monkeypatch, sensors):
    import dtcc_core.datasets as datasets

    monkeypatch.setattr(datasets, "air_quality", lambda **kwargs: sensors)


def _patch_smooth_reconstruction(monkeypatch, simulator_cls):
    module = ModuleType("dtcc_sim.smooth_reconstruction")

    class FakeParameters:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.SmoothReconstructionParameters = FakeParameters
    module.SmoothReconstructionSimulator = simulator_cls
    monkeypatch.setitem(sys.modules, "dtcc_sim.smooth_reconstruction", module)


def test_air_quality_field_xdmf_serializes_fenics_solution(monkeypatch):
    dataset = AirQualityFieldDataset()
    volume_mesh = object()
    solution = object()

    class FakeSimulator:
        def __init__(
            self,
            *,
            bounds,
            point_coords,
            point_values,
            field_name,
            field_unit,
            params,
        ):
            assert (bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax) == (
                0.0,
                0.0,
                1.0,
                1.0,
            )
            assert point_coords.shape == (2, 3)
            assert point_values.tolist() == [20.0, 25.0]
            assert field_name == "NO2"
            assert field_unit == "ug/m3"
            self.solution = solution

        def simulate(self):
            return volume_mesh

    def fake_export(obj, fmt):
        assert obj is solution
        assert fmt == "xdmf"
        return b"solution-xdmf"

    _patch_air_quality_fetch(monkeypatch)
    _patch_smooth_reconstruction(monkeypatch, FakeSimulator)
    monkeypatch.setattr(dataset, "export_to_bytes", fake_export)

    payload = dataset.build(
        AirQualityFieldArgs(bounds=(0.0, 0.0, 1.0, 1.0), format="xdmf")
    )

    assert payload == b"solution-xdmf"


def test_air_quality_field_xdmf_requires_fenics_solution(monkeypatch):
    dataset = AirQualityFieldDataset()
    volume_mesh = object()

    class FakeSimulator:
        def __init__(self, **kwargs):
            self.solution = None

        def simulate(self):
            return volume_mesh

    _patch_air_quality_fetch(monkeypatch)
    _patch_smooth_reconstruction(monkeypatch, FakeSimulator)

    with pytest.raises(RuntimeError, match="air_quality_field did not produce"):
        dataset.build(AirQualityFieldArgs(bounds=(0.0, 0.0, 1.0, 1.0), format="xdmf"))


def test_air_quality_field_without_format_returns_volume_mesh(monkeypatch):
    dataset = AirQualityFieldDataset()
    volume_mesh = object()

    class FakeSimulator:
        def __init__(self, **kwargs):
            self.solution = None

        def simulate(self):
            return volume_mesh

    _patch_air_quality_fetch(monkeypatch)
    _patch_smooth_reconstruction(monkeypatch, FakeSimulator)

    result = dataset.build(AirQualityFieldArgs(bounds=(0.0, 0.0, 1.0, 1.0)))

    assert result is volume_mesh


def test_air_quality_field_requires_station_unit(monkeypatch):
    dataset = AirQualityFieldDataset()

    class SensorsWithoutUnit(_FakeSensors):
        def stations(self):
            return [SimpleNamespace(attributes={})]

    _patch_air_quality_fetch_with(monkeypatch, SensorsWithoutUnit())

    with pytest.raises(RuntimeError, match="non-empty unit"):
        dataset.build(AirQualityFieldArgs(bounds=(0.0, 0.0, 1.0, 1.0)))


def test_air_quality_field_requires_observations(monkeypatch):
    dataset = AirQualityFieldDataset()

    class EmptySensors:
        def to_arrays(self, *, field_name):
            assert field_name == "NO2"
            return np.empty((0, 3)), np.empty(0)

        def stations(self):
            return [SimpleNamespace(attributes={"unit": "ug/m3"})]

    _patch_air_quality_fetch_with(monkeypatch, EmptySensors())

    with pytest.raises(RuntimeError, match="at least one station coordinate"):
        dataset.build(AirQualityFieldArgs(bounds=(0.0, 0.0, 1.0, 1.0)))
