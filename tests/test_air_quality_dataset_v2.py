import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dtcc_sim.datasets import AirQualityFieldArgs, AirQualityFieldDataset


pytestmark = pytest.mark.simulation


class _FakeSensors:
    def __init__(self, points=None, values=None, sources=None):
        self.points = np.asarray(
            points if points is not None else [[0., 0., 0.], [1., 1., 0.]], dtype=float
        )
        self.values = np.asarray(values if values is not None else [20., 25.], dtype=float)
        self.sources = sources if sources is not None else ["upstream"] * len(self.points)

    def to_arrays(self, *, field_name):
        assert field_name == "NO2"
        return self.points, self.values

    def stations(self):
        return [
            SimpleNamespace(attributes={
                "unit": "ug/m3",
                "station_id": f"s{i}",
                "station_name": f"Station {i}",
                "elevation_source": source,
            })
            for i, source in enumerate(self.sources)
        ]


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
            point_labels,
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
            assert point_labels == ["Station 0 (id=s0)", "Station 1 (id=s1)"]
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


@pytest.mark.parametrize("height", [None, 4.0], ids=["default-height", "configured-height"])
def test_missing_elevation_uses_local_ground_and_preserves_supplied_zero(monkeypatch, height):
    sensors = _FakeSensors(
        points=[[.25, .25, 0.], [.75, .25, 0.], [.9, .9, 0.]],
        values=[20., 25., np.nan], sources=["missing", "upstream", "missing"],
    )
    ground = SimpleNamespace(
        vertices=np.array([[0., 0., 3.], [1., 0., 5.], [0., 1., 3.]]),
        boundary_faces=np.array([[0, 1, 2]]), boundary_markers=np.array([-1]),
    )
    loaded = []

    class FakeSimulator:
        def __init__(self, **kwargs):
            self.point_coords = kwargs["point_coords"]
            self.volume_mesh_dtcc = ground
            assert kwargs["params"].kwargs["z_offset"] == .5

        def _load_if_needed(self):
            loaded.append(True)

        def simulate(self):
            expected_height = 2.0 if height is None else height
            assert self.point_coords[:, 2].tolist() == [3.5 + expected_height, 0., 0.]
            return ground

    _patch_air_quality_fetch_with(monkeypatch, sensors)
    _patch_smooth_reconstruction(monkeypatch, FakeSimulator)
    kwargs = {} if height is None else {"station_height_above_ground": height}
    args = AirQualityFieldArgs(bounds=(0., 0., 1., 1.), z_offset=.5, **kwargs)
    assert AirQualityFieldDataset().build(args) is ground
    assert loaded == [True]
    assert sensors.points[:, 2].tolist() == [0., 0., 0.]


def test_missing_elevation_reports_station_without_ground_coverage():
    mesh = SimpleNamespace(
        vertices=np.array([[0., 0., 3.], [1., 0., 3.], [0., 1., 3.]]),
        boundary_faces=np.array([[0, 1, 2]]), boundary_markers=np.array([-1]),
    )
    with pytest.raises(RuntimeError, match=r"Femman \(id=87\).*no ground boundary"):
        AirQualityFieldDataset._resolve_missing_elevations(
            mesh, np.array([[.9, .9, 0.]]), np.array([True]), ["Femman (id=87)"], 2.,
        )


def test_air_quality_field_requires_explicit_elevation_provenance(monkeypatch):
    _patch_air_quality_fetch_with(monkeypatch, _FakeSensors(sources=[None, "upstream"]))
    with pytest.raises(RuntimeError, match=r"Station 0 \(id=s0\).*elevation_source"):
        AirQualityFieldDataset().build(AirQualityFieldArgs(bounds=(0., 0., 1., 1.)))
