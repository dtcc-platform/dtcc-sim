"""Tests for urban heat Dataset v2 return behavior."""

from __future__ import annotations

import pytest

from dtcc_sim.datasets import UrbanHeatSimulationArgs, UrbanHeatSimulationDataset

pytest.importorskip("dolfinx", reason="urban heat dataset tests require dolfinx")

from dtcc_sim.urban_heat import UrbanHeatSimulator


def test_urban_heat_attaches_temperature_field_to_dtcc_mesh(monkeypatch):
    calls = {}
    volume_mesh = object()
    solution = object()
    dolfinx_mesh = object()

    def fake_attach(mesh, solved, dtcc_mesh, *, name, unit):
        calls.update(
            {
                "mesh": mesh,
                "solution": solved,
                "dtcc_mesh": dtcc_mesh,
                "name": name,
                "unit": unit,
            }
        )
        return dtcc_mesh

    monkeypatch.setattr("dtcc_sim.urban_heat._attach_scalar_field_to_volume_mesh", fake_attach)

    sim = UrbanHeatSimulator(bounds=[0.0, 0.0, 1.0, 1.0])
    sim.mesh = dolfinx_mesh
    sim.solution = solution
    sim.volume_mesh_dtcc = volume_mesh

    assert sim._attach_temperature_field() is volume_mesh
    assert calls == {
        "mesh": dolfinx_mesh,
        "solution": solution,
        "dtcc_mesh": volume_mesh,
        "name": "temperature",
        "unit": "degC",
    }


def test_urban_heat_dataset_xdmf_serializes_fenics_solution(monkeypatch):
    dataset = UrbanHeatSimulationDataset()
    volume_mesh = object()
    solution = object()

    class FakeSimulator:
        def __init__(self, *, bounds, params):
            self.bounds = bounds
            self.params = params
            self.solution = solution

        def simulate(self):
            return volume_mesh

    def fake_export(obj, fmt):
        assert obj is solution
        assert fmt == "xdmf"
        return b"xdmf"

    monkeypatch.setattr("dtcc_sim.urban_heat.UrbanHeatSimulator", FakeSimulator)
    monkeypatch.setattr(dataset, "export_to_bytes", fake_export)

    payload = dataset.build(
        UrbanHeatSimulationArgs(bounds=(0.0, 0.0, 1.0, 1.0), format="xdmf")
    )

    assert payload == b"xdmf"


def test_urban_heat_dataset_advertises_native_volume_mesh():
    descriptor = UrbanHeatSimulationDataset().describe()

    assert descriptor["python_return_type"] == "dtcc_core.model.VolumeMesh"
    assert descriptor["result_kind"] == "mesh"
