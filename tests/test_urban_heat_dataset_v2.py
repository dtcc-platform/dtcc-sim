"""Tests for urban heat Dataset v2 return behavior."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from dtcc_sim.datasets import UrbanHeatSimulationArgs, UrbanHeatSimulationDataset

pytestmark = [pytest.mark.simulation, pytest.mark.fenics]

dolfinx = pytest.importorskip(
    "dolfinx",
    reason="urban heat dataset tests require dolfinx",
)

from dtcc_sim.urban_heat import UrbanHeatParameters, UrbanHeatSimulator


def _box_mesh_with_heat_markers():
    import dolfinx.mesh
    from mpi4py import MPI

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [2, 2, 2],
        dolfinx.mesh.CellType.tetrahedron,
    )
    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, fdim, boundary_facets)
    marker_values = np.full(len(boundary_facets), -2, dtype=np.int32)
    tol = 1.0e-10
    for i, midpoint in enumerate(midpoints):
        x, _y, z = midpoint
        if abs(x) < tol:
            marker_values[i] = 0  # wall category after collapse
        elif abs(z - 1.0) < tol:
            marker_values[i] = 1  # roof category after collapse
        elif abs(z) < tol:
            marker_values[i] = -1  # ground category after collapse

    order = np.argsort(boundary_facets)
    return dolfinx.mesh.meshtags(
        mesh,
        fdim,
        boundary_facets[order],
        marker_values[order],
    ), mesh


def test_urban_heat_attaches_temperature_field_to_dtcc_mesh(monkeypatch):
    calls = {}
    volume_mesh = SimpleNamespace()
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

    monkeypatch.setattr(
        "dtcc_sim.urban_heat._attach_scalar_field_to_volume_mesh",
        fake_attach,
    )

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


def test_urban_heat_dataset_passes_boundary_parameters(monkeypatch):
    dataset = UrbanHeatSimulationDataset()
    volume_mesh = object()
    seen = {}

    class FakeSimulator:
        def __init__(self, *, bounds, params):
            seen["bounds"] = bounds
            seen["params"] = params
            self.solution = object()

        def simulate(self):
            return volume_mesh

    monkeypatch.setattr("dtcc_sim.urban_heat.UrbanHeatSimulator", FakeSimulator)

    result = dataset.build(
        UrbanHeatSimulationArgs(
            bounds=(0.0, 0.0, 1.0, 1.0),
            wall_bc_type="dirichlet",
            wall_value=12.0,
            roof_bc_type="neumann",
            roof_flux=0.2,
            ground_bc_type="robin",
            ground_h=3.5,
            open_bc_type="dirichlet",
            open_value=4.0,
        )
    )

    assert result is volume_mesh
    params = seen["params"]
    assert params.wall_bc_type == "dirichlet"
    assert params.wall_value == 12.0
    assert params.roof_bc_type == "neumann"
    assert params.roof_flux == 0.2
    assert params.ground_bc_type == "robin"
    assert params.ground_h == 3.5
    assert params.open_bc_type == "dirichlet"
    assert params.open_value == 4.0


def test_urban_heat_dataset_advertises_native_volume_mesh():
    descriptor = UrbanHeatSimulationDataset().describe()

    assert descriptor["python_return_type"] == "dtcc_core.model.VolumeMesh"
    assert descriptor["result_kind"] == "mesh"


@pytest.mark.slow
def test_urban_heat_tiny_box_dirichlet_solution_is_bounded():
    markers, mesh = _box_mesh_with_heat_markers()
    params = UrbanHeatParameters(
        kappa=1.0,
        sigma=0.0,
        degree=1,
        wall_bc_type="dirichlet",
        wall_value=10.0,
        roof_bc_type="dirichlet",
        roof_value=0.0,
        ground_bc_type="dirichlet",
        ground_value=0.0,
        open_bc_type="dirichlet",
        open_value=0.0,
    )
    sim = UrbanHeatSimulator(
        mesh=mesh,
        markers=markers,
        params=params,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    solution = sim.simulate()
    values = np.real(np.asarray(solution.x.array))

    assert np.all(np.isfinite(values))
    assert float(np.min(values)) >= -1.0e-8
    assert float(np.max(values)) <= 10.0 + 1.0e-8
    assert float(np.max(values)) > 1.0
    assert sim.diagnostics["degrees_of_freedom"] > 0
    assert sim.diagnostics["temperature_min"] >= -1.0e-8
    assert sim.diagnostics["temperature_max"] <= 10.0 + 1.0e-8
    assert sim.diagnostics["boundary_category_counts"]["wall"] > 0
    assert sim.diagnostics["boundary_category_counts"]["open"] > 0
    assert sim.diagnostics["residual_status"].startswith("not_available")
