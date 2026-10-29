"""Shared simulator interface and compatibility of parameter keywords."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from dtcc_core.model import VolumeMesh
from dtcc_sim.helmholtz import (
    HelmholtzParameters,
    HelmholtzSolver,
    HelmholtzWorkflow,
)
from dtcc_sim.helmholtz.results import attach_helmholtz_fields


@pytest.fixture(params=[HelmholtzSolver, HelmholtzWorkflow])
def constructor(request, tmp_path):
    (tmp_path / "mesh.xdmf").write_text(
        '<Xdmf><DataItem Format="HDF">mesh.h5:/Mesh</DataItem></Xdmf>'
    )
    (tmp_path / "mesh.h5").write_bytes(b"input mesh data")

    def create(**kwargs):
        return request.param(
            mesh_path=tmp_path / "mesh.xdmf", output_dir=tmp_path / "case", **kwargs
        )

    return create


def test_parameter_keywords_are_aliases_and_conflicts_fail(constructor):
    params = HelmholtzParameters(frequency_hz=5)
    for name in ["params", "parameters"]:
        sim = constructor(**{name: params})
        assert sim.params == params and sim.params is sim.parameters
        assert sim.params is not params
        sim.params = replace(params, frequency_hz=7)
        assert sim.parameters.frequency_hz == 7
        sim.parameters = replace(params, frequency_hz=9)
        assert sim.params.frequency_hz == 9
    with pytest.raises(ValueError, match="not both"):
        constructor(params=params, parameters=params)
    with pytest.raises(TypeError, match="HelmholtzParameters"):
        constructor(params={"frequency_hz": 5})


def test_simulate_returns_dtcc_mesh_fields_and_keeps_diagnostics(
    constructor, monkeypatch, tmp_path
):
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    values = np.array([1 + 2j, 3 - 4j, 0j, 2j])
    returned = VolumeMesh(vertices=vertices, cells=np.array([[0, 1, 2, 3]]))
    units = []

    def convert(*, unit):
        units.append(unit)
        return attach_helmholtz_fields(returned, vertices, values, unit=unit)

    result = SimpleNamespace(
        converged=True, iterations=8, relative_residual=1e-8, to_dtcc=convert,
        mesh_path=tmp_path / "mesh.xdmf"
    )
    sim = constructor(params=HelmholtzParameters())
    monkeypatch.setattr(sim, "run", lambda: result)
    saved = []
    monkeypatch.setattr(
        VolumeMesh, "save", lambda mesh, path: saved.append((mesh, path)), raising=False
    )
    destination = tmp_path / "result.vtu"
    assert sim.simulate(unit="Pa", output_path=destination) is returned
    assert sim.result is result
    assert sim.diagnostics["run_timings"]["status"] == "completed"
    assert {key: value for key, value in sim.diagnostics.items() if key != "run_timings"} == {
        "converged": True,
        "iterations": 8,
        "relative_residual": 1e-8,
    }
    assert units == ["Pa"] and saved == [(returned, destination)]
    fields = {f.name: f for f in returned.fields}
    np.testing.assert_array_equal(fields["helmholtz_real"].values.ravel(), values.real)
    np.testing.assert_array_equal(
        fields["helmholtz_imaginary"].values.ravel(), values.imag
    )
    np.testing.assert_array_equal(
        fields["helmholtz_amplitude"].values.ravel(), abs(values)
    )
    assert all(f.unit == "Pa" for f in fields.values())


@pytest.mark.parametrize(
    "converged,residual", [(False, 1.0), (True, float("nan")), (True, -1.0)]
)
def test_simulate_rejects_failed_or_invalid_results(
    constructor, monkeypatch, converged, residual
):
    sim = constructor()

    def forbidden(**kwargs):
        pytest.fail("Failed results must not become DTCC fields")

    monkeypatch.setattr(
        sim,
        "run",
        lambda: SimpleNamespace(
            converged=converged,
            iterations=1,
            relative_residual=residual,
            to_dtcc=forbidden,
        ),
    )
    with pytest.raises(RuntimeError):
        sim.simulate()
    assert sim.diagnostics["converged"] is converged


def test_simulate_rejects_higher_order_before_running(constructor, monkeypatch):
    sim = constructor(params=HelmholtzParameters(fe_degree=2))
    monkeypatch.setattr(
        sim, "run", lambda: pytest.fail("must fail before numerical work")
    )
    with pytest.raises(ValueError, match="degree 1"):
        sim.simulate()


@pytest.mark.parametrize("suffix", ["vtu", "dtcc", "xdmf"])
def test_simulate_replaces_previous_export(constructor, monkeypatch, tmp_path, suffix):
    sim = constructor()
    destination = tmp_path / f"result.{suffix}"
    destination.write_bytes(b"previous output")
    mesh = VolumeMesh()
    result = SimpleNamespace(
        converged=True, iterations=1, relative_residual=1e-9,
        mesh_path=sim.mesh_path, to_dtcc=lambda **kwargs: mesh,
    )
    monkeypatch.setattr(sim, "run", lambda: result)
    monkeypatch.setattr(VolumeMesh, "save", lambda mesh, path: path.write_bytes(b"new output"))
    assert sim.simulate(output_path=destination) is mesh
    assert destination.read_bytes() == b"new output"


@pytest.mark.parametrize("companion", [False, True])
def test_simulate_protects_input_before_running(constructor, monkeypatch, tmp_path, companion):
    sim = constructor()
    if companion:
        sim.mesh_path.write_text('<Xdmf><DataItem Format="HDF">result.h5:/Mesh</DataItem></Xdmf>')
        protected = tmp_path / "result.h5"
        protected.write_bytes(b"input data")
        destination = tmp_path / "result.xdmf"
    else:
        protected = destination = sim.mesh_path
    original = protected.read_bytes()
    monkeypatch.setattr(sim, "run", lambda: pytest.fail("must protect input before solving"))
    with pytest.raises(ValueError, match="input mesh"):
        sim.simulate(output_path=destination)
    assert protected.read_bytes() == original


def test_simulate_rejects_invalid_export_before_running(
    constructor, monkeypatch, tmp_path
):
    sim = constructor()
    monkeypatch.setattr(
        sim, "run", lambda: pytest.fail("must check format before solving")
    )
    with pytest.raises(ValueError, match="Choose"):
        sim.simulate(output_path=tmp_path / "result.csv")


def test_simulate_rechecks_input_protection_after_running(constructor, monkeypatch, tmp_path):
    sim = constructor()
    destination = tmp_path / "result.xdmf"

    def run():
        destination.write_text("<Xdmf/>")
        return SimpleNamespace(
            converged=True, iterations=1, relative_residual=1e-9,
            mesh_path=destination, to_dtcc=lambda **kwargs: VolumeMesh(),
        )

    monkeypatch.setattr(sim, "run", run)
    with pytest.raises(ValueError, match="input mesh"):
        sim.simulate(output_path=destination)
    assert destination.read_text() == "<Xdmf/>"
