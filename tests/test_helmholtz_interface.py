"""Shared simulator interface and compatibility of parameter keywords."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from dtcc_core.model import VolumeMesh
from dtcc_sim.helmholtz import (
    HelmholtzParameters,
    HelmholtzSimulator,
    HelmholtzWorkflow,
)
from dtcc_sim.helmholtz.results import attach_helmholtz_fields


@pytest.fixture(params=[HelmholtzSimulator, HelmholtzWorkflow])
def constructor(request, tmp_path):
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
        converged=True, iterations=8, relative_residual=1e-8, to_dtcc=convert
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
    assert sim.diagnostics == {
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


@pytest.mark.parametrize(
    "existing",
    [
        "result.vtu",
        "result.dtcc",
        "result.xdmf",
        "result.h5",
        "result_boundary_markers.xdmf",
    ],
)
def test_simulate_protects_exports_before_running(
    constructor, monkeypatch, tmp_path, existing
):
    protected = tmp_path / existing
    protected.write_bytes(b"original data")
    destination = (
        protected if protected.suffix in (".vtu", ".dtcc") else tmp_path / "result.xdmf"
    )
    sim = constructor()
    monkeypatch.setattr(
        sim, "run", lambda: pytest.fail("must check destination before solving")
    )
    with pytest.raises(FileExistsError):
        sim.simulate(output_path=destination)
    assert protected.read_bytes() == b"original data"


def test_simulate_rejects_invalid_export_before_running(
    constructor, monkeypatch, tmp_path
):
    sim = constructor()
    monkeypatch.setattr(
        sim, "run", lambda: pytest.fail("must check format before solving")
    )
    with pytest.raises(ValueError, match="Choose"):
        sim.simulate(output_path=tmp_path / "result.csv")


def test_simulate_rechecks_export_after_running(constructor, monkeypatch, tmp_path):
    sim = constructor()
    destination = tmp_path / "result.vtu"

    def run():
        destination.write_bytes(b"created during solve")
        return SimpleNamespace(
            converged=True,
            iterations=1,
            relative_residual=1e-9,
            to_dtcc=lambda **kw: object(),
        )

    monkeypatch.setattr(sim, "run", run)
    with pytest.raises(FileExistsError):
        sim.simulate(output_path=destination)
    assert destination.read_bytes() == b"created during solve"
