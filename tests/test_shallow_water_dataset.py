"""Flood dataset discovery, native context and real-worker execution."""

import os

import pytest
from pydantic import ValidationError

from dtcc_core import datasets
from dtcc_core import RectangleMesh
import dtcc_sim.datasets
from dtcc_sim.shallow_water.dataset import UrbanFloodSimulationArgs


def test_flood_dataset_contract(tmp_path):
    description = datasets.urban_flood_simulation.describe()
    assert description["name"] == "urban_flood_simulation"
    assert description["python_return_type"] == "dtcc_core.model.Mesh"
    assert set(description["supported_formats"]) == {"dtccpkg", "dtcc", "vtu"}
    args = UrbanFloodSimulationArgs(bounds=[0, 0, 1, 1], output_dir=tmp_path)
    assert args.parameters.gravity == 9.81
    with pytest.raises(ValidationError):
        UrbanFloodSimulationArgs(bounds=[0, 0, 1, 1], output_dir=tmp_path, parameters={"gravity": -1})
    with pytest.raises(ValidationError, match="Invalid bounds"):
        UrbanFloodSimulationArgs(bounds=[1, 0, 0, 1], output_dir=tmp_path)


@pytest.mark.skipif(not os.environ.get("DTCC_SWE_PYTHON"), reason="Set DTCC_SWE_PYTHON for real worker tests")
@pytest.mark.simulation
def test_native_file_dataset_and_package_use_real_solver(tmp_path):
    from dtcc_core.datasets import load_model_package

    mesh = RectangleMesh(4, 4); mesh.transform.srs = "EPSG:3006"
    source = tmp_path / "input.dtcc"; mesh.save(source)
    arguments = dict(bounds=[0, 0, 1, 1], mesh_path=source, output_dir=tmp_path / "case",
        parameters=dict(h0=0., compact_pulse=True, sigma=.2, final_time=.02, n_outputs=1))
    result = datasets.urban_flood_simulation(**arguments)
    assert result.transform.srs == "EPSG:3006"
    assert result.simulation_diagnostics["passed"]
    assert result.dataset_context.metadata.crs == ["EPSG:3006"]
    steps = result.dataset_context.provenance.processing_steps
    assert any(isinstance(step, dict) and step.get("backend") == "cecilia-reference" for step in steps)
    payload = datasets.urban_flood_simulation(**arguments, format="dtccpkg")
    package = tmp_path / "result.dtccpkg"; package.write_bytes(payload)
    restored = load_model_package(package)
    assert {field.name for field in restored.fields} == {field.name for field in result.fields}
    assert restored.dataset_context.metadata.crs == ["EPSG:3006"]
    assert any(isinstance(step, dict) and step.get("backend") == "cecilia-reference" for step in restored.dataset_context.provenance.processing_steps)
