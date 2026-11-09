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
    from dtcc_sim.shallow_water.solver import _reference
    completed = next(step for step in restored.dataset_context.provenance.processing_steps
                     if isinstance(step, dict) and step.get("backend") == "cecilia-reference")
    assert completed["reference"] == _reference()


@pytest.mark.skipif(not os.environ.get("DTCC_SWE_PYTHON"), reason="Set DTCC_SWE_PYTHON for real worker tests")
@pytest.mark.simulation
def test_moved_package_plots_without_numerical_run(tmp_path):
    import shutil
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image
    from dtcc_core.datasets import load_model_package
    from dtcc_sim.shallow_water.solver import _reference

    mesh = RectangleMesh(4, 4)
    centers = mesh.vertices[mesh.faces].mean(axis=1)
    mesh.markers = np.where(centers[:, 0] > .5, 0, -1).astype(np.int32)
    mesh.vertices[:, :2] += [319845., 6397785.]
    mesh.transform.srs = "EPSG:3006"
    source = tmp_path / "input.dtcc"
    mesh.save(source)
    run = tmp_path / "numerical"
    result = datasets.urban_flood_simulation(
        bounds=[319845., 6397785., 319846., 6397786.], mesh_path=source, output_dir=run,
        parameters=dict(h0=.01, amplitude=.2, sigma=.12, x_center=319845.8,
                        y_center=6397785.5, final_time=.002, n_outputs=1),
    )
    package = tmp_path / "flood.dtccpkg"
    result.export(package, canonical=True)
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    moved = consumer / package.name
    shutil.move(package, moved)
    shutil.rmtree(run)
    source.unlink()
    restored = load_model_package(moved)
    assert not run.exists() and not source.exists()
    np.testing.assert_array_equal(restored.vertices, result.vertices)
    np.testing.assert_array_equal(restored.faces, result.faces)
    np.testing.assert_array_equal(restored.markers, result.markers)
    assert restored.transform.srs == "EPSG:3006"
    np.testing.assert_array_equal(restored.transform.affine, result.transform.affine)
    for actual, expected in zip(restored.fields, result.fields, strict=True):
        assert (actual.name, actual.unit, actual.association, actual.dim) == (
            expected.name, expected.unit, expected.association, expected.dim)
        np.testing.assert_array_equal(actual.values, expected.values)
    completed = next(step for step in restored.dataset_context.provenance.processing_steps
                     if isinstance(step, dict) and step.get("backend") == "cecilia-reference")
    assert completed["reference"] == _reference()
    assert restored.dataset_context.metadata.crs == ["EPSG:3006"]
    assert not hasattr(restored, "simulation_diagnostics")
    context = restored.markers >= 0
    depth = next(f.values[:, 0] for f in restored.fields if f.name == "water_depth")
    assert depth[context].max() > 5 * depth[~context].max()
    before = moved.read_bytes()
    ax = restored.plot(projection="xy", field="water_depth", context_mask=context,
                       title="Water depth · t = 0.002 s", coordinate_unit="m",
                       scale_bar=True, vmin=0., cmap="Blues", show=False)
    layers = {c.get_gid(): c for c in ax.collections}
    water = layers["face_field"]
    np.testing.assert_array_equal(water.get_array(), depth[~context])
    assert water.norm.vmax == depth[~context].max()
    for path, face in zip(water.get_paths(), restored.faces[~context], strict=True):
        np.testing.assert_array_equal(path.vertices[:3], restored.vertices[face, :2])
    assert len(layers["context_outline"].get_segments()) == 12
    assert ax.figure.axes[1].get_ylabel() == "water_depth [m]"
    image = consumer / "flood.png"
    ax.figure.savefig(image, dpi=100)
    pixels = np.asarray(Image.open(image).convert("RGB"))
    assert np.unique(pixels.reshape(-1, 3), axis=0).shape[0] > 100
    assert moved.read_bytes() == before
    plt.close(ax.figure)
