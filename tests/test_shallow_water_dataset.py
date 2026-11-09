"""Flood dataset discovery, native context and real-worker execution."""

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


@pytest.mark.usefixtures("real_worker")
@pytest.mark.simulation
def test_generated_city_exports_aligned_context_and_supplied_mesh_clears_it(tmp_path, monkeypatch):
    import json
    import meshio
    import numpy as np
    from dtcc_core.model import Mesh

    flat = RectangleMesh(4, 4)
    flat.vertices[:, :2] += [319845., 6397785.]
    flat.transform.srs = "EPSG:3006"
    surface = Mesh(
        vertices=np.array([[319845., 6397785., 0.], [319846., 6397785., 0.],
                           [319845., 6397786., 0.], [319845., 6397785., 12.]]),
        faces=np.array([[0, 1, 2], [0, 1, 3]]), markers=np.array([-2, 0]),
    )
    original = surface.vertices.copy()
    requests = []

    def city_flat_mesh(**kwargs):
        requests.append(("flat", kwargs))
        return flat

    def city_surface_mesh(**kwargs):
        requests.append(("surface", kwargs))
        return surface

    monkeypatch.setattr(datasets, "city_flat_mesh", city_flat_mesh)
    monkeypatch.setattr(datasets, "city_surface_mesh", city_surface_mesh)
    output = tmp_path / "result"
    arguments = dict(bounds=[319845., 6397785., 319846., 6397786.], output_dir=output,
                     h=10., min_building_detail=.5, strict_live=True,
                     parameters=dict(amplitude=0., final_time=.001, n_outputs=1))
    for _ in range(2):
        result = datasets.urban_flood_simulation(**arguments)
        city = meshio.read(output / "city.vtu")
        np.testing.assert_array_equal(city.points, original)
        np.testing.assert_array_equal(city.cell_data_dict["markers"]["triangle"], [-2, 0])
        assert json.loads((output / "city.coordinates.json").read_text()) == json.loads(
            (output / "solution.coordinates.json").read_text())
        assert any(row["path"] == ["export", "city_surface"]
                   for row in result.simulation_diagnostics["run_timings"]["stages"])
        (output / "notes.txt").write_text("keep")
    assert [name for name, _ in requests] == ["flat", "surface", "flat", "surface"]
    flat_request, surface_request = requests[0][1], requests[1][1]
    for key in ("max_mesh_size", "min_building_detail", "center_on_origin", "strict_live"):
        assert surface_request[key] == flat_request[key]
    assert surface_request["bounds"] == arguments["bounds"]
    assert (surface_request["lod"], surface_request["flat_ground"], surface_request["ground_level"]) == ("lod1", True, 0.)
    np.testing.assert_array_equal(surface.vertices, original)

    source = tmp_path / "input.dtcc"
    flat.save(source)
    datasets.urban_flood_simulation(**arguments, mesh_path=source)
    assert len(requests) == 4
    assert not (output / "city.vtu").exists()
    assert not (output / "city.coordinates.json").exists()
    assert (output / "notes.txt").read_text() == "keep"


@pytest.mark.usefixtures("real_worker")
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
    import meshio
    payload = datasets.urban_flood_simulation(**arguments, format="vtu")
    vtu = tmp_path / "download.vtu"
    vtu.write_bytes(payload)
    downloaded = meshio.read(vtu)
    assert "water_depth" in downloaded.point_data
    assert "water_depth" in downloaded.cell_data_dict
    assert len(downloaded.points) == 3*len(downloaded.cells_dict["triangle"])


@pytest.mark.usefixtures("real_worker")
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
