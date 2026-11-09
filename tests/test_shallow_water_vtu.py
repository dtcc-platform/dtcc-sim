"""DG1 visualization preserves nodal values, jumps and input coordinates."""

import json
import shutil
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from dtcc_core import RectangleMesh
from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver
from dtcc_sim.shallow_water import ShallowWaterResult


@pytest.fixture(scope="module")
def dg_run(tmp_path_factory, real_worker):
    mesh = RectangleMesh(4, 4)
    mesh.faces = mesh.faces[::-1, [1, 2, 0]].copy()
    mesh.vertices[:, 2] = .02*mesh.vertices[:, 0]
    mesh.vertices[:, :2] += [319845., 6397785.]
    mesh.markers = np.full(len(mesh.faces), -1, dtype=np.int32)
    mesh.transform.srs = "EPSG:3006"
    params = ShallowWaterParameters(h0=.1, amplitude=.2, sigma=.2, bed_from_mesh=True,
        x_center=319845.3, y_center=6397785.4, final_time=.005, n_outputs=2)
    solver = ShallowWaterSolver(mesh=mesh, params=params,
                               output_dir=tmp_path_factory.mktemp("dg1"))
    native = solver.simulate()
    return solver, native


def test_exact_dg_nodes_and_cell_means_in_input_order(dg_run):
    import meshio

    solver, native = dg_run
    path = solver.output_dir / "solution_finaly.vtu"
    exported = meshio.read(path)
    count = len(native.faces)
    np.testing.assert_array_equal(exported.points, native.vertices[native.faces].reshape(-1, 3))
    np.testing.assert_array_equal(exported.cells_dict["triangle"], np.arange(3*count).reshape(-1, 3))
    np.testing.assert_array_equal(exported.cell_data_dict["markers"]["triangle"], native.markers)
    assert json.loads(path.with_suffix(".coordinates.json").read_text())["input_crs"] == "EPSG:3006"
    for field in native.fields:
        np.testing.assert_array_equal(exported.cell_data_dict[field.name]["triangle"],
                                      field.values.squeeze())
    with np.load(solver.result.case_dir / "solver/state.npz") as raw:
        cells = raw["cell_dofs"]
        xy = raw["coordinates"][cells, :2]
        point_values = np.stack([raw["state"][raw[key]][cells]
                                for key in ("h_dofs", "hu_dofs", "hv_dofs")], axis=-1)
        offset = native.vertices[:, :2].min(axis=0)
        # Match each complete triangle independently; coincident nodes can differ.
        key = lambda coordinates: tuple(sorted(tuple(v) for v in coordinates.round(12)))
        by_triangle = {key(coords): (coords, values, raw["bed"][dofs])
                       for coords, values, dofs in zip(xy, point_values, cells, strict=True)}
        expected, beds = [], []
        for triangle in native.vertices[native.faces, :2] - offset:
            coords, values, bed = by_triangle[key(triangle)]
            for node in triangle:
                index, = np.flatnonzero(np.linalg.norm(coords-node, axis=1) < 1e-12)
                expected.append(values[index]); beds.append(bed[index])
        expected = np.asarray(expected)
        np.testing.assert_array_equal(exported.point_data["water_depth"], expected[:, 0])
        np.testing.assert_array_equal(exported.point_data["water_momentum"][:, :2], expected[:, 1:])
        np.testing.assert_array_equal(exported.point_data["bed_elevation"], beds)
        np.testing.assert_array_equal(exported.point_data["water_surface_elevation"], beds + expected[:, 0])
    np.testing.assert_allclose(exported.point_data["water_velocity"][:, :2],
                               expected[:, 1:]/expected[:, :1])
    initial = solver.result.to_visualization_mesh(0)
    fields = {f.name: f.values for f in initial.fields if f.association == "vertex"}
    xy = initial.vertices[:, :2]
    expected_initial = .1 + .2*np.exp(-np.sum((xy-[319845.3, 6397785.4])**2, axis=1)/(2*.2**2))
    np.testing.assert_allclose(fields["water_depth"][:, 0], expected_initial, atol=1e-10)


def test_vtk_warp_uses_point_data_and_preserves_jumps(dg_run):
    pv = pytest.importorskip("pyvista")
    from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
    from vtkmodules.vtkFiltersGeneral import vtkWarpScalar
    solver, _ = dg_run
    water = pv.read(solver.output_dir / "solution_finaly.vtu")
    warp = vtkWarpScalar()
    warp.SetInputData(water)
    warp.SetInputArrayToProcess(0, 0, 0, 0, "water_depth")
    warp.SetScaleFactor(1.)
    warp.SetUseNormal(True)
    warp.SetNormal(0., 0., 1.)
    warp.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    warp.Update()
    warped = pv.wrap(warp.GetOutput())
    np.testing.assert_array_equal(warped.points[:, :2], water.points[:, :2])
    np.testing.assert_allclose(warped.points[:, 2], water.points[:, 2]+water.point_data["water_depth"])
    xy = water.points[:, :2]
    _, inverse = np.unique(xy, axis=0, return_inverse=True)
    assert any(np.ptp(water.point_data["water_depth"][inverse == i]) > 1e-8
               for i in np.unique(inverse)), "Adjacent DG traces must remain independent"
    assert warped.n_points == water.n_points == 3*water.n_cells


def test_series_has_dg_snapshots_and_paraview_time_index(dg_run, tmp_path):
    import meshio
    solver, _ = dg_run
    (tmp_path / "notes.txt").write_text("keep")
    entries = solver.result.export_series(tmp_path)
    index = ET.parse(tmp_path / "water.pvd").findall("./Collection/DataSet")
    assert [float(e.attrib["timestep"]) for e in index] == list(solver.result.snapshot_times)
    assert [e.attrib["file"] for e in index] == [e["vtu"] for e in entries]
    for entry in entries:
        water = meshio.read(tmp_path / entry["vtu"])
        for name in ("water_depth", "water_momentum", "bed_elevation"):
            nodal = water.point_data[name].reshape(len(water.cells_dict["triangle"]), 3, -1)
            means = water.cell_data_dict[name]["triangle"].reshape(len(nodal), -1)
            np.testing.assert_allclose(nodal.mean(axis=1), means, atol=1e-14)
    assert (tmp_path / "notes.txt").read_text() == "keep"
    pv = pytest.importorskip("pyvista")
    reader = pv.get_reader(tmp_path / "water.pvd")
    assert reader.time_values == list(solver.result.snapshot_times)
    assert "water_depth" in reader.read()[0].point_data


def test_cell_only_run_recovers_exact_endpoint_polynomials(dg_run, tmp_path):
    solver, _ = dg_run
    case = tmp_path / "run"
    shutil.copytree(solver.result.case_dir, case)
    for path in (case / "solver").glob("snapshot_*.npz"):
        with np.load(path) as raw:
            arrays = {k: raw[k] for k in raw.files if not k.startswith("dg_")}
        np.savez(path, **arrays)
    recovered = ShallowWaterResult.load(case)
    for index in (0, -1):
        expected = solver.result.to_visualization_mesh(index)
        actual = recovered.to_visualization_mesh(index)
        for a, b in zip(actual.fields, expected.fields, strict=True):
            np.testing.assert_array_equal(a.values, b.values)
    with pytest.raises(ValueError, match="rerun"):
        recovered.to_visualization_mesh(1)


@pytest.mark.parametrize("fault", ["missing", "shape", "means", "nonfinite"])
def test_invalid_dg_snapshot_fails_explicitly(dg_run, tmp_path, fault):
    solver, _ = dg_run
    case = tmp_path / "run"
    shutil.copytree(solver.result.case_dir, case)
    path = sorted((case / "solver").glob("snapshot_*.npz"))[-1]
    with np.load(path) as raw:
        arrays = {k: raw[k].copy() for k in raw.files}
    if fault == "missing":
        del arrays["dg_depth"]
    elif fault == "shape":
        arrays["dg_depth"] = arrays["dg_depth"][:, :2]
    elif fault == "means":
        arrays["dg_depth"] += 1.
    else:
        arrays["dg_depth"][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="DG1"):
        ShallowWaterResult.load(case).to_visualization_mesh()


def test_root_animation_references_series_and_removes_owned_old_names(dg_run, tmp_path):
    solver, _ = dg_run
    frames = tmp_path / 'series'
    frames.mkdir()
    (frames / 'water.pvd').write_text('old animation index')
    (frames / 'notes.txt').write_text('keep')
    solver.result.export_series(frames, index_path=tmp_path / 'solution.pvd')
    assert not (frames / 'water.pvd').exists()
    assert (frames / 'notes.txt').read_text() == 'keep'
    entries = ET.parse(tmp_path / 'solution.pvd').findall('./Collection/DataSet')
    assert all(e.attrib['file'].startswith('series/water_') for e in entries)
    assert all((tmp_path / e.attrib['file']).is_file() for e in entries)
    assert not (solver.output_dir / 'solution.vtu').exists()
    pv = pytest.importorskip('pyvista')
    reader = pv.get_reader(tmp_path / 'solution.pvd')
    assert reader.time_values == list(solver.result.snapshot_times)
    assert 'water_depth' in reader.read()[0].point_data
