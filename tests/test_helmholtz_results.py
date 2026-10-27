"""Check complex field mapping and persistent visualization exports."""

from types import SimpleNamespace

import numpy as np
import pytest

from dtcc_core.model import Field, VolumeMesh
from dtcc_sim.helmholtz.results import attach_helmholtz_fields, save_dtcc_result


@pytest.fixture
def vertices():
    return np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])


def test_reordered_vertices_receive_correct_complex_components(vertices):
    mesh = VolumeMesh(vertices=vertices[[2, 0, 3, 1]], cells=np.array([[0, 1, 2, 3]]))
    unrelated = Field(name="other", association="vertex", values=np.ones((4, 1)))
    mesh.fields = [unrelated]
    values = np.array([3 + 4j, -2 + 1j, 0 - 5j, 1 + 0j])
    assert attach_helmholtz_fields(mesh, vertices, values, unit="Pa") is mesh
    fields = {f.name: f for f in mesh.fields}
    np.testing.assert_allclose(fields["helmholtz_real"].values.ravel(), [0, 3, 1, -2])
    np.testing.assert_allclose(
        fields["helmholtz_imaginary"].values.ravel(), [-5, 4, 0, 1]
    )
    np.testing.assert_allclose(
        fields["helmholtz_amplitude"].values.ravel(), [5, 5, 1, np.sqrt(5)]
    )
    assert mesh.fields[0] is unrelated
    for name in ("helmholtz_real", "helmholtz_imaginary", "helmholtz_amplitude"):
        assert (
            fields[name].unit == "Pa"
            and fields[name].association == "vertex"
            and fields[name].dim == 1
        )
    attach_helmholtz_fields(mesh, vertices, values)
    assert len(mesh.fields) == 4


@pytest.mark.parametrize(
    "invalid", ["translated", "duplicate", "nonfinite", "wrong_count", "ambiguous"]
)
def test_invalid_mapping_is_rejected(vertices, invalid):
    target = vertices.copy()
    values = np.ones(4, dtype=complex)
    tolerance = 1.0e-8
    if invalid == "translated":
        target += 100
    elif invalid == "duplicate":
        target[1] = target[0]
    elif invalid == "nonfinite":
        values[0] = np.nan
    elif invalid == "wrong_count":
        values = values[:3]
    else:
        tolerance = 2.0
    mesh = VolumeMesh(vertices=target, cells=np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError):
        attach_helmholtz_fields(mesh, vertices, values, tolerance=tolerance)
    assert mesh.fields == []


def test_existing_export_is_not_overwritten(tmp_path):
    path = tmp_path / "result.vtu"
    path.write_bytes(b"keep original")
    with pytest.raises(FileExistsError):
        save_dtcc_result(SimpleNamespace(output_dir=tmp_path), path)
    assert path.read_bytes() == b"keep original"


def test_hdf5_companion_is_not_overwritten(tmp_path):
    (tmp_path / "result.h5").write_bytes(b"keep original")
    with pytest.raises(FileExistsError):
        save_dtcc_result(SimpleNamespace(output_dir=tmp_path), tmp_path / "result.xdmf")


def test_result_rejects_higher_degree_vertex_export(tmp_path):
    from dtcc_sim.helmholtz import HelmholtzResult

    result = HelmholtzResult(
        solution=np.ones(4),
        iterations=1,
        relative_residual=0,
        converged=True,
        gmres_history=(0,),
        output_dir=tmp_path,
        mesh_path=tmp_path / "mesh.xdmf",
        fe_degree=2,
    )
    with pytest.raises(ValueError, match="degree 1"):
        result.to_dtcc()


def test_real_vtu_export_roundtrip(tmp_path, vertices):
    from dtcc_core.io import load_volume_mesh

    mesh = VolumeMesh(vertices=vertices, cells=np.array([[0, 1, 2, 3]], dtype=np.int64))
    attach_helmholtz_fields(mesh, vertices, np.array([3 + 4j, 0 - 2j, 1j, -1]))
    path = tmp_path / "result.vtu"
    mesh.save(path)
    import meshio

    loaded = meshio.read(path)
    np.testing.assert_allclose(
        loaded.point_data["helmholtz_amplitude"].ravel(), [5, 2, 1, 1]
    )
    np.testing.assert_allclose(
        loaded.point_data["helmholtz_imaginary"].ravel(), [4, -2, 1, 0]
    )
    np.testing.assert_array_equal(loaded.cells_dict["tetra"], mesh.cells)
    native = tmp_path / "result.dtcc"
    mesh.save(native)
    restored = load_volume_mesh(native)
    fields = {f.name: f for f in restored.fields}
    np.testing.assert_allclose(
        fields["helmholtz_amplitude"].values.ravel(), [5, 2, 1, 1]
    )


def test_saved_case_load_preserves_solution_and_mesh_identity(tmp_path):
    import json
    from dtcc_sim.helmholtz import HelmholtzResult

    path = tmp_path / "input.xdmf"
    np.save(tmp_path / "solution.npy", np.array([3 + 4j, 1 - 2j]))
    (tmp_path / "prepared.json").write_text(
        json.dumps(
            {
                "inputs": {str(path): "hash"},
                "parameters": {"fe_degree": 1, "mesh_name": "mesh"},
            }
        )
    )
    (tmp_path / "diagnostics.json").write_text(
        json.dumps(
            {
                "iterations": 2,
                "relative_residual": 1e-7,
                "converged": True,
                "gmres_history": [0.1, 1e-7],
            }
        )
    )
    result = HelmholtzResult.load(tmp_path)
    assert result.mesh_path == path and result.gmres_history == (0.1, 1e-7)
    np.testing.assert_array_equal(result.solution, [3 + 4j, 1 - 2j])
    assert result.prepared_identity["inputs"][str(path)] == "hash"


@pytest.mark.parametrize("format", ["dtcc", "dtccpkg"])
def test_tagged_fields_survive_native_result_export(tmp_path, vertices, format):
    from dtcc_core.datasets import load_model_package
    from dtcc_core.datasets.package import export_model_package
    from dtcc_core.io import load_volume_mesh
    from dtcc_sim.helmholtz.dataset import (
        HelmholtzSimulationArgs,
        HelmholtzSimulationDataset,
    )

    mesh = VolumeMesh(vertices=vertices, cells=np.array([[0, 1, 2, 3]], dtype=np.int64))
    mesh.boundary_markers = {(0, 1, 2): -2, (0, 1, 3): 0, (0, 2, 3): 17}
    values = np.array([3 + 4j, -2j, 1j, -1])
    attach_helmholtz_fields(mesh, vertices, values, unit="Pa")
    args = HelmholtzSimulationArgs(
        bounds=[0, 0, 1, 1],
        h=1,
        mesh_path=tmp_path / "input.xdmf",
        output_dir=tmp_path / "case",
        field_unit="Pa",
    )
    mesh.dataset_context = HelmholtzSimulationDataset().create_context(args)
    path = tmp_path / ("field." + format)
    if format == "dtccpkg":
        export_model_package(mesh, path, canonical=True)
        restored = load_model_package(path)
    else:
        mesh.save(path)
        restored = load_volume_mesh(path)
    np.testing.assert_array_equal(restored.vertices, mesh.vertices)
    np.testing.assert_array_equal(restored.cells, mesh.cells)
    assert dict(
        zip(map(tuple, restored.boundary_faces), restored.boundary_markers)
    ) == mesh.boundary_markers
    fields = {field.name: field for field in restored.fields}
    for name, expected in zip(
        ("helmholtz_real", "helmholtz_imaginary", "helmholtz_amplitude"),
        (values.real, values.imag, abs(values)),
    ):
        np.testing.assert_array_equal(fields[name].values.ravel(), expected)
        assert fields[name].unit == "Pa"


@pytest.mark.parametrize("change", ["manifest", "mesh_name", "fe_degree"])
def test_export_rejects_changed_case_identity_before_reading_mesh(tmp_path, change):
    from copy import deepcopy
    from dataclasses import replace
    import json
    from dtcc_sim.helmholtz import HelmholtzResult

    manifest = {
        "inputs": {str(tmp_path / "mesh.xdmf"): "hash"},
        "parameters": {"fe_degree": 1, "mesh_name": "mesh", "frequency_hz": 10},
    }
    result = HelmholtzResult(
        solution=np.ones(4, dtype=complex), iterations=1, relative_residual=0,
        converged=True, gmres_history=(0,), output_dir=tmp_path,
        mesh_path=tmp_path / "mesh.xdmf", prepared_identity=deepcopy(manifest),
    )
    if change == "manifest":
        manifest["parameters"]["frequency_hz"] = 20
        message = "identity changed"
    else:
        # Also protect loaded legacy results without an in-memory identity.
        result = replace(result, prepared_identity=None)
        manifest["parameters"][change] = "other_mesh" if change == "mesh_name" else 2
        message = "does not match"
    (tmp_path / "prepared.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        result.save_dtcc(tmp_path / "result.vtu")
    assert not (tmp_path / "result.vtu").exists()


@pytest.mark.parametrize("legacy_values", [False, True])
@pytest.mark.parametrize(
    "runtime_change", [None, "missing_partitioner", "partitioner", "reader"]
)
def test_dolfinx_mesh_exports_fields_and_tags_without_core_reader(
    tmp_path, monkeypatch, legacy_values, runtime_change
):
    """Read tagged multi-grid XDMF with either boundary dataset capitalization."""
    pytest.importorskip("dolfinx")
    from dataclasses import asdict
    import json
    import xml.etree.ElementTree as ET
    import h5py
    import meshio
    from mpi4py import MPI
    from dolfinx import mesh as dmesh, fem, io
    import dtcc_core.io
    from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzResult
    from dtcc_sim.helmholtz.simulator import _mesh_input_hashes, _runtime_versions

    mesh = dmesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    mesh.topology.create_connectivity(2, 3)
    facets = dmesh.exterior_facet_indices(mesh.topology)
    tags = dmesh.meshtags(mesh, 2, facets, np.full(len(facets), -2, np.int32))
    tags.name = "boundary_markers"
    cell_indices = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    cell_tags = dmesh.meshtags(mesh, 3, cell_indices, np.full(len(cell_indices), 7, np.int32))
    cell_tags.name = "cell_tags"
    path = tmp_path / "mesh.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, str(path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(tags, mesh.geometry)
        xdmf.write_meshtags(cell_tags, mesh.geometry)
    if legacy_values:
        tree = ET.parse(path)
        grid = next(g for g in tree.findall(".//Grid") if g.get("Name") == tags.name)
        item = grid.find("Attribute/DataItem")
        filename, dataset = item.text.strip().split(":", 1)
        renamed = dataset.rsplit("/", 1)[0] + "/Values"
        with h5py.File(tmp_path / filename, "r+") as handle:
            if dataset != renamed:
                handle.move(dataset, renamed)
        item.text = filename + ":" + renamed
        tree.write(path)
    # Construct DOF values in precisely the same ordering as result_to_dtcc.
    with io.XDMFFile(MPI.COMM_WORLD, str(path), "r") as xdmf:
        loaded = xdmf.read_mesh(name="mesh")
    coords = fem.functionspace(loaded, ("Lagrange", 1)).tabulate_dof_coordinates()
    solution = coords[:, 0] + 2 * coords[:, 1] + 1j * coords[:, 2]
    parameters = HelmholtzParameters()
    runtime = _runtime_versions()
    if runtime_change == "missing_partitioner":
        runtime.pop("partitioner", None)
    elif runtime_change == "partitioner":
        runtime["partitioner"] = {"pymetis": "different-build"}
    elif runtime_change == "reader":
        runtime["dolfinx"] = "different-reader-version"
    (tmp_path / "prepared.json").write_text(json.dumps({
        "inputs": _mesh_input_hashes(path), "runtime": runtime,
        "parameters": asdict(parameters),
    }))
    result = HelmholtzResult(
        solution=solution, iterations=0, relative_residual=0, converged=True,
        gmres_history=(), output_dir=tmp_path, mesh_path=path,
    )
    if runtime_change == "reader":
        with pytest.raises(ValueError, match="runtime changed"):
            result.to_dtcc()
        return
    monkeypatch.setattr(
        dtcc_core.io, "load_volume_mesh",
        lambda *a, **k: pytest.fail("Export must use its checked numerical mesh"),
    )
    converted = result.to_dtcc()
    assert len(converted.cells) == 6
    assert len(converted.boundary_markers) == len(facets)
    assert set(converted.boundary_markers.values()) == {-2}
    np.testing.assert_array_equal(converted.markers, 7)
    fields = {f.name: f.values.ravel() for f in converted.fields}
    np.testing.assert_allclose(fields["helmholtz_real"], converted.vertices[:, 0] + 2 * converted.vertices[:, 1], atol=1e-14)
    np.testing.assert_allclose(fields["helmholtz_imaginary"], converted.vertices[:, 2], atol=1e-14)
    exported = result.save_dtcc(tmp_path / "result.vtu")
    vtu = meshio.read(exported)
    np.testing.assert_allclose(vtu.point_data["helmholtz_amplitude"].ravel(), fields["helmholtz_amplitude"])
    boundary_output = result.save_dtcc(tmp_path / "result.xdmf")
    with io.XDMFFile(MPI.COMM_WORLD, str(boundary_output), "r") as xdmf:
        restored = xdmf.read_mesh(name="mesh")
        restored.topology.create_connectivity(2, 3)
        restored_tags = xdmf.read_meshtags(restored, name="boundary_markers")
    assert len(restored_tags.values) == len(facets)
    assert set(restored_tags.values) == {-2}
