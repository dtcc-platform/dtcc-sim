"""Physical forcing and map-coordinate persistence under local assembly."""

import json
import numpy as np
import pytest

pytest.importorskip("dolfinx")
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from dtcc_core import BoxMesh
from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzResult, HelmholtzSolver
from dtcc_sim.helmholtz._backend import mesh as backend_mesh

pytestmark = [pytest.mark.fenics, pytest.mark.simulation]


@pytest.fixture(autouse=True)
def complex_runtime():
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        pytest.skip("Helmholtz requires complex PETSc")


def parameters(**kwargs):
    return HelmholtzParameters(
        boundary_mode="impedance_only",
        frequency_hz=50.0,
        n_subdomains=2,
        partitioner="structured",
        nx_sub=2,
        ny_sub=1,
        nz_sub=1,
        workers=1,
        overlap_layers=1,
        oversample_layers=1,
        n_modes_per_patch=2,
        outer_rtol=1e-10,
        verbose=False,
        **kwargs,
    )


def complex_samples(mesh):
    values = {field.name: field.values.ravel() for field in mesh.fields}
    complex_values = values["helmholtz_real"] + 1j * values["helmholtz_imaginary"]
    local = np.round(mesh.vertices - mesh.vertices.min(axis=0), 6)
    order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
    return complex_values[order]


def test_manual_source_translation_invariance_and_persistence(tmp_path, monkeypatch):
    assemble = backend_mesh.assemble_global_matrix
    assembled = []

    def assemble_locally(mesh, *args, **kwargs):
        np.testing.assert_array_equal(mesh.geometry.x.min(axis=0), 0.0)
        np.testing.assert_array_equal(
            backend_mesh.source_point(mesh), [0.25, 0.5, 0.75]
        )
        assembled.append(mesh)
        return assemble(mesh, *args, **kwargs)

    monkeypatch.setattr(backend_mesh, "assemble_global_matrix", assemble_locally)
    samples = []
    for index, offset in enumerate(([0.0, 0.0, 0.0], [319895.0, 6398909.0, 12.5])):
        native = BoxMesh(4, 4, 4)
        native.vertices += offset
        native.transform.srs = "EPSG:3006"
        original = native.vertices.copy()
        point = np.asarray(offset) + [0.25, 0.5, 0.75]
        solver = HelmholtzSolver(
            mesh=native,
            output_dir=tmp_path / str(index),
            params=parameters(
                source_location_mode="manual",
                source_x=point[0],
                source_y=point[1],
                source_z=point[2],
                gaussian_source_sigma=0.2,
            ),
        )
        result = solver.simulate()
        np.testing.assert_array_equal(native.vertices, original)
        np.testing.assert_array_equal(result.vertices.min(axis=0), offset)
        assert result.transform.srs == "EPSG:3006"
        record = solver.result.coordinate_frame
        assert (
            json.loads((solver.output_dir / "solution.coordinates.json").read_text())
            == record
        )
        assert record["local_to_input_translation"] == list(offset)
        assert record["input_crs"] == "EPSG:3006"
        assert (
            json.loads(solver.manifest_path.read_text())["coordinate_frame"] == record
        )
        assert (
            json.loads((solver.output_dir / "diagnostics.json").read_text())[
                "coordinate_frame"
            ]
            == record
        )
        restored = HelmholtzResult.load(solver.output_dir).to_dtcc()
        np.testing.assert_allclose(
            complex_samples(restored), complex_samples(result), atol=1e-12
        )
        with XDMFFile(
            MPI.COMM_WORLD, str(solver.output_dir / "solution.xdmf"), "r"
        ) as file:
            exported = file.read_mesh()
        assert exported.geometry.x.dtype == np.float64
        np.testing.assert_array_equal(exported.geometry.x.min(axis=0), offset)
        np.testing.assert_array_equal(
            exported.geometry.x.max(axis=0), np.asarray(offset) + 1
        )
        samples.append(complex_samples(result))
        manifest = json.loads(solver.manifest_path.read_text())
        manifest["coordinate_frame"]["local_to_input_translation"][0] += 1
        solver.manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="coordinate frame"):
            solver.solve()
    assert len(assembled) == 4
    np.testing.assert_allclose(samples[0], samples[1], atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("mixed", [False, True])
def test_plane_wave_phase_and_boundary_data_use_input_coordinates(tmp_path, mixed):
    samples = []
    direction = np.array([1.0, 0.3, 0.2])
    direction /= np.linalg.norm(direction)
    offset = np.array([319895.0, 6398909.0, 12.5])
    params = parameters(
        rhs_mode="plane_wave_exact",
        plane_wave_direction_x=direction[0],
        plane_wave_direction_y=direction[1],
        plane_wave_direction_z=direction[2],
    )
    if mixed:
        from dataclasses import replace

        params = replace(params, boundary_mode="mixed", dirichlet_facet_tags=(-3,))
    wave_number = 2 * np.pi * params.frequency_hz / params.sound_speed
    for index, shift in enumerate((np.zeros(3), offset)):
        native = BoxMesh(4, 4, 4)
        native.vertices += shift
        solver = HelmholtzSolver(
            mesh=native, output_dir=tmp_path / str(index), params=params
        )
        result = solver.simulate()
        numerical = complex_samples(result)
        exact = np.exp(1j * wave_number * (result.vertices @ direction))
        values = {field.name: field.values.ravel() for field in result.fields}
        original_order = values["helmholtz_real"] + 1j * values["helmholtz_imaginary"]
        assert np.linalg.norm(original_order - exact) / np.linalg.norm(exact) < 0.02
        samples.append(numerical)
    phase = float(
        np.remainder(
            np.longdouble(wave_number)
            * np.dot(direction.astype(np.longdouble), offset.astype(np.longdouble)),
            2 * np.longdouble(np.pi),
        )
    )
    np.testing.assert_allclose(
        samples[1], samples[0] * np.exp(1j * phase), atol=1e-8, rtol=1e-7
    )


def test_weighted_source_height_preserves_its_input_frame_rule():
    import dolfinx.mesh as dm
    from dtcc_sim.coordinates import localize_mesh
    from dtcc_sim.helmholtz._backend.params import Params, bind_parameters

    mesh = dm.create_box(
        MPI.COMM_WORLD,
        [[319895.0, 6398909.0, 10.0], [319896.0, 6398910.0, 11.0]],
        [1, 1, 1],
    )
    frame = localize_mesh(mesh)
    with bind_parameters(
        Params(
            source_location_mode="mesh_bounds",
            source_z_weight_min=1.4,
            source_z_weight_max=0.4,
        )
    ):
        actual = frame.to_input(backend_mesh.source_point(mesh))
    np.testing.assert_allclose(
        actual, [319895.5, 6398909.5, 1.4 * 10 + 0.4 * 11], atol=1e-10
    )
