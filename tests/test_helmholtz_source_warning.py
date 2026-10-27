"""Source checks report exterior centers without moving or changing the forcing."""

import warnings

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from dolfinx import mesh as dmesh, fem
from mpi4py import MPI
from dtcc_sim.helmholtz._backend import mesh as backend
from dtcc_sim.helmholtz._backend.params import Params, bind_parameters


@pytest.fixture
def cube():
    return dmesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)


def test_exterior_source_warns_and_keeps_coordinates(cube):
    params = Params(
        source_location_mode="manual", source_x=-0.1, source_y=0.5, source_z=0.5
    )
    with bind_parameters(params):
        before = backend.source_point(cube).copy()
        with pytest.warns(RuntimeWarning, match="outside the computational mesh"):
            backend._gaussian_source_expression(cube)
        np.testing.assert_array_equal(backend.source_point(cube), before)


def test_existing_bounding_box_rule_is_unchanged(cube):
    params = Params(source_location_mode="mesh_bounds")
    with bind_parameters(params), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend._gaussian_source_expression(cube)
        np.testing.assert_array_equal(backend.source_point(cube), [0.5, 0.5, 0.1])
    assert not caught


def test_source_on_boundary_does_not_warn(cube):
    with bind_parameters(
        Params(source_location_mode="manual", source_x=0, source_y=0.5, source_z=0.5)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend._gaussian_source_expression(cube)
    assert not caught


def test_warning_does_not_change_assembled_forcing(cube, monkeypatch):
    import ufl

    params = Params(
        source_location_mode="manual",
        source_x=-0.1,
        source_y=0.5,
        source_z=0.5,
        gaussian_source_sigma=0.2,
    )
    with bind_parameters(params):
        with pytest.warns(RuntimeWarning):
            expression = backend._gaussian_source_expression(cube)
        actual = fem.assemble_scalar(fem.form(expression * ufl.dx(domain=cube)))
        monkeypatch.setattr(backend, "_warn_if_source_outside", lambda *args: None)
        reference = fem.assemble_scalar(
            fem.form(backend._gaussian_source_expression(cube) * ufl.dx(domain=cube))
        )
    assert abs(actual) > 0
    assert actual == reference
