"""Check that visualization cuts sample the actual tetrahedral complex field."""

import numpy as np
import pytest

from tools import validate_helmholtz_box as plot


def test_complex_linear_field_slice():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    values = vertices[:, 0] + 2 * vertices[:, 1] + 1j * vertices[:, 2]
    xy, triangles, u = plot.horizontal_slice(
        vertices, np.array([[0, 1, 2, 3]]), values, 0.25
    )
    np.testing.assert_allclose(u, xy[:, 0] + 2 * xy[:, 1] + 0.25j)
    assert triangles.shape == (1, 3)
    assert np.all(xy.sum(axis=1) <= 0.75 + 1e-12)
    with pytest.raises(ValueError, match="does not intersect"):
        plot.horizontal_slice(vertices, np.array([[0, 1, 2, 3]]), values, 2.0)


def test_slice_interpolates_complex_components_before_magnitude():
    vertices = np.array(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    )
    values = np.array([1j, 1j, 1j, -1j])
    _, _, samples = plot.horizontal_slice(
        vertices, np.array([[0, 1, 2, 3]]), values, 0.5
    )
    np.testing.assert_allclose(np.abs(samples), 0.)
