"""Input-frame transformations and CRS declarations without a solver runtime."""

from types import SimpleNamespace
import numpy as np
import pytest
from dtcc_sim.coordinates import MeshCoordinateFrame, input_crs, write_coordinate_frame


def test_coordinate_frame_transforms_points_and_input_callbacks_without_mutation():
    frame = MeshCoordinateFrame((319895.0, 6398909.0, 12.5), "EPSG:3006")
    local = np.array([[0.25, 0.5, 0.75], [1.0, 2.0, 3.0]])
    original = local.copy()
    world = frame.to_input(local)
    np.testing.assert_array_equal(frame.to_local(world), local)
    np.testing.assert_array_equal(local, original)
    observed = []

    def callback(points):
        observed.append(points.copy())
        return points[2] - 12.5

    np.testing.assert_array_equal(frame.input_callback(callback)(local.T), local[:, 2])
    np.testing.assert_array_equal(observed[0], world.T)
    assert frame.as_dict() == {
        "local_to_input_translation": [319895.0, 6398909.0, 12.5],
        "input_crs": "EPSG:3006",
        "computation_coordinates": "local",
        "output_coordinates": "input",
    }


def test_crs_is_declared_or_retained_without_inference():
    assert input_crs() is None
    assert input_crs(generated=True) == "EPSG:3006"
    assert input_crs(declared=" EPSG:3006 ") == "EPSG:3006"
    native = SimpleNamespace(transform=SimpleNamespace(srs="EPSG:3006"))
    assert input_crs(native, declared="EPSG:3006") == "EPSG:3006"
    with pytest.raises(ValueError, match="conflicts"):
        input_crs(native, declared="EPSG:4326")
    with pytest.raises(ValueError, match="conflicts"):
        input_crs(generated=True, declared="EPSG:4326")
    with pytest.raises(ValueError, match="nonempty"):
        input_crs(declared=" ")


def test_coordinate_metadata_is_persisted_and_replaced(tmp_path):
    import json

    output = tmp_path / "solution.xdmf"
    frame = MeshCoordinateFrame((319895.0, 6398909.0, 12.5), "EPSG:3006")
    path = write_coordinate_frame(output, frame)
    assert path.name == "solution.coordinates.json"
    assert json.loads(path.read_text()) == frame.as_dict()
    local = MeshCoordinateFrame((0.0, 0.0, 0.0))
    write_coordinate_frame(output, local.as_dict())
    assert json.loads(path.read_text()) == local.as_dict()
