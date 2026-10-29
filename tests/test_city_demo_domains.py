"""Wind and noise demonstrations use one geographic domain convention."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name,height_parameter",
    [
        ("urban_wind_simulation", "mesh_domain_height"),
        ("urban_wind_stokes_simulation", "mesh_domain_height"),
        ("dataset_urban_wind", "mesh_domain_height"),
        ("dataset_urban_wind_stokes", "mesh_domain_height"),
        ("dataset_urban_noise", "domain_height"),
        ("dataset_urban_heat", "mesh_domain_height"),
        ("urban_heat_cooling", "mesh_domain_height"),
        ("urban_heat_heatwave", "mesh_domain_height"),
        ("smooth_reconstruction", "mesh_domain_height"),
    ],
)
def test_poseidon_demo_domain(name, height_parameter):
    path = Path(__file__).resolve().parents[1] / "demos" / f"{name}.py"
    tree = ast.parse(path.read_text())
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"center_x", "center_y", "L"}
    }
    assert constants == {
        "center_x": 319995.962899,
        "center_y": 6399009.716755,
        "L": 200.0,
    }
    heights = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == height_parameter
    ]
    assert heights == [80.0]


@pytest.mark.parametrize(
    "name",
    [
        "urban_wind_simulation",
        "urban_wind_stokes_simulation",
        "dataset_urban_wind",
        "dataset_urban_wind_stokes",
        "dataset_urban_heat",
        "urban_heat_cooling",
        "urban_heat_heatwave",
        "smooth_reconstruction",
        "dataset_urban_noise",
        "dataset_air_quality_field",
    ],
)
def test_city_demos_select_explicit_lod_and_share_cache(name):
    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / "demos" / f"{name}.py").read_text()
    )
    keywords = {
        keyword.arg: keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    }
    assert ast.literal_eval(keywords["mesh_cache_dir"]) == "output/city-mesh-cache"
    if name == "dataset_urban_noise":
        options = ast.literal_eval(keywords["mesh_options"])
        assert options == {
            "lod": "lod2",
            "flat_ground": False,
            "repair_boundary_corners": True,
        }
        assert ast.literal_eval(keywords["min_building_detail"]) == 0.5
    else:
        assert ast.literal_eval(keywords["mesh_lod"]) == (
            "lod1" if name == "dataset_air_quality_field" else "lod2"
        )


def test_sensor_demo_retains_stockholm_observation_coverage():
    tree = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "demos"
            / "dataset_air_quality_field.py"
        ).read_text()
    )
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"center_x", "center_y", "L"}
    }
    assert constants == {"center_x": 674295.0, "center_y": 6581875.0, "L": 750.0}
