from __future__ import annotations

from dataclasses import fields
from inspect import signature

import h5py
import numpy as np
import pytest

import dtcc_core.io  # noqa: F401  Registers VolumeMesh.save.
import dtcc_core.datasets as datasets
from dtcc_core.model import DeSO, Field, RoadNetwork, SensorCollection, VolumeMesh


UPSTREAM_DATASET_CONTRACTS = [
    (
        "city_volume_mesh",
        {
            "bounds",
            "max_mesh_size",
            "domain_height",
            "min_building_detail",
            "raster_cell_size",
            "raster_radius",
            "repair_boundary_corners",
        },
        "dtcc_core.model.VolumeMesh",
    ),
    (
        "air_quality",
        {
            "bounds",
            "phenomenon",
            "crs",
            "timeout_s",
            "max_stations",
            "drop_missing",
            "base_url",
        },
        "dtcc_core.model.SensorCollection",
    ),
    (
        "weather",
        {"bounds", "parameters"},
        "dtcc_core.model.SensorCollection",
    ),
    (
        "roads",
        {"bounds"},
        "dtcc_core.model.RoadNetwork",
    ),
    (
        "deso",
        {"bounds", "statistics", "statistics_year"},
        "dtcc_core.model.DeSO",
    ),
]


@pytest.mark.parametrize(
    ("dataset_name", "required_args", "return_type"),
    UPSTREAM_DATASET_CONTRACTS,
    ids=[contract[0] for contract in UPSTREAM_DATASET_CONTRACTS],
)
def test_upstream_dataset_contract(dataset_name, required_args, return_type):
    contract = getattr(datasets, dataset_name).describe()
    properties = contract["args_schema"]["properties"]

    missing_args = required_args.difference(properties)
    assert not missing_args, f"{dataset_name} removed arguments: {sorted(missing_args)}"
    assert contract["python_return_type"] == return_type


def test_upstream_result_interfaces_used_by_sim():
    sensor_parameters = signature(SensorCollection.to_arrays).parameters
    assert "field_name" in sensor_parameters
    assert callable(SensorCollection.stations)

    road_fields = {field.name for field in fields(RoadNetwork)}
    assert {"vertices", "edges", "length", "attributes"} <= road_fields
    assert callable(RoadNetwork.to_proto)

    assert callable(DeSO.to_arrays)
    assert isinstance(DeSO.fields, property)
    assert callable(VolumeMesh.save)


@pytest.mark.parametrize("field_name", ["temperature", "NO2"])
def test_volume_mesh_xdmf_keeps_vertex_fields(tmp_path, field_name):
    mesh = VolumeMesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    mesh.add_field(
        Field(name=field_name, values=np.array([18.0, 19.0, 20.0, 21.0]), dim=1,
              association="vertex", unit="degC")
    )

    mesh.save(tmp_path / "result.xdmf")

    assert f'Attribute Name="{field_name}"' in (tmp_path / "result.xdmf").read_text()
    with h5py.File(tmp_path / "result.h5", "r") as h5_file:
        saved = {
            dataset.attrs["name"]: dataset[()]
            for dataset in h5_file["Mesh/mesh/fields"].values()
        }
    np.testing.assert_allclose(saved[field_name], [18.0, 19.0, 20.0, 21.0])
