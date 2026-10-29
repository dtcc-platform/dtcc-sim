"""Keep local Helmholtz execution outside the general queued worker."""

import dtcc_core.datasets as datasets
from fastapi import HTTPException
import pytest


def test_local_dataset_is_registered_but_not_exposed_to_general_service():
    from service import routes, tasks

    assert "urban_noise_simulation" in datasets.list()
    assert "urban_noise_simulation" not in routes._get_service_datasets()
    assert "dataset.urban_noise_simulation" not in tasks.celery_app.tasks
    assert "urban_heat_simulation" in routes._get_service_datasets()
    assert "dataset.urban_heat_simulation" in tasks.celery_app.tasks
    with pytest.raises(HTTPException) as error:
        routes._require_dataset("urban_noise_simulation")
    assert error.value.status_code == 404


@pytest.mark.parametrize("name", ["urban_heat_simulation", "urban_wind_simulation", "air_quality_field", "traffic_simulation"])
def test_service_rejects_client_selected_local_output(name, tmp_path):
    from service import routes, tasks

    schema = routes.list_datasets()["datasets"][name]["args_schema"]
    assert "output_dir" not in schema["properties"]
    params = {"bounds": [0, 0, 1, 1], "output_dir": str(tmp_path / "client-selected")}
    with pytest.raises(HTTPException, match="local-only") as error:
        routes.submit_job(name, params)
    assert error.value.status_code == 422
    with pytest.raises(ValueError, match="local-only"):
        tasks.celery_app.tasks[f"dataset.{name}"].run(params)
    assert not (tmp_path / "client-selected").exists()
