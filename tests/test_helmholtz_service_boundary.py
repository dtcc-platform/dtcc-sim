"""Keep local Helmholtz execution outside the general queued worker."""

import dtcc_core.datasets as datasets
from fastapi import HTTPException
import pytest


def test_local_dataset_is_registered_but_not_exposed_to_general_service():
    from service import routes, tasks

    assert "helmholtz_simulation" in datasets.list()
    assert "helmholtz_simulation" not in routes._get_service_datasets()
    assert "dataset.helmholtz_simulation" not in tasks.celery_app.tasks
    assert "urban_heat_simulation" in routes._get_service_datasets()
    assert "dataset.urban_heat_simulation" in tasks.celery_app.tasks
    with pytest.raises(HTTPException) as error:
        routes._require_dataset("helmholtz_simulation")
    assert error.value.status_code == 404
