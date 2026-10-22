import importlib
import logging

from celery import Celery
from service.config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    MODULE_PREFIX,
    JOB_TIMEOUT,
    HELMHOLTZ_QUEUE,
    DATASET_ALLOWLIST,
)

logger = logging.getLogger(__name__)

# Import the DTCC package to trigger dataset auto-registration
try:
    importlib.import_module(MODULE_PREFIX)
except ImportError as e:
    raise ImportError(
        f"Cannot import DTCC package '{MODULE_PREFIX}'. "
        f"Ensure it is installed in the worker container. "
        f"Original error: {e}"
    ) from e

import dtcc_core.datasets as datasets
from service.results import handle_result
from service.progress import progress_bridge

celery_app = Celery(
    "dtcc_service",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)
celery_app.conf.task_default_queue = MODULE_PREFIX.replace("_", "-")
celery_app.conf.task_time_limit = JOB_TIMEOUT
if MODULE_PREFIX == "dtcc_sim":
    celery_app.conf.task_routes = {
        "dataset.helmholtz_simulation": {"queue": HELMHOLTZ_QUEUE},
    }


def create_dataset_task(dataset_name, dataset_instance):
    @celery_app.task(
        bind=True,
        name=f"dataset.{dataset_name}",
        track_started=dataset_name == "helmholtz_simulation",
    )
    def run(self, params):
        task_params = dict(params)
        # Extract format before calling the dataset so build() always returns
        # a Python object. This prevents export_to_bytes() from discarding
        # companion files (e.g., XDMF's .h5). Serialization happens in
        # handle_result() via save(), which preserves multi-file outputs.
        format_ext = task_params.pop("format", None)
        if dataset_name == "helmholtz_simulation":
            from service.helmholtz_control import run_managed_helmholtz

            return run_managed_helmholtz(self, task_params, format_ext)
        with progress_bridge(self):
            result = dataset_instance(**task_params)
            return handle_result(result, format_ext, self.request.id)
    return run


# Register Celery tasks for datasets matching this service's module prefix
_service_datasets = {}
for name, ds in datasets.list().items():
    if ds.__class__.__module__.startswith(MODULE_PREFIX) and (
        not DATASET_ALLOWLIST or name in DATASET_ALLOWLIST
    ):
        create_dataset_task(name, ds)
        _service_datasets[name] = ds

logger.info(f"Registered {len(_service_datasets)} dataset tasks: {list(_service_datasets.keys())}")
