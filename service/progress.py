import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def progress_bridge(celery_task):
    """Bridge dtcc-core ProgressTracker updates to Celery task state.

    Sets a thread-local progress callback so that any ProgressTracker
    created during the dataset computation will inherit it and forward
    updates to the Celery task state.

    Usage:
        with progress_bridge(self):
            result = dataset(**params)
    """
    from dtcc_core.common.progress import (
        set_progress_callback,
        get_progress_callback,
    )

    previous_callback = get_progress_callback()

    def celery_progress_callback(state_dict):
        try:
            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": state_dict.get("percent", 0) / 100.0,
                    "message": state_dict.get("message", ""),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to update Celery state: {e}")

    set_progress_callback(celery_progress_callback)
    try:
        yield
    finally:
        set_progress_callback(previous_callback)
