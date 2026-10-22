import os

SERVICE_NAME = os.environ.get("DTCC_SERVICE_NAME", "dtcc-sim")
SERVICE_VERSION = os.environ.get("DTCC_SERVICE_VERSION", "0.1.0")
MODULE_PREFIX = os.environ.get("DTCC_MODULE_PREFIX", "dtcc_sim")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
SHARED_RESULTS_DIR = os.environ.get("SHARED_RESULTS_DIR", "/shared/results")
JOB_TIMEOUT = int(os.environ.get("DTCC_JOB_TIMEOUT", "3600"))
