"""Environment-based configuration for the MCP server."""

import os


API_URL = os.environ.get("DTCC_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.environ.get("DTCC_API_TIMEOUT", "30.0"))
JOB_POLL_INTERVAL = float(os.environ.get("DTCC_JOB_POLL_INTERVAL", "5.0"))
JOB_TIMEOUT = float(os.environ.get("DTCC_JOB_TIMEOUT", "300.0"))
