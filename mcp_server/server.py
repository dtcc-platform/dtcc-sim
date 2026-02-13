"""MCP server for dtcc-sim.

Translates MCP tool calls into HTTP requests against the FastAPI backend.
Run with: python -m mcp_server.server
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .client import DTCCClient
from .config import API_URL, JOB_POLL_INTERVAL, JOB_TIMEOUT

mcp = FastMCP("dtcc-sim")
client = DTCCClient()


# -- Helpers -----------------------------------------------------------------

def _format(data: Any) -> str:
    """Pretty-print a dict/list as JSON text for the LLM."""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, default=str)


# -- Dataset Discovery -------------------------------------------------------

@mcp.tool()
async def list_datasets() -> str:
    """List all available datasets with names and types.

    Returns a JSON object with a 'datasets' key containing an array of
    datasets, each with 'name', 'type', and 'source' fields.
    Use this to discover what datasets are available before submitting jobs.
    """
    result = await client.list_datasets()
    return _format(result)


@mcp.tool()
async def get_dataset_schema(dataset_name: str) -> str:
    """Get the parameter schema for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g. 'urban_heat_simulation',
            'air_quality_field'). Use list_datasets() to see available names.

    Returns the JSON Schema describing accepted parameters including bounds,
    mesh settings, and simulation-specific coefficients.
    """
    result = await client.get_dataset_schema(dataset_name)
    return _format(result)


# -- Job Management ----------------------------------------------------------

@mcp.tool()
async def submit_job(
    dataset: str,
    bounds: list[float],
    parameters: dict[str, Any] | None = None,
    filename: str | None = None,
) -> str:
    """Submit an async simulation/download job.

    Args:
        dataset: Dataset name (e.g. 'urban_heat_simulation').
        bounds: Bounding box as [minX, minY, maxX, maxY] in the dataset's CRS.
        parameters: Optional dict of dataset-specific parameters. Use
            get_dataset_schema() to see what is accepted.
        filename: Optional output filename (without extension).

    Returns a JSON object with 'job_id' and 'status'.
    """
    result = await client.submit_job(dataset, bounds, parameters, filename)
    return _format(result)


@mcp.tool()
async def get_job_status(job_id: str) -> str:
    """Get the current status of a job.

    Args:
        job_id: The job ID returned by submit_job().

    Returns job details including status ('queued', 'processing', 'complete',
    'failed'), progress info, and download_url when complete.
    """
    result = await client.get_job_status(job_id)
    return _format(result)


@mcp.tool()
async def list_jobs(limit: int = 20) -> str:
    """List recent jobs.

    Args:
        limit: Maximum number of jobs to return (default 20).

    Returns a JSON object with a 'jobs' key containing an array of job summaries.
    """
    result = await client.list_jobs(limit)
    return _format(result)


@mcp.tool()
async def cancel_job(job_id: str) -> str:
    """Cancel a running or queued job.

    Args:
        job_id: The job ID to cancel.

    Returns success/failure status. Cannot cancel already-completed jobs.
    """
    result = await client.cancel_job(job_id)
    return _format(result)


@mcp.tool()
async def download_job_result(job_id: str, save_path: str | None = None) -> str:
    """Download the result file from a completed job.

    Args:
        job_id: The job ID of a completed job.
        save_path: Local file path to save the result. If not provided,
            saves to the current directory using the server-provided filename.

    Returns the path where the file was saved.
    """
    try:
        resp = await client.download_job_result(job_id)
    except (ConnectionError, ValueError) as exc:
        return _format({"error": str(exc)})

    # Extract filename from Content-Disposition header
    disposition = resp.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^"]+)"?', disposition)
    server_filename = match.group(1) if match else f"job_{job_id}_result"

    if save_path:
        out = Path(save_path)
    else:
        out = Path.cwd() / server_filename

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    return _format({"saved_to": str(out), "size_bytes": len(resp.content)})


# -- Convenience -------------------------------------------------------------

@mcp.tool()
async def run_simulation(
    simulation_type: str,
    bounds: list[float],
    parameters: dict[str, Any] | None = None,
    poll_interval: float = JOB_POLL_INTERVAL,
    timeout: float = JOB_TIMEOUT,
) -> str:
    """Submit a simulation job and wait for it to complete.

    This is a blocking convenience wrapper that submits a job, polls for
    status, and returns the final result. Use this when you want to wait
    for the simulation to finish rather than managing the job lifecycle
    manually.

    Args:
        simulation_type: Dataset/simulation name (e.g. 'urban_heat_simulation').
        bounds: Bounding box as [minX, minY, maxX, maxY].
        parameters: Optional simulation parameters.
        poll_interval: Seconds between status checks (default from config).
        timeout: Maximum seconds to wait before returning (default from config).
            On timeout the job is NOT cancelled -- use the returned job_id to
            check later.

    Returns the final job status including download_url on success, or
    the last known status on timeout.
    """
    submit_result = await client.submit_job(simulation_type, bounds, parameters)
    if "error" in submit_result:
        return _format(submit_result)

    job_id = submit_result["job_id"]
    elapsed = 0.0

    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        status = await client.get_job_status(job_id)
        if "error" in status:
            return _format(status)

        job_status = status.get("status", "")
        if job_status in ("complete", "failed", "cancelled"):
            return _format(status)

    # Timed out -- return last known status so the user can follow up
    status = await client.get_job_status(job_id)
    if isinstance(status, dict):
        status["_timeout"] = True
        status["_message"] = (
            f"Timed out after {timeout}s. Job {job_id} is still running. "
            "Use get_job_status() to check progress."
        )
    return _format(status)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
