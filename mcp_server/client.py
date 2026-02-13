"""Async HTTP client wrapping the FastAPI backend API."""

from typing import Any

import httpx

from .config import API_URL, API_TIMEOUT


class DTCCClient:
    """Thin async wrapper around the dtcc-sim FastAPI backend."""

    def __init__(self, base_url: str = API_URL, timeout: float = API_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Issue an HTTP request; translate connection errors into a clear message."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(method, self._url(path), **kwargs)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError:
            return {
                "error": f"Cannot connect to DTCC backend at {self._base_url}. "
                "Is the server running?"
            }
        except httpx.TimeoutException:
            return {
                "error": f"Request to {self._base_url}{path} timed out "
                f"after {self._timeout}s."
            }
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            return {"error": detail, "status_code": exc.response.status_code}

    async def _download(self, path: str) -> httpx.Response:
        """Issue a GET and return the raw response (for binary downloads).

        Raises ConnectionError on connect failure, ValueError on HTTP errors.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self._url(path))
                resp.raise_for_status()
                return resp
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to DTCC backend at {self._base_url}. "
                "Is the server running?"
            )
        except httpx.TimeoutException:
            raise ConnectionError(
                f"Request to {self._base_url}{path} timed out "
                f"after {self._timeout}s."
            )
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = exc.response.text or str(exc)
            raise ValueError(f"HTTP {exc.response.status_code}: {detail}")

    # -- Dataset endpoints ---------------------------------------------------

    async def list_datasets(self) -> Any:
        return await self._request("GET", "/api/v1/datasets/list")

    async def get_dataset_schema(self, dataset_name: str) -> Any:
        return await self._request(
            "GET", f"/api/v1/datasets/get_args/{dataset_name}"
        )

    # -- Job endpoints -------------------------------------------------------

    async def submit_job(
        self,
        dataset: str,
        bounds: list[float],
        parameters: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> Any:
        body: dict[str, Any] = {
            "dataset": dataset,
            "bounds": bounds,
            "parameters": parameters or {},
        }
        if filename:
            body["filename"] = filename
        return await self._request("POST", "/api/v1/jobs/submit", json=body)

    async def get_job_status(self, job_id: str) -> Any:
        return await self._request("GET", f"/api/v1/jobs/{job_id}/status")

    async def list_jobs(self, limit: int = 20) -> Any:
        return await self._request(
            "GET", "/api/v1/jobs/list", params={"limit": limit}
        )

    async def cancel_job(self, job_id: str) -> Any:
        return await self._request("POST", f"/api/v1/jobs/{job_id}/cancel")

    async def download_job_result(self, job_id: str) -> httpx.Response:
        return await self._download(f"/api/v1/jobs/{job_id}/download")
