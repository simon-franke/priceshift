"""Base httpx client with tenacity retry logic."""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def make_retry_decorator(max_attempts: int = 5) -> Any:
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )


class BaseAPIClient:
    """Shared httpx client with retry, logging, and timeout."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 30,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_seconds,
            headers=headers or {},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BaseAPIClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @make_retry_decorator()
    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        url = path if path.startswith("http") else f"{self._base_url}/{path.lstrip('/')}"
        logger.debug("GET %s params=%s", url, params)
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
