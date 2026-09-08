"""Small stdlib HTTP client used by the CLI and MCP adapter."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class SuperGPUError(RuntimeError):
    """An error answer from the controller, carrying its stable ``code``.

    ``code`` is one of the values catalogued by ``GET /api/meta`` (for example
    ``not_found`` or ``idempotency_conflict``); ``status`` is the HTTP status.
    A controller that cannot be reached at all has ``code == "unreachable"``
    and ``status == 0``.
    """

    def __init__(self, message: str, *, code: str, status: int, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.payload = payload or {}

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": str(self), "code": self.code, "status": self.status}


class SuperGPUClient:
    def __init__(self, url: str, token: str = "", *, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.token = token or os.environ.get("SUPER_GPU_API_TOKEN", "")
        self.timeout = float(timeout)

    def get(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        return self._request("GET", path, timeout=timeout)

    def get_text(self, path: str, *, timeout: float | None = None) -> str:
        return self._request("GET", path, timeout=timeout, raw=True)

    def post(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", path, payload or {}, timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Generic call for ``super-gpu api``; GET sends no body."""
        method = method.upper()
        if method == "GET":
            return self._request("GET", path, timeout=timeout)
        return self._request(method, path, payload or {}, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        raw: bool = False,
    ) -> Any:
        headers = {"Accept": "text/plain" if raw else "application/json"}
        data = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read().decode("utf-8")
                return body if raw else json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            detail = str(parsed.get("error") or body or exc.reason)
            code = str(parsed.get("code") or f"http_{exc.code}")
            raise SuperGPUError(
                f"super_gpu HTTP {exc.code} ({code}): {detail}",
                code=code,
                status=exc.code,
                payload=parsed,
            ) from exc
        except urllib.error.URLError as exc:
            raise SuperGPUError(
                f"super_gpu is unreachable at {self.url}: {exc.reason}",
                code="unreachable",
                status=0,
            ) from exc
