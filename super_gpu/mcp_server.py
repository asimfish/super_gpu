"""MCP adapter exposing super_gpu scheduling tools to AI agents."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]

from urllib.parse import urlencode

from .api_manifest import LOG_LINES_DEFAULT, LOG_LINES_MAX, WAIT_TIMEOUT_MAX_SECONDS
from .client import SuperGPUClient


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def create_mcp(
    *,
    url: str | None = None,
    token: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> Any:
    if FastMCP is None:
        raise RuntimeError("install the optional MCP dependency: pip install 'super-gpu[mcp]'")
    client = SuperGPUClient(
        url or os.environ.get("SUPER_GPU_URL", "http://127.0.0.1:8765"),
        token or os.environ.get("SUPER_GPU_API_TOKEN", ""),
    )
    mcp = FastMCP("super_gpu", host=host, port=port, streamable_http_path="/mcp")

    @mcp.tool()
    def cluster_snapshot() -> str:
        """Return live server/GPU utilization, leases, running jobs, and scheduler health."""
        return _dumps(client.get("/api/state"))

    @mcp.tool()
    def scheduler_status() -> str:
        """Return scheduler heartbeat and queue counts."""
        return _dumps(client.get("/api/scheduler"))

    @mcp.tool()
    def experiment_brief(hours: float = 6.0, format: str = "text") -> str:
        """One-screen situation brief: controller health, attention items (failures, blocked
        jobs, offline nodes, idle-yet-occupied GPUs), queue, free capacity, changes in the
        window, and recommendations. Use this first when supervising. format: text | json."""
        if format == "json":
            return _dumps(client.get(f"/api/brief?hours={float(hours):g}"))
        return client.get_text(f"/api/brief?hours={float(hours):g}&format=text")

    @mcp.tool()
    def capacity_query(
        gpus: int = 1,
        memory_mib: int = 0,
        nodes: str = "",
        labels: str = "",
    ) -> str:
        """Dry-run placement: how many jobs needing `gpus` GPUs and `memory_mib` MiB per GPU
        (0 = estimate) could start right now, and on which GPUs. Observation only, not a
        reservation. `nodes`/`labels` are comma-separated constraints."""
        query = {"gpus": str(int(gpus))}
        if memory_mib:
            query["memory_mib"] = str(int(memory_mib))
        if nodes:
            query["nodes"] = nodes
        if labels:
            query["labels"] = labels
        return _dumps(client.get(f"/api/capacity?{urlencode(query)}"))

    @mcp.tool()
    def plan_preview(plan_json: str) -> str:
        """Validate a plan and show, per job, its resource estimate and whether it would
        start now (and where) or queue (and why). Nothing is submitted."""
        plan = json.loads(plan_json)
        if not isinstance(plan, dict):
            raise ValueError("plan_json must contain a JSON object")
        return _dumps(client.post("/api/plans/preview", {"plan": plan}))

    @mcp.tool()
    def experiment_submit(plan_json: str, request_id: str) -> str:
        """Submit a plan with a stable request ID; retries return the original plan."""
        plan = json.loads(plan_json)
        if not isinstance(plan, dict):
            raise ValueError("plan_json must contain a JSON object")
        return _dumps(client.post("/api/plans", {"plan": plan, "request_id": request_id}))

    @mcp.tool()
    def experiment_wait(plan_id: str, timeout_seconds: float = 60.0) -> str:
        """Block until the plan is terminal or `timeout_seconds` (max 300) elapse; the answer
        carries `terminal` and the plan either way. Prefer this over polling in a loop."""
        timeout = max(0.0, min(float(timeout_seconds), WAIT_TIMEOUT_MAX_SECONDS))
        return _dumps(
            client.get(f"/api/plans/{plan_id}/wait?timeout={timeout:g}", timeout=timeout + 30)
        )

    @mcp.tool()
    def experiment_status(plan_id: str) -> str:
        """Return one plan and all of its jobs, placements, estimates, and logs."""
        return _dumps(client.get(f"/api/plans/{plan_id}"))

    @mcp.tool()
    def experiment_jobs(status: str = "", limit: int = 200) -> str:
        """List jobs, optionally filtered by comma-separated status values."""
        query = f"?limit={limit}"
        if status:
            query += f"&status={status}"
        return _dumps(client.get(f"/api/jobs{query}"))

    @mcp.tool()
    def experiment_logs(job_id: str, lines: int = LOG_LINES_DEFAULT) -> str:
        """Fetch the last `lines` lines (max 5000) of a job's stdout and stderr from the
        node that ran it, plus its result.json and exit code when finished."""
        lines = max(1, min(int(lines), LOG_LINES_MAX))
        return _dumps(client.get(f"/api/jobs/{job_id}/logs?lines={lines}"))

    @mcp.tool()
    def experiment_outputs(job_id: str) -> str:
        """List a finished job's declared output files (expanded on the node, no transfer)."""
        return _dumps(client.get(f"/api/jobs/{job_id}/outputs"))

    @mcp.tool()
    def experiment_cancel(job_id: str) -> str:
        """Request graceful cancellation of a queued or running job."""
        return _dumps(client.post(f"/api/jobs/{job_id}/cancel"))

    @mcp.tool()
    def scheduler_events(limit: int = 100, since: float = 0.0) -> str:
        """Return recent placement, completion, retry, and controller events, optionally
        only those created after `since` (epoch seconds)."""
        query = f"?limit={limit}"
        if since:
            query += f"&since={float(since):g}"
        return _dumps(client.get(f"/api/events{query}"))

    @mcp.tool()
    def anomaly_report() -> str:
        """Return sustained idle-GPU findings and the active watchdog policy."""
        return _dumps(client.get("/api/anomalies"))

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="super_gpu MCP server")
    parser.add_argument("--url", default=os.environ.get("SUPER_GPU_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--token", default=os.environ.get("SUPER_GPU_API_TOKEN", ""))
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    mcp = create_mcp(url=args.url, token=args.token, host=args.host, port=args.port)
    if args.transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
