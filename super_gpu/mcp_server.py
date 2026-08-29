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

from .client import SuperGPUClient


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
        return json.dumps(client.get("/api/state"), ensure_ascii=False, indent=2)

    @mcp.tool()
    def scheduler_status() -> str:
        """Return scheduler heartbeat and queue counts."""
        return json.dumps(client.get("/api/scheduler"), ensure_ascii=False, indent=2)

    @mcp.tool()
    def experiment_submit(plan_json: str, request_id: str) -> str:
        """Submit a plan with a stable request ID; retries return the original plan."""
        plan = json.loads(plan_json)
        if not isinstance(plan, dict):
            raise ValueError("plan_json must contain a JSON object")
        return json.dumps(
            client.post("/api/plans", {"plan": plan, "request_id": request_id}),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def experiment_status(plan_id: str) -> str:
        """Return one plan and all of its jobs, placements, estimates, and logs."""
        return json.dumps(
            client.get(f"/api/plans/{plan_id}"),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def experiment_jobs(status: str = "", limit: int = 200) -> str:
        """List jobs, optionally filtered by comma-separated status values."""
        query = f"?limit={limit}"
        if status:
            query += f"&status={status}"
        return json.dumps(client.get(f"/api/jobs{query}"), ensure_ascii=False, indent=2)

    @mcp.tool()
    def experiment_cancel(job_id: str) -> str:
        """Request graceful cancellation of a queued or running job."""
        return json.dumps(
            client.post(f"/api/jobs/{job_id}/cancel"),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def experiment_outputs(job_id: str) -> str:
        """List a finished job's declared output files (expanded on the node, no transfer)."""
        return json.dumps(
            client.get(f"/api/jobs/{job_id}/outputs"),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def scheduler_events(limit: int = 100) -> str:
        """Return recent placement, completion, retry, and controller events."""
        return json.dumps(
            client.get(f"/api/events?limit={limit}"),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    def anomaly_report() -> str:
        """Return sustained idle-GPU findings and the active watchdog policy."""
        return json.dumps(client.get("/api/anomalies"), ensure_ascii=False, indent=2)

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
