"""Machine-readable contract of the HTTP API and MCP surface.

``GET /api/meta`` serializes this module so an agent can discover what a
controller supports before calling it, branch on stable error ``code`` values
instead of parsing English prose, and find the matching documentation for the
exact version it is talking to. The README route table, the MCP tool list in
``AGENTS.md``, and the tools registered by ``mcp_server`` are tested against
this manifest so the three can never disagree.
"""
from __future__ import annotations

from typing import Any

from . import __version__

API_VERSION = "1"
DOCUMENTATION_URL = "https://github.com/asimfish/super_gpu/blob/main/README.md"
SCHEMA_URL = (
    "https://github.com/asimfish/super_gpu/blob/main/schemas/experiment-plan.schema.json"
)

# Access tiers: ``public`` never requires the API token (discovery and health
# only); ``token`` requires it whenever one is configured.
API_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("GET", "/api/meta", "public", "API manifest: routes, error codes, MCP tools, capability flags"),
    ("GET", "/healthz", "public", "liveness: the HTTP server answers"),
    ("GET", "/readyz", "public", "readiness: scheduler thread alive and at least one tick completed (503 otherwise)"),
    ("GET", "/api/state", "token", "full cluster, job, lease, and event state for the dashboard"),
    ("GET", "/api/brief", "token", "one-screen situation brief: health, attention items, queue, capacity, changes, recommendations"),
    ("GET", "/api/capacity", "token", "dry-run placement: how many jobs of a given shape could start right now, and where"),
    ("GET", "/api/snapshot", "token", "latest GPU snapshot and active leases"),
    ("GET", "/api/scheduler", "token", "scheduler heartbeat, queue counts, watchdog status"),
    ("GET", "/api/config", "token", "public view of the controller configuration (no secrets)"),
    ("GET", "/api/plans", "token", "list plans"),
    ("POST", "/api/plans", "token", "submit a plan JSON; supports top-level request_id, conflict -> 409"),
    ("POST", "/api/plans/preview", "token", "validate a plan and preview estimates and placement without submitting"),
    ("GET", "/api/plans/<id>", "token", "one plan with all jobs"),
    ("GET", "/api/plans/<id>/wait", "token", "long-poll (<= 300 s) until the plan is terminal"),
    ("GET", "/api/jobs", "token", "query jobs (each carries pending_reason and result_state)"),
    ("GET", "/api/jobs/<id>", "token", "one job including command, stored log tails, and result"),
    ("GET", "/api/jobs/<id>/wait", "token", "long-poll (<= 300 s) until the job is terminal"),
    ("GET", "/api/jobs/<id>/logs", "token", "fetch the last N lines of stdout/stderr from the node that ran the job"),
    ("GET", "/api/jobs/<id>/outputs", "token", "expand a finished job's declared outputs on its node"),
    ("POST", "/api/jobs/<id>/cancel", "token", "cancel a job"),
    ("POST", "/api/scan", "token", "refresh GPU state immediately"),
    ("GET", "/api/events", "token", "scheduler events; optional since=<epoch seconds>"),
    ("GET", "/api/anomalies", "token", "current low-utilization occupancy and watchdog policy"),
)

# Stable error codes and the HTTP status each is served with. Every error
# body is ``{"ok": false, "error": <human text>, "code": <one of these>}``.
ERROR_CODES: tuple[tuple[str, int, str], ...] = (
    ("invalid_request", 400, "malformed JSON, schema violation, or an out-of-range parameter"),
    ("unauthorized", 401, "API token missing or wrong"),
    ("not_found", 404, "unknown route, plan, or job"),
    ("method_not_allowed", 405, "the route exists but not for this HTTP method"),
    ("idempotency_conflict", 409, "request_id reused with different plan content"),
    ("remote_failure", 502, "the node could not be reached or the remote command failed"),
    ("internal_error", 500, "unexpected controller failure; see scheduler events"),
    ("not_ready", 503, "scheduler has not completed its first tick or is not running"),
)

# MCP tools exposed by ``super_gpu.mcp_server``; AGENTS.md lists exactly these.
MCP_TOOLS: tuple[tuple[str, str], ...] = (
    ("cluster_snapshot", "latest GPU telemetry, leases, and scheduler health for every node"),
    ("scheduler_status", "tick, queue depth, watchdog, and lease state"),
    ("experiment_brief", "one-screen situation brief: what needs attention, queue, free capacity, recommendations"),
    ("capacity_query", "dry-run placement: how many jobs of a given GPU/memory shape could start now"),
    ("plan_preview", "validate a plan and preview estimates and placement without submitting"),
    ("experiment_submit", "idempotent plan submission with request_id"),
    ("experiment_wait", "block (bounded) until a plan is terminal; returns the plan either way"),
    ("experiment_status", "one plan with all of its jobs"),
    ("experiment_jobs", "filterable job listing"),
    ("experiment_logs", "last N lines of a job's stdout/stderr fetched from its node"),
    ("experiment_outputs", "expand a finished job's declared outputs on its node"),
    ("experiment_cancel", "cancel a single job"),
    ("scheduler_events", "recent scheduling decisions and transitions"),
    ("anomaly_report", "idle-yet-occupied GPU findings and watchdog policy"),
)

WAIT_TIMEOUT_MAX_SECONDS = 300.0
LOG_LINES_DEFAULT = 200
LOG_LINES_MAX = 5000
CAPACITY_MAX_GPUS = 64


def route_paths() -> set[str]:
    return {path for _, path, _, _ in API_ROUTES}


def error_code_status(code: str) -> int:
    for name, status, _ in ERROR_CODES:
        if name == code:
            return status
    raise KeyError(code)


def build_meta(*, capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``GET /api/meta`` document."""
    return {
        "ok": True,
        "service": "super_gpu",
        "version": __version__,
        "api_version": API_VERSION,
        "documentation_url": DOCUMENTATION_URL,
        "plan_schema_url": SCHEMA_URL,
        "authentication": {
            "header": "Authorization: Bearer <token>",
            "alternative_header": "X-Super-GPU-Token",
            "tiers": {
                "public": "never requires the API token",
                "token": "requires the API token when one is configured",
            },
        },
        "routes": [
            {"method": method, "path": path, "tier": tier, "purpose": purpose}
            for method, path, tier, purpose in API_ROUTES
        ],
        "error_codes": [
            {"code": code, "status": status, "meaning": meaning}
            for code, status, meaning in ERROR_CODES
        ],
        "mcp_tools": [{"name": name, "purpose": purpose} for name, purpose in MCP_TOOLS],
        "limits": {
            "wait_timeout_max_seconds": WAIT_TIMEOUT_MAX_SECONDS,
            "log_lines_default": LOG_LINES_DEFAULT,
            "log_lines_max": LOG_LINES_MAX,
            "capacity_max_gpus": CAPACITY_MAX_GPUS,
        },
        "capabilities": dict(capabilities or {}),
    }
