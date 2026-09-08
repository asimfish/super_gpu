"""REST API and packaged monitoring dashboard.

Every error body is ``{"ok": false, "error": <text>, "code": <stable code>}``
with the codes catalogued in :mod:`super_gpu.api_manifest`, so an agent can
branch on ``code`` instead of parsing prose. ``GET /api/meta`` is public and
describes every route; unknown API paths and wrong methods answer in the same
JSON envelope.
"""
from __future__ import annotations

import hmac
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .api_manifest import (
    API_ROUTES,
    LOG_LINES_DEFAULT,
    WAIT_TIMEOUT_MAX_SECONDS,
    build_meta,
    error_code_status,
)
from .brief import DEFAULT_WINDOW_SECONDS, WINDOW_MAX_SECONDS, build_brief, render_brief
from .capacity import CapacityRequest, dry_run_capacity, preview_plan, snapshots_from_store
from .logs import fetch_job_logs
from .models import TERMINAL_JOB_STATES, ExperimentPlan
from .outputs import describe_job_outputs
from .scheduler import Scheduler
from .store import TERMINAL_PLAN_STATES, IdempotencyConflict, StateStore


WEB_ROOT = Path(__file__).with_name("web")

_ROUTE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (method, re.compile("^" + re.escape(path).replace(re.escape("<id>"), r"[^/]+") + "$"))
    for method, path, _, _ in API_ROUTES
]


def allowed_methods(path: str) -> set[str]:
    """HTTP methods the manifest declares for ``path`` (empty when unknown)."""
    return {method for method, pattern in _ROUTE_PATTERNS if pattern.match(path)}


def dashboard_state(scheduler: Scheduler) -> dict[str, Any]:
    snapshots = scheduler.store.latest_snapshots()
    leases = scheduler.store.leases(active_only=True)
    leases_by_gpu: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for lease in leases:
        leases_by_gpu.setdefault(
            (str(lease["node"]), int(lease["gpu_index"])),
            [],
        ).append(lease)
    full_jobs = scheduler.store.list_jobs(limit=500)
    job_by_id = {job["id"]: job for job in full_jobs}
    jobs = []
    for full_job in full_jobs:
        job = dict(full_job)
        for private_or_heavy in (
            "command",
            "env",
            "params",
            "stdout_tail",
            "stderr_tail",
            "handle",
        ):
            job.pop(private_or_heavy, None)
        jobs.append(job)
    plans = []
    for full_plan in scheduler.store.list_plans(limit=100):
        plan = dict(full_plan)
        plan.pop("plan", None)
        plans.append(plan)
    for snapshot in snapshots:
        try:
            node_cfg = scheduler.config.node(snapshot["node"])
            snapshot["priority"] = node_cfg.priority
            snapshot["policy"] = {
                "max_gpu_utilization": node_cfg.policy.max_gpu_utilization,
                "max_memory_used_ratio": node_cfg.policy.max_memory_used_ratio,
                "reserve_memory_mib": node_cfg.policy.reserve_memory_mib,
                "stabilization_samples": node_cfg.policy.stabilization_samples,
                "allow_colocation": node_cfg.policy.allow_colocation,
                "max_jobs_per_gpu": node_cfg.policy.max_jobs_per_gpu,
            }
        except KeyError:
            snapshot["priority"] = 0
            snapshot["policy"] = {}
        for gpu in snapshot.get("gpus", []):
            gpu_leases = leases_by_gpu.get(
                (snapshot["node"], int(gpu["index"])),
                [],
            )
            gpu["leases"] = [
                {
                    **lease,
                    "job_name": job_by_id.get(lease["job_id"], {}).get("name", lease["job_id"]),
                    "job_status": job_by_id.get(lease["job_id"], {}).get("status", ""),
                }
                for lease in gpu_leases
            ]
    return {
        "ok": True,
        "scheduler": scheduler.status(),
        "config": scheduler.config.public_dict(),
        "snapshots": snapshots,
        "plans": plans,
        "jobs": jobs,
        "events": scheduler.store.list_events(limit=100),
        "leases": leases,
        "anomalies": scheduler.guardian.findings(),
    }


def readiness(scheduler: Scheduler) -> tuple[bool, dict[str, Any]]:
    """Ready means the scheduler thread is alive and has completed a tick."""
    status = scheduler.status()
    running = bool(status.get("running"))
    ticked = float(status.get("last_tick") or 0.0) > 0
    reasons = []
    if not running:
        reasons.append("scheduler thread is not running")
    if not ticked:
        reasons.append("no scheduling tick has completed yet")
    return (running and ticked), {
        "ok": running and ticked,
        "ready": running and ticked,
        "service": "super_gpu",
        "version": __version__,
        "reasons": reasons,
        "scheduler": status,
    }


def capacity_report(scheduler: Scheduler, request: CapacityRequest) -> dict[str, Any]:
    store = scheduler.store
    return dry_run_capacity(
        scheduler.config,
        scheduler.placement,
        scheduler.estimator,
        snapshots_from_store(store.latest_snapshots()),
        store.leases(active_only=True),
        request,
        active_jobs=store.active_count(),
    )


def plan_preview(scheduler: Scheduler, plan: ExperimentPlan) -> dict[str, Any]:
    store = scheduler.store
    return preview_plan(
        scheduler.config,
        scheduler.placement,
        scheduler.estimator,
        snapshots_from_store(store.latest_snapshots()),
        store.leases(active_only=True),
        plan,
        active_jobs=store.active_count(),
    )


def _number(
    params: dict[str, list[str]],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    values = params.get(key) or []
    if not values or not values[0].strip():
        return default
    try:
        value = float(values[0])
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")
    return value


class SuperGPUHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        scheduler: Scheduler,
    ) -> None:
        super().__init__(address, SuperGPUHandler)
        self.scheduler = scheduler


class SuperGPUHandler(BaseHTTPRequestHandler):
    server: SuperGPUHTTPServer
    server_version = f"super-gpu/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Access logs go to stderr so stdout stays clean for data (the CLI
        # and demo print JSON there).
        print(f"[super-gpu] {self.address_string()} {fmt % args}", file=sys.stderr)

    # ----------------------------------------------------------------- GET
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        scheduler = self.server.scheduler
        store = scheduler.store

        if path == "/api/meta":
            self._json(200, build_meta(capabilities=self._capabilities()))
            return
        if path in {"/health", "/api/health", "/healthz"}:
            self._json(200, {"ok": True, "service": "super_gpu", "version": __version__, "scheduler": scheduler.status()})
            return
        if path == "/readyz":
            ready, payload = readiness(scheduler)
            if ready:
                self._json(200, payload)
            else:
                payload.update({"error": "; ".join(payload["reasons"]), "code": "not_ready"})
                self._json(503, payload)
            return
        if path.startswith("/api/") and not self._authorized():
            return
        try:
            if self._get_api(path, params, scheduler, store):
                return
        except ValueError as exc:
            self._error("invalid_request", str(exc))
            return
        if path == "/" or path == "/index.html":
            self._static("index.html")
            return
        if path.startswith("/static/"):
            self._static(path.removeprefix("/static/"))
            return
        self._unknown(path, "GET")

    def _get_api(
        self,
        path: str,
        params: dict[str, list[str]],
        scheduler: Scheduler,
        store: StateStore,
    ) -> bool:
        if path == "/api/state":
            self._json(200, dashboard_state(scheduler))
            return True
        if path == "/api/brief":
            hours = _number(
                params,
                "hours",
                default=DEFAULT_WINDOW_SECONDS / 3600,
                minimum=1 / 60,
                maximum=WINDOW_MAX_SECONDS / 3600,
            )
            brief = build_brief(scheduler, window_seconds=hours * 3600)
            if (params.get("format") or [""])[0] == "text":
                self._text(200, render_brief(brief))
            else:
                self._json(200, brief)
            return True
        if path == "/api/capacity":
            request = CapacityRequest.from_query(params)
            self._json(200, {"ok": True, "capacity": capacity_report(scheduler, request)})
            return True
        if path == "/api/snapshot":
            self._json(
                200,
                {
                    "ok": True,
                    "snapshots": store.latest_snapshots(),
                    "leases": store.leases(active_only=True),
                },
            )
            return True
        if path == "/api/scheduler":
            self._json(200, {"ok": True, "scheduler": scheduler.status()})
            return True
        if path == "/api/config":
            self._json(200, {"ok": True, "config": scheduler.config.public_dict()})
            return True
        if path == "/api/plans":
            limit = int(_number(params, "limit", default=100, minimum=1, maximum=10000))
            self._json(200, {"ok": True, "plans": store.list_plans(limit=limit)})
            return True
        if path.startswith("/api/plans/") and path.endswith("/wait"):
            plan_id = path.removeprefix("/api/plans/").removesuffix("/wait").strip("/")
            timeout = _number(params, "timeout", default=30, minimum=0, maximum=WAIT_TIMEOUT_MAX_SECONDS)
            try:
                plan = scheduler.wait_for_plan(plan_id, timeout=timeout)
            except KeyError:
                self._error("not_found", f"unknown plan {plan_id}")
                return True
            self._json(
                200,
                {"ok": True, "terminal": plan["status"] in TERMINAL_PLAN_STATES, "plan": plan},
            )
            return True
        if path.startswith("/api/plans/"):
            plan_id = path.removeprefix("/api/plans/").strip("/")
            plan = store.get_plan(plan_id, include_jobs=True)
            if not plan:
                self._error("not_found", f"unknown plan {plan_id}")
            else:
                self._json(200, {"ok": True, "plan": plan})
            return True
        if path == "/api/jobs":
            plan_id = params.get("plan_id", [None])[0]
            statuses = [
                value
                for item in params.get("status", [])
                for value in item.split(",")
                if value
            ]
            limit = int(_number(params, "limit", default=500, minimum=1, maximum=100000))
            jobs = store.list_jobs(plan_id=plan_id, statuses=statuses or None, limit=limit)
            self._json(200, {"ok": True, "jobs": jobs})
            return True
        if path.startswith("/api/jobs/") and path.endswith("/outputs"):
            job_id = path.removeprefix("/api/jobs/").removesuffix("/outputs").strip("/")
            try:
                manifest = describe_job_outputs(scheduler.config, store, job_id)
            except ValueError as exc:
                if str(exc).startswith("unknown job"):
                    self._error("not_found", str(exc))
                else:
                    self._error("invalid_request", str(exc))
                return True
            except Exception as exc:  # noqa: BLE001 - remote expansion may fail
                self._error("remote_failure", str(exc))
                return True
            self._json(200, {"ok": True, "outputs": manifest})
            return True
        if path.startswith("/api/jobs/") and path.endswith("/logs"):
            job_id = path.removeprefix("/api/jobs/").removesuffix("/logs").strip("/")
            lines = int(_number(params, "lines", default=LOG_LINES_DEFAULT, minimum=1, maximum=5000))
            try:
                logs = fetch_job_logs(scheduler.config, store, job_id, lines=lines)
            except ValueError as exc:
                if str(exc).startswith("unknown job"):
                    self._error("not_found", str(exc))
                else:
                    self._error("invalid_request", str(exc))
                return True
            except Exception as exc:  # noqa: BLE001 - remote tail may fail
                self._error("remote_failure", str(exc))
                return True
            self._json(200, {"ok": True, "logs": logs})
            return True
        if path.startswith("/api/jobs/") and path.endswith("/wait"):
            job_id = path.removeprefix("/api/jobs/").removesuffix("/wait").strip("/")
            timeout = _number(params, "timeout", default=30, minimum=0, maximum=WAIT_TIMEOUT_MAX_SECONDS)
            try:
                job = scheduler.wait_for_job(job_id, timeout=timeout)
            except KeyError:
                self._error("not_found", f"unknown job {job_id}")
                return True
            self._json(
                200,
                {"ok": True, "terminal": job["status"] in TERMINAL_JOB_STATES, "job": job},
            )
            return True
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            job = store.get_job(job_id)
            if not job:
                self._error("not_found", f"unknown job {job_id}")
            else:
                self._json(200, {"ok": True, "job": job})
            return True
        if path == "/api/events":
            limit = int(_number(params, "limit", default=100, minimum=1, maximum=10000))
            since_values = params.get("since") or []
            since = float(since_values[0]) if since_values and since_values[0].strip() else None
            kinds = [
                value
                for item in params.get("kind", [])
                for value in item.split(",")
                if value
            ]
            events = store.list_events(limit=limit, since=since, kinds=kinds or None)
            self._json(200, {"ok": True, "events": events})
            return True
        if path == "/api/anomalies":
            self._json(
                200,
                {
                    "ok": True,
                    "watchdog": scheduler.guardian.status(),
                    "anomalies": scheduler.guardian.findings(),
                },
            )
            return True
        return False

    # ---------------------------------------------------------------- POST
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/") and not self._authorized():
            return
        try:
            body = self._body()
        except ValueError as exc:
            self._error("invalid_request", str(exc))
            return
        scheduler = self.server.scheduler
        store = scheduler.store
        try:
            if path == "/api/plans":
                raw_plan = body.get("plan", body)
                if body.get("request_id"):
                    raw_plan = dict(raw_plan)
                    raw_plan["request_id"] = body["request_id"]
                plan = ExperimentPlan.from_dict(raw_plan)
                submitted = scheduler.submit(plan)
                status = 200 if submitted.get("submission", {}).get("replayed") else 201
                self._json(status, {"ok": True, "plan": submitted})
                return
            if path == "/api/plans/preview":
                raw_plan = body.get("plan", body)
                plan = ExperimentPlan.from_dict(raw_plan)
                self._json(200, {"ok": True, "preview": plan_preview(scheduler, plan)})
                return
            if path == "/api/scan":
                snapshots = scheduler.monitor.collect_all()
                store.save_snapshots(snapshots)
                self._json(
                    200,
                    {"ok": True, "snapshots": [snapshot.as_dict() for snapshot in snapshots]},
                )
                return
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                job = store.request_cancel(job_id)
                if not job:
                    self._error("not_found", f"unknown job {job_id}")
                else:
                    self._json(200, {"ok": True, "job": job})
                return
        except IdempotencyConflict as exc:
            self._error("idempotency_conflict", str(exc))
            return
        except (ValueError, KeyError) as exc:
            self._error("invalid_request", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._error("internal_error", str(exc))
            return
        self._unknown(path, "POST")

    # ------------------------------------------------------------- helpers
    def _capabilities(self) -> dict[str, Any]:
        config = self.server.scheduler.config
        try:
            import mcp  # noqa: F401

            mcp_available = True
        except ModuleNotFoundError:
            mcp_available = False
        return {
            "api_token_required": bool(config.api_token),
            "mcp_extra_installed": mcp_available,
            "notifications": len(config.notifications),
            "watchdog_action": self.server.scheduler.guardian.status()["policy"].get("action"),
            "nodes": len(config.nodes),
            "source_snapshot": True,
            "typed_results": True,
            "declared_outputs": True,
        }

    def _unknown(self, path: str, method: str) -> None:
        methods = allowed_methods(path)
        if methods and method not in methods:
            self._error(
                "method_not_allowed",
                f"{method} is not allowed for {path}; use {', '.join(sorted(methods))}",
                headers={"Allow": ", ".join(sorted(methods))},
            )
            return
        self._error("not_found", f"unknown route {method} {path}; see GET /api/meta")

    def _authorized(self) -> bool:
        expected = self.server.scheduler.config.api_token
        if not expected:
            return True
        supplied = self.headers.get("X-Super-GPU-Token", "")
        authorization = self.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if hmac.compare_digest(supplied, expected):
            return True
        self._error("unauthorized", "invalid or missing API token")
        return False

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _error(self, code: str, message: str, *, headers: dict[str, str] | None = None) -> None:
        self._json(
            error_code_status(code),
            {"ok": False, "error": message, "code": code},
            headers=headers,
        )

    def _json(self, status: int, payload: Any, *, headers: dict[str, str] | None = None) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, encoded, "application/json; charset=utf-8", headers)

    def _text(self, status: int, text: str) -> None:
        self._send(status, text.encode("utf-8"), "text/plain; charset=utf-8", None)

    def _send(
        self,
        status: int,
        encoded: bytes,
        content_type: str,
        headers: dict[str, str] | None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _static(self, name: str) -> None:
        target = (WEB_ROOT / name).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._error("not_found", f"no static file {name}")
            return
        if not target.is_file():
            self._error("not_found", f"no static file {name}")
            return
        payload = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def create_server(
    scheduler: Scheduler,
    *,
    host: str | None = None,
    port: int | None = None,
) -> SuperGPUHTTPServer:
    resolved_host = host if host is not None else scheduler.config.api_host
    resolved_port = int(port if port is not None else scheduler.config.api_port)
    if resolved_host not in {"127.0.0.1", "localhost", "::1"} and not scheduler.config.api_token:
        raise ValueError(
            "refusing to bind a non-loopback address without api_token or "
            "SUPER_GPU_API_TOKEN"
        )
    return SuperGPUHTTPServer((resolved_host, resolved_port), scheduler)


def serve(
    scheduler: Scheduler,
    *,
    host: str | None = None,
    port: int | None = None,
    start_scheduler: bool = True,
) -> None:
    if start_scheduler:
        scheduler.start()
    server = create_server(scheduler, host=host, port=port)
    address, actual_port = server.server_address[:2]
    print(f"[super-gpu] dashboard: http://{address}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if start_scheduler:
            scheduler.stop()
