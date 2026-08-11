"""REST API and packaged monitoring dashboard."""
from __future__ import annotations

import hmac
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .models import ExperimentPlan
from .scheduler import Scheduler


WEB_ROOT = Path(__file__).with_name("web")


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
    server_version = "super-gpu/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[super-gpu] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/health", "/api/health"}:
            self._json(
                200,
                {
                    "ok": True,
                    "service": "super_gpu",
                    "scheduler": self.server.scheduler.status(),
                },
            )
            return
        if path.startswith("/api/") and not self._authorized():
            return
        if path == "/api/state":
            self._json(200, dashboard_state(self.server.scheduler))
            return
        if path == "/api/snapshot":
            self._json(
                200,
                {
                    "ok": True,
                    "snapshots": self.server.scheduler.store.latest_snapshots(),
                    "leases": self.server.scheduler.store.leases(active_only=True),
                },
            )
            return
        if path == "/api/scheduler":
            self._json(200, {"ok": True, "scheduler": self.server.scheduler.status()})
            return
        if path == "/api/config":
            self._json(200, {"ok": True, "config": self.server.scheduler.config.public_dict()})
            return
        if path == "/api/plans":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["100"])[0])
            self._json(
                200,
                {"ok": True, "plans": self.server.scheduler.store.list_plans(limit=limit)},
            )
            return
        if path.startswith("/api/plans/"):
            plan_id = path.removeprefix("/api/plans/").strip("/")
            plan = self.server.scheduler.store.get_plan(plan_id, include_jobs=True)
            self._json(200 if plan else 404, {"ok": bool(plan), "plan": plan})
            return
        if path == "/api/jobs":
            params = parse_qs(parsed.query)
            plan_id = params.get("plan_id", [None])[0]
            statuses = [
                value
                for item in params.get("status", [])
                for value in item.split(",")
                if value
            ]
            jobs = self.server.scheduler.store.list_jobs(
                plan_id=plan_id,
                statuses=statuses or None,
                limit=int(params.get("limit", ["500"])[0]),
            )
            self._json(200, {"ok": True, "jobs": jobs})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            job = self.server.scheduler.store.get_job(job_id)
            self._json(200 if job else 404, {"ok": bool(job), "job": job})
            return
        if path == "/api/events":
            params = parse_qs(parsed.query)
            events = self.server.scheduler.store.list_events(
                limit=int(params.get("limit", ["100"])[0])
            )
            self._json(200, {"ok": True, "events": events})
            return
        if path == "/api/anomalies":
            self._json(
                200,
                {
                    "ok": True,
                    "watchdog": self.server.scheduler.guardian.status(),
                    "anomalies": self.server.scheduler.guardian.findings(),
                },
            )
            return
        if path == "/" or path == "/index.html":
            self._static("index.html")
            return
        if path.startswith("/static/"):
            self._static(path.removeprefix("/static/"))
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/") and not self._authorized():
            return
        try:
            body = self._body()
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        try:
            if path == "/api/plans":
                raw_plan = body.get("plan", body)
                plan = ExperimentPlan.from_dict(raw_plan)
                submitted = self.server.scheduler.submit(plan)
                self._json(201, {"ok": True, "plan": submitted})
                return
            if path == "/api/scan":
                snapshots = self.server.scheduler.monitor.collect_all()
                self.server.scheduler.store.save_snapshots(snapshots)
                self._json(
                    200,
                    {"ok": True, "snapshots": [snapshot.as_dict() for snapshot in snapshots]},
                )
                return
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                job = self.server.scheduler.store.request_cancel(job_id)
                self._json(200 if job else 404, {"ok": bool(job), "job": job})
                return
        except (ValueError, KeyError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"ok": False, "error": str(exc)})
            return
        self._json(404, {"ok": False, "error": "not found"})

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
        self._json(401, {"ok": False, "error": "invalid or missing API token"})
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

    def _json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _static(self, name: str) -> None:
        target = (WEB_ROOT / name).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._json(403, {"ok": False, "error": "forbidden"})
            return
        if not target.is_file():
            self._json(404, {"ok": False, "error": "not found"})
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
