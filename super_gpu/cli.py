"""Command-line interface for operators and shell-capable agents."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .api import serve
from .client import SuperGPUClient
from .config import load_config, load_plan
from .models import ExperimentPlan
from .monitor import ClusterMonitor
from .scheduler import Scheduler
from .store import StateStore


def _print(value: Any, *, pretty: bool = True) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )
    )


def _client(args: argparse.Namespace) -> SuperGPUClient | None:
    if getattr(args, "url", ""):
        return SuperGPUClient(args.url, getattr(args, "token", ""))
    return None


def _scheduler(args: argparse.Namespace) -> Scheduler:
    return Scheduler(load_config(args.config))


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    payload: dict[str, Any] = {"ok": True, "config": cfg.public_dict()}
    if args.plan:
        plan = load_plan(args.plan)
        payload["plan"] = {
            "name": plan.name,
            "jobs": len(plan.jobs),
            "max_parallel": plan.max_parallel,
        }
    _print(payload)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = StateStore(cfg.database)
    snapshots = ClusterMonitor(cfg).collect_all()
    store.save_snapshots(snapshots)
    ok = all(snapshot.reachable for snapshot in snapshots)
    _print(
        {
            "ok": ok,
            "database": str(store.path),
            "nodes": [snapshot.as_dict() for snapshot in snapshots],
        }
    )
    return 0 if ok else 2


def cmd_scan(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        _print(client.post("/api/scan"))
        return 0
    cfg = load_config(args.config)
    monitor = ClusterMonitor(cfg)
    snapshots = monitor.collect_all()
    StateStore(cfg.database).save_snapshots(snapshots)
    payload = {
        "ok": all(snapshot.reachable for snapshot in snapshots),
        "snapshots": [snapshot.as_dict() for snapshot in snapshots],
    }
    _print(payload)
    return 0 if payload["ok"] else 2


def cmd_serve(args: argparse.Namespace) -> int:
    scheduler = _scheduler(args)
    serve(
        scheduler,
        host=args.host,
        port=args.port,
        start_scheduler=not args.no_scheduler,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scheduler = _scheduler(args)
    plan = load_plan(args.plan)
    scheduler.start()
    try:
        submitted = scheduler.submit(plan)
        _print({"ok": True, "submitted": submitted})
        result = scheduler.wait_for_plan(submitted["id"], timeout=args.wait_timeout)
        _print({"ok": result.get("status") == "completed", "plan": result})
        return 0 if result.get("status") == "completed" else 1
    finally:
        scheduler.stop()


def cmd_submit(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).expanduser()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    client = _client(args)
    if client:
        _print(client.post("/api/plans", {"plan": raw}))
        return 0
    plan = ExperimentPlan.from_dict(raw)
    scheduler = _scheduler(args)
    _print({"ok": True, "plan": scheduler.submit(plan)})
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        _print(client.get("/api/state"))
        return 0
    scheduler = _scheduler(args)
    from .api import dashboard_state

    _print(dashboard_state(scheduler))
    return 0


def cmd_plans(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        _print(client.get(f"/api/plans?limit={args.limit}"))
    else:
        store = StateStore(load_config(args.config).database)
        _print({"ok": True, "plans": store.list_plans(limit=args.limit)})
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    client = _client(args)
    query = f"?limit={args.limit}"
    if args.plan_id:
        query += f"&plan_id={args.plan_id}"
    if args.status:
        query += f"&status={args.status}"
    if client:
        _print(client.get(f"/api/jobs{query}"))
    else:
        store = StateStore(load_config(args.config).database)
        statuses = [value for value in args.status.split(",") if value] if args.status else None
        _print(
            {
                "ok": True,
                "jobs": store.list_jobs(
                    plan_id=args.plan_id or None,
                    statuses=statuses,
                    limit=args.limit,
                ),
            }
        )
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        payload = client.post(f"/api/jobs/{args.job_id}/cancel")
    else:
        store = StateStore(load_config(args.config).database)
        job = store.request_cancel(args.job_id)
        payload = {"ok": bool(job), "job": job}
    _print(payload)
    return 0 if payload.get("ok") else 1


def cmd_events(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        payload = client.get(f"/api/events?limit={args.limit}")
    else:
        store = StateStore(load_config(args.config).database)
        payload = {"ok": True, "events": store.list_events(limit=args.limit)}
    _print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="super-gpu",
        description="Resource-aware multi-server GPU experiment scheduler",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("SUPER_GPU_CONFIG", "config.json"),
        help="cluster config JSON (or SUPER_GPU_CONFIG)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate config and optional plan")
    validate.add_argument("--plan")
    validate.set_defaults(func=cmd_validate)

    doctor = sub.add_parser("doctor", help="check DB, SSH, and NVIDIA telemetry")
    doctor.set_defaults(func=cmd_doctor)

    scan = sub.add_parser("scan", help="collect an immediate cluster snapshot")
    _remote_args(scan)
    scan.set_defaults(func=cmd_scan)

    serve_parser = sub.add_parser("serve", aliases=["dashboard"], help="run scheduler and dashboard")
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.add_argument("--no-scheduler", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    run = sub.add_parser("run", help="submit a plan and wait for completion")
    run.add_argument("plan")
    run.add_argument("--wait-timeout", type=float)
    run.set_defaults(func=cmd_run)

    submit = sub.add_parser("submit", help="submit a plan locally or to a daemon")
    submit.add_argument("plan")
    _remote_args(submit)
    submit.set_defaults(func=cmd_submit)

    state = sub.add_parser("state", help="show scheduler, fleet, and job state")
    _remote_args(state)
    state.set_defaults(func=cmd_state)

    plans = sub.add_parser("plans", help="list experiment plans")
    plans.add_argument("--limit", type=int, default=100)
    _remote_args(plans)
    plans.set_defaults(func=cmd_plans)

    jobs = sub.add_parser("jobs", help="list jobs")
    jobs.add_argument("--plan-id", default="")
    jobs.add_argument("--status", default="")
    jobs.add_argument("--limit", type=int, default=500)
    _remote_args(jobs)
    jobs.set_defaults(func=cmd_jobs)

    cancel = sub.add_parser("cancel", help="cancel one job")
    cancel.add_argument("job_id")
    _remote_args(cancel)
    cancel.set_defaults(func=cmd_cancel)

    events = sub.add_parser("events", help="show scheduler events")
    events.add_argument("--limit", type=int, default=100)
    _remote_args(events)
    events.set_defaults(func=cmd_events)
    return parser


def _remote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        default=os.environ.get("SUPER_GPU_URL", ""),
        help="running super_gpu daemon URL",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SUPER_GPU_API_TOKEN", ""),
        help="daemon API token",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        _print({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
