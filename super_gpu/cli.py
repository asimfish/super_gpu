"""Command-line interface for operators and shell-capable agents."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .api import capacity_report, plan_preview, serve
from .api_manifest import LOG_LINES_DEFAULT, WAIT_TIMEOUT_MAX_SECONDS, build_meta
from .brief import DEFAULT_WINDOW_SECONDS, build_brief, render_brief
from .capacity import CapacityRequest
from .client import SuperGPUClient, SuperGPUError
from .config import import_gpumgr_inventory, load_config, load_plan
from .logs import fetch_job_logs
from .models import TERMINAL_JOB_STATES, ExperimentPlan
from .monitor import ClusterMonitor
from .scheduler import Scheduler
from .store import TERMINAL_PLAN_STATES, StateStore

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNHEALTHY = 2
EXIT_TIMEOUT = 3


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
            "notifications": [webhook.public_dict() for webhook in cfg.notifications],
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
    if args.request_id:
        plan.request_id = args.request_id
        plan.validate()
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
    if args.request_id:
        raw["request_id"] = args.request_id
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


def cmd_pull(args: argparse.Namespace) -> int:
    from .outputs import pull_job_outputs

    cfg = load_config(args.config)
    store = StateStore(cfg.database)
    summary = pull_job_outputs(cfg, store, args.job_id, args.dest)
    _print({"ok": True, "pull": summary})
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        payload = client.get(f"/api/events?limit={args.limit}")
    else:
        store = StateStore(load_config(args.config).database)
        payload = {"ok": True, "events": store.list_events(limit=args.limit)}
    _print(payload)
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    client = _client(args)
    hours = float(args.hours)
    if client:
        if args.json:
            payload = client.get(f"/api/brief?hours={hours:g}")
        else:
            print(client.get_text(f"/api/brief?hours={hours:g}&format=text"), end="")
            return 0
    else:
        payload = build_brief(_scheduler(args), window_seconds=hours * 3600)
        if not args.json:
            print(render_brief(payload), end="")
            return 0
    _print(payload)
    return 0


def cmd_capacity(args: argparse.Namespace) -> int:
    query: dict[str, list[str]] = {"gpus": [str(args.gpus)]}
    if args.memory_mib:
        query["memory_mib"] = [str(args.memory_mib)]
    if args.gpu_utilization is not None:
        query["gpu_utilization"] = [str(args.gpu_utilization)]
    if args.nodes:
        query["nodes"] = [args.nodes]
    if args.labels:
        query["labels"] = [args.labels]
    if args.allow_colocation:
        query["allow_colocation"] = [args.allow_colocation]
    if args.max_slots:
        query["max_slots"] = [str(args.max_slots)]
    client = _client(args)
    if client:
        from urllib.parse import urlencode

        flat = {key: values[0] for key, values in query.items()}
        _print(client.get(f"/api/capacity?{urlencode(flat)}"))
        return 0
    request = CapacityRequest.from_query(query)
    _print({"ok": True, "capacity": capacity_report(_scheduler(args), request)})
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).expanduser()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    client = _client(args)
    if client:
        payload = client.post("/api/plans/preview", {"plan": raw})
    else:
        plan = ExperimentPlan.from_dict(raw)
        payload = {"ok": True, "preview": plan_preview(_scheduler(args), plan)}
    _print(payload)
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    """Block until a plan (or job) is terminal.

    Exit 0 when it completed, 1 when it failed or was cancelled, 3 when the
    overall timeout elapsed first. Remote waits are chained bounded
    long-polls so a controller behind a proxy never sees a request longer
    than WAIT_TIMEOUT_MAX_SECONDS.
    """
    kind = "jobs" if args.job else "plans"
    key = "job" if args.job else "plan"
    terminal_states = TERMINAL_JOB_STATES if args.job else TERMINAL_PLAN_STATES
    overall = float(args.timeout) if args.timeout is not None else None
    deadline = time.monotonic() + overall if overall is not None else None
    client = _client(args)
    scheduler = None if client else _scheduler(args)
    while True:
        remaining = (deadline - time.monotonic()) if deadline is not None else None
        if remaining is not None and remaining <= 0:
            remaining = 0.0
        if client:
            step = WAIT_TIMEOUT_MAX_SECONDS if remaining is None else min(remaining, WAIT_TIMEOUT_MAX_SECONDS)
            payload = client.get(
                f"/api/{kind}/{args.id}/wait?timeout={step:g}",
                timeout=step + 30,
            )
            record = payload[key]
        else:
            step = None if remaining is None else min(remaining, WAIT_TIMEOUT_MAX_SECONDS)
            record = (
                scheduler.wait_for_job(args.id, timeout=step)
                if args.job
                else scheduler.wait_for_plan(args.id, timeout=step)
            )
        status = str(record.get("status"))
        if status in terminal_states:
            _print({"ok": status == "completed", "terminal": True, key: record})
            return EXIT_OK if status == "completed" else EXIT_FAILED
        if deadline is not None and time.monotonic() >= deadline:
            _print({"ok": False, "terminal": False, "code": "timeout", key: record})
            return EXIT_TIMEOUT


def cmd_logs(args: argparse.Namespace) -> int:
    client = _client(args)
    if client:
        payload = client.get(f"/api/jobs/{args.job_id}/logs?lines={int(args.lines)}")
        logs = payload["logs"]
    else:
        cfg = load_config(args.config)
        logs = fetch_job_logs(cfg, StateStore(cfg.database), args.job_id, lines=int(args.lines))
        payload = {"ok": True, "logs": logs}
    if args.text:
        print(
            f"# job {logs.get('job_name')} [{logs.get('job_id')}] status={logs.get('status')} "
            f"node={logs.get('node') or '-'} exit={logs.get('exit_code')} source={logs.get('source')}"
        )
        print(f"# --- stdout (last {logs.get('lines')} lines) ---")
        print(logs.get("stdout", ""))
        print(f"# --- stderr (last {logs.get('lines')} lines) ---")
        print(logs.get("stderr", ""))
        if logs.get("result_raw"):
            print("# --- result.json ---")
            print(logs["result_raw"])
        return 0
    _print(payload)
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    """Call any route of a running controller; the manifest is GET /api/meta."""
    client = _client(args)
    path = args.path if args.path.startswith("/") else f"/{args.path}"
    if client is None:
        if path == "/api/meta":
            _print(build_meta())
            return 0
        raise ValueError("super-gpu api needs a running controller: pass --url or set SUPER_GPU_URL")
    payload = None
    if args.data:
        payload = json.loads(args.data)
        if not isinstance(payload, dict):
            raise ValueError("--data must be a JSON object")
    method = args.method or ("POST" if payload is not None else "GET")
    _print(client.request(method, path, payload))
    return 0


def cmd_import_gpumgr(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        raise ValueError(f"refusing to overwrite existing config: {output}; use --force")
    payload = import_gpumgr_inventory(
        args.source,
        role=args.role,
        workspace=args.workspace,
        database=args.database,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(output)
    _print(
        {
            "ok": True,
            "output": str(output),
            "nodes": len(payload["nodes"]),
            "role": args.role,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="super-gpu",
        description="Resource-aware multi-server GPU experiment scheduler",
    )
    parser.add_argument("--version", action="version", version=f"super-gpu {__version__}")
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
    run.add_argument("--request-id", default="", help="stable retry-safe submission ID")
    run.set_defaults(func=cmd_run)

    submit = sub.add_parser("submit", help="submit a plan locally or to a daemon")
    submit.add_argument("plan")
    submit.add_argument("--request-id", default="", help="stable retry-safe submission ID")
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

    pull = sub.add_parser("pull", help="fetch a finished job's declared outputs")
    pull.add_argument("job_id")
    pull.add_argument(
        "--dest",
        default="./outputs",
        help="local directory; files land in <dest>/<job-name>/",
    )
    pull.set_defaults(func=cmd_pull)

    events = sub.add_parser("events", help="show scheduler events")
    events.add_argument("--limit", type=int, default=100)
    _remote_args(events)
    events.set_defaults(func=cmd_events)

    brief = sub.add_parser(
        "brief",
        help="one-screen situation brief: health, attention items, queue, capacity, recommendations",
    )
    brief.add_argument("--hours", type=float, default=DEFAULT_WINDOW_SECONDS / 3600, help="change window")
    brief.add_argument("--json", action="store_true", help="print the JSON document instead of text")
    _remote_args(brief)
    brief.set_defaults(func=cmd_brief)

    capacity = sub.add_parser(
        "capacity",
        help="dry-run placement: how many jobs of this shape could start right now",
    )
    capacity.add_argument("--gpus", type=int, default=1)
    capacity.add_argument("--memory-mib", type=int, default=0, help="per-GPU budget; 0 = estimate")
    capacity.add_argument("--gpu-utilization", type=int, default=None)
    capacity.add_argument("--nodes", default="", help="comma-separated node names")
    capacity.add_argument("--labels", default="", help="comma-separated required labels")
    capacity.add_argument("--allow-colocation", choices=["true", "false"], default="")
    capacity.add_argument("--max-slots", type=int, default=0)
    _remote_args(capacity)
    capacity.set_defaults(func=cmd_capacity)

    preview = sub.add_parser(
        "preview",
        help="validate a plan and show per-job estimates and placement without submitting",
    )
    preview.add_argument("plan")
    _remote_args(preview)
    preview.set_defaults(func=cmd_preview)

    wait = sub.add_parser(
        "wait",
        help="block until a plan (or --job) is terminal; exit 0 completed, 1 failed, 3 timeout",
    )
    wait.add_argument("id")
    wait.add_argument("--job", action="store_true", help="the ID is a job, not a plan")
    wait.add_argument("--timeout", type=float, default=None, help="overall seconds; default: forever")
    _remote_args(wait)
    wait.set_defaults(func=cmd_wait)

    logs = sub.add_parser("logs", help="fetch a job's stdout/stderr tail from its node")
    logs.add_argument("job_id")
    logs.add_argument("--lines", type=int, default=LOG_LINES_DEFAULT)
    logs.add_argument("--text", action="store_true", help="print plain text instead of JSON")
    _remote_args(logs)
    logs.set_defaults(func=cmd_logs)

    api = sub.add_parser("api", help="call any controller route (see GET /api/meta)")
    api.add_argument("path", help="for example /api/meta or /api/jobs?status=failed")
    api.add_argument("--data", default="", help="JSON object body; implies POST")
    api.add_argument("--method", default="", help="override the HTTP method")
    _remote_args(api)
    api.set_defaults(func=cmd_api)

    import_gpumgr = sub.add_parser(
        "import-gpumgr",
        help="create a private super_gpu config from gpumgr's node inventory",
    )
    import_gpumgr.add_argument(
        "--source",
        default=os.environ.get(
            "GPUMGR_CONFIG",
            str(Path.home() / ".config" / "gpumgr" / "nodes.json"),
        ),
    )
    import_gpumgr.add_argument(
        "--output",
        default=str(Path.home() / ".config" / "super_gpu" / "config.json"),
    )
    import_gpumgr.add_argument("--role", choices=["shared", "dedicated"], default="shared")
    import_gpumgr.add_argument("--workspace", default="~/.super_gpu/work")
    import_gpumgr.add_argument("--database", default="~/.super_gpu/state.sqlite3")
    import_gpumgr.add_argument("--force", action="store_true")
    import_gpumgr.set_defaults(func=cmd_import_gpumgr)
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
    except SuperGPUError as exc:
        _print(exc.as_dict())
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        _print({"ok": False, "error": str(exc), "code": "cli_error"})
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
