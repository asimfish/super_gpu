"""Situation brief: the supervising agent's status check as one document.

``GET /api/brief`` and ``super-gpu brief`` answer, in reading order, whether
the controller is alive, what needs a decision now (failures, blocked jobs,
offline nodes, idle-yet-occupied GPUs), what the queue looks like, how much
capacity could start work this instant, what changed over the window, and
what to do about it. Every number comes from projections the dashboard and
the other endpoints already serve, so the brief cannot disagree with them;
what it adds is the ordering, the grouping of pending reasons into causes,
and the recommendations an operator would otherwise derive by eye.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from . import __version__
from .capacity import CapacityRequest, dry_run_capacity, snapshots_from_store
from .models import ACTIVE_JOB_STATES, TERMINAL_JOB_STATES

DEFAULT_WINDOW_SECONDS = 6 * 3600
WINDOW_MAX_SECONDS = 30 * 24 * 3600
ITEM_LIMIT = 10
IDLE_UTILIZATION_PCT = 5
IDLE_MEMORY_RATIO = 0.10
OOM_MARKERS = ("cuda out of memory", "outofmemoryerror", "cuda oom", "hip out of memory")

CHANGE_KINDS = (
    "job_started",
    "job_completed",
    "job_failed",
    "job_retry",
    "job_skipped",
    "job_cancelled",
    "job_launch_failed",
    "plan_completed",
    "plan_failed",
    "plan_cancelled",
    "node_offline",
    "node_online",
    "watchdog_anomaly_detected",
    "watchdog_cancel_requested",
    "notification_failed",
    "scheduler_error",
)


def _iso(timestamp: float) -> str:
    if not timestamp:
        return ""
    moment = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}{'' if value == 1 else 's'}"


def _first_line(text: str, limit: int = 160) -> str:
    for line in (text or "").strip().splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def _looks_like_oom(*texts: str) -> bool:
    combined = "\n".join(text or "" for text in texts).lower()
    return any(marker in combined for marker in OOM_MARKERS)


def pending_cause(reason: str) -> str:
    """Collapse a free-text ``pending_reason`` into one of a few causes."""
    text = (reason or "").strip()
    if not text:
        return "not_evaluated_yet"
    if text.startswith("cluster max_parallel"):
        return "cluster_max_parallel"
    if text.startswith("plan max_parallel"):
        return "plan_max_parallel"
    if text.startswith("waiting for dependencies"):
        return "dependencies"
    if text.startswith("in retry backoff"):
        return "retry_backoff"
    if "no enabled node matches" in text:
        return "no_matching_node"
    if "unreachable" in text or "no telemetry" in text:
        return "nodes_unreachable"
    return "no_capacity"


def build_brief(scheduler: Any, *, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> dict[str, Any]:
    now = time.time()
    window_seconds = max(60.0, min(float(window_seconds), WINDOW_MAX_SECONDS))
    since = now - window_seconds
    store = scheduler.store
    config = scheduler.config

    status = scheduler.status()
    snapshot_rows = store.latest_snapshots()
    snapshots = snapshots_from_store(snapshot_rows)
    leases = store.leases(active_only=True)
    active_jobs = store.list_jobs(statuses=sorted(ACTIVE_JOB_STATES), limit=10000)
    pending_jobs = store.list_jobs(statuses=["pending"], limit=10000)
    finished_jobs = store.list_jobs(
        statuses=sorted(TERMINAL_JOB_STATES), limit=10000, finished_since=since
    )
    plans = store.list_plans(limit=200)
    events = store.list_events(limit=5000, since=since, kinds=CHANGE_KINDS)
    findings = list(scheduler.guardian.findings())

    controller = _controller(
        status,
        config,
        now,
        lock=store.controller_lock(),
        latest_sample=max((float(s.collected_at or 0.0) for s in snapshots), default=0.0),
    )
    fleet = _fleet(config, snapshots, leases, now)
    queue = _queue(store, plans, active_jobs, pending_jobs, finished_jobs, now, since)
    capacity = _capacity(scheduler, config, snapshots, leases, status)
    attention = _attention(
        controller, fleet, capacity, pending_jobs, active_jobs, finished_jobs, findings, events, now
    )
    changes = _changes(events, plans, since)
    recommendations = _recommendations(
        controller, fleet, capacity, queue, attention, config
    )
    brief_status = _status(controller, fleet, attention, queue)
    return {
        "ok": True,
        "generated_at": _iso(now),
        "since": _iso(since),
        "window_seconds": window_seconds,
        "status": brief_status,
        "controller": controller,
        "fleet": fleet,
        "queue": queue,
        "attention": attention,
        "capacity": capacity,
        "changes": changes,
        "recommendations": recommendations,
    }


def _controller(
    status: dict[str, Any],
    config: Any,
    now: float,
    *,
    lock: dict[str, Any] | None = None,
    latest_sample: float = 0.0,
) -> dict[str, Any]:
    """Describe the controller that owns the database.

    When the brief is built inside the serving process, the scheduler's own
    status is authoritative. When it is built by a CLI process against the
    same database, the scheduler object never ran; the live controller lock
    says whether another process is in charge, and the freshest telemetry
    sample stands in for its last tick, because every tick saves snapshots.
    """
    poll = float(config.poll_interval)
    # A tick can legitimately take a few poll intervals when SSH is slow;
    # beyond the lease TTL the controller can no longer be trusted to hold
    # its leases either.
    stale_after = max(3 * poll, float(config.lease_ttl))
    running = bool(status.get("running"))
    external = False
    owner = status.get("owner")
    tick_count = status.get("tick_count")
    last_error = _first_line(str(status.get("last_error") or ""), 300)
    if running:
        last_tick = float(status.get("last_tick") or 0.0)
    elif lock is not None and lock.get("owner") != status.get("owner"):
        external = True
        running = True
        owner = lock.get("owner")
        tick_count = None
        last_error = ""
        last_tick = float(latest_sample or 0.0)
    else:
        last_tick = float(status.get("last_tick") or 0.0)
    tick_age = (now - last_tick) if last_tick else None
    if not running:
        state = "stopped"
    elif tick_age is None:
        state = "starting"
    elif tick_age > stale_after:
        state = "stale"
    elif last_error:
        state = "erroring"
    else:
        state = "running"
    return {
        "state": state,
        "running": running,
        "external": external,
        "version": __version__,
        "owner": owner,
        "tick_count": tick_count,
        "last_tick": _iso(last_tick),
        "tick_age_seconds": round(tick_age, 1) if tick_age is not None else None,
        "stale_after_seconds": stale_after,
        "poll_interval": poll,
        "last_error": last_error,
        "watchdog_action": (status.get("watchdog") or {}).get("policy", {}).get("action"),
    }


def _fleet(config: Any, snapshots: list[Any], leases: list[dict[str, Any]], now: float) -> dict[str, Any]:
    leased: set[tuple[str, int]] = {
        (str(lease["node"]), int(lease["gpu_index"])) for lease in leases
    }
    per_node = []
    totals = Counter()
    offline: list[str] = []
    observed: list[float] = []
    for snapshot in snapshots:
        try:
            node_cfg = config.node(snapshot.node)
            role = node_cfg.role
            enabled = node_cfg.enabled
        except KeyError:
            role, enabled = snapshot.role, True
        observed.append(float(snapshot.collected_at or 0.0))
        entry: dict[str, Any] = {
            "node": snapshot.node,
            "role": role,
            "enabled": enabled,
            "reachable": snapshot.reachable,
            "error": _first_line(snapshot.error, 120),
            "gpus": len(snapshot.gpus),
            "leased": 0,
            "idle": 0,
            "busy_unmanaged": 0,
            "avg_utilization": None,
            "free_memory_mib": 0,
            "observed_age_seconds": round(max(0.0, now - float(snapshot.collected_at or now)), 1),
        }
        if not snapshot.reachable:
            offline.append(snapshot.node)
        utilizations = []
        for gpu in snapshot.gpus:
            totals["gpus"] += 1
            utilizations.append(int(gpu.utilization))
            entry["free_memory_mib"] += max(0, int(gpu.memory_free_mib))
            if (snapshot.node, int(gpu.index)) in leased:
                entry["leased"] += 1
                totals["leased"] += 1
            elif (
                int(gpu.utilization) < IDLE_UTILIZATION_PCT
                and gpu.memory_used_ratio < IDLE_MEMORY_RATIO
            ):
                entry["idle"] += 1
                totals["idle"] += 1
            else:
                entry["busy_unmanaged"] += 1
                totals["busy_unmanaged"] += 1
        if utilizations:
            entry["avg_utilization"] = round(sum(utilizations) / len(utilizations), 1)
        per_node.append(entry)
    configured = [node.name for node in config.nodes if node.enabled]
    unsampled = sorted(set(configured) - {snapshot.node for snapshot in snapshots})
    return {
        "nodes": len(configured),
        "online": sum(1 for snapshot in snapshots if snapshot.reachable),
        "offline": sorted(offline),
        "unsampled": unsampled,
        "gpus": totals["gpus"],
        "gpus_leased": totals["leased"],
        "gpus_idle": totals["idle"],
        "gpus_busy_unmanaged": totals["busy_unmanaged"],
        "managed_leases": len(leases),
        "observed_at": _iso(max(observed, default=0.0)),
        "per_node": per_node,
    }


def _plan_counts(store: Any, plan: dict[str, Any]) -> dict[str, int]:
    jobs = store.list_jobs(plan_id=plan["id"], limit=10000)
    return dict(Counter(str(job["status"]) for job in jobs))


def _queue(
    store: Any,
    plans: list[dict[str, Any]],
    active_jobs: list[dict[str, Any]],
    pending_jobs: list[dict[str, Any]],
    finished_jobs: list[dict[str, Any]],
    now: float,
    since: float,
) -> dict[str, Any]:
    active_plans = []
    for plan in plans:
        if plan.get("status") in {"queued", "running"}:
            counts = _plan_counts(store, plan)
            active_plans.append(
                {
                    "id": plan["id"],
                    "name": plan["name"],
                    "status": plan["status"],
                    "total_jobs": plan.get("total_jobs"),
                    "job_counts": counts,
                    "created_at": _iso(float(plan.get("created_at") or 0.0)),
                    "age_seconds": round(max(0.0, now - float(plan.get("created_at") or now)), 1),
                }
            )
    oldest_pending = None
    for job in pending_jobs:
        submitted = float(job.get("submitted_at") or now)
        age = now - submitted
        if oldest_pending is None or age > oldest_pending:
            oldest_pending = age
    return {
        "jobs": {
            "pending": len(pending_jobs),
            "active": len(active_jobs),
            **dict(Counter(str(job["status"]) for job in active_jobs)),
            "finished_in_window": dict(Counter(str(job["status"]) for job in finished_jobs)),
        },
        "oldest_pending_seconds": round(oldest_pending, 1) if oldest_pending is not None else None,
        "plans_active": active_plans,
        "plans_total": len(plans),
    }


def _capacity(
    scheduler: Any,
    config: Any,
    snapshots: list[Any],
    leases: list[dict[str, Any]],
    status: dict[str, Any],
) -> dict[str, Any]:
    probe = dry_run_capacity(
        config,
        scheduler.placement,
        scheduler.estimator,
        snapshots,
        leases,
        CapacityRequest(gpus=1),
        active_jobs=int(status.get("active_jobs") or 0),
    )
    per_node = [
        {
            "node": node,
            "role": entry["role"],
            "slots": entry["slots"],
            "gpu_indices": entry["gpu_indices"],
            "blocked_by": entry["blocked_by"] if entry["slots"] == 0 else "",
        }
        for node, entry in sorted(probe["per_node"].items())
    ]
    return {
        "probe": "one job of 1 GPU with the default memory estimate",
        "estimate_memory_mib": probe["estimate"]["memory_mib"],
        "estimate_source": probe["estimate"]["source"],
        "slots": probe["slots"],
        "cluster_slots_remaining": probe["cluster_slots_remaining"],
        "effective_slots": probe["effective_slots"],
        "per_node": per_node,
        "note": probe["note"],
    }


def _attention(
    controller: dict[str, Any],
    fleet: dict[str, Any],
    capacity: dict[str, Any],
    pending_jobs: list[dict[str, Any]],
    active_jobs: list[dict[str, Any]],
    finished_jobs: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    events: list[dict[str, Any]],
    now: float,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    if controller["state"] in {"stopped", "stale", "erroring"}:
        detail = {
            "stopped": "scheduler thread is not running; nothing will be placed or supervised",
            "stale": (
                f"last tick {_age(controller['tick_age_seconds'])} ago exceeds "
                f"{controller['stale_after_seconds']:g}s; leases may expire"
            ),
            "erroring": f"last tick failed: {controller['last_error']}",
        }[controller["state"]]
        items.append(
            {
                "severity": "critical",
                "kind": f"controller_{controller['state']}",
                "message": detail,
                "hint": "check the serve process (`supergpu status`, `super-gpu events`)",
                "ref": {},
            }
        )

    all_offline = fleet["nodes"] > 0 and fleet["online"] == 0
    for node in fleet["per_node"]:
        if node["reachable"]:
            continue
        items.append(
            {
                "severity": "critical" if all_offline else "warning",
                "kind": "node_offline",
                "message": f"node {node['node']} unreachable"
                + (f": {node['error']}" if node["error"] else ""),
                "hint": f"check SSH: `ssh {node['node']} nvidia-smi`",
                "ref": {"node": node["node"]},
            }
        )
    for node in fleet["unsampled"]:
        items.append(
            {
                "severity": "info",
                "kind": "node_unsampled",
                "message": f"node {node} has no telemetry sample yet",
                "hint": "wait for the next tick or run `super-gpu scan`",
                "ref": {"node": node},
            }
        )

    failed = [job for job in finished_jobs if job.get("status") == "failed"]
    failed.sort(key=lambda job: float(job.get("finished_at") or 0.0), reverse=True)
    for job in failed[:ITEM_LIMIT]:
        oom = _looks_like_oom(job.get("error", ""), job.get("stderr_tail", ""))
        peak = int(job.get("peak_memory_mib") or 0)
        reason = _first_line(job.get("error") or job.get("stderr_tail") or "", 140)
        hint = f"super-gpu logs {job['id']}"
        if oom:
            suggested = int(((peak * 1.25) // 1024 + 1) * 1024) if peak else None
            hint = (
                f"CUDA OOM: declare resources.memory_mib >= {suggested} for the retry"
                if suggested
                else "CUDA OOM: declare a larger resources.memory_mib for the retry"
            )
        items.append(
            {
                "severity": "warning",
                "kind": "job_failed",
                "message": (
                    f"job {job['name']} failed on {job.get('node') or '?'} "
                    f"(exit {job.get('exit_code')}, attempt {job.get('attempt')})"
                    + (f": {reason}" if reason else "")
                ),
                "hint": hint,
                "ref": {"job_id": job["id"], "plan_id": job.get("plan_id"), "oom": oom},
            }
        )
    if len(failed) > ITEM_LIMIT:
        items.append(
            {
                "severity": "warning",
                "kind": "job_failed_more",
                "message": f"{len(failed) - ITEM_LIMIT} more failed job(s) in the window",
                "hint": "super-gpu jobs --status failed",
                "ref": {},
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for job in pending_jobs:
        groups.setdefault(pending_cause(job.get("pending_reason", "")), []).append(job)
    free_slots = int(capacity.get("slots") or 0)
    for cause, jobs in sorted(groups.items(), key=lambda item: -len(item[1])):
        example = jobs[0]
        oldest = max(now - float(job.get("submitted_at") or now) for job in jobs)
        severity = "info"
        hint = ""
        if cause == "cluster_max_parallel":
            severity = "warning" if free_slots > 0 else "info"
            hint = (
                "raise max_parallel in config.json; GPUs are free"
                if free_slots > 0
                else "no action: the cluster limit matches its capacity"
            )
        elif cause == "plan_max_parallel":
            hint = "no action unless the plan should run wider"
        elif cause == "dependencies":
            hint = "no action: upstream jobs are still running"
        elif cause == "retry_backoff":
            severity = "warning"
            hint = "a previous attempt failed; inspect with `super-gpu logs <job-id>`"
        elif cause == "no_matching_node":
            severity = "warning"
            hint = "fix the job's nodes/required_labels or enable a matching node"
        elif cause == "nodes_unreachable":
            severity = "warning"
            hint = "restore SSH to the constrained nodes"
        elif cause == "no_capacity":
            severity = "warning" if free_slots > 0 else "info"
            hint = (
                "free GPU slots exist but do not fit these jobs: compare their gpus/memory_mib "
                "with `super-gpu capacity`"
                if free_slots > 0
                else "no action: waiting for GPUs to free up"
            )
        elif cause == "not_evaluated_yet":
            hint = "the next tick will evaluate them"
        items.append(
            {
                "severity": severity,
                "kind": f"pending_{cause}",
                "message": (
                    f"{_count(len(jobs), 'pending job')} blocked by {cause.replace('_', ' ')}"
                    f" (e.g. {example['name']}: {_first_line(example.get('pending_reason', ''), 120) or 'n/a'};"
                    f" oldest {_age(oldest)})"
                ),
                "hint": hint,
                "ref": {"job_ids": [job["id"] for job in jobs[:ITEM_LIMIT]], "count": len(jobs)},
            }
        )

    retrying = [job for job in active_jobs if int(job.get("attempt") or 0) > 1]
    for job in retrying[:ITEM_LIMIT]:
        items.append(
            {
                "severity": "info",
                "kind": "job_retrying",
                "message": f"job {job['name']} is on attempt {job.get('attempt')} on {job.get('node')}",
                "hint": f"super-gpu logs {job['id']}",
                "ref": {"job_id": job["id"], "plan_id": job.get("plan_id")},
            }
        )

    active_findings = [finding for finding in findings if finding.get("status") == "active"]
    for finding in active_findings[:ITEM_LIMIT]:
        processes = finding.get("processes") or []
        owner = ", ".join(
            sorted({str(proc.get("user") or "?") for proc in processes if isinstance(proc, dict)})
        )
        managed_ids = list(finding.get("managed_job_ids") or [])
        items.append(
            {
                "severity": "warning",
                "kind": "watchdog_anomaly",
                "message": (
                    f"GPU {finding.get('node')}:{finding.get('gpu_index')} holds "
                    f"{finding.get('memory_used_mib', '?')} MiB at ~{finding.get('utilization', '?')}% "
                    f"utilization for {_age(finding.get('idle_seconds'))}"
                    + (f" (user {owner})" if owner else "")
                    + (
                        f"; managed job(s) {', '.join(managed_ids)}"
                        if managed_ids
                        else "; unmanaged"
                    )
                ),
                "hint": (
                    "managed job: inspect its logs and cancel it if it is stuck"
                    if managed_ids
                    else "report-only: super_gpu never signals other users' processes"
                ),
                "ref": {
                    "node": finding.get("node"),
                    "gpu_index": finding.get("gpu_index"),
                    "managed_job_ids": managed_ids,
                },
            }
        )

    failed_notifications = [event for event in events if event.get("kind") == "notification_failed"]
    if failed_notifications:
        latest = failed_notifications[0]
        items.append(
            {
                "severity": "warning",
                "kind": "notification_failed",
                "message": (
                    f"{_count(len(failed_notifications), 'webhook delivery')} failed in the window; "
                    f"latest: {_first_line(latest.get('message', ''), 120)}"
                ),
                "hint": "check the webhook URL/secret and network from the controller host",
                "ref": {},
            }
        )
    scheduler_errors = [event for event in events if event.get("kind") == "scheduler_error"]
    if scheduler_errors and controller["state"] not in {"erroring"}:
        latest = scheduler_errors[0]
        items.append(
            {
                "severity": "warning",
                "kind": "scheduler_error_recent",
                "message": (
                    f"{_count(len(scheduler_errors), 'scheduler tick')} failed in the window "
                    f"(latest {_iso(float(latest.get('created_at') or 0.0))})"
                ),
                "hint": "super-gpu events --limit 50",
                "ref": {},
            }
        )

    order = {"critical": 0, "warning": 1, "info": 2}
    items.sort(key=lambda item: order.get(item["severity"], 3))
    return {
        "critical": sum(1 for item in items if item["severity"] == "critical"),
        "warning": sum(1 for item in items if item["severity"] == "warning"),
        "info": sum(1 for item in items if item["severity"] == "info"),
        "items": items,
    }


def _changes(events: list[dict[str, Any]], plans: list[dict[str, Any]], since: float) -> dict[str, Any]:
    counts = Counter(str(event.get("kind")) for event in events)
    finished_plans = [
        {
            "id": plan["id"],
            "name": plan["name"],
            "status": plan["status"],
            "total_jobs": plan.get("total_jobs"),
            "finished_at": _iso(float(plan.get("finished_at") or 0.0)),
        }
        for plan in plans
        if plan.get("finished_at") and float(plan["finished_at"]) >= since
    ]
    finished_plans.sort(key=lambda plan: plan["finished_at"], reverse=True)
    return {
        "events": dict(sorted(counts.items())),
        "plans_finished": finished_plans[:ITEM_LIMIT],
        "plans_finished_total": len(finished_plans),
    }


def _recommendations(
    controller: dict[str, Any],
    fleet: dict[str, Any],
    capacity: dict[str, Any],
    queue: dict[str, Any],
    attention: dict[str, Any],
    config: Any,
) -> list[str]:
    tips: list[str] = []
    if controller["state"] in {"stopped", "stale", "erroring"}:
        tips.append(
            f"controller is {controller['state']}: restart `super-gpu serve` before submitting more work"
        )
    pending = int(queue["jobs"].get("pending") or 0)
    active = int(queue["jobs"].get("active") or 0)
    slots = int(capacity.get("slots") or 0)
    effective = int(capacity.get("effective_slots") or 0)
    if effective > 0 and pending == 0 and active == 0:
        tips.append(
            f"cluster is idle: {_count(effective, 'GPU slot')} can start 1-GPU jobs now "
            f"({capacity['estimate_memory_mib']} MiB each) - submit a plan"
        )
    elif effective > 0 and pending == 0:
        tips.append(
            f"{_count(effective, 'GPU slot')} free while nothing is pending - "
            "submit more work to keep GPUs busy"
        )
    kinds = {item["kind"] for item in attention["items"]}
    if "pending_cluster_max_parallel" in kinds and slots > 0:
        tips.append(
            f"cluster max_parallel ({config.max_parallel}) is the bottleneck while "
            f"{_count(slots, 'GPU slot')} are free: raise max_parallel in config.json"
        )
    if "pending_no_capacity" in kinds and slots > 0:
        tips.append(
            "pending jobs do not fit the free GPUs: check their gpus/memory_mib against "
            "`super-gpu capacity` and lower the budget or split the job"
        )
    for item in attention["items"]:
        if item["kind"] == "job_failed" and item["ref"].get("oom"):
            tips.append(f"{item['message'].split(' failed')[0]}: {item['hint']}")
    if fleet["offline"]:
        tips.append(
            f"{_count(len(fleet['offline']), 'node')} offline ({', '.join(fleet['offline'])}): "
            "restore SSH or disable them in config.json so placement stops considering them"
        )
    if "watchdog_anomaly" in kinds:
        tips.append(
            "idle-yet-occupied GPUs detected: contact the owners, or enable "
            "SUPER_GPU_WATCHDOG_ACTION=cancel_managed only for jobs super_gpu launched"
        )
    return tips[:ITEM_LIMIT]


def _status(
    controller: dict[str, Any],
    fleet: dict[str, Any],
    attention: dict[str, Any],
    queue: dict[str, Any],
) -> str:
    if controller["state"] in {"stopped", "stale", "erroring"}:
        return "critical"
    if attention["critical"] > 0:
        return "critical"
    if attention["warning"] > 0:
        return "attention"
    if controller["state"] == "starting":
        return "starting"
    if int(queue["jobs"].get("pending") or 0) == 0 and int(queue["jobs"].get("active") or 0) == 0:
        return "idle"
    return "healthy"


def render_brief(brief: dict[str, Any]) -> str:
    """The same document as plain text: one screen, worst first."""
    controller = brief["controller"]
    fleet = brief["fleet"]
    queue = brief["queue"]
    attention = brief["attention"]
    capacity = brief["capacity"]
    changes = brief["changes"]
    window_hours = float(brief["window_seconds"]) / 3600
    lines = [
        f"super_gpu brief {brief['generated_at']} | window {window_hours:g}h | status {brief['status'].upper()}",
        (
            f"controller: {controller['state']}"
            + (" (separate process)" if controller.get("external") else "")
            + (
                f" | tick {_age(controller['tick_age_seconds'])} ago"
                + (f" (#{controller['tick_count']})" if controller["tick_count"] is not None else "")
                if controller["tick_age_seconds"] is not None
                else ""
            )
            + f" | poll {controller['poll_interval']:g}s | v{controller['version']}"
            + (f" | error: {controller['last_error']}" if controller["last_error"] else "")
        ),
        (
            f"fleet: {fleet['online']}/{fleet['nodes']} nodes online"
            + (f" (offline: {', '.join(fleet['offline'])})" if fleet["offline"] else "")
            + f" | {fleet['gpus']} GPUs: {fleet['gpus_leased']} leased, {fleet['gpus_idle']} idle, "
            f"{fleet['gpus_busy_unmanaged']} busy (unmanaged)"
        ),
    ]
    finished = queue["jobs"].get("finished_in_window") or {}
    finished_text = ", ".join(f"{count} {status}" for status, count in sorted(finished.items()))
    lines.append(
        f"queue: {queue['jobs'].get('active', 0)} active, {queue['jobs'].get('pending', 0)} pending"
        + (f" | window: {finished_text}" if finished_text else "")
        + f" | plans active: {len(queue['plans_active'])}"
    )
    for plan in queue["plans_active"][:ITEM_LIMIT]:
        counts = ", ".join(f"{count} {status}" for status, count in sorted(plan["job_counts"].items()))
        lines.append(f"  {plan['name']} [{plan['id']}]: {plan['status']} | {counts}")

    lines.append("")
    total_items = len(attention["items"])
    lines.append(
        f"attention ({total_items}): {attention['critical']} critical, "
        f"{attention['warning']} warning, {attention['info']} info"
    )
    marker = {"critical": "!!", "warning": " !", "info": " -"}
    for item in attention["items"][: ITEM_LIMIT * 2]:
        lines.append(f"  {marker.get(item['severity'], '  ')} {item['message']}")
        if item.get("hint"):
            lines.append(f"       -> {item['hint']}")
    hidden = total_items - min(total_items, ITEM_LIMIT * 2)
    if hidden > 0:
        lines.append(f"  ... {hidden} more in GET /api/brief")

    lines.append("")
    lines.append(
        f"capacity now: {_count(capacity['slots'], 'slot')} for 1-GPU jobs "
        f"({capacity['estimate_memory_mib']} MiB {capacity['estimate_source']})"
        + f" | cluster max_parallel leaves {capacity['cluster_slots_remaining']}"
        + f" | effective {capacity['effective_slots']}"
    )
    for node in capacity["per_node"]:
        if node["slots"]:
            indices = ",".join(str(index) for index in node["gpu_indices"])
            lines.append(f"  {node['node']} ({node['role']}): {_count(node['slots'], 'slot')} on gpu {indices}")
        else:
            lines.append(
                f"  {node['node']} ({node['role']}): 0 slots"
                + (f" - {node['blocked_by'][:100]}" if node["blocked_by"] else "")
            )

    lines.append("")
    event_text = ", ".join(f"{count} {kind}" for kind, count in changes["events"].items())
    lines.append(f"changes ({window_hours:g}h): {event_text or 'none'}")
    for plan in changes["plans_finished"]:
        lines.append(f"  plan {plan['name']} {plan['status']} ({plan['total_jobs']} jobs) at {plan['finished_at']}")

    if brief["recommendations"]:
        lines.append("")
        lines.append("recommendations:")
        for tip in brief["recommendations"]:
            lines.append(f"  - {tip}")
    return "\n".join(lines) + "\n"
