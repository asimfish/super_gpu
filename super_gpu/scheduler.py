"""Continuous resource-aware experiment scheduler."""
from __future__ import annotations

import os
import socket
import threading
import time
import traceback
from typing import Any

from .estimator import ResourceEstimator
from .models import (
    ExperimentPlan,
    JobSpec,
    NodeSnapshot,
    RunnerHandle,
    SystemConfig,
    new_id,
)
from .monitor import ClusterMonitor
from .placement import PlacementEngine
from .runner import PersistentRunner
from .store import StateStore


class Scheduler:
    def __init__(
        self,
        config: SystemConfig,
        *,
        store: StateStore | None = None,
        monitor: ClusterMonitor | None = None,
        runner: PersistentRunner | None = None,
        estimator: ResourceEstimator | None = None,
        placement: PlacementEngine | None = None,
    ) -> None:
        self.config = config
        self.store = store or StateStore(config.database)
        self.monitor = monitor or ClusterMonitor(config)
        self.runner = runner or PersistentRunner(self.monitor.transport)
        self.estimator = estimator or ResourceEstimator(config, self.store)
        self.placement = placement or PlacementEngine(config, self.monitor)
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{new_id('controller')}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._last_tick = 0.0
        self._last_error = ""
        self._tick_count = 0

    def submit(self, plan: ExperimentPlan) -> dict[str, Any]:
        return self.store.submit_plan(plan)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.store.acquire_controller(self.owner, self.config.lease_ttl):
            raise RuntimeError("another super_gpu scheduler is already controlling this database")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="super-gpu-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, timeout))
        if not self._thread or not self._thread.is_alive():
            self.store.release_controller(self.owner)
        else:
            # A tick may still be blocked in SSH monitoring. Keep the lock
            # until its TTL expires rather than allowing a second controller
            # to overlap the unfinished placement cycle.
            self.store.add_event(
                "scheduler_stopping",
                "scheduler thread is still finishing; controller lock retained until TTL",
                {"owner": self.owner},
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                error = traceback.format_exc()
                with self._state_lock:
                    self._last_error = error
                self.store.add_event("scheduler_error", "scheduler tick failed", {"error": error[-4000:]})
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.05, self.config.poll_interval - elapsed))

    def run_once(self) -> None:
        if not self.store.acquire_controller(self.owner, self.config.lease_ttl):
            raise RuntimeError("scheduler controller lock was lost")
        snapshots = self.monitor.collect_all()
        self.store.save_snapshots(snapshots)
        self._reconcile_running(snapshots)
        for blocked in self.store.fail_blocked_dependencies():
            self.store.add_event(
                "job_blocked",
                f"job {blocked['name']} blocked by failed dependencies",
                blocked,
            )
        self._schedule_pending(snapshots)
        self.store.refresh_all_plans()
        with self._state_lock:
            self._last_tick = time.time()
            self._last_error = ""
            self._tick_count += 1

    def _reconcile_running(self, snapshots: list[NodeSnapshot]) -> None:
        snapshot_by_node = {snapshot.node: snapshot for snapshot in snapshots}
        active_jobs = self.store.list_jobs(
            statuses=["starting", "running", "cancelling"],
            limit=10000,
        )
        for row in active_jobs:
            job = self._spec_from_row(row)
            handle_data = row.get("handle") or {}
            if not handle_data:
                # Controller may have stopped between acquiring the lease and
                # persisting the remote handle. Safely retry instead of leaving
                # a permanent starting record.
                self.store.retry_or_fail(
                    row["id"],
                    exit_code=None,
                    error="controller restarted before runner handle was recorded",
                    stdout_tail=row.get("stdout_tail", ""),
                    stderr_tail=row.get("stderr_tail", ""),
                )
                continue
            handle = RunnerHandle.from_dict(handle_data)
            try:
                node = self.config.node(row["node"])
            except KeyError:
                self.store.retry_or_fail(
                    row["id"],
                    exit_code=None,
                    error=f"assigned node {row['node']!r} is no longer configured",
                    stdout_tail=row.get("stdout_tail", ""),
                    stderr_tail=row.get("stderr_tail", ""),
                )
                continue

            if row["cancel_requested"] or row["status"] == "cancelling":
                self.runner.cancel(node, handle)
                self.store.finish_job(
                    row["id"],
                    status="cancelled",
                    exit_code=143,
                    error="cancelled by user",
                    stdout_tail=row.get("stdout_tail", ""),
                    stderr_tail=row.get("stderr_tail", ""),
                )
                self.store.add_event(
                    "job_cancelled",
                    f"job {row['name']} cancelled",
                    {"job_id": row["id"]},
                )
                continue

            if time.time() - handle.started_at > float(row["timeout"]):
                self.runner.cancel(node, handle)
                self._finish_unsuccessful(
                    row,
                    job,
                    exit_code=124,
                    error=f"timeout after {row['timeout']:g}s",
                    stdout_tail=row.get("stdout_tail", ""),
                    stderr_tail=row.get("stderr_tail", ""),
                )
                continue

            status = self.runner.poll(node, handle)
            peak_memory = self._observed_peak(row, snapshot_by_node)
            self.store.update_runtime(
                row["id"],
                peak_memory_mib=peak_memory,
                stdout_tail=status.stdout_tail,
                stderr_tail=status.stderr_tail,
            )
            if status.state in {"running", "unknown"}:
                # A transient SSH failure must not release a live GPU lease.
                self.store.heartbeat_leases(row["id"], self.config.lease_ttl)
                continue

            duration = max(0.0, time.time() - handle.started_at)
            peak_memory = max(int(row.get("peak_memory_mib") or 0), peak_memory)
            if status.exit_code == 0:
                self.store.finish_job(
                    row["id"],
                    status="completed",
                    exit_code=0,
                    stdout_tail=status.stdout_tail,
                    stderr_tail=status.stderr_tail,
                )
                self.estimator.observe(
                    job,
                    peak_memory_mib=max(1, peak_memory),
                    duration_seconds=duration,
                )
                self.store.add_event(
                    "job_completed",
                    f"job {row['name']} completed on {row['node']}",
                    {
                        "job_id": row["id"],
                        "node": row["node"],
                        "gpus": row["gpus"],
                        "duration_seconds": duration,
                    },
                )
            else:
                error = status.error or status.stderr_tail or f"exit code {status.exit_code}"
                self._finish_unsuccessful(
                    row,
                    job,
                    exit_code=status.exit_code,
                    error=error,
                    stdout_tail=status.stdout_tail,
                    stderr_tail=status.stderr_tail,
                    duration=duration,
                    peak_memory=peak_memory,
                )

    def _finish_unsuccessful(
        self,
        row: dict[str, Any],
        job: JobSpec,
        *,
        exit_code: int | None,
        error: str,
        stdout_tail: str,
        stderr_tail: str,
        duration: float | None = None,
        peak_memory: int | None = None,
    ) -> None:
        combined = f"{error}\n{stdout_tail}\n{stderr_tail}".lower()
        oom = any(
            marker in combined
            for marker in (
                "cuda out of memory",
                "outofmemoryerror",
                "cuda oom",
                "hip out of memory",
            )
        )
        if peak_memory or oom:
            self.estimator.observe(
                job,
                peak_memory_mib=max(1, int(peak_memory or self.config.default_memory_mib)),
                duration_seconds=duration,
                oom=oom,
            )
        updated = self.store.retry_or_fail(
            row["id"],
            exit_code=exit_code,
            error=error,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        self.store.add_event(
            "job_retry" if updated.get("status") == "pending" else "job_failed",
            (
                f"job {row['name']} will retry"
                if updated.get("status") == "pending"
                else f"job {row['name']} failed"
            ),
            {"job_id": row["id"], "error": error[-1000:], "oom": oom},
        )

    def _schedule_pending(self, snapshots: list[NodeSnapshot]) -> None:
        global_slots = self.config.max_parallel - self.store.active_count()
        if global_slots <= 0:
            return
        leases = self.store.leases(active_only=True)
        for row in self.store.runnable_jobs():
            if global_slots <= 0:
                break
            plan = self.store.get_plan(row["plan_id"])
            plan_limit = plan.get("max_parallel")
            if plan_limit is not None and self.store.active_count(row["plan_id"]) >= int(plan_limit):
                continue
            job = self._spec_from_row(row)
            estimate = self.estimator.estimate(job)
            placement = self.placement.choose(job, estimate, snapshots, leases)
            if placement is None:
                continue
            if not self.store.acquire_placement(row["id"], placement, self.config.lease_ttl):
                continue
            job.id = row["id"]
            try:
                node = self.config.node(placement.node)
                env = self._job_env(row, placement.gpu_indices)
                handle = self.runner.launch(
                    node,
                    job,
                    placement.gpu_indices,
                    env=env,
                )
                self.store.mark_running(row["id"], handle.as_dict())
                self.store.add_event(
                    "job_started",
                    f"job {row['name']} started on {placement.node}",
                    {
                        "job_id": row["id"],
                        "plan_id": row["plan_id"],
                        "node": placement.node,
                        "gpus": placement.gpu_indices,
                        "estimate": estimate.as_dict(),
                        "score": placement.score,
                    },
                )
                leases.extend(
                    {
                        "job_id": row["id"],
                        "node": placement.node,
                        "gpu_index": gpu_index,
                        "memory_mib": placement.memory_mib_per_gpu,
                    }
                    for gpu_index in placement.gpu_indices
                )
                global_slots -= 1
            except Exception as exc:  # noqa: BLE001
                self.store.retry_or_fail(
                    row["id"],
                    exit_code=None,
                    error=str(exc),
                    stdout_tail="",
                    stderr_tail=traceback.format_exc(),
                )
                self.store.add_event(
                    "job_launch_failed",
                    f"job {row['name']} could not launch",
                    {"job_id": row["id"], "error": str(exc)},
                )

    def _observed_peak(
        self,
        row: dict[str, Any],
        snapshot_by_node: dict[str, NodeSnapshot],
    ) -> int:
        snapshot = snapshot_by_node.get(row.get("node", ""))
        if snapshot is None or not snapshot.reachable:
            return int(row.get("peak_memory_mib") or 0)
        indices = {int(index) for index in row.get("gpus", [])}
        values = [
            gpu.memory_used_mib
            for gpu in snapshot.gpus
            if gpu.index in indices
        ]
        return max(values, default=int(row.get("peak_memory_mib") or 0))

    @staticmethod
    def _spec_from_row(row: dict[str, Any]) -> JobSpec:
        job = JobSpec.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "command": row["command"],
                "priority": row["priority"],
                "nodes": row["nodes"],
                "required_labels": row["required_labels"],
                "params": row["params"],
                "env": row["env"],
                "resources": row["resources"],
                "timeout": row["timeout"],
                "max_retries": row["max_retries"],
                "retry_delay": row["retry_delay"],
                "dependencies": row["dependencies"],
            }
        )
        job.id = row["id"]
        return job

    @staticmethod
    def _job_env(row: dict[str, Any], gpu_indices: list[int]) -> dict[str, str]:
        env = {str(k): str(v) for k, v in row.get("env", {}).items()}
        env.update(
            {
                "SUPER_GPU_PLAN_ID": row["plan_id"],
                "SUPER_GPU_JOB_ID": row["id"],
                "SUPER_GPU_JOB_NAME": row["name"],
                "SUPER_GPU_GPU_INDICES": ",".join(str(index) for index in gpu_indices),
                "PYTHONUNBUFFERED": "1",
            }
        )
        for key, value in row.get("params", {}).items():
            normalized = "".join(
                character if character.isalnum() else "_"
                for character in str(key).upper()
            )
            env[f"SUPER_GPU_PARAM_{normalized}"] = str(value)
        return env

    def wait_for_plan(self, plan_id: str, timeout: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            plan = self.store.get_plan(plan_id, include_jobs=True)
            if not plan:
                raise KeyError(f"unknown plan {plan_id}")
            if plan["status"] in {"completed", "failed", "cancelled"}:
                return plan
            if deadline is not None and time.monotonic() >= deadline:
                return plan
            time.sleep(min(1.0, self.config.poll_interval))

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "owner": self.owner,
                "last_tick": self._last_tick,
                "last_error": self._last_error,
                "tick_count": self._tick_count,
                "poll_interval": self.config.poll_interval,
                "active_jobs": self.store.active_count(),
                "pending_jobs": len(self.store.list_jobs(statuses=["pending"], limit=10000)),
            }
