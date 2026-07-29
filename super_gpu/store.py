"""SQLite-backed desired state, leases, telemetry, and resource history."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .models import (
    ExperimentPlan,
    JobSpec,
    NodeSnapshot,
    Placement,
    new_id,
    now_ts,
)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS plans (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  max_parallel INTEGER,
  total_jobs INTEGER NOT NULL,
  plan_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  finished_at REAL
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  plan_id TEXT NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  command TEXT NOT NULL,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL,
  nodes_json TEXT NOT NULL,
  labels_json TEXT NOT NULL,
  params_json TEXT NOT NULL,
  env_json TEXT NOT NULL,
  resources_json TEXT NOT NULL,
  dependencies_json TEXT NOT NULL,
  timeout REAL NOT NULL,
  max_retries INTEGER NOT NULL,
  retry_delay REAL NOT NULL,
  fingerprint TEXT NOT NULL,
  estimate_json TEXT NOT NULL DEFAULT '{}',
  attempt INTEGER NOT NULL DEFAULT 0,
  node TEXT NOT NULL DEFAULT '',
  gpus_json TEXT NOT NULL DEFAULT '[]',
  handle_json TEXT NOT NULL DEFAULT '{}',
  peak_memory_mib INTEGER NOT NULL DEFAULT 0,
  submitted_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  next_run_at REAL NOT NULL DEFAULT 0,
  exit_code INTEGER,
  error TEXT NOT NULL DEFAULT '',
  stdout_tail TEXT NOT NULL DEFAULT '',
  stderr_tail TEXT NOT NULL DEFAULT '',
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  UNIQUE(plan_id, name)
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, priority, submitted_at);
CREATE INDEX IF NOT EXISTS jobs_plan_idx ON jobs(plan_id, submitted_at);

CREATE TABLE IF NOT EXISTS leases (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  node TEXT NOT NULL,
  gpu_index INTEGER NOT NULL,
  memory_mib INTEGER NOT NULL,
  created_at REAL NOT NULL,
  heartbeat_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  UNIQUE(job_id, node, gpu_index)
);

CREATE INDEX IF NOT EXISTS leases_gpu_idx ON leases(node, gpu_index, expires_at);

CREATE TABLE IF NOT EXISTS profiles (
  fingerprint TEXT PRIMARY KEY,
  samples INTEGER NOT NULL,
  peak_memory_mib INTEGER NOT NULL,
  avg_duration_seconds REAL,
  oom_count INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
  node TEXT PRIMARY KEY,
  payload_json TEXT NOT NULL,
  collected_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  message TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS events_created_idx ON events(created_at DESC);

CREATE TABLE IF NOT EXISTS controller_lock (
  name TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  expires_at REAL NOT NULL
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class StateStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._init_lock:
            with self.connect() as conn:
                conn.executescript(SCHEMA)

    @staticmethod
    def _job_row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        item = dict(row)
        for source, target, fallback in (
            ("nodes_json", "nodes", []),
            ("labels_json", "required_labels", []),
            ("params_json", "params", {}),
            ("env_json", "env", {}),
            ("resources_json", "resources", {}),
            ("dependencies_json", "dependencies", []),
            ("estimate_json", "estimate", {}),
            ("gpus_json", "gpus", []),
            ("handle_json", "handle", {}),
        ):
            item[target] = _loads(item.pop(source), fallback)
        item["cancel_requested"] = bool(item["cancel_requested"])
        return item

    @staticmethod
    def _plan_row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        item = dict(row)
        item["plan"] = _loads(item.pop("plan_json"), {})
        return item

    def submit_plan(self, plan: ExperimentPlan) -> dict[str, Any]:
        plan_id = plan.id or new_id("plan")
        created = now_ts()
        plan_payload = {
            "id": plan_id,
            "name": plan.name,
            "max_parallel": plan.max_parallel,
            "defaults": plan.defaults,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "command": job.command,
                    "priority": job.priority,
                    "nodes": job.nodes,
                    "required_labels": job.required_labels,
                    "params": job.params,
                    "env": job.env,
                    "resources": {
                        "gpus": job.resources.gpus,
                        "memory_mib": job.resources.memory_mib,
                        "gpu_utilization": job.resources.gpu_utilization,
                        "duration_seconds": job.resources.duration_seconds,
                        "allow_colocation": job.resources.allow_colocation,
                    },
                    "timeout": job.timeout,
                    "max_retries": job.max_retries,
                    "retry_delay": job.retry_delay,
                    "dependencies": job.dependencies,
                }
                for job in plan.jobs
            ],
        }
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO plans
                      (id, name, status, max_parallel, total_jobs, plan_json, created_at, updated_at)
                    VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        plan_id,
                        plan.name,
                        plan.max_parallel,
                        len(plan.jobs),
                        _json(plan_payload),
                        created,
                        created,
                    ),
                )
                for job in plan.jobs:
                    job_id = job.id or new_id("job")
                    conn.execute(
                        """
                        INSERT INTO jobs (
                          id, plan_id, name, command, status, priority,
                          nodes_json, labels_json, params_json, env_json,
                          resources_json, dependencies_json, timeout,
                          max_retries, retry_delay, fingerprint, submitted_at
                        ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job_id,
                            plan_id,
                            job.name,
                            job.command,
                            job.priority,
                            _json(job.nodes),
                            _json(job.required_labels),
                            _json(job.params),
                            _json(job.env),
                            _json(
                                {
                                    "gpus": job.resources.gpus,
                                    "memory_mib": job.resources.memory_mib,
                                    "gpu_utilization": job.resources.gpu_utilization,
                                    "duration_seconds": job.resources.duration_seconds,
                                    "allow_colocation": job.resources.allow_colocation,
                                }
                            ),
                            _json(job.dependencies),
                            job.timeout,
                            job.max_retries,
                            job.retry_delay,
                            job.fingerprint(),
                            created,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        self.add_event("plan_submitted", f"plan {plan.name} submitted", {"plan_id": plan_id})
        return self.get_plan(plan_id, include_jobs=True)

    def get_plan(self, plan_id: str, *, include_jobs: bool = False) -> dict[str, Any]:
        with self.connect() as conn:
            plan = self._plan_row(conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone())
            if plan and include_jobs:
                plan["jobs"] = [
                    self._job_row(row)
                    for row in conn.execute(
                        "SELECT * FROM jobs WHERE plan_id=? ORDER BY submitted_at, name",
                        (plan_id,),
                    ).fetchall()
                ]
            return plan

    def list_plans(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                self._plan_row(row)
                for row in conn.execute(
                    "SELECT * FROM plans ORDER BY created_at DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
            ]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            return self._job_row(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list_jobs(
        self,
        *,
        plan_id: str | None = None,
        statuses: Iterable[str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        args: list[Any] = []
        if plan_id:
            where.append("plan_id=?")
            args.append(plan_id)
        status_values = list(statuses or [])
        if status_values:
            where.append(f"status IN ({','.join('?' for _ in status_values)})")
            args.extend(status_values)
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY priority DESC, submitted_at, name LIMIT ?"
        args.append(max(1, int(limit)))
        with self.connect() as conn:
            return [self._job_row(row) for row in conn.execute(sql, args).fetchall()]

    def runnable_jobs(self, now: float | None = None) -> list[dict[str, Any]]:
        timestamp = now if now is not None else now_ts()
        pending = self.list_jobs(statuses=["pending"], limit=10000)
        if not pending:
            return []
        plan_jobs: dict[str, dict[str, str]] = {}
        with self.connect() as conn:
            for plan_id in {job["plan_id"] for job in pending}:
                plan_jobs[plan_id] = {
                    row["name"]: row["status"]
                    for row in conn.execute(
                        "SELECT name, status FROM jobs WHERE plan_id=?",
                        (plan_id,),
                    ).fetchall()
                }
        runnable = []
        for job in pending:
            if job["next_run_at"] > timestamp or job["cancel_requested"]:
                continue
            statuses = plan_jobs.get(job["plan_id"], {})
            if all(statuses.get(dep) == "completed" for dep in job["dependencies"]):
                runnable.append(job)
        return runnable

    def fail_blocked_dependencies(self) -> list[dict[str, Any]]:
        """Fail pending jobs whose upstream dependency cannot complete.

        Propagation is iterative so A -> B -> C resolves in one scheduler tick
        when A fails.
        """
        blocked: list[dict[str, Any]] = []
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                while True:
                    changed = 0
                    plan_ids = [
                        row["plan_id"]
                        for row in conn.execute(
                            "SELECT DISTINCT plan_id FROM jobs WHERE status='pending'"
                        ).fetchall()
                    ]
                    for plan_id in plan_ids:
                        rows = conn.execute(
                            "SELECT id, name, status, dependencies_json FROM jobs WHERE plan_id=?",
                            (plan_id,),
                        ).fetchall()
                        statuses = {row["name"]: row["status"] for row in rows}
                        for row in rows:
                            if row["status"] != "pending":
                                continue
                            dependencies = _loads(row["dependencies_json"], [])
                            failed = [
                                dependency
                                for dependency in dependencies
                                if statuses.get(dependency) in {"failed", "cancelled"}
                            ]
                            if not failed:
                                continue
                            error = (
                                "blocked because dependencies did not complete: "
                                + ", ".join(failed)
                            )
                            conn.execute(
                                """
                                UPDATE jobs
                                SET status='failed', error=?, finished_at=?
                                WHERE id=? AND status='pending'
                                """,
                                (error, now_ts(), row["id"]),
                            )
                            blocked.append(
                                {
                                    "id": row["id"],
                                    "name": row["name"],
                                    "plan_id": plan_id,
                                    "dependencies": failed,
                                    "error": error,
                                }
                            )
                            statuses[row["name"]] = "failed"
                            changed += 1
                    if not changed:
                        break
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        for item in blocked:
            self.refresh_plan_for_job(item["id"])
        return blocked

    def active_count(self, plan_id: str | None = None) -> int:
        args: list[Any] = []
        where = "status IN ('starting','running','cancelling')"
        if plan_id:
            where += " AND plan_id=?"
            args.append(plan_id)
        with self.connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM jobs WHERE {where}", args).fetchone()
            return int(row["n"])

    def acquire_placement(self, job_id: str, placement: Placement, ttl: float) -> bool:
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row is None or row["status"] != "pending":
                    conn.execute("ROLLBACK")
                    return False
                conn.execute(
                    """
                    UPDATE jobs
                    SET status='starting', node=?, gpus_json=?, estimate_json=?,
                        attempt=attempt+1, started_at=COALESCE(started_at, ?), error=''
                    WHERE id=?
                    """,
                    (
                        placement.node,
                        _json(placement.gpu_indices),
                        _json(placement.estimate.as_dict()),
                        timestamp,
                        job_id,
                    ),
                )
                for gpu_index in placement.gpu_indices:
                    conn.execute(
                        """
                        INSERT INTO leases
                          (id, job_id, node, gpu_index, memory_mib, created_at, heartbeat_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("lease"),
                            job_id,
                            placement.node,
                            gpu_index,
                            placement.memory_mib_per_gpu,
                            timestamp,
                            timestamp,
                            timestamp + ttl,
                        ),
                    )
                conn.execute("COMMIT")
                return True
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                return False
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def mark_running(self, job_id: str, handle: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', handle_json=?, error='' WHERE id=?",
                (_json(handle), job_id),
            )
        self.refresh_plan_for_job(job_id)

    def update_runtime(
        self,
        job_id: str,
        *,
        peak_memory_mib: int | None = None,
        stdout_tail: str | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        sets: list[str] = []
        args: list[Any] = []
        if peak_memory_mib is not None:
            sets.append("peak_memory_mib=MAX(peak_memory_mib, ?)")
            args.append(max(0, int(peak_memory_mib)))
        if stdout_tail is not None:
            sets.append("stdout_tail=?")
            args.append(stdout_tail[-12000:])
        if stderr_tail is not None:
            sets.append("stderr_tail=?")
            args.append(stderr_tail[-12000:])
        if not sets:
            return
        args.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", args)

    def heartbeat_leases(self, job_id: str, ttl: float) -> None:
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute(
                "UPDATE leases SET heartbeat_at=?, expires_at=? WHERE job_id=?",
                (timestamp, timestamp + ttl, job_id),
            )

    def leases(self, *, active_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM leases"
        args: tuple[Any, ...] = ()
        if active_only:
            sql += " WHERE expires_at>?"
            args = (now_ts(),)
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, args).fetchall()]

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        exit_code: int | None,
        error: str = "",
        stdout_tail: str = "",
        stderr_tail: str = "",
    ) -> dict[str, Any]:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid terminal status: {status}")
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE jobs
                SET status=?, exit_code=?, error=?, stdout_tail=?, stderr_tail=?,
                    finished_at=?, handle_json='{}'
                WHERE id=?
                """,
                (
                    status,
                    exit_code,
                    error[-12000:],
                    stdout_tail[-12000:],
                    stderr_tail[-12000:],
                    timestamp,
                    job_id,
                ),
            )
            conn.execute("DELETE FROM leases WHERE job_id=?", (job_id,))
            conn.execute("COMMIT")
        self.refresh_plan_for_job(job_id)
        return self.get_job(job_id)

    def retry_or_fail(
        self,
        job_id: str,
        *,
        exit_code: int | None,
        error: str,
        stdout_tail: str,
        stderr_tail: str,
    ) -> dict[str, Any]:
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT attempt, max_retries, retry_delay, cancel_requested FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return {}
            if row["cancel_requested"]:
                status = "cancelled"
                next_run_at = 0.0
                finished_at: float | None = timestamp
            elif int(row["attempt"]) <= int(row["max_retries"]):
                status = "pending"
                next_run_at = timestamp + float(row["retry_delay"])
                finished_at = None
            else:
                status = "failed"
                next_run_at = 0.0
                finished_at = timestamp
            conn.execute(
                """
                UPDATE jobs
                SET status=?, exit_code=?, error=?, stdout_tail=?, stderr_tail=?,
                    finished_at=?, next_run_at=?, handle_json='{}', node='', gpus_json='[]'
                WHERE id=?
                """,
                (
                    status,
                    exit_code,
                    error[-12000:],
                    stdout_tail[-12000:],
                    stderr_tail[-12000:],
                    finished_at,
                    next_run_at,
                    job_id,
                ),
            )
            conn.execute("DELETE FROM leases WHERE job_id=?", (job_id,))
            conn.execute("COMMIT")
        self.refresh_plan_for_job(job_id)
        return self.get_job(job_id)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return {}
            if row["status"] == "pending":
                conn.execute(
                    """
                    UPDATE jobs
                    SET status='cancelled', cancel_requested=1, finished_at=?
                    WHERE id=?
                    """,
                    (now_ts(), job_id),
                )
            elif row["status"] in {"starting", "running"}:
                conn.execute(
                    "UPDATE jobs SET status='cancelling', cancel_requested=1 WHERE id=?",
                    (job_id,),
                )
            conn.execute("COMMIT")
        self.refresh_plan_for_job(job_id)
        return self.get_job(job_id)

    def save_snapshots(self, snapshots: list[NodeSnapshot]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO snapshots(node, payload_json, collected_at)
                VALUES (?, ?, ?)
                ON CONFLICT(node) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  collected_at=excluded.collected_at
                """,
                [
                    (snapshot.node, _json(snapshot.as_dict()), snapshot.collected_at)
                    for snapshot in snapshots
                ],
            )

    def latest_snapshots(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                _loads(row["payload_json"], {})
                for row in conn.execute(
                    "SELECT payload_json FROM snapshots ORDER BY node"
                ).fetchall()
            ]

    def get_profile(self, fingerprint: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
            return dict(row) if row else {}

    def record_profile(
        self,
        fingerprint: str,
        *,
        peak_memory_mib: int,
        duration_seconds: float | None,
        oom: bool = False,
    ) -> None:
        timestamp = now_ts()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO profiles
                      (fingerprint, samples, peak_memory_mib, avg_duration_seconds, oom_count, updated_at)
                    VALUES (?, 1, ?, ?, ?, ?)
                    """,
                    (
                        fingerprint,
                        max(1, int(peak_memory_mib)),
                        duration_seconds,
                        1 if oom else 0,
                        timestamp,
                    ),
                )
            else:
                samples = int(row["samples"]) + 1
                old_duration = row["avg_duration_seconds"]
                if duration_seconds is None:
                    avg_duration = old_duration
                elif old_duration is None:
                    avg_duration = duration_seconds
                else:
                    avg_duration = (
                        float(old_duration) * (samples - 1) + duration_seconds
                    ) / samples
                observed_peak = max(1, int(peak_memory_mib))
                if oom:
                    observed_peak = max(observed_peak, int(row["peak_memory_mib"] * 1.35))
                conn.execute(
                    """
                    UPDATE profiles
                    SET samples=?, peak_memory_mib=MAX(peak_memory_mib, ?),
                        avg_duration_seconds=?, oom_count=oom_count+?, updated_at=?
                    WHERE fingerprint=?
                    """,
                    (
                        samples,
                        observed_peak,
                        avg_duration,
                        1 if oom else 0,
                        timestamp,
                        fingerprint,
                    ),
                )

    def add_event(self, kind: str, message: str, payload: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO events(kind, message, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (kind, message, _json(payload or {}), now_ts()),
            )

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            result = []
            for row in conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall():
                item = dict(row)
                item["payload"] = _loads(item.pop("payload_json"), {})
                result.append(item)
            return result

    def refresh_plan_for_job(self, job_id: str) -> None:
        with self.connect() as conn:
            row = conn.execute("SELECT plan_id FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row:
                self._refresh_plan(conn, row["plan_id"])

    def refresh_all_plans(self) -> None:
        with self.connect() as conn:
            ids = [row["id"] for row in conn.execute("SELECT id FROM plans").fetchall()]
            for plan_id in ids:
                self._refresh_plan(conn, plan_id)

    @staticmethod
    def _refresh_plan(conn: sqlite3.Connection, plan_id: str) -> None:
        statuses = [
            row["status"]
            for row in conn.execute("SELECT status FROM jobs WHERE plan_id=?", (plan_id,)).fetchall()
        ]
        if not statuses:
            return
        if all(status == "completed" for status in statuses):
            status = "completed"
            finished = now_ts()
        elif all(status in {"completed", "failed", "cancelled"} for status in statuses):
            status = "failed" if "failed" in statuses else "cancelled"
            finished = now_ts()
        elif any(status in {"starting", "running", "cancelling"} for status in statuses):
            status = "running"
            finished = None
        else:
            status = "queued"
            finished = None
        conn.execute(
            "UPDATE plans SET status=?, updated_at=?, finished_at=? WHERE id=?",
            (status, now_ts(), finished, plan_id),
        )

    def acquire_controller(self, owner: str, ttl: float) -> bool:
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner, expires_at FROM controller_lock WHERE name='scheduler'"
            ).fetchone()
            if row and row["owner"] != owner and float(row["expires_at"]) > timestamp:
                conn.execute("ROLLBACK")
                return False
            conn.execute(
                """
                INSERT INTO controller_lock(name, owner, expires_at)
                VALUES ('scheduler', ?, ?)
                ON CONFLICT(name) DO UPDATE SET owner=excluded.owner, expires_at=excluded.expires_at
                """,
                (owner, timestamp + ttl),
            )
            conn.execute("COMMIT")
            return True

    def release_controller(self, owner: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM controller_lock WHERE name='scheduler' AND owner=?",
                (owner,),
            )
