"""SQLite-backed desired state, leases, telemetry, and resource history."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import (
    RESULT_STATES,
    ExperimentPlan,
    JobDependency,
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

CREATE TABLE IF NOT EXISTS submission_requests (
  request_id TEXT PRIMARY KEY,
  intent_digest TEXT NOT NULL,
  plan_id TEXT NOT NULL UNIQUE REFERENCES plans(id) ON DELETE CASCADE,
  created_at REAL NOT NULL
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
  pending_reason TEXT NOT NULL DEFAULT '',
  result_state TEXT NOT NULL DEFAULT '',
  result_json TEXT NOT NULL DEFAULT '{}',
  outputs_json TEXT NOT NULL DEFAULT '[]',
  run_dir TEXT NOT NULL DEFAULT '',
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

CREATE TABLE IF NOT EXISTS watchdog_observations (
  key TEXT PRIMARY KEY,
  node TEXT NOT NULL,
  first_seen REAL NOT NULL,
  last_seen REAL NOT NULL,
  samples INTEGER NOT NULL
);
"""

# Columns added after the first release; applied to databases created by
# older versions. SQLite CREATE TABLE IF NOT EXISTS never alters existing
# tables, so each new jobs column needs an entry here as well.
JOBS_COLUMN_MIGRATIONS: dict[str, str] = {
    "pending_reason": "TEXT NOT NULL DEFAULT ''",
    "result_state": "TEXT NOT NULL DEFAULT ''",
    "result_json": "TEXT NOT NULL DEFAULT '{}'",
    "outputs_json": "TEXT NOT NULL DEFAULT '[]'",
    "run_dir": "TEXT NOT NULL DEFAULT ''",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class IdempotencyConflict(ValueError):
    """A request ID was already bound to a different submission intent."""


def _plan_payload(plan: ExperimentPlan, plan_id: str) -> dict[str, Any]:
    return {
        "id": plan_id,
        "name": plan.name,
        "max_parallel": plan.max_parallel,
        "defaults": plan.defaults,
        "request_id": plan.request_id,
        "source": plan.source,
        "source_snapshot": plan.source_snapshot,
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
                "dependencies": [dep.as_value() for dep in job.dependencies],
                "outputs": job.outputs,
            }
            for job in plan.jobs
        ],
    }


def _intent_digest(plan: ExperimentPlan) -> str:
    import hashlib

    payload = _plan_payload(plan, plan.id)
    payload.pop("request_id", None)
    source = dict(payload.get("source") or {})
    if source.get("mode") == "snapshot":
        source.pop("path", None)
        source.pop("exclude", None)
    payload["source"] = source
    snapshot = dict(payload.get("source_snapshot") or {})
    payload["source_snapshot"] = {
        key: snapshot[key]
        for key in ("digest", "format")
        if key in snapshot
    }
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


TERMINAL_PLAN_STATES = {"completed", "failed", "cancelled"}


class StateStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._initialize()

    def add_event_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Receive every recorded event after it is committed.

        Listeners run on the caller's thread and must return quickly; any
        exception they raise is swallowed so observers can never break the
        scheduler.
        """
        self._event_listeners.append(callback)

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
                existing = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                for column, ddl in JOBS_COLUMN_MIGRATIONS.items():
                    if column not in existing:
                        conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")

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
            ("result_json", "result", {}),
            ("outputs_json", "outputs", []),
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
        plan_payload = _plan_payload(plan, plan_id)
        intent_digest = _intent_digest(plan)
        replayed = False
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if plan.request_id:
                    receipt = conn.execute(
                        "SELECT intent_digest, plan_id FROM submission_requests WHERE request_id=?",
                        (plan.request_id,),
                    ).fetchone()
                    if receipt:
                        if receipt["intent_digest"] != intent_digest:
                            raise IdempotencyConflict(
                                f"request_id {plan.request_id!r} is already bound to a different plan intent"
                            )
                        plan_id = str(receipt["plan_id"])
                        replayed = True
                if not replayed:
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
                              max_retries, retry_delay, fingerprint, submitted_at,
                              outputs_json
                            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                _json([dep.as_value() for dep in job.dependencies]),
                                job.timeout,
                                job.max_retries,
                                job.retry_delay,
                                job.fingerprint(),
                                created,
                                _json(job.outputs),
                            ),
                        )
                    if plan.request_id:
                        conn.execute(
                            """
                            INSERT INTO submission_requests
                              (request_id, intent_digest, plan_id, created_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (plan.request_id, intent_digest, plan_id, created),
                        )
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        if not replayed:
            self.add_event(
                "plan_submitted",
                f"plan {plan.name} submitted",
                {"plan_id": plan_id, "request_id": plan.request_id, "intent_digest": intent_digest},
            )
        result = self.get_plan(plan_id, include_jobs=True)
        result["submission"] = {
            "request_id": plan.request_id,
            "intent_digest": intent_digest,
            "replayed": replayed,
        }
        return result

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

    def runnable_jobs(
        self,
        now: float | None = None,
        *,
        annotate: bool = False,
    ) -> list[dict[str, Any]]:
        """Return pending jobs whose gates are open.

        With ``annotate=True`` (used by the scheduler tick) jobs held back by
        retry backoff or dependencies get a human-readable ``pending_reason``
        so agents can see why a job is not running yet.
        """
        timestamp = now if now is not None else now_ts()
        pending = self.list_jobs(statuses=["pending"], limit=10000)
        if not pending:
            return []
        plan_jobs: dict[str, dict[str, tuple[str, str]]] = {}
        with self.connect() as conn:
            for plan_id in {job["plan_id"] for job in pending}:
                plan_jobs[plan_id] = {
                    row["name"]: (row["status"], row["result_state"])
                    for row in conn.execute(
                        "SELECT name, status, result_state FROM jobs WHERE plan_id=?",
                        (plan_id,),
                    ).fetchall()
                }
        runnable = []
        for job in pending:
            if job["cancel_requested"]:
                continue
            if job["next_run_at"] > timestamp:
                if annotate:
                    self.set_pending_reason(
                        job["id"],
                        f"in retry backoff after attempt {job['attempt']}",
                    )
                continue
            states = plan_jobs.get(job["plan_id"], {})
            unmet = []
            for entry in job["dependencies"]:
                dependency = JobDependency.from_value(entry)
                status, result_state = states.get(dependency.job, ("missing", ""))
                if dependency.gate(status, result_state) != "satisfied":
                    unmet.append((dependency.job, status))
            if unmet:
                if annotate:
                    detail = ", ".join(
                        f"{name} ({status})" for name, status in unmet[:5]
                    )
                    self.set_pending_reason(
                        job["id"], f"waiting for dependencies: {detail}"
                    )
                continue
            runnable.append(job)
        return runnable

    def set_pending_reason(self, job_id: str, reason: str) -> None:
        """Record why a pending job is not running; writes only on change."""
        normalized = (reason or "")[:1000]
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET pending_reason=?
                WHERE id=? AND status='pending' AND pending_reason<>?
                """,
                (normalized, job_id, normalized),
            )

    def resolve_dependency_gates(self) -> list[dict[str, Any]]:
        """Skip pending jobs whose dependency predicate can no longer hold.

        A dependency that reached a terminal state without satisfying its
        predicate (default: completed with result state success) makes the
        dependent job terminal as ``skipped`` with result state
        ``dependency_skipped``. Propagation is iterative so A -> B -> C
        resolves in one scheduler tick when A fails.
        """
        skipped: list[dict[str, Any]] = []
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
                            """
                            SELECT id, name, status, result_state, dependencies_json
                            FROM jobs WHERE plan_id=?
                            """,
                            (plan_id,),
                        ).fetchall()
                        states = {
                            row["name"]: (row["status"], row["result_state"])
                            for row in rows
                        }
                        for row in rows:
                            if row["status"] != "pending":
                                continue
                            unsatisfiable = []
                            for entry in _loads(row["dependencies_json"], []):
                                dependency = JobDependency.from_value(entry)
                                status, result_state = states.get(
                                    dependency.job, ("missing", "")
                                )
                                if dependency.gate(status, result_state) == "skip":
                                    unsatisfiable.append(
                                        f"{dependency.job} finished as "
                                        f"{result_state or status} "
                                        f"(needs after={dependency.after}"
                                        + (
                                            f" {dependency.result_states}"
                                            if dependency.after == "result"
                                            else ""
                                        )
                                        + ")"
                                    )
                            if not unsatisfiable:
                                continue
                            error = "dependency not satisfiable: " + "; ".join(
                                unsatisfiable
                            )
                            conn.execute(
                                """
                                UPDATE jobs
                                SET status='skipped', result_state='dependency_skipped',
                                    error=?, finished_at=?, pending_reason=''
                                WHERE id=? AND status='pending'
                                """,
                                (error, now_ts(), row["id"]),
                            )
                            skipped.append(
                                {
                                    "id": row["id"],
                                    "name": row["name"],
                                    "plan_id": plan_id,
                                    "error": error,
                                }
                            )
                            states[row["name"]] = ("skipped", "dependency_skipped")
                            changed += 1
                    if not changed:
                        break
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        for item in skipped:
            self.refresh_plan_for_job(item["id"])
        return skipped

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
                        attempt=attempt+1, started_at=COALESCE(started_at, ?), error='',
                        pending_reason=''
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
            # run_dir survives job completion (handle_json is wiped) so that
            # declared outputs stay pullable after the job finishes.
            conn.execute(
                "UPDATE jobs SET status='running', handle_json=?, run_dir=?, error='' WHERE id=?",
                (_json(handle), str(handle.get("run_dir") or ""), job_id),
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
        result_state: str = "",
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid terminal status: {status}")
        if result_state and result_state not in RESULT_STATES:
            raise ValueError(f"invalid result state: {result_state}")
        if not result_state:
            result_state = {
                "completed": "success",
                "failed": "execution_failure",
                "cancelled": "cancelled",
            }[status]
        result_payload = _json(result or {})
        if len(result_payload) > 8000:
            result_payload = _json(
                {"result_warning": "result metadata exceeded 8000 characters and was dropped"}
            )
        timestamp = now_ts()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE jobs
                SET status=?, exit_code=?, error=?, stdout_tail=?, stderr_tail=?,
                    finished_at=?, handle_json='{}', result_state=?, result_json=?
                WHERE id=?
                """,
                (
                    status,
                    exit_code,
                    error[-12000:],
                    stdout_tail[-12000:],
                    stderr_tail[-12000:],
                    timestamp,
                    result_state,
                    result_payload,
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
                pending_reason = ""
                result_state = "cancelled"
            elif int(row["attempt"]) <= int(row["max_retries"]):
                status = "pending"
                next_run_at = timestamp + float(row["retry_delay"])
                finished_at = None
                pending_reason = (
                    f"retry scheduled in {float(row['retry_delay']):g}s "
                    f"(attempt {int(row['attempt'])} of {int(row['max_retries']) + 1} failed)"
                )
                result_state = ""
            else:
                status = "failed"
                next_run_at = 0.0
                finished_at = timestamp
                pending_reason = ""
                # exit_code None means the process never ran or was lost by
                # the platform rather than failing on its own.
                result_state = "infra_failure" if exit_code is None else "execution_failure"
            conn.execute(
                """
                UPDATE jobs
                SET status=?, exit_code=?, error=?, stdout_tail=?, stderr_tail=?,
                    finished_at=?, next_run_at=?, handle_json='{}', node='', gpus_json='[]',
                    pending_reason=?, result_state=?
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
                    pending_reason,
                    result_state,
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

    def save_watchdog_observations(self, observations: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM watchdog_observations")
                conn.executemany(
                    """
                    INSERT INTO watchdog_observations(key, node, first_seen, last_seen, samples)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(observation["key"]),
                            str(observation["node"]),
                            float(observation["first_seen"]),
                            float(observation["last_seen"]),
                            int(observation["samples"]),
                        )
                        for observation in observations
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def load_watchdog_observations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT key, node, first_seen, last_seen, samples FROM watchdog_observations"
            ).fetchall()
        return [dict(row) for row in rows]

    def add_event(self, kind: str, message: str, payload: dict[str, Any] | None = None) -> None:
        created = now_ts()
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events(kind, message, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (kind, message, _json(payload or {}), created),
            )
            event_id = cursor.lastrowid
        if not self._event_listeners:
            return
        event = {
            "id": event_id,
            "kind": kind,
            "message": message,
            "payload": dict(payload or {}),
            "created_at": created,
        }
        for listener in list(self._event_listeners):
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - observers must never break the store
                pass

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
            transition = self._refresh_plan(conn, row["plan_id"]) if row else None
        if transition:
            self._announce_plan_transition(transition)

    def refresh_all_plans(self) -> list[dict[str, Any]]:
        """Recompute every plan's status; return plans that just became terminal."""
        transitions: list[dict[str, Any]] = []
        with self.connect() as conn:
            ids = [row["id"] for row in conn.execute("SELECT id FROM plans").fetchall()]
            for plan_id in ids:
                transition = self._refresh_plan(conn, plan_id)
                if transition:
                    transitions.append(transition)
        for transition in transitions:
            self._announce_plan_transition(transition)
        return transitions

    def _announce_plan_transition(self, transition: dict[str, Any]) -> None:
        counts = transition["job_counts"]
        summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        self.add_event(
            f"plan_{transition['status']}",
            f"plan {transition['name']} {transition['status']}: {summary}",
            transition,
        )

    @staticmethod
    def _refresh_plan(conn: sqlite3.Connection, plan_id: str) -> dict[str, Any] | None:
        """Update one plan's derived status.

        Returns a transition record the first time the plan reaches a terminal
        state, and None otherwise. finished_at is written once and preserved.
        """
        current = conn.execute(
            "SELECT id, name, status, finished_at FROM plans WHERE id=?", (plan_id,)
        ).fetchone()
        if current is None:
            return None
        statuses = [
            row["status"]
            for row in conn.execute("SELECT status FROM jobs WHERE plan_id=?", (plan_id,)).fetchall()
        ]
        if not statuses:
            return None
        if all(status == "completed" for status in statuses):
            status = "completed"
        elif all(
            status in {"completed", "failed", "cancelled", "skipped"}
            for status in statuses
        ):
            # Skipped jobs reflect deliberate routing (for example a
            # scientific_reject upstream), not a plan failure by themselves.
            if "failed" in statuses:
                status = "failed"
            elif "cancelled" in statuses:
                status = "cancelled"
            else:
                status = "completed"
        elif any(status in {"starting", "running", "cancelling"} for status in statuses):
            status = "running"
        else:
            status = "queued"
        terminal = status in TERMINAL_PLAN_STATES
        previous = str(current["status"])
        if status == previous and (not terminal or current["finished_at"] is not None):
            return None
        finished = None
        if terminal:
            finished = current["finished_at"] if current["finished_at"] is not None else now_ts()
        conn.execute(
            "UPDATE plans SET status=?, updated_at=?, finished_at=? WHERE id=?",
            (status, now_ts(), finished, plan_id),
        )
        if not terminal or previous in TERMINAL_PLAN_STATES:
            return None
        counts: dict[str, int] = {}
        for item in statuses:
            counts[item] = counts.get(item, 0) + 1
        return {
            "plan_id": plan_id,
            "name": str(current["name"]),
            "status": status,
            "previous_status": previous,
            "finished_at": finished,
            "total_jobs": len(statuses),
            "job_counts": counts,
        }

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
