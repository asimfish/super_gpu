"""Conservative detection of sustained low-utilization GPU occupancy."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .models import NodeSnapshot


WATCHDOG_ACTIONS = {"report", "cancel_managed"}

# Idle accumulation pauses (instead of resetting) when consecutive samples for
# an observation are further apart than this, e.g. while a node is unreachable.
STALE_SAMPLE_GAP_SECONDS = 30.0


def _env_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class WatchdogPolicy:
    enabled: bool = True
    low_utilization_threshold: int = 3
    min_memory_used_mib: int = 1024
    grace_seconds: float = 900.0
    min_runtime_seconds: float = 1800.0
    action: str = "report"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WatchdogPolicy":
        values = os.environ if env is None else env
        policy = cls(
            enabled=_env_bool(
                values.get("SUPER_GPU_WATCHDOG_ENABLED", "true"),
                name="SUPER_GPU_WATCHDOG_ENABLED",
            ),
            low_utilization_threshold=int(
                values.get("SUPER_GPU_WATCHDOG_LOW_UTILIZATION", "3")
            ),
            min_memory_used_mib=int(
                values.get("SUPER_GPU_WATCHDOG_MIN_MEMORY_MIB", "1024")
            ),
            grace_seconds=float(values.get("SUPER_GPU_WATCHDOG_GRACE_SECONDS", "900")),
            min_runtime_seconds=float(
                values.get("SUPER_GPU_WATCHDOG_MIN_RUNTIME_SECONDS", "1800")
            ),
            action=values.get("SUPER_GPU_WATCHDOG_ACTION", "report").strip().lower(),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not 0 <= self.low_utilization_threshold <= 100:
            raise ValueError("watchdog low utilization threshold must be between 0 and 100")
        if self.min_memory_used_mib < 1:
            raise ValueError("watchdog minimum memory must be positive")
        if self.grace_seconds < 0:
            raise ValueError("watchdog grace must be non-negative")
        if self.min_runtime_seconds < 0:
            raise ValueError("watchdog minimum runtime must be non-negative")
        if self.action not in WATCHDOG_ACTIONS:
            raise ValueError(
                "SUPER_GPU_WATCHDOG_ACTION must be report or cancel_managed"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardianResult:
    findings: list[dict[str, Any]]
    newly_active: list[dict[str, Any]]
    resolved_ids: list[str]
    cancel_job_ids: list[str]


@dataclass
class _Observation:
    finding_id: str
    node: str
    first_seen: float
    last_seen: float
    samples: int


class IdleGpuGuardian:
    """Track idle occupancy and propose cancellation only for owned jobs."""

    def __init__(self, policy: WatchdogPolicy | None = None) -> None:
        self.policy = policy or WatchdogPolicy.from_env()
        self._observations: dict[str, _Observation] = {}
        self._findings: list[dict[str, Any]] = []
        self._active_ids: set[str] = set()
        self._last_inspection = 0.0

    @staticmethod
    def _observation_key(
        node: str,
        gpu_index: int,
        managed_job_ids: list[str],
    ) -> str:
        # Process ids are deliberately excluded: on shared GPUs neighbour
        # processes churn constantly, and keying on the pid set would reset
        # the idle timer so often that sustained idleness never accumulates.
        identity = f"{node}:{gpu_index}:{','.join(managed_job_ids)}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    def inspect(
        self,
        snapshots: list[NodeSnapshot],
        leases: list[dict[str, Any]],
        jobs: list[dict[str, Any]],
        *,
        now: float | None = None,
    ) -> GuardianResult:
        timestamp = time.time() if now is None else float(now)
        self._last_inspection = timestamp
        previous_active = set(self._active_ids)
        if not self.policy.enabled:
            self._observations.clear()
            self._findings = []
            self._active_ids = set()
            return GuardianResult([], [], sorted(previous_active), [])

        active_jobs = {
            str(job["id"]): job
            for job in jobs
            if job.get("status") in {"starting", "running", "cancelling"}
        }
        leases_by_gpu: dict[tuple[str, int], list[dict[str, Any]]] = {}
        leases_by_job: dict[str, list[dict[str, Any]]] = {}
        for lease in leases:
            job_id = str(lease["job_id"])
            if job_id not in active_jobs:
                continue
            leases_by_gpu.setdefault(
                (str(lease["node"]), int(lease["gpu_index"])), []
            ).append(lease)
            leases_by_job.setdefault(job_id, []).append(lease)

        findings: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        findings_by_gpu: dict[tuple[str, int], dict[str, Any]] = {}
        reachable_nodes = {
            snapshot.node for snapshot in snapshots if snapshot.reachable
        }
        for snapshot in snapshots:
            if not snapshot.reachable:
                continue
            for gpu in snapshot.gpus:
                if (
                    gpu.utilization > self.policy.low_utilization_threshold
                    or gpu.memory_used_mib < self.policy.min_memory_used_mib
                    or not gpu.processes
                ):
                    continue
                gpu_leases = leases_by_gpu.get((snapshot.node, gpu.index), [])
                managed_job_ids = sorted({str(lease["job_id"]) for lease in gpu_leases})
                key = self._observation_key(
                    snapshot.node,
                    gpu.index,
                    managed_job_ids,
                )
                seen_keys.add(key)
                observation = self._observations.get(key)
                if observation is None:
                    observation = _Observation(
                        key, snapshot.node, timestamp, timestamp, 1
                    )
                    self._observations[key] = observation
                else:
                    gap = timestamp - observation.last_seen
                    if gap > STALE_SAMPLE_GAP_SECONDS:
                        # Sampling was interrupted; don't count the blind window
                        # as idle time, but don't reset accumulation either.
                        observation.first_seen += gap
                    observation.last_seen = timestamp
                    observation.samples += 1
                idle_seconds = max(0.0, timestamp - observation.first_seen)
                status = "active" if idle_seconds >= self.policy.grace_seconds else "observing"
                kind = (
                    "unmanaged_idle"
                    if not managed_job_ids
                    else "managed_idle"
                    if len(managed_job_ids) == 1
                    else "colocated_idle"
                )
                finding = {
                    "id": observation.finding_id,
                    "kind": kind,
                    "status": status,
                    "node": snapshot.node,
                    "role": snapshot.role,
                    "gpu_index": gpu.index,
                    "gpu_uuid": gpu.uuid,
                    "gpu_name": gpu.name,
                    "utilization": gpu.utilization,
                    "memory_used_mib": gpu.memory_used_mib,
                    "memory_total_mib": gpu.memory_total_mib,
                    "processes": [asdict(process) for process in gpu.processes],
                    "managed_job_ids": managed_job_ids,
                    "first_seen": observation.first_seen,
                    "last_seen": observation.last_seen,
                    "idle_seconds": idle_seconds,
                    "samples": observation.samples,
                    "automatic_action": "none",
                }
                findings.append(finding)
                findings_by_gpu[(snapshot.node, gpu.index)] = finding

        for key in set(self._observations) - seen_keys:
            # Keep observations for nodes we could not sample this round so a
            # transient poll failure does not zero out accumulated idle time.
            if self._observations[key].node in reachable_nodes:
                self._observations.pop(key, None)

        cancel_job_ids: list[str] = []
        if self.policy.action == "cancel_managed":
            for job_id, job in active_jobs.items():
                if job.get("cancel_requested"):
                    continue
                started_at = float(job.get("started_at") or 0.0)
                if started_at <= 0:
                    continue
                if timestamp - started_at < self.policy.min_runtime_seconds:
                    continue
                job_leases = leases_by_job.get(job_id, [])
                if not job_leases:
                    continue
                owned_findings = [
                    findings_by_gpu.get((str(lease["node"]), int(lease["gpu_index"])))
                    for lease in job_leases
                ]
                if not all(
                    finding
                    and finding["status"] == "active"
                    and finding["managed_job_ids"] == [job_id]
                    for finding in owned_findings
                ):
                    continue
                cancel_job_ids.append(job_id)
                for finding in owned_findings:
                    if finding is not None:
                        finding["automatic_action"] = "cancel_managed"

        current_active = {
            str(finding["id"])
            for finding in findings
            if finding["status"] == "active"
        }
        newly_active_ids = current_active - previous_active
        newly_active = [
            finding for finding in findings if finding["id"] in newly_active_ids
        ]
        resolved_ids = sorted(previous_active - current_active)
        self._findings = sorted(
            findings,
            key=lambda finding: (
                finding["status"] != "active",
                -float(finding["idle_seconds"]),
                finding["node"],
                int(finding["gpu_index"]),
            ),
        )
        self._active_ids = current_active
        return GuardianResult(
            findings=self.findings(),
            newly_active=newly_active,
            resolved_ids=resolved_ids,
            cancel_job_ids=sorted(set(cancel_job_ids)),
        )

    def export_observations(self) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "node": observation.node,
                "first_seen": observation.first_seen,
                "last_seen": observation.last_seen,
                "samples": observation.samples,
            }
            for key, observation in sorted(self._observations.items())
        ]

    def restore_observations(self, rows: list[Mapping[str, Any]]) -> None:
        self._observations = {
            str(row["key"]): _Observation(
                str(row["key"]),
                str(row["node"]),
                float(row["first_seen"]),
                float(row["last_seen"]),
                int(row["samples"]),
            )
            for row in rows
        }

    def findings(self) -> list[dict[str, Any]]:
        return [dict(finding) for finding in self._findings]

    def status(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict(),
            "active": sum(1 for finding in self._findings if finding["status"] == "active"),
            "observing": sum(
                1 for finding in self._findings if finding["status"] == "observing"
            ),
            "last_inspection": self._last_inspection,
        }
