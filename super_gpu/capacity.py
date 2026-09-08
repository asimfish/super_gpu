"""Dry-run placement: what the cluster could accept right now.

``GET /api/capacity`` answers "how many jobs of this shape could start this
instant, and on which GPUs" by running the real ``PlacementEngine`` against
the latest telemetry and the active leases, adding a synthetic lease after
each successful placement until the engine refuses. ``POST /api/plans/preview``
applies the same walk to every job of a plan before it is submitted, so an
agent learns that a job asks for four GPUs on a node with two, or that the
memory budget is an unmeasured heuristic, without creating a plan and reading
its ``pending_reason`` afterwards.

Both are observations, not reservations: nothing is written, and another
submission may take the GPUs first. Submitting the plan is what holds them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .api_manifest import CAPACITY_MAX_GPUS
from .estimator import ResourceEstimator
from .models import (
    ExperimentPlan,
    JobResources,
    JobSpec,
    NodeSnapshot,
    Placement,
    SystemConfig,
)
from .placement import PlacementEngine

OBSERVATION_NOTE = (
    "observation, not a reservation: GPUs are held only by submitting a plan"
)


@dataclass
class CapacityRequest:
    """The shape of a hypothetical job."""

    gpus: int = 1
    memory_mib: int | None = None
    gpu_utilization: int | None = None
    nodes: list[str] = field(default_factory=list)
    required_labels: list[str] = field(default_factory=list)
    allow_colocation: bool | None = None
    max_slots: int | None = None

    @classmethod
    def from_query(cls, params: dict[str, list[str]]) -> "CapacityRequest":
        def first(key: str) -> str:
            values = params.get(key) or []
            return values[0].strip() if values else ""

        def integer(key: str, *, minimum: int, maximum: int) -> int | None:
            raw = first(key)
            if not raw or raw.lower() == "auto":
                return None
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer") from exc
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
            return value

        def csv(key: str) -> list[str]:
            return [
                item.strip()
                for value in params.get(key) or []
                for item in value.split(",")
                if item.strip()
            ]

        colocation_raw = first("allow_colocation").lower()
        allow_colocation: bool | None
        if not colocation_raw:
            allow_colocation = None
        elif colocation_raw in {"1", "true", "yes"}:
            allow_colocation = True
        elif colocation_raw in {"0", "false", "no"}:
            allow_colocation = False
        else:
            raise ValueError("allow_colocation must be true or false")
        return cls(
            gpus=integer("gpus", minimum=1, maximum=CAPACITY_MAX_GPUS) or 1,
            memory_mib=integer("memory_mib", minimum=1, maximum=4_000_000),
            gpu_utilization=integer("gpu_utilization", minimum=0, maximum=100),
            nodes=csv("nodes"),
            required_labels=csv("labels") + csv("required_labels"),
            allow_colocation=allow_colocation,
            max_slots=integer("max_slots", minimum=1, maximum=10_000),
        )

    def as_job(self) -> JobSpec:
        return JobSpec(
            name="capacity-probe",
            command="true",
            nodes=list(self.nodes),
            required_labels=list(self.required_labels),
            resources=JobResources(
                gpus=self.gpus,
                memory_mib=self.memory_mib,
                gpu_utilization=self.gpu_utilization,
                allow_colocation=self.allow_colocation,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpus": self.gpus,
            "memory_mib": self.memory_mib if self.memory_mib is not None else "auto",
            "gpu_utilization": (
                self.gpu_utilization if self.gpu_utilization is not None else "auto"
            ),
            "nodes": list(self.nodes),
            "required_labels": list(self.required_labels),
            "allow_colocation": self.allow_colocation,
            "max_slots": self.max_slots,
        }


def snapshots_from_store(rows: list[dict[str, Any]]) -> list[NodeSnapshot]:
    return [NodeSnapshot.from_dict(dict(row)) for row in rows]


def _synthetic_leases(placement: Placement, tag: str) -> list[dict[str, Any]]:
    return [
        {
            "job_id": tag,
            "node": placement.node,
            "gpu_index": gpu_index,
            "memory_mib": placement.memory_mib_per_gpu,
        }
        for gpu_index in placement.gpu_indices
    ]


def _slot_ceiling(config: SystemConfig, snapshots: list[NodeSnapshot]) -> int:
    """Upper bound on placements so the walk always terminates."""
    total = 0
    for snapshot in snapshots:
        if not snapshot.reachable:
            continue
        try:
            node = config.node(snapshot.node)
        except KeyError:
            continue
        total += len(snapshot.gpus) * max(1, int(node.policy.max_jobs_per_gpu))
    return total


def dry_run_capacity(
    config: SystemConfig,
    placement: PlacementEngine,
    estimator: ResourceEstimator,
    snapshots: list[NodeSnapshot],
    leases: list[dict[str, Any]],
    request: CapacityRequest,
    *,
    active_jobs: int,
) -> dict[str, Any]:
    """Count how many copies of ``request`` could be placed right now."""
    job = request.as_job()
    estimate = estimator.estimate(job)
    synthetic = [dict(lease) for lease in leases]
    ceiling = _slot_ceiling(config, snapshots)
    if request.max_slots is not None:
        ceiling = min(ceiling, request.max_slots)
    placements: list[Placement] = []
    rejections: dict[str, str] = {}
    while len(placements) < ceiling:
        decision = placement.decide(job, estimate, snapshots, synthetic)
        if decision.placement is None:
            rejections = decision.rejections
            break
        placements.append(decision.placement)
        synthetic.extend(
            _synthetic_leases(decision.placement, f"capacity-probe-{len(placements)}")
        )
    else:
        # The ceiling stopped the walk; still explain why no further copy fits,
        # unless the caller's own max_slots was the only limit.
        decision = placement.decide(job, estimate, snapshots, synthetic)
        if decision.placement is None:
            rejections = decision.rejections
        else:
            rejections = {"limit": f"max_slots ({request.max_slots}) reached"}

    per_node: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        try:
            role = config.node(snapshot.node).role
        except KeyError:
            role = snapshot.role
        per_node[snapshot.node] = {
            "role": role,
            "reachable": snapshot.reachable,
            "gpus": len(snapshot.gpus),
            "slots": 0,
            "gpu_indices": [],
            "blocked_by": rejections.get(snapshot.node, ""),
        }
    for item in placements:
        entry = per_node.setdefault(
            item.node,
            {"role": "", "reachable": True, "gpus": 0, "slots": 0, "gpu_indices": [], "blocked_by": ""},
        )
        entry["slots"] += 1
        for gpu_index in item.gpu_indices:
            if gpu_index not in entry["gpu_indices"]:
                entry["gpu_indices"].append(gpu_index)
    for entry in per_node.values():
        entry["gpu_indices"].sort()

    cluster_remaining = max(0, int(config.max_parallel) - int(active_jobs))
    observed_at = max((snapshot.collected_at for snapshot in snapshots), default=0.0)
    return {
        "request": request.as_dict(),
        "estimate": estimate.as_dict(),
        "slots": len(placements),
        "cluster_slots_remaining": cluster_remaining,
        "effective_slots": min(len(placements), cluster_remaining),
        "placements": [
            {"node": item.node, "gpu_indices": list(item.gpu_indices)} for item in placements
        ],
        "per_node": per_node,
        "rejections": dict(rejections),
        "observed_at": observed_at,
        "note": OBSERVATION_NOTE,
    }


def _ordered_like_scheduler(plan: ExperimentPlan) -> list[JobSpec]:
    # StateStore.list_jobs orders runnable work by priority DESC, then
    # submitted_at (identical within one plan), then name.
    return sorted(plan.jobs, key=lambda job: (-int(job.priority), job.name))


def preview_plan(
    config: SystemConfig,
    placement: PlacementEngine,
    estimator: ResourceEstimator,
    snapshots: list[NodeSnapshot],
    leases: list[dict[str, Any]],
    plan: ExperimentPlan,
    *,
    active_jobs: int,
) -> dict[str, Any]:
    """Explain, per job, what would happen if the plan were submitted now."""
    synthetic = [dict(lease) for lease in leases]
    cluster_remaining = max(0, int(config.max_parallel) - int(active_jobs))
    plan_remaining = int(plan.max_parallel) if plan.max_parallel is not None else math.inf
    max_gpus_on_a_node = max(
        (len(snapshot.gpus) for snapshot in snapshots if snapshot.reachable),
        default=0,
    )
    jobs: list[dict[str, Any]] = []
    warnings: list[str] = []
    heuristic_jobs: list[str] = []
    immediate = 0

    for job in _ordered_like_scheduler(plan):
        estimate = estimator.estimate(job)
        # Structural feasibility: could this job be placed on the fleet as
        # sampled if no managed job held any GPU? Catches shapes that will
        # never run (too many GPUs per node, impossible memory, no matching
        # node) before the plan is submitted.
        unloaded = placement.decide(job, estimate, snapshots, [])
        entry: dict[str, Any] = {
            "name": job.name,
            "priority": job.priority,
            "resources": {
                "gpus": job.resources.gpus,
                "memory_mib": job.resources.memory_mib,
                "gpu_utilization": job.resources.gpu_utilization,
            },
            "estimate": estimate.as_dict(),
            "fits_fleet": unloaded.placement is not None,
            "fits_fleet_reason": unloaded.summary(300),
            "would_start_now": False,
            "placement": None,
            "blocked_by": "",
        }
        if estimate.source == "heuristic":
            heuristic_jobs.append(job.name)
        if unloaded.placement is None:
            warnings.append(
                f"job {job.name!r} fits no node even with every managed GPU free: "
                f"{unloaded.summary(200)}"
            )
        elif job.resources.gpus > max_gpus_on_a_node:
            warnings.append(
                f"job {job.name!r} needs {job.resources.gpus} GPUs on one node but the "
                f"largest reachable node reports {max_gpus_on_a_node}"
            )
        if job.dependencies:
            names = ", ".join(dependency.job for dependency in job.dependencies)
            entry["blocked_by"] = f"waits for dependencies: {names}"
        elif cluster_remaining <= 0:
            entry["blocked_by"] = f"cluster max_parallel ({config.max_parallel}) reached"
        elif plan_remaining <= 0:
            entry["blocked_by"] = f"plan max_parallel ({plan.max_parallel}) reached"
        else:
            decision = placement.decide(job, estimate, snapshots, synthetic)
            if decision.placement is None:
                entry["blocked_by"] = decision.summary()
                if "cluster" in decision.rejections:
                    warnings.append(
                        f"job {job.name!r}: {decision.rejections['cluster']}"
                    )
            else:
                entry["would_start_now"] = True
                entry["placement"] = {
                    "node": decision.placement.node,
                    "gpu_indices": list(decision.placement.gpu_indices),
                    "memory_mib_per_gpu": decision.placement.memory_mib_per_gpu,
                }
                synthetic.extend(_synthetic_leases(decision.placement, f"preview-{job.name}"))
                cluster_remaining -= 1
                plan_remaining -= 1
                immediate += 1
        jobs.append(entry)

    if heuristic_jobs:
        shown = ", ".join(heuristic_jobs[:5]) + (" ..." if len(heuristic_jobs) > 5 else "")
        warnings.append(
            f"{len(heuristic_jobs)} job(s) have no measured or declared memory budget "
            f"({shown}); the {config.default_memory_mib} MiB heuristic is used until a run "
            "records a peak. Declare resources.memory_mib for memory-sensitive work."
        )
    source_mode = str(plan.source.get("mode") or "workspace")
    if source_mode == "snapshot":
        source_path = Path(str(plan.source.get("path") or "")).expanduser()
        if not source_path.is_dir():
            warnings.append(
                f"source.path {str(source_path)!r} is not a directory on the controller host; "
                "submission would fail"
            )
    if jobs and immediate == 0:
        warnings.append("no job would start immediately; all would queue")

    return {
        "plan": {
            "name": plan.name,
            "request_id": plan.request_id,
            "jobs": len(plan.jobs),
            "max_parallel": plan.max_parallel,
            "source_mode": source_mode,
        },
        "jobs": jobs,
        "immediate": immediate,
        "queued": len(jobs) - immediate,
        "cluster_slots_remaining_after": cluster_remaining,
        "warnings": warnings,
        "note": OBSERVATION_NOTE + "; nothing was submitted",
    }
