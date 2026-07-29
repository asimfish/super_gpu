"""Policy-aware placement for dedicated and shared GPU servers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .models import (
    GpuSnapshot,
    JobSpec,
    NodeConfig,
    NodeSnapshot,
    Placement,
    ResourceEstimate,
    SystemConfig,
)
from .monitor import ClusterMonitor


@dataclass
class _GpuCandidate:
    node: NodeConfig
    gpu: GpuSnapshot
    available_mib: int
    active_jobs: int
    reserved_mib: int
    score: float
    reasons: list[str]


class PlacementEngine:
    def __init__(self, config: SystemConfig, monitor: ClusterMonitor) -> None:
        self.config = config
        self.monitor = monitor

    def choose(
        self,
        job: JobSpec,
        estimate: ResourceEstimate,
        snapshots: list[NodeSnapshot],
        leases: list[dict[str, Any]],
    ) -> Placement | None:
        snapshot_by_node = {snapshot.node: snapshot for snapshot in snapshots}
        leases_by_gpu: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for lease in leases:
            leases_by_gpu[(str(lease["node"]), int(lease["gpu_index"]))].append(lease)

        placements: list[Placement] = []
        for node in self._candidate_nodes(job):
            snapshot = snapshot_by_node.get(node.name)
            if snapshot is None or not snapshot.reachable:
                continue
            gpu_candidates: list[_GpuCandidate] = []
            for gpu in snapshot.gpus:
                candidate = self._evaluate_gpu(
                    node,
                    gpu,
                    job,
                    estimate,
                    leases_by_gpu[(node.name, gpu.index)],
                )
                if candidate is not None:
                    gpu_candidates.append(candidate)
            if len(gpu_candidates) < job.resources.gpus:
                continue
            gpu_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
            selected = gpu_candidates[: job.resources.gpus]
            # Multi-GPU jobs must fit entirely on one server.
            placements.append(
                Placement(
                    node=node.name,
                    gpu_indices=[candidate.gpu.index for candidate in selected],
                    memory_mib_per_gpu=estimate.memory_mib,
                    score=sum(candidate.score for candidate in selected),
                    estimate=estimate,
                    reasons=[
                        reason
                        for candidate in selected
                        for reason in candidate.reasons
                    ],
                )
            )
        if not placements:
            return None
        placements.sort(key=lambda placement: placement.score, reverse=True)
        return placements[0]

    def _candidate_nodes(self, job: JobSpec) -> list[NodeConfig]:
        allowed = set(job.nodes)
        required = set(job.required_labels)
        result = []
        for node in self.config.nodes:
            if not node.enabled:
                continue
            if allowed and node.name not in allowed and node.ssh not in allowed:
                continue
            if required and not required.issubset(set(node.labels)):
                continue
            result.append(node)
        return result

    def _evaluate_gpu(
        self,
        node: NodeConfig,
        gpu: GpuSnapshot,
        job: JobSpec,
        estimate: ResourceEstimate,
        active_leases: list[dict[str, Any]],
    ) -> _GpuCandidate | None:
        policy = node.policy
        active_jobs = len(active_leases)
        reserved_mib = sum(int(lease["memory_mib"]) for lease in active_leases)
        allow_colocation = (
            job.resources.allow_colocation
            if job.resources.allow_colocation is not None
            else policy.allow_colocation
        )
        if active_jobs and not allow_colocation:
            return None
        if active_jobs >= policy.max_jobs_per_gpu:
            return None
        if gpu.utilization >= policy.max_gpu_utilization:
            return None

        # Current memory already includes workloads that have ramped up while
        # reservations cover workloads that have not. max() avoids counting
        # the same workload twice and remains conservative during startup.
        committed_mib = max(gpu.memory_used_mib, reserved_mib)
        available_mib = (
            gpu.memory_total_mib
            - committed_mib
            - policy.reserve_memory_mib
        )
        if available_mib < estimate.memory_mib:
            return None
        predicted_ratio = (
            committed_mib + estimate.memory_mib + policy.reserve_memory_mib
        ) / max(1, gpu.memory_total_mib)
        if predicted_ratio > policy.max_memory_used_ratio:
            return None

        if node.role == "shared":
            # Shared nodes must remain quiet for multiple consecutive samples.
            # Total usage is intentionally used here: we should not disturb
            # another user's bursty process merely because it is unregistered.
            def quiet(item: GpuSnapshot) -> bool:
                return (
                    item.utilization < policy.max_gpu_utilization
                    and item.memory_used_ratio < policy.max_memory_used_ratio
                )

            if not self.monitor.stable_for(
                node.name,
                gpu.index,
                policy.stabilization_samples,
                quiet,
            ):
                return None
            if (
                policy.cooldown_seconds > 0
                and self.monitor.seconds_since_pressure(node.name, gpu.index)
                < policy.cooldown_seconds
            ):
                return None

        role_boost = 10000 if node.role == "dedicated" else 0
        priority_score = node.priority * 100
        leftover = available_mib - estimate.memory_mib
        # Best-fit packing minimizes fragmented memory on dedicated nodes.
        memory_fit_score = -leftover / 64.0
        predicted_util = min(100, gpu.utilization + estimate.gpu_utilization)
        util_fit_score = -abs(policy.max_gpu_utilization - predicted_util) * 4
        shared_pressure_penalty = (
            (gpu.utilization * 8 + gpu.memory_used_ratio * 1000)
            if node.role == "shared"
            else 0
        )
        score = (
            role_boost
            + priority_score
            + memory_fit_score
            + util_fit_score
            - shared_pressure_penalty
        )
        reasons = [
            f"role={node.role}",
            f"priority={node.priority}",
            f"available={available_mib}MiB",
            f"util={gpu.utilization}%",
            f"active_jobs={active_jobs}",
        ]
        return _GpuCandidate(
            node=node,
            gpu=gpu,
            available_mib=available_mib,
            active_jobs=active_jobs,
            reserved_mib=reserved_mib,
            score=score,
            reasons=reasons,
        )
