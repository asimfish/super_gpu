"""History-calibrated GPU memory, utilization, and duration estimates."""
from __future__ import annotations

import math

from .models import JobSpec, ResourceEstimate, SystemConfig
from .store import StateStore


class ResourceEstimator:
    def __init__(
        self,
        config: SystemConfig,
        store: StateStore,
        *,
        memory_safety_factor: float = 1.15,
    ) -> None:
        self.config = config
        self.store = store
        self.memory_safety_factor = max(1.0, float(memory_safety_factor))

    def estimate(self, job: JobSpec) -> ResourceEstimate:
        explicit_memory = job.resources.memory_mib
        explicit_util = job.resources.gpu_utilization
        explicit_duration = job.resources.duration_seconds
        if explicit_memory is not None:
            return ResourceEstimate(
                memory_mib=explicit_memory,
                gpu_utilization=(
                    explicit_util
                    if explicit_util is not None
                    else self.config.default_gpu_utilization
                ),
                duration_seconds=explicit_duration,
                source="plan",
                confidence=1.0,
            )

        profile = self.store.get_profile(job.fingerprint())
        if profile:
            observed = int(profile["peak_memory_mib"])
            oom_count = int(profile.get("oom_count") or 0)
            oom_factor = 1.1 if oom_count else 1.0
            memory = int(math.ceil(observed * self.memory_safety_factor * oom_factor / 256) * 256)
            samples = int(profile.get("samples") or 1)
            confidence = min(0.95, 0.55 + samples * 0.08)
            return ResourceEstimate(
                memory_mib=max(256, memory),
                gpu_utilization=(
                    explicit_util
                    if explicit_util is not None
                    else self.config.default_gpu_utilization
                ),
                duration_seconds=(
                    explicit_duration
                    if explicit_duration is not None
                    else profile.get("avg_duration_seconds")
                ),
                source="history",
                confidence=confidence,
            )

        # A few common knobs provide a weak but useful first-run adjustment.
        # History takes over after the first successful observation.
        batch_size = _positive_number(job.params.get("batch_size"))
        base = self.config.default_memory_mib
        if batch_size is not None:
            # Scale gently rather than linearly; model/optimizer state is often
            # the dominant fixed cost.
            base = int(base * max(0.75, min(2.0, (batch_size / 16.0) ** 0.35)))
        memory = int(math.ceil(base / 256) * 256)
        return ResourceEstimate(
            memory_mib=max(256, memory),
            gpu_utilization=(
                explicit_util
                if explicit_util is not None
                else self.config.default_gpu_utilization
            ),
            duration_seconds=explicit_duration,
            source="heuristic",
            confidence=0.3,
        )

    def observe(
        self,
        job: JobSpec,
        *,
        peak_memory_mib: int,
        duration_seconds: float | None,
        oom: bool = False,
    ) -> None:
        self.store.record_profile(
            job.fingerprint(),
            peak_memory_mib=max(1, int(peak_memory_mib)),
            duration_seconds=duration_seconds,
            oom=oom,
        )


def _positive_number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
