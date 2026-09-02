"""super_gpu: resource-aware scheduling for multi-server GPU experiments."""

from .models import (
    ExperimentPlan,
    GpuSnapshot,
    JobResources,
    JobSpec,
    NodeConfig,
    NodePolicy,
    NodeSnapshot,
    SystemConfig,
)

__all__ = [
    "ExperimentPlan",
    "GpuSnapshot",
    "JobResources",
    "JobSpec",
    "NodeConfig",
    "NodePolicy",
    "NodeSnapshot",
    "SystemConfig",
]

__version__ = "0.2.0"
