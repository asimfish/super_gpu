"""Typed configuration and runtime models used across super_gpu."""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


TERMINAL_JOB_STATES = {"completed", "failed", "cancelled", "skipped"}
ACTIVE_JOB_STATES = {"starting", "running", "cancelling"}
VALID_NODE_ROLES = {"dedicated", "shared"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Closed taxonomy of job outcomes, richer than an exit code. Lifecycle status
# says whether a process ran to completion; result_state says what the run
# *means*: a clean training run and a run that finished only to report "this
# hypothesis is dead" both exit 0 but must route dependents differently.
RESULT_STATES = {
    "success",
    "scientific_reject",
    "execution_failure",
    "infra_failure",
    "cancelled",
    "dependency_skipped",
}
# States an application may emit via $SUPER_GPU_RESULT_FILE. Control-plane
# states (infra_failure, cancelled, dependency_skipped, execution_failure)
# are owned by the scheduler and cannot be claimed by job code.
APP_RESULT_STATES = {"success", "scientific_reject"}
DEPENDENCY_MODES = {"success", "complete", "result"}


def effective_result_state(status: str, result_state: str) -> str:
    """Resolve a job's result state, deriving it for legacy rows."""
    if result_state:
        return result_state
    return {
        "completed": "success",
        "failed": "execution_failure",
        "cancelled": "cancelled",
        "skipped": "dependency_skipped",
    }.get(status, "")


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class NodePolicy:
    """Admission and packing policy for one server.

    Dedicated nodes are intentionally aggressive. Shared nodes default to a
    conservative policy and require multiple stable samples before admitting
    a new workload.
    """

    max_gpu_utilization: int = 92
    max_memory_used_ratio: float = 0.92
    reserve_memory_mib: int = 2048
    stabilization_samples: int = 1
    allow_colocation: bool = True
    max_jobs_per_gpu: int = 2
    cooldown_seconds: float = 0.0

    @classmethod
    def defaults_for_role(cls, role: str) -> "NodePolicy":
        if role == "shared":
            return cls(
                max_gpu_utilization=35,
                max_memory_used_ratio=0.35,
                reserve_memory_mib=4096,
                stabilization_samples=3,
                allow_colocation=False,
                max_jobs_per_gpu=1,
                cooldown_seconds=20.0,
            )
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, role: str) -> "NodePolicy":
        base = asdict(cls.defaults_for_role(role))
        base.update(data or {})
        policy = cls(**base)
        policy.validate()
        return policy

    def validate(self) -> None:
        if not 0 <= int(self.max_gpu_utilization) <= 100:
            raise ValueError("policy.max_gpu_utilization must be between 0 and 100")
        if not 0 <= float(self.max_memory_used_ratio) <= 1:
            raise ValueError("policy.max_memory_used_ratio must be between 0 and 1")
        if int(self.reserve_memory_mib) < 0:
            raise ValueError("policy.reserve_memory_mib must be >= 0")
        if int(self.stabilization_samples) < 1:
            raise ValueError("policy.stabilization_samples must be >= 1")
        if int(self.max_jobs_per_gpu) < 1:
            raise ValueError("policy.max_jobs_per_gpu must be >= 1")


@dataclass
class NodeConfig:
    name: str
    ssh: str
    role: str = "shared"
    priority: int = 0
    enabled: bool = True
    workspace: str = "~/.super_gpu/work"
    labels: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    policy: NodePolicy = field(default_factory=lambda: NodePolicy.defaults_for_role("shared"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeConfig":
        role = str(data.get("role", "shared")).lower()
        node = cls(
            name=str(data.get("name") or "").strip(),
            ssh=str(data.get("ssh") or data.get("name") or "").strip(),
            role=role,
            priority=int(data.get("priority", 100 if role == "dedicated" else 0)),
            enabled=bool(data.get("enabled", True)),
            workspace=str(data.get("workspace") or "~/.super_gpu/work"),
            labels=[str(x) for x in data.get("labels", [])],
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            policy=NodePolicy.from_dict(data.get("policy"), role),
        )
        node.validate()
        return node

    def validate(self) -> None:
        if not self.name:
            raise ValueError("node.name is required")
        if not self.ssh:
            raise ValueError(f"node {self.name!r}: ssh is required")
        if self.role not in VALID_NODE_ROLES:
            raise ValueError(f"node {self.name!r}: role must be dedicated or shared")
        invalid_env = [key for key in self.env if not SAFE_ENV_KEY.fullmatch(key)]
        if invalid_env:
            raise ValueError(f"node {self.name!r}: invalid environment keys {invalid_env}")
        self.policy.validate()

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = asdict(self.policy)
        data["env_keys"] = sorted(self.env)
        data.pop("env", None)
        return data


@dataclass
class SystemConfig:
    nodes: list[NodeConfig]
    database: str = "~/.super_gpu/state.sqlite3"
    poll_interval: float = 5.0
    lease_ttl: float = 60.0
    default_memory_mib: int = 16384
    default_gpu_utilization: int = 70
    max_parallel: int = 64
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_token: str = ""

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("at least one node is required")
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("node names must be unique")
        if float(self.poll_interval) < 0.2:
            raise ValueError("poll_interval must be >= 0.2 seconds")
        if float(self.lease_ttl) < self.poll_interval * 2:
            raise ValueError("lease_ttl must be at least twice poll_interval")
        if int(self.default_memory_mib) < 1:
            raise ValueError("default_memory_mib must be positive")
        if int(self.max_parallel) < 1:
            raise ValueError("max_parallel must be positive")
        for node in self.nodes:
            node.validate()

    def node(self, name: str) -> NodeConfig:
        for node in self.nodes:
            if node.name == name or node.ssh == name:
                return node
        raise KeyError(f"unknown node {name!r}")

    def public_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.public_dict() for n in self.nodes],
            "database": self.database,
            "poll_interval": self.poll_interval,
            "lease_ttl": self.lease_ttl,
            "default_memory_mib": self.default_memory_mib,
            "default_gpu_utilization": self.default_gpu_utilization,
            "max_parallel": self.max_parallel,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_token_configured": bool(self.api_token),
        }


@dataclass
class JobResources:
    gpus: int = 1
    memory_mib: int | None = None
    gpu_utilization: int | None = None
    duration_seconds: float | None = None
    allow_colocation: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "JobResources":
        raw = data or {}
        memory = raw.get("memory_mib", raw.get("gpu_memory_mib"))
        if isinstance(memory, str) and memory.lower() == "auto":
            memory = None
        util = raw.get("gpu_utilization")
        if isinstance(util, str) and util.lower() == "auto":
            util = None
        duration = raw.get("duration_seconds")
        if isinstance(duration, str) and duration.lower() == "auto":
            duration = None
        resources = cls(
            gpus=int(raw.get("gpus", raw.get("num_gpus", 1))),
            memory_mib=int(memory) if memory is not None else None,
            gpu_utilization=int(util) if util is not None else None,
            duration_seconds=float(duration) if duration is not None else None,
            allow_colocation=(
                bool(raw["allow_colocation"]) if "allow_colocation" in raw else None
            ),
        )
        resources.validate()
        return resources

    def validate(self) -> None:
        if self.gpus < 1:
            raise ValueError("resources.gpus must be >= 1")
        if self.memory_mib is not None and self.memory_mib < 1:
            raise ValueError("resources.memory_mib must be positive or auto")
        if self.gpu_utilization is not None and not 0 <= self.gpu_utilization <= 100:
            raise ValueError("resources.gpu_utilization must be between 0 and 100")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("resources.duration_seconds must be positive")


@dataclass
class JobDependency:
    """One dependency edge with an explicit satisfaction predicate.

    ``after`` semantics:
      - ``success``  (default): upstream completed with result state success.
      - ``complete``: upstream reached any terminal state.
      - ``result``: upstream terminal and its result state is listed in
        ``result_states``.
    When the upstream is terminal but the predicate cannot hold, the
    dependent job is skipped with result state ``dependency_skipped``.
    """

    job: str
    after: str = "success"
    result_states: list[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: Any) -> "JobDependency":
        if isinstance(value, str):
            dependency = cls(job=value)
        elif isinstance(value, dict):
            dependency = cls(
                job=str(value.get("job") or "").strip(),
                after=str(value.get("after") or ("result" if value.get("result_states") else "success")),
                result_states=[str(item) for item in value.get("result_states", [])],
            )
        else:
            raise ValueError(f"dependency entries must be strings or objects, got {value!r}")
        dependency.validate()
        return dependency

    def validate(self) -> None:
        if not self.job:
            raise ValueError("dependency.job is required")
        if self.after not in DEPENDENCY_MODES:
            raise ValueError(
                f"dependency on {self.job!r}: after must be one of {sorted(DEPENDENCY_MODES)}"
            )
        if self.after == "result":
            if not self.result_states:
                raise ValueError(
                    f"dependency on {self.job!r}: after=result requires result_states"
                )
            unknown = set(self.result_states) - RESULT_STATES
            if unknown:
                raise ValueError(
                    f"dependency on {self.job!r}: unknown result_states {sorted(unknown)}"
                )
        elif self.result_states:
            raise ValueError(
                f"dependency on {self.job!r}: result_states requires after=result"
            )

    def as_value(self) -> Any:
        """Compact serialization: plain string for the default predicate."""
        if self.after == "success":
            return self.job
        payload: dict[str, Any] = {"job": self.job, "after": self.after}
        if self.after == "result":
            payload["result_states"] = list(self.result_states)
        return payload

    def gate(self, status: str, result_state: str) -> str:
        """Return 'satisfied', 'wait', or 'skip' for the upstream's state."""
        terminal = status in TERMINAL_JOB_STATES
        if self.after == "complete":
            return "satisfied" if terminal else "wait"
        if not terminal:
            return "wait"
        effective = effective_result_state(status, result_state)
        if self.after == "success":
            return "satisfied" if effective == "success" else "skip"
        return "satisfied" if effective in self.result_states else "skip"


@dataclass
class JobSpec:
    name: str
    command: str
    id: str = ""
    priority: int = 0
    nodes: list[str] = field(default_factory=list)
    required_labels: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    resources: JobResources = field(default_factory=JobResources)
    timeout: float = 86400.0
    max_retries: int = 0
    retry_delay: float = 10.0
    dependencies: list[JobDependency] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], defaults: dict[str, Any] | None = None) -> "JobSpec":
        merged = dict(defaults or {})
        merged.update(data)
        resources = dict((defaults or {}).get("resources") or {})
        resources.update(data.get("resources") or {})
        nodes = merged.get("nodes") or ([merged["node"]] if merged.get("node") else [])
        job = cls(
            id=str(merged.get("id") or ""),
            name=str(merged.get("name") or "").strip(),
            command=str(merged.get("command") or "").strip(),
            priority=int(merged.get("priority", 0)),
            nodes=[str(x) for x in nodes],
            required_labels=[str(x) for x in merged.get("required_labels", [])],
            params=dict(merged.get("params") or {}),
            env={str(k): str(v) for k, v in (merged.get("env") or {}).items()},
            resources=JobResources.from_dict(resources),
            timeout=float(merged.get("timeout", 86400)),
            max_retries=int(merged.get("max_retries", 0)),
            retry_delay=float(merged.get("retry_delay", 10)),
            dependencies=[
                JobDependency.from_value(entry)
                for entry in merged.get("dependencies", [])
            ],
            outputs=[str(x) for x in merged.get("outputs", [])],
        )
        job.validate()
        return job

    def validate(self) -> None:
        if self.id and not SAFE_ID.fullmatch(self.id):
            raise ValueError(
                f"job id {self.id!r} must contain only letters, digits, dot, underscore, or dash"
            )
        if not self.name:
            raise ValueError("job.name is required")
        if not self.command:
            raise ValueError(f"job {self.name!r}: command is required")
        if self.timeout <= 0:
            raise ValueError(f"job {self.name!r}: timeout must be positive")
        if self.max_retries < 0:
            raise ValueError(f"job {self.name!r}: max_retries must be >= 0")
        invalid_env = [key for key in self.env if not SAFE_ENV_KEY.fullmatch(key)]
        if invalid_env:
            raise ValueError(f"job {self.name!r}: invalid environment keys {invalid_env}")
        for dependency in self.dependencies:
            dependency.validate()
        for pattern in self.outputs:
            if not pattern.strip():
                raise ValueError(f"job {self.name!r}: output patterns must be non-empty")
            if pattern.startswith("/") or pattern.startswith("~"):
                raise ValueError(
                    f"job {self.name!r}: output pattern {pattern!r} must be workspace-relative"
                )
            if ".." in pattern.split("/"):
                raise ValueError(
                    f"job {self.name!r}: output pattern {pattern!r} must not contain '..'"
                )
            if any(ch.isspace() for ch in pattern):
                raise ValueError(
                    f"job {self.name!r}: output pattern {pattern!r} must not contain whitespace"
                )
        self.resources.validate()

    def fingerprint(self) -> str:
        payload = {
            "command": " ".join(self.command.split()),
            "params": self.params,
            "gpus": self.resources.gpus,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ExperimentPlan:
    name: str
    jobs: list[JobSpec]
    id: str = ""
    max_parallel: int | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    source_snapshot: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentPlan":
        defaults = dict(data.get("defaults") or {})
        jobs = [JobSpec.from_dict(dict(item), defaults) for item in data.get("jobs", [])]
        plan = cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or "").strip(),
            jobs=jobs,
            max_parallel=int(data["max_parallel"]) if data.get("max_parallel") is not None else None,
            defaults=defaults,
            request_id=str(data.get("request_id") or ""),
            source=dict(data.get("source") or {}),
            source_snapshot=dict(data.get("source_snapshot") or {}),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.id and not SAFE_ID.fullmatch(self.id):
            raise ValueError(
                f"plan id {self.id!r} must contain only letters, digits, dot, underscore, or dash"
            )
        if not self.name:
            raise ValueError("plan.name is required")
        if not self.jobs:
            raise ValueError("plan.jobs must contain at least one job")
        if self.max_parallel is not None and self.max_parallel < 1:
            raise ValueError("plan.max_parallel must be positive")
        if self.request_id and not SAFE_ID.fullmatch(self.request_id):
            raise ValueError(
                "request_id must contain only letters, digits, dot, underscore, or dash"
            )
        source_mode = str(self.source.get("mode") or "workspace")
        if source_mode not in {"workspace", "snapshot"}:
            raise ValueError("source.mode must be workspace or snapshot")
        if source_mode == "snapshot" and not str(self.source.get("path") or "").strip():
            raise ValueError("source.path is required when source.mode is snapshot")
        excludes = self.source.get("exclude", [])
        if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
            raise ValueError("source.exclude must be an array of strings")
        names = [job.name for job in self.jobs]
        if len(names) != len(set(names)):
            raise ValueError("job names must be unique within a plan")
        known = set(names)
        for job in self.jobs:
            missing = {dependency.job for dependency in job.dependencies} - known
            if missing:
                raise ValueError(f"job {job.name!r}: unknown dependencies {sorted(missing)}")


@dataclass
class GpuProcess:
    pid: int
    process_name: str
    used_memory_mib: int
    user: str = ""
    command: str = ""
    elapsed_seconds: int | None = None
    cwd: str = ""


@dataclass
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    utilization: int
    memory_used_mib: int
    memory_free_mib: int
    memory_total_mib: int
    temperature_c: int | None = None
    power_w: float | None = None
    processes: list[GpuProcess] = field(default_factory=list)

    @property
    def memory_used_ratio(self) -> float:
        if not self.memory_total_mib:
            return 1.0
        return self.memory_used_mib / self.memory_total_mib

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_used_ratio"] = round(self.memory_used_ratio, 4)
        data["processes"] = [asdict(p) for p in self.processes]
        return data


@dataclass
class NodeSnapshot:
    node: str
    role: str
    reachable: bool
    gpus: list[GpuSnapshot] = field(default_factory=list)
    error: str = ""
    collected_at: float = field(default_factory=now_ts)
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "role": self.role,
            "reachable": self.reachable,
            "gpus": [gpu.as_dict() for gpu in self.gpus],
            "error": self.error,
            "collected_at": self.collected_at,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class ResourceEstimate:
    memory_mib: int
    gpu_utilization: int
    duration_seconds: float | None
    source: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Placement:
    node: str
    gpu_indices: list[int]
    memory_mib_per_gpu: int
    score: float
    estimate: ResourceEstimate
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["estimate"] = self.estimate.as_dict()
        return data


@dataclass
class RunnerHandle:
    kind: str
    node: str
    pid: int
    run_dir: str
    started_at: float
    launch_token: str = ""
    process_start_ticks: int = 0
    snapshot_digest: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunnerHandle":
        return cls(
            kind=str(data["kind"]),
            node=str(data["node"]),
            pid=int(data["pid"]),
            run_dir=str(data["run_dir"]),
            started_at=float(data["started_at"]),
            launch_token=str(data.get("launch_token") or ""),
            process_start_ticks=int(data.get("process_start_ticks") or 0),
            snapshot_digest=str(data.get("snapshot_digest") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
