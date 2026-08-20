from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from super_gpu.api import create_server
from super_gpu.cli import main as cli_main
from super_gpu.config import config_from_dict, import_gpumgr_inventory, load_config, load_plan
from super_gpu.estimator import ResourceEstimator
from super_gpu.guardian import IdleGpuGuardian, WatchdogPolicy
from super_gpu.models import (
    ExperimentPlan,
    GpuProcess,
    GpuSnapshot,
    JobSpec,
    NodeConfig,
    NodeSnapshot,
    Placement,
    ResourceEstimate,
    RunnerHandle,
)
from super_gpu.monitor import ClusterMonitor, parse_nvidia_output, redact_command
from super_gpu.placement import PlacementEngine
from super_gpu.runner import PersistentRunner, RunnerStatus
from super_gpu.scheduler import Scheduler
from super_gpu.source_snapshot import SnapshotError, SourceSnapshotStore
from super_gpu.store import IdempotencyConflict, StateStore


def _config(tmp_path, *, nodes=None, default_memory=4096, max_parallel=8):
    return config_from_dict(
        {
            "database": str(tmp_path / "state.sqlite3"),
            "poll_interval": 0.2,
            "lease_ttl": 2,
            "default_memory_mib": default_memory,
            "max_parallel": max_parallel,
            "nodes": nodes
            or [
                {
                    "name": "gpu-main",
                    "ssh": "local",
                    "role": "dedicated",
                    "workspace": str(tmp_path),
                    "policy": {
                        "max_gpu_utilization": 95,
                        "max_memory_used_ratio": 0.95,
                        "reserve_memory_mib": 1024,
                        "allow_colocation": False,
                        "max_jobs_per_gpu": 1,
                    },
                }
            ],
        }
    )


def _snapshot(node="gpu-main", *, util=0, used=0, total=24576, role="dedicated"):
    return NodeSnapshot(
        node=node,
        role=role,
        reachable=True,
        gpus=[
            GpuSnapshot(
                index=0,
                uuid=f"GPU-{node}",
                name="NVIDIA Test GPU",
                utilization=util,
                memory_used_mib=used,
                memory_free_mib=total - used,
                memory_total_mib=total,
                temperature_c=42,
                power_w=80.0,
            )
        ],
    )


def _idle_snapshot(node="gpu-main", *, util=0, used=8192, role="dedicated"):
    snapshot = _snapshot(node, util=util, used=used, role=role)
    snapshot.gpus[0].processes = [
        GpuProcess(
            pid=4321,
            process_name="python train.py",
            used_memory_mib=max(1, used - 64),
            user="researcher",
        )
    ]
    return snapshot


def test_example_config_and_plan_validate(tmp_path):
    config = load_config("examples/nodes.example.json")
    plan = load_plan("examples/experiment.example.json")
    assert {node.role for node in config.nodes} == {"dedicated", "shared"}
    assert config.nodes[0].policy.allow_colocation is True
    assert config.nodes[1].policy.stabilization_samples == 3
    assert "env" not in config.public_dict()["nodes"][0]
    assert config.public_dict()["nodes"][0]["env_keys"] == []
    assert len(plan.jobs) == 3
    assert plan.jobs[2].dependencies == ["baseline"]

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one node"):
        load_config(path)

    with pytest.raises(ValueError, match="job id"):
        JobSpec.from_dict(
            {
                "id": "../../escape",
                "name": "unsafe",
                "command": "echo unsafe",
            }
        )
    with pytest.raises(ValueError, match="environment keys"):
        JobSpec.from_dict(
            {
                "name": "unsafe-env",
                "command": "echo unsafe",
                "env": {"BAD;KEY": "value"},
            }
        )


def test_import_gpumgr_inventory_defaults_to_shared_and_writes_private_config(
    tmp_path,
) -> None:
    source = tmp_path / "nodes.json"
    source.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "name": "a100-a",
                        "ssh": "a100-a",
                        "vendor": "nvidia",
                        "gpus": 8,
                        "can_vllm": True,
                        "note": "private endpoint is intentionally not copied",
                    },
                    {
                        "name": "rtx-a",
                        "ssh": "rtx-a",
                        "vendor": "rtx4090",
                        "gpus": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = import_gpumgr_inventory(source)
    config = config_from_dict(payload)

    assert [node.role for node in config.nodes] == ["shared", "shared"]
    assert config.nodes[0].labels == ["nvidia", "gpu-count-8", "vllm"]
    assert config.nodes[1].labels == ["rtx4090", "gpu-count-1"]
    assert "note" not in payload["nodes"][0]

    output = tmp_path / "private" / "config.json"
    assert cli_main(
        [
            "import-gpumgr",
            "--source",
            str(source),
            "--output",
            str(output),
        ]
    ) == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert load_config(output).nodes[0].role == "shared"
    assert cli_main(
        [
            "import-gpumgr",
            "--source",
            str(source),
            "--output",
            str(output),
        ]
    ) == 1


def test_parse_nvidia_output_with_processes():
    sep = "\x1f"
    raw = f"""
__SUPER_GPU_GPU__
0, GPU-a, NVIDIA A100-SXM4-80GB, 71, 42000, 39000, 81000, 67, 298.50
1, GPU-b, NVIDIA A100-SXM4-80GB, 0, 100, 80900, 81000, 35, [Not Supported]
__SUPER_GPU_PROCESS__
GPU-a, 1234, python3, 41000
__SUPER_GPU_PROCESS_DETAIL__
1234{sep}researcher{sep}7325{sep}/home/researcher/proj{sep}python3 train.py --lr 1e-4, --api-key hunter2
"""
    gpus = parse_nvidia_output(raw)
    assert len(gpus) == 2
    assert gpus[0].utilization == 71
    process = gpus[0].processes[0]
    assert process.pid == 1234
    assert process.user == "researcher"
    assert process.elapsed_seconds == 7325
    assert process.cwd == "/home/researcher/proj"
    assert "--lr 1e-4," in process.command
    assert "hunter2" not in process.command
    assert "[REDACTED]" in process.command
    assert gpus[1].power_w is None


def test_parse_nvidia_output_accepts_legacy_user_rows():
    raw = """
__SUPER_GPU_GPU__
0, GPU-a, NVIDIA A100-SXM4-80GB, 71, 42000, 39000, 81000, 67, 298.50
__SUPER_GPU_PROCESS__
GPU-a, 1234, python3, 41000
__SUPER_GPU_PROCESS_USER__
1234,researcher
"""
    gpus = parse_nvidia_output(raw)
    process = gpus[0].processes[0]
    assert process.user == "researcher"
    assert process.command == ""
    assert process.elapsed_seconds is None


def test_redact_command_masks_credentials():
    assert redact_command("python serve.py --token abc123 --port 8000") == (
        "python serve.py --token [REDACTED] --port 8000"
    )
    assert redact_command("run --api-key=sk-secret --safe") == (
        "run --api-key=[REDACTED] --safe"
    )


def test_watchdog_reports_unmanaged_idle_gpu_without_cancellation():
    guardian = IdleGpuGuardian(
        WatchdogPolicy(
            low_utilization_threshold=3,
            min_memory_used_mib=1024,
            grace_seconds=10,
            min_runtime_seconds=0,
            action="cancel_managed",
        )
    )
    snapshot = _idle_snapshot()

    observing = guardian.inspect([snapshot], [], [], now=100)
    assert observing.findings[0]["status"] == "observing"
    assert observing.findings[0]["kind"] == "unmanaged_idle"
    assert observing.cancel_job_ids == []

    active = guardian.inspect([snapshot], [], [], now=111)
    assert active.findings[0]["status"] == "active"
    assert active.findings[0]["automatic_action"] == "none"
    assert active.cancel_job_ids == []

    resolved = guardian.inspect([_idle_snapshot(util=40)], [], [], now=112)
    assert resolved.findings == []
    assert resolved.resolved_ids == [active.findings[0]["id"]]


def test_watchdog_cancels_only_exclusive_managed_job_after_grace():
    policy = WatchdogPolicy(
        low_utilization_threshold=3,
        min_memory_used_mib=1024,
        grace_seconds=10,
        min_runtime_seconds=30,
        action="cancel_managed",
    )
    jobs = [
        {
            "id": "job-a",
            "name": "job-a",
            "status": "running",
            "started_at": 50.0,
            "cancel_requested": False,
        },
        {
            "id": "job-b",
            "name": "job-b",
            "status": "running",
            "started_at": 50.0,
            "cancel_requested": False,
        },
    ]
    lease_a = {"job_id": "job-a", "node": "gpu-main", "gpu_index": 0}

    guardian = IdleGpuGuardian(policy)
    guardian.inspect([_idle_snapshot()], [lease_a], jobs, now=100)
    result = guardian.inspect([_idle_snapshot()], [lease_a], jobs, now=111)
    assert result.cancel_job_ids == ["job-a"]
    assert result.findings[0]["automatic_action"] == "cancel_managed"

    colocated = IdleGpuGuardian(policy)
    shared_leases = [
        lease_a,
        {"job_id": "job-b", "node": "gpu-main", "gpu_index": 0},
    ]
    colocated.inspect([_idle_snapshot()], shared_leases, jobs, now=100)
    result = colocated.inspect([_idle_snapshot()], shared_leases, jobs, now=111)
    assert result.findings[0]["kind"] == "colocated_idle"
    assert result.cancel_job_ids == []


def _watchdog_policy(**overrides):
    defaults = dict(
        low_utilization_threshold=3,
        min_memory_used_mib=1024,
        grace_seconds=10,
        min_runtime_seconds=0,
        action="report",
    )
    defaults.update(overrides)
    return WatchdogPolicy(**defaults)


def test_watchdog_idle_timer_survives_process_churn():
    guardian = IdleGpuGuardian(_watchdog_policy())
    first = guardian.inspect([_idle_snapshot()], [], [], now=100)

    churned = _idle_snapshot()
    churned.gpus[0].processes[0].pid = 9999
    active = guardian.inspect([churned], [], [], now=111)

    assert active.findings[0]["id"] == first.findings[0]["id"]
    assert active.findings[0]["status"] == "active"
    assert active.findings[0]["idle_seconds"] == pytest.approx(11.0)


def test_watchdog_idle_timer_freezes_across_sampling_gap():
    guardian = IdleGpuGuardian(_watchdog_policy())
    guardian.inspect([_idle_snapshot()], [], [], now=100)

    resumed = guardian.inspect([_idle_snapshot()], [], [], now=140)
    assert resumed.findings[0]["status"] == "observing"
    assert resumed.findings[0]["idle_seconds"] == pytest.approx(0.0)

    active = guardian.inspect([_idle_snapshot()], [], [], now=151)
    assert active.findings[0]["status"] == "active"
    assert active.findings[0]["idle_seconds"] == pytest.approx(11.0)


def test_watchdog_keeps_observations_while_node_unreachable():
    guardian = IdleGpuGuardian(_watchdog_policy())
    started = guardian.inspect([_idle_snapshot()], [], [], now=100)

    offline = _snapshot()
    offline.reachable = False
    during_outage = guardian.inspect([offline], [], [], now=105)
    assert during_outage.findings == []
    assert during_outage.resolved_ids == []

    recovered = guardian.inspect([_idle_snapshot()], [], [], now=112)
    assert recovered.findings[0]["id"] == started.findings[0]["id"]
    assert recovered.findings[0]["status"] == "active"


def test_watchdog_observation_state_survives_restart(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    policy = _watchdog_policy()
    guardian = IdleGpuGuardian(policy)
    guardian.inspect([_idle_snapshot()], [], [], now=100)
    guardian.inspect([_idle_snapshot()], [], [], now=108)
    store.save_watchdog_observations(guardian.export_observations())

    reborn = IdleGpuGuardian(policy)
    reborn.restore_observations(store.load_watchdog_observations())
    resumed = reborn.inspect([_idle_snapshot()], [], [], now=158)
    assert resumed.findings[0]["idle_seconds"] == pytest.approx(8.0)
    active = reborn.inspect([_idle_snapshot()], [], [], now=161)
    assert active.findings[0]["status"] == "active"


def test_scheduler_wires_watchdog_observation_persistence(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    store.save_watchdog_observations(
        [{"key": "k1", "node": "gpu-main", "first_seen": 1.0, "last_seen": 2.0, "samples": 3}]
    )

    scheduler = Scheduler(config, store=store)
    assert scheduler.guardian.export_observations() == [
        {"key": "k1", "node": "gpu-main", "first_seen": 1.0, "last_seen": 2.0, "samples": 3}
    ]

    scheduler._inspect_watchdog([_idle_snapshot()])
    persisted = store.load_watchdog_observations()
    assert len(persisted) == 1
    assert persisted[0]["node"] == "gpu-main"
    assert persisted[0]["key"] != "k1"


def test_scheduler_emits_node_reachability_transition_events(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    scheduler = Scheduler(config, store=store)
    up = _snapshot()
    down = _snapshot()
    down.reachable = False
    down.error = "ssh timeout"

    scheduler._track_reachability([up])
    scheduler._track_reachability([down])
    scheduler._track_reachability([down])
    scheduler._track_reachability([up])

    kinds = [event["kind"] for event in store.list_events()]
    assert kinds.count("node_offline") == 1
    assert kinds.count("node_online") == 1

    rebooted = Scheduler(config, store=store)
    rebooted._track_reachability([down])
    kinds = [event["kind"] for event in store.list_events()]
    assert kinds.count("node_offline") == 1


def test_scheduler_watchdog_requests_managed_job_cancellation(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    submitted = store.submit_plan(
        ExperimentPlan.from_dict(
            {
                "name": "watchdog-plan",
                "jobs": [{"name": "idle", "command": "python train.py"}],
            }
        )
    )
    job = submitted["jobs"][0]
    estimate = ResourceEstimate(4096, 60, None, "test", 1.0)
    assert store.acquire_placement(
        job["id"],
        Placement("gpu-main", [0], 4096, 1.0, estimate),
        config.lease_ttl,
    )
    store.mark_running(
        job["id"],
        RunnerHandle("local", "gpu-main", 1234, "/tmp/run", time.time()).as_dict(),
    )
    guardian = IdleGpuGuardian(
        WatchdogPolicy(
            low_utilization_threshold=3,
            min_memory_used_mib=1024,
            grace_seconds=0,
            min_runtime_seconds=0,
            action="cancel_managed",
        )
    )
    scheduler = Scheduler(config, store=store, guardian=guardian)

    scheduler._inspect_watchdog([_idle_snapshot()])

    cancelled = store.get_job(job["id"])
    assert cancelled["status"] == "cancelling"
    assert cancelled["cancel_requested"] is True
    assert any(
        event["kind"] == "watchdog_cancel_requested"
        for event in store.list_events()
    )


def test_shared_node_requires_stable_samples(tmp_path):
    config = _config(
        tmp_path,
        nodes=[
            {
                "name": "shared",
                "ssh": "local",
                "role": "shared",
                "workspace": str(tmp_path),
                "policy": {
                    "max_gpu_utilization": 30,
                    "max_memory_used_ratio": 0.6,
                    "reserve_memory_mib": 1024,
                    "stabilization_samples": 3,
                    "allow_colocation": False,
                    "max_jobs_per_gpu": 1,
                },
            }
        ],
        default_memory=2048,
    )
    monitor = ClusterMonitor(config)
    engine = PlacementEngine(config, monitor)
    job = JobSpec.from_dict(
        {
            "name": "quiet-only",
            "command": "python train.py",
            "nodes": ["shared"],
            "resources": {"gpus": 1, "memory_mib": 2048},
        }
    )
    estimate = ResourceEstimate(2048, 50, None, "plan", 1.0)
    snapshot = _snapshot("shared", util=5, used=500, role="shared")

    monitor._record(snapshot)
    monitor._record(snapshot)
    assert engine.choose(job, estimate, [snapshot], []) is None

    monitor._record(snapshot)
    placement = engine.choose(job, estimate, [snapshot], [])
    assert placement is not None
    assert placement.node == "shared"


def test_shared_node_cooldown_after_pressure(tmp_path):
    config = _config(
        tmp_path,
        nodes=[
            {
                "name": "shared",
                "ssh": "local",
                "role": "shared",
                "workspace": str(tmp_path),
                "policy": {
                    "max_gpu_utilization": 30,
                    "max_memory_used_ratio": 0.7,
                    "reserve_memory_mib": 512,
                    "stabilization_samples": 1,
                    "cooldown_seconds": 0.05,
                },
            }
        ],
        default_memory=1024,
    )
    monitor = ClusterMonitor(config)
    engine = PlacementEngine(config, monitor)
    job = JobSpec.from_dict(
        {
            "name": "cooldown",
            "command": "echo ok",
            "resources": {"memory_mib": 1024},
        }
    )
    estimate = ResourceEstimate(1024, 20, None, "plan", 1)
    busy = _snapshot("shared", util=90, used=1000, role="shared")
    quiet = _snapshot("shared", util=0, used=100, role="shared")
    monitor._record(busy)
    monitor._record(quiet)
    assert engine.choose(job, estimate, [quiet], []) is None
    time.sleep(0.06)
    assert engine.choose(job, estimate, [quiet], []) is not None


def test_dedicated_node_is_preferred_and_shared_pressure_blocks(tmp_path):
    config = _config(
        tmp_path,
        nodes=[
            {
                "name": "main",
                "ssh": "local",
                "role": "dedicated",
                "priority": 100,
                "workspace": str(tmp_path),
                "policy": {
                    "max_gpu_utilization": 95,
                    "max_memory_used_ratio": 0.95,
                    "reserve_memory_mib": 512,
                    "allow_colocation": True,
                    "max_jobs_per_gpu": 2,
                },
            },
            {
                "name": "shared",
                "ssh": "local",
                "role": "shared",
                "priority": 5,
                "workspace": str(tmp_path),
                "policy": {
                    "max_gpu_utilization": 25,
                    "max_memory_used_ratio": 0.5,
                    "reserve_memory_mib": 1024,
                    "stabilization_samples": 1,
                    "allow_colocation": False,
                },
            },
        ],
    )
    monitor = ClusterMonitor(config)
    main = _snapshot("main", util=55, used=8000, role="dedicated")
    shared = _snapshot("shared", util=10, used=1000, role="shared")
    monitor._record(main)
    monitor._record(shared)
    engine = PlacementEngine(config, monitor)
    job = JobSpec.from_dict(
        {
            "name": "job",
            "command": "python train.py",
            "resources": {"memory_mib": 4096},
        }
    )
    estimate = ResourceEstimate(4096, 30, None, "plan", 1)
    placement = engine.choose(job, estimate, [main, shared], [])
    assert placement is not None
    assert placement.node == "main"

    busy_shared = _snapshot("shared", util=80, used=1000, role="shared")
    monitor._record(busy_shared)
    shared_job = JobSpec.from_dict(
        {
            "name": "shared-job",
            "command": "python train.py",
            "nodes": ["shared"],
            "resources": {"memory_mib": 1024},
        }
    )
    assert engine.choose(shared_job, estimate, [busy_shared], []) is None


def test_estimator_learns_from_history(tmp_path):
    config = _config(tmp_path, default_memory=8192)
    store = StateStore(config.database)
    estimator = ResourceEstimator(config, store)
    job = JobSpec.from_dict(
        {
            "name": "train",
            "command": "python train.py --model x",
            "params": {"batch_size": 16},
            "resources": {"memory_mib": "auto"},
        }
    )
    first = estimator.estimate(job)
    assert first.source == "heuristic"
    estimator.observe(job, peak_memory_mib=12000, duration_seconds=50)
    second = estimator.estimate(job)
    assert second.source == "history"
    assert second.memory_mib >= 13800
    assert second.duration_seconds == 50


def test_store_dependency_gating(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    plan = ExperimentPlan.from_dict(
        {
            "name": "dependency-plan",
            "jobs": [
                {"name": "a", "command": "echo a"},
                {"name": "b", "command": "echo b", "dependencies": ["a"]},
            ],
        }
    )
    submitted = store.submit_plan(plan)
    runnable = store.runnable_jobs()
    assert [job["name"] for job in runnable] == ["a"]
    first = next(job for job in submitted["jobs"] if job["name"] == "a")
    store.finish_job(first["id"], status="completed", exit_code=0)
    assert [job["name"] for job in store.runnable_jobs()] == ["b"]


def test_submission_request_is_transactionally_idempotent(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    first_plan = ExperimentPlan.from_dict(
        {
            "request_id": "agent-retry-001",
            "name": "retry-safe",
            "jobs": [{"name": "train", "command": "echo first"}],
        }
    )
    first = store.submit_plan(first_plan)
    replay = store.submit_plan(first_plan)
    assert replay["id"] == first["id"]
    assert replay["submission"]["replayed"] is True
    assert len(store.list_plans()) == 1
    assert len(store.list_jobs()) == 1

    conflicting = ExperimentPlan.from_dict(
        {
            "request_id": "agent-retry-001",
            "name": "retry-safe",
            "jobs": [{"name": "train", "command": "echo changed"}],
        }
    )
    with pytest.raises(IdempotencyConflict, match="different plan intent"):
        store.submit_plan(conflicting)
    assert len(store.list_plans()) == 1


def test_source_snapshot_is_deterministic_and_rejects_escaping_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('v1')\n", encoding="utf-8")
    (source / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    snapshots = SourceSnapshotStore(tmp_path / "objects")
    first = snapshots.create(source)
    (source / "train.py").touch()
    second = snapshots.create(source)
    assert first.digest == second.digest
    assert first.file_count == 1
    assert snapshots.verify(first.digest).is_file()
    (source / "train.py").write_text("print('v2')\n", encoding="utf-8")
    assert snapshots.create(source).digest != first.digest

    (source / "outside").symlink_to(tmp_path / "outside")
    with pytest.raises(SnapshotError, match="escapes source root"):
        snapshots.create(source)


def test_failed_dependency_propagates_to_downstream_jobs(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    submitted = store.submit_plan(
        ExperimentPlan.from_dict(
            {
                "name": "blocked-plan",
                "jobs": [
                    {"name": "a", "command": "exit 1"},
                    {"name": "b", "command": "echo b", "dependencies": ["a"]},
                    {"name": "c", "command": "echo c", "dependencies": ["b"]},
                ],
            }
        )
    )
    first = next(job for job in submitted["jobs"] if job["name"] == "a")
    store.finish_job(first["id"], status="failed", exit_code=1, error="boom")
    blocked = store.fail_blocked_dependencies()
    assert {item["name"] for item in blocked} == {"b", "c"}
    plan = store.get_plan(submitted["id"], include_jobs=True)
    assert plan["status"] == "failed"
    assert all(job["status"] == "failed" for job in plan["jobs"])


class FakeMonitor:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def collect_all(self):
        return self.snapshots

    def stable_for(self, node, gpu_index, samples, predicate):
        gpu = self.snapshots[0].gpus[0]
        return predicate(gpu)


class FakeRunner:
    def __init__(self):
        self.handles = {}
        self.next_pid = 100

    def launch(self, node, job, gpu_indices, env=None):
        self.next_pid += 1
        handle = RunnerHandle("fake", node.name, self.next_pid, f"/fake/{job.id}", time.time())
        self.handles[handle.pid] = "running"
        return handle

    def poll(self, node, handle):
        state = self.handles[handle.pid]
        if state == "finished":
            return RunnerStatus("finished", 0, f"done {handle.pid}", "")
        return RunnerStatus("running", None, "", "")

    def cancel(self, node, handle):
        self.handles[handle.pid] = "finished"

    def finish_all(self):
        for pid in self.handles:
            self.handles[pid] = "finished"


class UnverifiedIdentityRunner(FakeRunner):
    def launch(self, node, job, gpu_indices, env=None):
        handle = super().launch(node, job, gpu_indices, env=env)
        handle.launch_token = "expected-token"
        handle.process_start_ticks = 123
        return handle

    def poll(self, node, handle):
        return RunnerStatus("lost", 255, "", "", "process identity mismatch")


def test_scheduler_backfills_newly_available_gpu(tmp_path):
    config = _config(tmp_path, default_memory=4096, max_parallel=2)
    store = StateStore(config.database)
    monitor = FakeMonitor([_snapshot()])
    runner = FakeRunner()
    scheduler = Scheduler(config, store=store, monitor=monitor, runner=runner)
    submitted = scheduler.submit(
        ExperimentPlan.from_dict(
            {
                "name": "backfill",
                "max_parallel": 2,
                "jobs": [
                    {"name": "first", "command": "echo first"},
                    {"name": "second", "command": "echo second"},
                ],
            }
        )
    )

    scheduler.run_once()
    jobs = store.list_jobs(plan_id=submitted["id"])
    assert [job["status"] for job in jobs].count("running") == 1
    assert [job["status"] for job in jobs].count("pending") == 1

    runner.finish_all()
    scheduler.run_once()
    jobs = store.list_jobs(plan_id=submitted["id"])
    assert [job["status"] for job in jobs].count("completed") == 1
    assert [job["status"] for job in jobs].count("running") == 1
    assert any(event["kind"] == "job_started" for event in store.list_events())
    store.release_controller(scheduler.owner)


def test_scheduler_keeps_lease_when_process_identity_is_unverified(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    scheduler = Scheduler(
        config,
        store=store,
        monitor=FakeMonitor([_snapshot()]),
        runner=UnverifiedIdentityRunner(),
    )
    submitted = scheduler.submit(
        ExperimentPlan.from_dict(
            {"name": "identity", "jobs": [{"name": "train", "command": "sleep 10"}]}
        )
    )
    scheduler.run_once()
    scheduler.run_once()
    job = store.list_jobs(plan_id=submitted["id"])[0]
    assert job["status"] == "running"
    assert store.leases(active_only=True)[0]["job_id"] == job["id"]
    assert any(event["kind"] == "job_identity_unverified" for event in store.list_events())
    store.release_controller(scheduler.owner)


def test_local_persistent_runner_writes_logs(tmp_path):
    node = NodeConfig.from_dict(
        {
            "name": "local",
            "ssh": "local",
            "role": "dedicated",
            "workspace": str(tmp_path),
        }
    )
    job = JobSpec.from_dict(
        {
            "id": "runner-test",
            "name": "runner-test",
            "command": "python3 -c 'print(\"hello-super\")'",
        }
    )
    job.id = "runner-test"
    runner = PersistentRunner()
    handle = runner.launch(node, job, [0])
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = runner.poll(node, handle)
        if status.finished:
            break
        time.sleep(0.05)
    assert status is not None
    assert status.exit_code == 0
    assert "hello-super" in status.stdout_tail


def test_runner_executes_captured_snapshot_and_refuses_wrong_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("captured\n", encoding="utf-8")
    snapshots = SourceSnapshotStore(tmp_path / "objects")
    snapshot = snapshots.create(source)
    (source / "value.txt").write_text("mutated\n", encoding="utf-8")
    node = NodeConfig.from_dict(
        {
            "name": "local",
            "ssh": "local",
            "role": "dedicated",
            "workspace": str(source),
        }
    )
    job = JobSpec.from_dict(
        {
            "id": f"snapshot-{time.time_ns()}",
            "name": "snapshot-test",
            "command": "cat value.txt; sleep 2",
        }
    )
    job.id = job.id
    runner = PersistentRunner()
    handle = runner.launch(
        node,
        job,
        [0],
        snapshot_digest=snapshot.digest,
        snapshot_path=str(snapshots.verify(snapshot.digest)),
    )
    wrong = RunnerHandle.from_dict({**handle.as_dict(), "launch_token": "wrong-token"})
    assert runner.poll(node, wrong).state == "lost"
    assert runner.cancel(node, wrong) is False
    assert runner.cancel(node, handle) is True

    deadline = time.time() + 5
    status = runner.poll(node, handle)
    while not status.finished and time.time() < deadline:
        time.sleep(0.05)
        status = runner.poll(node, handle)
    assert "captured" in status.stdout_tail
    assert "mutated" not in status.stdout_tail

    execution_file = Path(handle.run_dir) / "workspace" / "value.txt"
    execution_file.write_text("dirty-attempt\n", encoding="utf-8")
    retry = runner.launch(
        node,
        job,
        [0],
        snapshot_digest=snapshot.digest,
        snapshot_path=str(snapshots.verify(snapshot.digest)),
    )
    assert (Path(retry.run_dir) / "workspace" / "value.txt").read_text(encoding="utf-8") == "captured\n"
    assert runner.cancel(node, retry) is True


def test_runner_repeated_launch_adopts_existing_process(tmp_path):
    node = NodeConfig.from_dict(
        {
            "name": "local",
            "ssh": "local",
            "role": "dedicated",
            "workspace": str(tmp_path),
        }
    )
    job = JobSpec.from_dict(
        {
            "id": f"adopt-{time.time_ns()}",
            "name": "adopt-existing",
            "command": "sleep 10",
        }
    )
    runner = PersistentRunner()
    first = runner.launch(node, job, [0])
    replay = runner.launch(node, job, [0])
    assert replay.pid == first.pid
    assert replay.launch_token == first.launch_token
    assert replay.process_start_ticks == first.process_start_ticks
    assert replay.started_at == first.started_at
    assert runner.cancel(node, replay) is True


def test_dashboard_and_rest_plan_submission(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    store.save_snapshots([_snapshot()])
    scheduler = Scheduler(config, store=store, monitor=FakeMonitor([_snapshot()]), runner=FakeRunner())
    server = create_server(scheduler, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(base + "/", timeout=3).read().decode("utf-8")
        assert "SUPER_GPU" in html
        state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=3).read())
        assert state["ok"] is True
        assert state["snapshots"][0]["node"] == "gpu-main"
        assert state["anomalies"] == []
        anomalies = json.loads(
            urllib.request.urlopen(base + "/api/anomalies", timeout=3).read()
        )
        assert anomalies["ok"] is True
        assert anomalies["watchdog"]["policy"]["action"] == "report"

        payload = json.dumps(
            {
                "plan": {
                    "name": "api-plan",
                    "jobs": [{"name": "job", "command": "echo ok"}],
                }
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/plans",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        submitted = json.loads(urllib.request.urlopen(request, timeout=3).read())
        assert submitted["ok"] is True
        assert submitted["plan"]["name"] == "api-plan"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_rest_plan_submission_replay_and_conflict(tmp_path):
    config = _config(tmp_path)
    scheduler = Scheduler(
        config,
        store=StateStore(config.database),
        monitor=FakeMonitor([_snapshot()]),
        runner=FakeRunner(),
    )
    server = create_server(scheduler, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/api/plans"

    def submit(command):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(
                {
                    "request_id": "rest-retry-001",
                    "plan": {
                        "name": "rest-idempotency",
                        "jobs": [{"name": "job", "command": command}],
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=3)

    try:
        with submit("echo ok") as first_response:
            assert first_response.status == 201
            first = json.loads(first_response.read())
        with submit("echo ok") as replay_response:
            assert replay_response.status == 200
            replay = json.loads(replay_response.read())
        assert replay["plan"]["id"] == first["plan"]["id"]
        assert replay["plan"]["submission"]["replayed"] is True
        with pytest.raises(urllib.error.HTTPError) as conflict:
            submit("echo changed")
        assert conflict.value.code == 409
        payload = json.loads(conflict.value.read())
        assert payload["code"] == "idempotency_conflict"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
