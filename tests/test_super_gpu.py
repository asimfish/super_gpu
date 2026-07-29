from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from super_gpu.api import create_server
from super_gpu.config import config_from_dict, load_config, load_plan
from super_gpu.estimator import ResourceEstimator
from super_gpu.models import (
    ExperimentPlan,
    GpuSnapshot,
    JobSpec,
    NodeConfig,
    NodeSnapshot,
    ResourceEstimate,
    RunnerHandle,
)
from super_gpu.monitor import ClusterMonitor, parse_nvidia_output
from super_gpu.placement import PlacementEngine
from super_gpu.runner import PersistentRunner, RunnerStatus
from super_gpu.scheduler import Scheduler
from super_gpu.store import StateStore


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


def test_parse_nvidia_output_with_processes():
    raw = """
__SUPER_GPU_GPU__
0, GPU-a, NVIDIA A100-SXM4-80GB, 71, 42000, 39000, 81000, 67, 298.50
1, GPU-b, NVIDIA A100-SXM4-80GB, 0, 100, 80900, 81000, 35, [Not Supported]
__SUPER_GPU_PROCESS__
GPU-a, 1234, python3, 41000
"""
    gpus = parse_nvidia_output(raw)
    assert len(gpus) == 2
    assert gpus[0].utilization == 71
    assert gpus[0].processes[0].pid == 1234
    assert gpus[1].power_w is None


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
