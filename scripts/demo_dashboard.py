#!/usr/bin/env python3
"""Run the super_gpu dashboard against a simulated cluster - no GPUs needed.

Seeds a three-node demo fleet (two dedicated, one shared with external
workloads), submits a small ablation plan, drives the scheduler through a few
ticks so jobs land in completed / running / pending states, then serves the
real dashboard and REST API on top of that state.

    python3 scripts/demo_dashboard.py [--port 8899]

Nothing here touches SSH or real servers; state lives in a throwaway
directory under the system temp path.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from super_gpu.api import create_server
from super_gpu.config import config_from_dict
from super_gpu.models import (
    ExperimentPlan,
    GpuProcess,
    GpuSnapshot,
    NodeSnapshot,
    RunnerHandle,
)
from super_gpu.runner import RunnerStatus
from super_gpu.scheduler import Scheduler
from super_gpu.store import StateStore

A100 = "NVIDIA A100-SXM4-80GB"
RTX4090 = "NVIDIA GeForce RTX 4090"
V100 = "Tesla V100-SXM2-32GB"


def _gpu(index, name, total, used, util, temp, power, procs=()):
    return GpuSnapshot(
        index=index,
        uuid=f"GPU-demo-{name.split()[-1]}-{index}",
        name=name,
        utilization=util,
        memory_used_mib=used,
        memory_free_mib=total - used,
        memory_total_mib=total,
        temperature_c=temp,
        power_w=power,
        processes=list(procs),
    )


def _external(pid, cmd, mem, user):
    return GpuProcess(
        pid=pid,
        process_name=cmd,
        used_memory_mib=mem,
        user=user,
        command=cmd,
        elapsed_seconds=8123,
    )


def _snapshots(busy: bool) -> list[NodeSnapshot]:
    a100 = [
        _gpu(0, A100, 81920, 68232 if busy else 0, 97 if busy else 0, 71, 341.2),
        _gpu(1, A100, 81920, 68104 if busy else 0, 95 if busy else 0, 69, 338.7),
        _gpu(2, A100, 81920, 44120 if busy else 0, 88 if busy else 0, 64, 296.4),
        _gpu(3, A100, 81920, 44236 if busy else 0, 91 if busy else 0, 66, 301.9),
        _gpu(4, A100, 81920, 512, 0, 34, 62.1),
        _gpu(5, A100, 81920, 512, 0, 33, 61.4),
        _gpu(6, A100, 81920, 512, 0, 35, 63.0),
        _gpu(7, A100, 81920, 512, 0, 33, 60.8),
    ]
    rtx = [
        _gpu(0, RTX4090, 24564, 21730 if busy else 0, 99 if busy else 0, 74, 428.5),
        _gpu(1, RTX4090, 24564, 21688 if busy else 0, 98 if busy else 0, 76, 431.0),
        _gpu(2, RTX4090, 24564, 356, 0, 38, 41.2),
        _gpu(3, RTX4090, 24564, 356, 0, 37, 39.8),
    ]
    v100 = [
        _gpu(0, V100, 32768, 30890, 92, 72, 264.3,
             [_external(48231, "python train_gan.py", 30720, "alice")]),
        _gpu(1, V100, 32768, 30412, 89, 70, 259.1,
             [_external(48232, "python train_gan.py", 30208, "alice")]),
        _gpu(2, V100, 32768, 17120, 64, 61, 188.6,
             [_external(51194, "python finetune.py", 16896, "bob")]),
        _gpu(3, V100, 32768, 812, 4, 41, 55.0),
    ]
    return [
        NodeSnapshot(node="main-a100", role="dedicated", reachable=True, gpus=a100, latency_ms=42.0),
        NodeSnapshot(node="main-4090", role="dedicated", reachable=True, gpus=rtx, latency_ms=18.0),
        NodeSnapshot(node="lab-shared-v100", role="shared", reachable=True, gpus=v100, latency_ms=63.0),
    ]


class DemoMonitor:
    """Stands in for the SSH/nvidia-smi monitor with canned telemetry."""

    def __init__(self):
        self.busy = False

    def collect_all(self):
        return _snapshots(self.busy)

    def stable_for(self, node, gpu_index, samples, predicate):
        for snapshot in _snapshots(self.busy):
            if snapshot.node == node:
                for gpu in snapshot.gpus:
                    if gpu.index == gpu_index:
                        return predicate(gpu)
        return False


class DemoRunner:
    """Stands in for the SSH runner; jobs 'run' until finish() is called."""

    def __init__(self):
        self.handles = {}
        self.next_pid = 41000

    def launch(self, node, job, gpu_indices, env=None):
        self.next_pid += 1
        handle = RunnerHandle(
            "demo", node.name, self.next_pid, f"/tmp/super-gpu-demo/{job.id}", time.time()
        )
        self.handles[handle.pid] = "running"
        return handle

    def poll(self, node, handle):
        if self.handles.get(handle.pid) == "finished":
            return RunnerStatus("finished", 0, f"done {handle.pid}", "")
        return RunnerStatus("running", None, "epoch 12/40 loss=1.842", "")

    def cancel(self, node, handle):
        self.handles[handle.pid] = "finished"

    def finish(self, count):
        for pid in list(self.handles):
            if count <= 0:
                break
            if self.handles[pid] == "running":
                self.handles[pid] = "finished"
                count -= 1


def build_config(state_dir: str):
    return config_from_dict(
        {
            "database": f"{state_dir}/state.sqlite3",
            "poll_interval": 5,
            "lease_ttl": 60,
            "default_memory_mib": 20000,
            "max_parallel": 6,
            "nodes": [
                {
                    "name": "main-a100",
                    "ssh": "main-a100",
                    "role": "dedicated",
                    "priority": 100,
                    "workspace": state_dir,
                    "policy": {
                        "max_gpu_utilization": 94,
                        "max_memory_used_ratio": 0.94,
                        "reserve_memory_mib": 2048,
                        "allow_colocation": True,
                        "max_jobs_per_gpu": 2,
                    },
                },
                {
                    "name": "main-4090",
                    "ssh": "main-4090",
                    "role": "dedicated",
                    "priority": 80,
                    "workspace": state_dir,
                    "policy": {
                        "max_gpu_utilization": 95,
                        "max_memory_used_ratio": 0.95,
                        "reserve_memory_mib": 1024,
                        "allow_colocation": False,
                        "max_jobs_per_gpu": 1,
                    },
                },
                {
                    "name": "lab-shared-v100",
                    "ssh": "lab-shared-v100",
                    "role": "shared",
                    "priority": 10,
                    "workspace": state_dir,
                    "policy": {
                        "max_gpu_utilization": 35,
                        "max_memory_used_ratio": 0.35,
                        "reserve_memory_mib": 2048,
                        "stabilization_samples": 3,
                        "allow_colocation": False,
                        "max_jobs_per_gpu": 1,
                    },
                },
            ],
        }
    )


DEMO_PLAN = {
    "request_id": "llm-scaling-ablation-001",
    "name": "llm-scaling-ablation",
    "max_parallel": 6,
    "defaults": {"max_retries": 1, "resources": {"gpus": 1, "memory_mib": 40960}},
    "jobs": [
        {"name": "baseline-1.3b", "command": "python train.py --config base.yaml",
         "priority": 30, "resources": {"gpus": 2, "memory_mib": 65536}},
        {"name": "lr-3e-4", "command": "python train.py --lr 3e-4",
         "priority": 20, "resources": {"gpus": 2, "memory_mib": 40960}},
        {"name": "lr-1e-3", "command": "python train.py --lr 1e-3",
         "priority": 20, "resources": {"gpus": 1, "memory_mib": 20480}},
        {"name": "wd-sweep-0.1", "command": "python train.py --wd 0.1",
         "priority": 20, "resources": {"gpus": 1, "memory_mib": 20480}},
        {"name": "big-batch-bs256", "command": "python train.py --bs 256",
         "priority": 10, "resources": {"gpus": 2, "memory_mib": 65536}},
        {"name": "ctx-8k-probe", "command": "python train.py --ctx 8192",
         "priority": 10, "resources": {"gpus": 1, "memory_mib": 30720}},
        {"name": "eval-suite", "command": "python eval.py --all", "priority": 5,
         "dependencies": ["baseline-1.3b", "lr-3e-4"],
         "resources": {"gpus": 1, "memory_mib": 16384}},
    ],
}


def seed(state_dir: str) -> Scheduler:
    config = build_config(state_dir)
    store = StateStore(config.database)
    monitor = DemoMonitor()
    runner = DemoRunner()
    scheduler = Scheduler(config, store=store, monitor=monitor, runner=runner)

    scheduler.submit(ExperimentPlan.from_dict(DEMO_PLAN))
    scheduler.run_once()   # place the first wave on the idle cluster
    runner.finish(2)       # two jobs complete
    scheduler.run_once()   # record completions, backfill freed GPUs
    monitor.busy = True    # switch telemetry to a busy cluster
    scheduler.run_once()   # record busy snapshots and pending reasons
    return scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    state_dir = tempfile.mkdtemp(prefix="super-gpu-demo-")
    scheduler = seed(state_dir)
    server = create_server(scheduler, host=args.host, port=args.port)
    address, port = server.server_address[:2]
    print(f"[demo] simulated cluster ready - open http://{address}:{port}")
    print("[demo] Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        scheduler.store.release_controller(scheduler.owner)
        shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
