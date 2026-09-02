"""End-to-end test over real SSH against docker "GPU nodes".

Two containers run sshd plus a fake nvidia-smi, so the full production path
is exercised: ClusterMonitor over SSH, source snapshot upload with digest
verification, PersistentRunner launch/poll with process identity, typed
results, dependency routing and skip cascade, and declared-output pull.

Opt in with SUPER_GPU_E2E=1 (requires a running docker daemon):

    SUPER_GPU_E2E=1 python -m pytest tests/test_e2e_docker.py -v

The ssh binary is wrapped through a PATH shim that injects -F <test config>,
so the product code keeps its normal "aliases come from ssh config" contract
and nothing on the host machine is modified.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from super_gpu.config import config_from_dict
from super_gpu.models import ExperimentPlan
from super_gpu.outputs import describe_job_outputs, pull_job_outputs
from super_gpu.scheduler import Scheduler

pytestmark = pytest.mark.skipif(
    os.environ.get("SUPER_GPU_E2E") != "1",
    reason="docker end-to-end test; set SUPER_GPU_E2E=1 with a docker daemon running",
)

E2E_DIR = Path(__file__).parent / "e2e"
IMAGE = "super-gpu-e2e-node"
NODES = ("e2e-node-a", "e2e-node-b")


def _run(argv, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, check=True, **kwargs)


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    if shutil.which("docker") is None:
        pytest.skip("docker binary not found")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("docker daemon not running")

    root = tmp_path_factory.mktemp("e2e-fleet")
    _run(["docker", "build", "-t", IMAGE, str(E2E_DIR)])

    key = root / "id_ed25519"
    _run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key)])
    pubkey = (root / "id_ed25519.pub").read_text().strip()

    containers = []
    ports = {}
    try:
        for alias in NODES:
            cid = _run(
                [
                    "docker", "run", "--detach", "--rm",
                    "-e", f"SSH_PUBKEY={pubkey}",
                    "-p", "127.0.0.1::22",
                    IMAGE,
                ]
            ).stdout.strip()
            containers.append(cid)
            mapped = _run(["docker", "port", cid, "22/tcp"]).stdout.strip().splitlines()[0]
            ports[alias] = int(mapped.rsplit(":", 1)[1])

        ssh_config = root / "ssh_config"
        ssh_config.write_text(
            "".join(
                f"Host {alias}\n"
                f"  HostName 127.0.0.1\n"
                f"  Port {ports[alias]}\n"
                f"  User sguser\n"
                f"  IdentityFile {key}\n"
                f"  IdentitiesOnly yes\n"
                f"  StrictHostKeyChecking no\n"
                f"  UserKnownHostsFile /dev/null\n"
                f"  LogLevel ERROR\n\n"
                for alias in NODES
            )
        )

        real_ssh = shutil.which("ssh")
        shim_dir = root / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "ssh"
        shim.write_text(f'#!/bin/sh\nexec "{real_ssh}" -F "{ssh_config}" "$@"\n')
        shim.chmod(0o755)

        original_path = os.environ["PATH"]
        os.environ["PATH"] = f"{shim_dir}{os.pathsep}{original_path}"
        try:
            for alias in NODES:
                deadline = time.time() + 90
                while True:
                    probe = subprocess.run(
                        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                         alias, "true"],
                        capture_output=True,
                    )
                    if probe.returncode == 0:
                        break
                    if time.time() > deadline:
                        raise RuntimeError(
                            f"sshd in {alias} never became reachable: "
                            f"{probe.stderr.decode(errors='replace')}"
                        )
                    time.sleep(1)
            yield {"root": root}
        finally:
            os.environ["PATH"] = original_path
    finally:
        for cid in containers:
            subprocess.run(["docker", "stop", cid], capture_output=True)


def _build_project(root: Path) -> Path:
    project = root / "project"
    project.mkdir()
    (project / "train.py").write_text(
        "import json, os, pathlib, sys\n"
        "mode = sys.argv[1]\n"
        "pathlib.Path('results').mkdir(exist_ok=True)\n"
        "(pathlib.Path('results') / (mode + '.json')).write_text(\n"
        "    json.dumps({'mode': mode, 'ok': True}))\n"
        "if mode == 'probe':\n"
        "    pathlib.Path(os.environ['SUPER_GPU_RESULT_FILE']).write_text(\n"
        "        json.dumps({'state': 'scientific_reject', 'metrics': {'val': 0.5}}))\n"
        "print('done', mode)\n"
    )
    return project


def _config(root: Path):
    return config_from_dict(
        {
            "database": str(root / "state.sqlite3"),
            "poll_interval": 1,
            "lease_ttl": 30,
            "default_memory_mib": 4096,
            "max_parallel": 4,
            "nodes": [
                {
                    "name": "node-a",
                    "ssh": "e2e-node-a",
                    "role": "dedicated",
                    "priority": 100,
                    "workspace": "~/work",
                    "policy": {
                        "max_gpu_utilization": 95,
                        "max_memory_used_ratio": 0.95,
                        "reserve_memory_mib": 512,
                        "allow_colocation": True,
                        "max_jobs_per_gpu": 2,
                    },
                },
                {
                    "name": "node-b",
                    "ssh": "e2e-node-b",
                    "role": "shared",
                    "priority": 10,
                    "workspace": "~/work",
                },
            ],
        }
    )


def test_full_plan_lifecycle_over_ssh(fleet, tmp_path):
    project = _build_project(tmp_path)
    config = _config(tmp_path)
    scheduler = Scheduler(config)
    try:
        submitted = scheduler.submit(
            ExperimentPlan.from_dict(
                {
                    "request_id": "e2e-lifecycle-001",
                    "name": "e2e-lifecycle",
                    "source": {"mode": "snapshot", "path": str(project)},
                    "max_parallel": 4,
                    "defaults": {
                        "max_retries": 0,
                        "resources": {"gpus": 1, "memory_mib": 4096},
                    },
                    "jobs": [
                        {
                            "name": "probe",
                            "command": "python3 train.py probe",
                            "outputs": ["results/*.json"],
                        },
                        {
                            "name": "train",
                            "command": "python3 train.py train",
                            "outputs": ["results/*.json"],
                        },
                        {
                            "name": "scale-up",
                            "command": "python3 train.py scale",
                            "dependencies": [
                                {
                                    "job": "probe",
                                    "after": "result",
                                    "result_states": ["success"],
                                }
                            ],
                        },
                        {
                            "name": "report",
                            "command": "python3 train.py report",
                            "dependencies": ["train"],
                        },
                    ],
                }
            )
        )

        deadline = time.time() + 240
        terminal = {"completed", "failed", "cancelled", "skipped"}
        while True:
            scheduler.run_once()
            jobs = scheduler.store.list_jobs(plan_id=submitted["id"])
            if all(job["status"] in terminal for job in jobs):
                break
            if time.time() > deadline:
                states = {job["name"]: job["status"] for job in jobs}
                pytest.fail(f"plan did not reach terminal state in time: {states}")
            time.sleep(1)

        by_name = {job["name"]: job for job in jobs}
        assert by_name["probe"]["status"] == "completed"
        assert by_name["probe"]["result_state"] == "scientific_reject"
        assert by_name["train"]["status"] == "completed"
        assert by_name["report"]["status"] == "completed"
        assert by_name["scale-up"]["status"] == "skipped"
        assert by_name["scale-up"]["result_state"] == "dependency_skipped"

        described = describe_job_outputs(config, scheduler.store, by_name["train"]["id"])
        assert "results/train.json" in described["files"]

        dest = tmp_path / "pulled"
        pull_job_outputs(config, scheduler.store, by_name["train"]["id"], dest)
        pulled = list(dest.rglob("train.json"))
        assert pulled, f"expected pulled output under {dest}"
        assert json.loads(pulled[0].read_text()) == {"mode": "train", "ok": True}
    finally:
        scheduler.store.release_controller(scheduler.owner)


def test_snapshot_reupload_is_cached(fleet, tmp_path):
    project = _build_project(tmp_path)
    config = _config(tmp_path)
    scheduler = Scheduler(config)
    try:
        first = scheduler.source_snapshots.create(str(project), [])
        second = scheduler.source_snapshots.create(str(project), [])
        assert first.digest == second.digest, "snapshot digests must be deterministic"

        node = config.node("node-a")
        archive = scheduler.source_snapshots.archive_path(first.digest)
        destination = f".super_gpu/source-snapshots/{first.digest}.tar.gz"
        uploaded = scheduler.monitor.transport.upload_file(
            node, archive, destination, expected_sha256=first.digest
        )
        assert uploaded.ok
        cached = scheduler.monitor.transport.upload_file(
            node, archive, destination, expected_sha256=first.digest
        )
        assert cached.ok
        assert "cached" in cached.stdout, "second upload of same digest must be a cache hit"
    finally:
        scheduler.store.release_controller(scheduler.owner)
