"""Agent-facing API contract: manifest, error codes, brief, capacity, wait, logs."""
from __future__ import annotations

import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from super_gpu import api_manifest
from super_gpu.api import create_server, readiness
from super_gpu.brief import build_brief, pending_cause, render_brief
from super_gpu.capacity import CapacityRequest, dry_run_capacity, preview_plan
from super_gpu.cli import main as cli_main
from super_gpu.client import SuperGPUClient, SuperGPUError
from super_gpu.config import config_from_dict
from super_gpu.estimator import ResourceEstimator
from super_gpu.logs import fetch_job_logs
from super_gpu.models import ExperimentPlan, GpuSnapshot, NodeSnapshot, RunnerHandle
from super_gpu.placement import PlacementEngine
from super_gpu.runner import PersistentRunner, RunnerStatus
from super_gpu.scheduler import Scheduler
from super_gpu.store import StateStore

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- fixtures
def _config(tmp_path, *, gpus_per_node=2, max_parallel=8, token="", nodes=None, default_memory=4096):
    return config_from_dict(
        {
            "database": str(tmp_path / "state.sqlite3"),
            "poll_interval": 0.2,
            "lease_ttl": 2,
            "default_memory_mib": default_memory,
            "max_parallel": max_parallel,
            "api_token": token,
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


def _snapshot(node="gpu-main", *, gpus=2, util=0, used=0, total=24576, role="dedicated", reachable=True):
    return NodeSnapshot(
        node=node,
        role=role,
        reachable=reachable,
        error="" if reachable else "ssh: connect to host timed out",
        gpus=[
            GpuSnapshot(
                index=index,
                uuid=f"GPU-{node}-{index}",
                name="NVIDIA Test GPU",
                utilization=util,
                memory_used_mib=used,
                memory_free_mib=total - used,
                memory_total_mib=total,
                temperature_c=40,
                power_w=60.0,
            )
            for index in range(gpus)
        ],
    )


class FakeMonitor:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def collect_all(self):
        return self.snapshots

    def stable_for(self, node, gpu_index, samples, predicate):
        for snapshot in self.snapshots:
            if snapshot.node == node:
                for gpu in snapshot.gpus:
                    if gpu.index == gpu_index:
                        return predicate(gpu)
        return False

    def seconds_since_pressure(self, node, gpu_index):
        return 10_000.0


class FakeRunner:
    """Jobs run until finish_all(); a job named oom-* fails with a CUDA OOM."""

    def __init__(self):
        self.handles = {}
        self.next_pid = 100

    def launch(self, node, job, gpu_indices, env=None, **_):
        self.next_pid += 1
        handle = RunnerHandle("fake", node.name, self.next_pid, f"/fake/{job.id}", time.time())
        self.handles[handle.pid] = ("running", job.name)
        return handle

    def poll(self, node, handle):
        state, name = self.handles[handle.pid]
        if state != "finished":
            return RunnerStatus("running", None, "", "")
        if name.startswith("oom-"):
            return RunnerStatus(
                "finished", 1, "", "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB"
            )
        return RunnerStatus("finished", 0, f"done {handle.pid}", "")

    def cancel(self, node, handle):
        _, name = self.handles[handle.pid]
        self.handles[handle.pid] = ("finished", name)

    def finish_all(self):
        for pid, (_, name) in list(self.handles.items()):
            self.handles[pid] = ("finished", name)


def _scheduler(tmp_path, **kwargs):
    config = kwargs.pop("config", None) or _config(tmp_path, **kwargs)
    store = StateStore(config.database)
    monitor = FakeMonitor([_snapshot()])
    runner = FakeRunner()
    scheduler = Scheduler(config, store=store, monitor=monitor, runner=runner)
    return scheduler, store, monitor, runner


class _Server:
    def __init__(self, scheduler):
        self.server = create_server(scheduler, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def get(self, path, *, token="", timeout=5):
        request = urllib.request.Request(self.base + path, headers=self._headers(token))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read()), dict(response.headers)

    def get_text(self, path, *, token=""):
        request = urllib.request.Request(self.base + path, headers=self._headers(token))
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.read().decode("utf-8")

    def post(self, path, payload=None, *, token=""):
        headers = self._headers(token)
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload or {}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read()), dict(response.headers)

    @staticmethod
    def _headers(token):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _error(call):
    with pytest.raises(urllib.error.HTTPError) as caught:
        call()
    payload = json.loads(caught.value.read())
    return caught.value.code, payload, dict(caught.value.headers)


# ------------------------------------------------------------- manifest
def test_meta_is_public_and_errors_carry_stable_codes(tmp_path):
    scheduler, store, *_ = _scheduler(tmp_path, token="secret-token")
    with _Server(scheduler) as server:
        status, meta, _ = server.get("/api/meta")
        assert status == 200
        assert meta["version"] == api_manifest.__version__
        assert {route["path"] for route in meta["routes"]} == api_manifest.route_paths()
        assert {item["code"] for item in meta["error_codes"]} == {
            code for code, _, _ in api_manifest.ERROR_CODES
        }
        assert [tool["name"] for tool in meta["mcp_tools"]] == [
            name for name, _ in api_manifest.MCP_TOOLS
        ]
        assert meta["capabilities"]["api_token_required"] is True

        status, health, _ = server.get("/healthz")
        assert status == 200 and health["ok"] is True

        code, payload, _ = _error(lambda: server.get("/api/state"))
        assert (code, payload["code"]) == (401, "unauthorized")

        status, state, _ = server.get("/api/state", token="secret-token")
        assert status == 200 and state["ok"] is True

        code, payload, _ = _error(lambda: server.get("/api/does-not-exist", token="secret-token"))
        assert (code, payload["code"]) == (404, "not_found")
        assert "/api/meta" in payload["error"]

        code, payload, headers = _error(lambda: server.post("/api/state", token="secret-token"))
        assert (code, payload["code"]) == (405, "method_not_allowed")
        assert headers["Allow"] == "GET"

        code, payload, _ = _error(lambda: server.get("/api/plans/nope", token="secret-token"))
        assert (code, payload["code"]) == (404, "not_found")

        code, payload, _ = _error(
            lambda: server.post("/api/plans", {"plan": {"name": "x", "jobs": []}}, token="secret-token")
        )
        assert (code, payload["code"]) == (400, "invalid_request")

        code, payload, _ = _error(lambda: server.get("/api/brief?hours=abc", token="secret-token"))
        assert (code, payload["code"]) == (400, "invalid_request")

        code, payload, _ = _error(
            lambda: server.get("/api/capacity?gpus=0", token="secret-token")
        )
        assert (code, payload["code"]) == (400, "invalid_request")


def test_readyz_turns_ready_after_first_tick(tmp_path):
    scheduler, store, *_ = _scheduler(tmp_path)
    ready, payload = readiness(scheduler)
    assert ready is False
    assert "not running" in " ".join(payload["reasons"])
    with _Server(scheduler) as server:
        code, body, _ = _error(lambda: server.get("/readyz"))
        assert (code, body["code"], body["ready"]) == (503, "not_ready", False)
        scheduler.start()
        try:
            deadline = time.time() + 5
            while scheduler.status()["tick_count"] < 1 and time.time() < deadline:
                time.sleep(0.05)
            status, body, _ = server.get("/readyz")
            assert status == 200 and body["ready"] is True
        finally:
            scheduler.stop()


def test_client_error_exposes_code_and_status(tmp_path):
    scheduler, *_ = _scheduler(tmp_path, token="tok")
    with _Server(scheduler) as server:
        client = SuperGPUClient(server.base, "wrong")
        with pytest.raises(SuperGPUError) as caught:
            client.get("/api/state")
        assert caught.value.code == "unauthorized"
        assert caught.value.status == 401
        assert caught.value.as_dict()["ok"] is False
    unreachable = SuperGPUClient("http://127.0.0.1:9", "", timeout=1)
    with pytest.raises(SuperGPUError) as caught:
        unreachable.get("/api/meta")
    assert caught.value.code == "unreachable"


# ------------------------------------------------------------- capacity
def test_capacity_dry_run_counts_slots_and_respects_cluster_limit(tmp_path):
    config = _config(tmp_path, max_parallel=1)
    store = StateStore(config.database)
    monitor = FakeMonitor([_snapshot(gpus=2)])
    placement = PlacementEngine(config, monitor)
    estimator = ResourceEstimator(config, store)
    snapshots = monitor.collect_all()

    result = dry_run_capacity(
        config, placement, estimator, snapshots, [], CapacityRequest(gpus=1, memory_mib=4096), active_jobs=0
    )
    assert result["slots"] == 2
    assert result["cluster_slots_remaining"] == 1
    assert result["effective_slots"] == 1
    assert sorted(item["gpu_indices"][0] for item in result["placements"]) == [0, 1]
    assert result["per_node"]["gpu-main"]["slots"] == 2
    assert "colocation is disabled" in result["rejections"]["gpu-main"]
    assert "reservation" in result["note"]

    # An existing lease consumes one GPU; a two-GPU request no longer fits.
    lease = [{"job_id": "j1", "node": "gpu-main", "gpu_index": 0, "memory_mib": 4096}]
    result = dry_run_capacity(
        config, placement, estimator, snapshots, lease, CapacityRequest(gpus=2, memory_mib=4096), active_jobs=1
    )
    assert result["slots"] == 0
    assert result["per_node"]["gpu-main"]["blocked_by"]

    # A budget larger than the card is rejected with the memory rule.
    result = dry_run_capacity(
        config, placement, estimator, snapshots, [], CapacityRequest(gpus=1, memory_mib=30000), active_jobs=0
    )
    assert result["slots"] == 0
    assert "free memory" in result["rejections"]["gpu-main"]

    # max_slots bounds the walk.
    result = dry_run_capacity(
        config, placement, estimator, snapshots, [], CapacityRequest(gpus=1, memory_mib=4096, max_slots=1), active_jobs=0
    )
    assert result["slots"] == 1


def test_capacity_request_parsing():
    request = CapacityRequest.from_query(
        {"gpus": ["2"], "memory_mib": ["auto"], "nodes": ["a,b"], "labels": ["x"], "allow_colocation": ["false"]}
    )
    assert (request.gpus, request.memory_mib, request.nodes, request.required_labels, request.allow_colocation) == (
        2,
        None,
        ["a", "b"],
        ["x"],
        False,
    )
    assert request.as_dict()["memory_mib"] == "auto"
    with pytest.raises(ValueError):
        CapacityRequest.from_query({"gpus": ["0"]})
    with pytest.raises(ValueError):
        CapacityRequest.from_query({"memory_mib": ["lots"]})
    with pytest.raises(ValueError):
        CapacityRequest.from_query({"allow_colocation": ["maybe"]})


def test_plan_preview_explains_each_job_without_submitting(tmp_path):
    scheduler, store, *_ = _scheduler(tmp_path)
    store.save_snapshots([_snapshot(gpus=2)])
    plan = {
        "name": "preview",
        "max_parallel": 2,
        "jobs": [
            {"name": "fits", "command": "echo a", "priority": 10, "resources": {"memory_mib": 4096}},
            {"name": "too-wide", "command": "echo b", "priority": 5, "resources": {"gpus": 4}},
            {"name": "second", "command": "echo c", "priority": 1},
            {"name": "third", "command": "echo e", "priority": 0},
            {"name": "after", "command": "echo d", "dependencies": ["fits"]},
        ],
    }
    with _Server(scheduler) as server:
        status, payload, _ = server.post("/api/plans/preview", {"plan": plan})
    assert status == 200
    preview = payload["preview"]
    by_name = {job["name"]: job for job in preview["jobs"]}
    assert [job["name"] for job in preview["jobs"]] == ["fits", "too-wide", "second", "after", "third"]
    assert by_name["fits"]["would_start_now"] is True
    assert by_name["fits"]["placement"]["node"] == "gpu-main"
    assert by_name["fits"]["fits_fleet"] is True
    assert by_name["too-wide"]["would_start_now"] is False
    assert by_name["too-wide"]["fits_fleet"] is False
    assert "4 GPUs" in by_name["too-wide"]["blocked_by"]
    assert by_name["second"]["would_start_now"] is True
    assert by_name["third"]["blocked_by"].startswith("plan max_parallel")
    assert by_name["after"]["blocked_by"].startswith("waits for dependencies")
    assert preview["immediate"] == 2 and preview["queued"] == 3
    assert any("fits no node" in warning for warning in preview["warnings"])
    assert any("no measured or declared memory budget" in warning for warning in preview["warnings"])
    assert store.list_plans() == []

    snapshot_plan = ExperimentPlan.from_dict(
        {
            "name": "snap",
            "source": {"mode": "snapshot", "path": str(tmp_path / "missing-project")},
            "jobs": [{"name": "j", "command": "echo"}],
        }
    )
    result = preview_plan(
        scheduler.config,
        scheduler.placement,
        scheduler.estimator,
        [_snapshot(gpus=2)],
        [],
        snapshot_plan,
        active_jobs=0,
    )
    assert any("not a directory" in warning for warning in result["warnings"])


# ---------------------------------------------------------------- brief
def test_pending_cause_groups_reasons():
    assert pending_cause("") == "not_evaluated_yet"
    assert pending_cause("cluster max_parallel (8) reached") == "cluster_max_parallel"
    assert pending_cause("plan max_parallel (2) reached") == "plan_max_parallel"
    assert pending_cause("waiting for dependencies: a (running)") == "dependencies"
    assert pending_cause("in retry backoff after attempt 1") == "retry_backoff"
    assert pending_cause("cluster: no enabled node matches the job's nodes/required_labels constraints") == "no_matching_node"
    assert pending_cause("gpu-main: unreachable: ssh timeout") == "nodes_unreachable"
    assert pending_cause("gpu-main: gpu0 free memory 100MiB < required 4096MiB") == "no_capacity"


def test_brief_reports_failures_blocked_jobs_and_capacity(tmp_path):
    scheduler, store, monitor, runner = _scheduler(tmp_path, max_parallel=1)
    scheduler.submit(
        ExperimentPlan.from_dict(
            {
                "name": "brief-plan",
                "jobs": [
                    {"name": "oom-train", "command": "python train.py", "priority": 10},
                    {"name": "next", "command": "python next.py"},
                ],
            }
        )
    )
    scheduler.run_once()  # oom-train starts; next blocked by cluster max_parallel
    runner.finish_all()
    scheduler.run_once()  # oom-train fails; next starts
    brief = build_brief(scheduler, window_seconds=3600)

    assert brief["status"] == "critical"  # in-process scheduler thread is not running
    assert brief["controller"]["state"] == "stopped"
    kinds = {item["kind"] for item in brief["attention"]["items"]}
    assert "job_failed" in kinds
    failed = next(item for item in brief["attention"]["items"] if item["kind"] == "job_failed")
    assert failed["ref"]["oom"] is True
    assert "memory_mib" in failed["hint"]
    assert brief["fleet"]["gpus"] == 2
    assert brief["fleet"]["gpus_leased"] == 1
    assert brief["queue"]["jobs"]["active"] == 1
    assert brief["queue"]["jobs"]["finished_in_window"] == {"failed": 1}
    assert brief["capacity"]["slots"] == 1
    assert brief["capacity"]["effective_slots"] == 0  # cluster max_parallel is 1
    assert brief["changes"]["events"]["job_failed"] == 1
    assert any("memory_mib" in tip for tip in brief["recommendations"])

    text = render_brief(brief)
    assert text.startswith("super_gpu brief ")
    assert "status CRITICAL" in text
    assert "oom-train failed" in text
    assert "capacity now: 1 slot" in text
    store.release_controller(scheduler.owner)


def test_brief_recognizes_external_controller_and_idle_cluster(tmp_path):
    scheduler, store, *_ = _scheduler(tmp_path)
    store.save_snapshots([_snapshot(gpus=2)])
    assert store.acquire_controller("other-host:1:controller-x", 60)
    brief = build_brief(scheduler)
    assert brief["controller"]["state"] == "running"
    assert brief["controller"]["external"] is True
    assert brief["controller"]["owner"] == "other-host:1:controller-x"
    assert brief["status"] == "idle"
    assert brief["attention"]["items"] == []
    assert any("submit a plan" in tip for tip in brief["recommendations"])
    text = render_brief(brief)
    assert "separate process" in text
    assert "recommendations:" in text


def test_brief_flags_offline_nodes_and_blocked_jobs_with_free_slots(tmp_path):
    config = _config(
        tmp_path,
        nodes=[
            {"name": "alive", "ssh": "local", "role": "dedicated", "workspace": str(tmp_path),
             "policy": {"allow_colocation": False, "max_jobs_per_gpu": 1}},
            {"name": "dead", "ssh": "dead", "role": "dedicated", "workspace": str(tmp_path)},
        ],
    )
    store = StateStore(config.database)
    monitor = FakeMonitor([_snapshot("alive", gpus=1), _snapshot("dead", gpus=0, reachable=False)])
    scheduler = Scheduler(config, store=store, monitor=monitor, runner=FakeRunner())
    scheduler.submit(
        ExperimentPlan.from_dict(
            {"name": "constrained", "jobs": [{"name": "needs-dead", "command": "x", "nodes": ["dead"]}]}
        )
    )
    scheduler.run_once()
    brief = build_brief(scheduler)
    kinds = {item["kind"]: item for item in brief["attention"]["items"]}
    assert "node_offline" in kinds
    assert kinds["node_offline"]["severity"] == "warning"
    assert "pending_nodes_unreachable" in kinds
    assert brief["fleet"]["offline"] == ["dead"]
    assert brief["capacity"]["slots"] == 1
    assert any("offline" in tip for tip in brief["recommendations"])
    store.release_controller(scheduler.owner)


def test_brief_endpoint_serves_json_and_text(tmp_path):
    scheduler, store, *_ = _scheduler(tmp_path)
    store.save_snapshots([_snapshot()])
    with _Server(scheduler) as server:
        status, payload, _ = server.get("/api/brief?hours=2")
        assert status == 200
        assert payload["window_seconds"] == 7200
        assert set(payload) >= {"status", "controller", "fleet", "queue", "attention", "capacity", "changes", "recommendations"}
        text = server.get_text("/api/brief?format=text")
        assert text.startswith("super_gpu brief ")
        status, capacity, _ = server.get("/api/capacity?gpus=1&memory_mib=2048")
        assert status == 200 and capacity["capacity"]["slots"] == 2


# ----------------------------------------------------------------- wait
def test_wait_endpoints_long_poll_until_terminal(tmp_path):
    scheduler, store, monitor, runner = _scheduler(tmp_path)
    submitted = scheduler.submit(
        ExperimentPlan.from_dict({"name": "wait-plan", "jobs": [{"name": "job", "command": "echo"}]})
    )
    plan_id = submitted["id"]
    job_id = store.list_jobs(plan_id=plan_id)[0]["id"]
    with _Server(scheduler) as server:
        started = time.monotonic()
        status, payload, _ = server.get(f"/api/plans/{plan_id}/wait?timeout=0.5")
        assert status == 200 and payload["terminal"] is False
        assert 0.4 <= time.monotonic() - started < 3

        def finish_later():
            time.sleep(0.3)
            scheduler.run_once()
            runner.finish_all()
            scheduler.run_once()

        worker = threading.Thread(target=finish_later)
        worker.start()
        status, payload, _ = server.get(f"/api/plans/{plan_id}/wait?timeout=10", timeout=15)
        worker.join()
        assert payload["terminal"] is True
        assert payload["plan"]["status"] == "completed"

        status, payload, _ = server.get(f"/api/jobs/{job_id}/wait?timeout=1")
        assert payload["terminal"] is True and payload["job"]["status"] == "completed"

        code, payload, _ = _error(lambda: server.get("/api/plans/missing/wait?timeout=0"))
        assert (code, payload["code"]) == (404, "not_found")
        code, payload, _ = _error(lambda: server.get(f"/api/plans/{plan_id}/wait?timeout=9999"))
        assert (code, payload["code"]) == (400, "invalid_request")
    store.release_controller(scheduler.owner)


# ----------------------------------------------------------------- logs
def test_logs_are_fetched_from_the_node_that_ran_the_job(tmp_path):
    config = _config(tmp_path)
    store = StateStore(config.database)
    scheduler = Scheduler(
        config,
        store=store,
        monitor=FakeMonitor([_snapshot()]),
        runner=PersistentRunner(),
    )
    submitted = scheduler.submit(
        ExperimentPlan.from_dict(
            {
                "name": "logs-plan",
                "jobs": [
                    {
                        "name": "chatty",
                        "command": "printf 'one\\ntwo\\nthree\\n'; echo warn >&2",
                        "resources": {"memory_mib": 1024},
                    }
                ],
            }
        )
    )
    job_id = store.list_jobs(plan_id=submitted["id"])[0]["id"]
    stored = fetch_job_logs(config, store, job_id)
    assert stored["source"] == "stored" and stored["stdout"] == ""

    scheduler.run_once()
    deadline = time.time() + 10
    while store.get_job(job_id)["status"] != "completed" and time.time() < deadline:
        time.sleep(0.1)
        scheduler.run_once()
    assert store.get_job(job_id)["status"] == "completed"

    logs = fetch_job_logs(config, store, job_id, lines=2)
    assert logs["source"] == "node"
    assert logs["stdout"].splitlines() == ["two", "three"]
    assert logs["stderr"].strip() == "warn"
    assert logs["exit_code"] == 0
    with pytest.raises(ValueError):
        fetch_job_logs(config, store, job_id, lines=0)
    with pytest.raises(ValueError):
        fetch_job_logs(config, store, "missing")

    with _Server(scheduler) as server:
        status, payload, _ = server.get(f"/api/jobs/{job_id}/logs?lines=1")
        assert status == 200 and payload["logs"]["stdout"] == "three"
        code, payload, _ = _error(lambda: server.get("/api/jobs/missing/logs"))
        assert (code, payload["code"]) == (404, "not_found")
    store.release_controller(scheduler.owner)


# ------------------------------------------------------------------ CLI
def _run_cli(argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli_main(argv)
    return code, buffer.getvalue()


def test_cli_agent_commands_against_running_controller(tmp_path):
    scheduler, store, monitor, runner = _scheduler(tmp_path)
    store.save_snapshots([_snapshot()])
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps({"name": "cli-plan", "jobs": [{"name": "job", "command": "echo hi"}]}),
        encoding="utf-8",
    )
    with _Server(scheduler) as server:
        code, out = _run_cli(["api", "/api/meta", "--url", server.base])
        assert code == 0 and json.loads(out)["api_version"] == api_manifest.API_VERSION

        code, out = _run_cli(["brief", "--url", server.base])
        assert code == 0 and out.startswith("super_gpu brief ")

        code, out = _run_cli(["brief", "--url", server.base, "--json", "--hours", "1"])
        assert code == 0 and json.loads(out)["window_seconds"] == 3600

        code, out = _run_cli(["capacity", "--url", server.base, "--gpus", "1", "--memory-mib", "2048"])
        assert code == 0 and json.loads(out)["capacity"]["slots"] == 2

        code, out = _run_cli(["preview", str(plan_file), "--url", server.base])
        assert code == 0 and json.loads(out)["preview"]["immediate"] == 1

        code, out = _run_cli(["submit", str(plan_file), "--url", server.base, "--request-id", "cli-001"])
        assert code == 0
        plan_id = json.loads(out)["plan"]["id"]

        code, out = _run_cli(["wait", plan_id, "--url", server.base, "--timeout", "0.3"])
        assert code == 3 and json.loads(out)["code"] == "timeout"

        scheduler.run_once()
        runner.finish_all()
        scheduler.run_once()
        code, out = _run_cli(["wait", plan_id, "--url", server.base, "--timeout", "5"])
        assert code == 0 and json.loads(out)["terminal"] is True

        # The fake runner recorded a run directory that does not exist, so the
        # remote tail fails and the error code says so.
        job_id = store.list_jobs(plan_id=plan_id)[0]["id"]
        code, out = _run_cli(["logs", job_id, "--url", server.base, "--lines", "5"])
        assert code == 1 and json.loads(out)["code"] == "remote_failure"

        code, out = _run_cli(["api", "/api/plans/nope", "--url", server.base])
        assert code == 1 and json.loads(out)["code"] == "not_found"
    store.release_controller(scheduler.owner)


def test_cli_local_mode_brief_and_capacity(tmp_path, monkeypatch):
    # Local mode means "no controller URL"; a developer's shell may have one.
    monkeypatch.delenv("SUPER_GPU_URL", raising=False)
    monkeypatch.delenv("SUPER_GPU_API_TOKEN", raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "database": str(tmp_path / "state.sqlite3"),
                "nodes": [{"name": "gpu-main", "ssh": "local", "role": "dedicated", "workspace": str(tmp_path)}],
            }
        ),
        encoding="utf-8",
    )
    StateStore(tmp_path / "state.sqlite3").save_snapshots([_snapshot()])
    code, out = _run_cli(["--config", str(config_path), "brief", "--json"])
    assert code == 0
    brief = json.loads(out)
    assert brief["controller"]["state"] == "stopped"
    code, out = _run_cli(["--config", str(config_path), "capacity", "--memory-mib", "1024"])
    assert code == 0 and json.loads(out)["capacity"]["slots"] >= 1
    code, out = _run_cli(["--config", str(config_path), "api", "/api/meta"])
    assert code == 0 and "routes" in json.loads(out)


# ------------------------------------------------------ contract drift
def test_readme_documents_every_route_error_code_and_tool():
    for readme in ("README.md", "README.zh-CN.md"):
        text = (ROOT / readme).read_text(encoding="utf-8")
        for _, path, _, _ in api_manifest.API_ROUTES:
            assert f"`{path}`" in text, f"{readme} lacks route {path}"
        for code, _, _ in api_manifest.ERROR_CODES:
            assert f"`{code}`" in text, f"{readme} lacks error code {code}"
        for name, _ in api_manifest.MCP_TOOLS:
            assert f"`{name}`" in text, f"{readme} lacks MCP tool {name}"


def test_agents_contract_lists_exactly_the_manifest_tools():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    section = text.split("## MCP tools", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"^- `([a-z_]+)`", section, flags=re.M)
    assert listed == [name for name, _ in api_manifest.MCP_TOOLS]


def test_mcp_server_registers_exactly_the_manifest_tools():
    pytest.importorskip("mcp")
    from super_gpu.mcp_server import create_mcp

    server = create_mcp(url="http://127.0.0.1:1")
    registered = sorted(tool.name for tool in server._tool_manager.list_tools())
    assert registered == sorted(name for name, _ in api_manifest.MCP_TOOLS)
