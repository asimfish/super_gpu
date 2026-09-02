from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from super_gpu.config import config_from_dict
from super_gpu.models import WebhookConfig
from super_gpu.notify import Notifier, build_body, feishu_sign
from super_gpu.store import StateStore


class _Receiver:
    """Tiny HTTP sink that records every POST and can fail on demand."""

    def __init__(self, fail_first: int = 0, feishu_reject: bool = False):
        self.requests: list[dict] = []
        self.fail_first = fail_first
        self.feishu_reject = feishu_reject
        self.lock = threading.Lock()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D401 - silence test output
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                with receiver.lock:
                    receiver.requests.append(
                        {
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "json": json.loads(body.decode("utf-8")),
                        }
                    )
                    should_fail = receiver.fail_first > 0
                    if should_fail:
                        receiver.fail_first -= 1
                if should_fail:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"boom")
                    return
                answer = b'{"code": 19021, "msg": "sign mismatch"}' if receiver.feishu_reject else b'{"code": 0}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(answer)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/hook"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver():
    sink = _Receiver()
    yield sink
    sink.close()


def _event(kind="job_failed", message="job train failed on gpu-main", payload=None):
    return {
        "id": 7,
        "kind": kind,
        "message": message,
        "payload": payload if payload is not None else {"node": "gpu-main", "error": "CUDA OOM"},
        "created_at": 1_700_000_000.0,
    }


def test_feishu_sign_matches_documented_algorithm():
    timestamp = 1_700_000_000
    secret = "s3cr3t"
    expected = base64.b64encode(
        hmac.new(f"{timestamp}\n{secret}".encode(), msg=b"", digestmod=hashlib.sha256).digest()
    ).decode()
    assert feishu_sign(timestamp, secret) == expected
    assert feishu_sign(timestamp, secret) != feishu_sign(timestamp + 1, secret)


def test_build_body_per_dialect():
    event = _event(payload={"node": "gpu-main", "job_counts": {"completed": 3, "skipped": 1}})
    generic = json.loads(build_body(WebhookConfig(url="http://x/", kind="generic"), event))
    assert generic["source"] == "super_gpu"
    assert generic["event"] == "job_failed"
    assert generic["payload"]["job_counts"] == {"completed": 3, "skipped": 1}

    slack = json.loads(build_body(WebhookConfig(url="http://x/", kind="slack"), event))
    assert slack["text"].startswith("[super_gpu] job_failed")
    assert "3 completed, 1 skipped" in slack["text"]

    feishu = json.loads(
        build_body(WebhookConfig(url="http://x/", kind="feishu", secret="k"), event, now=1_700_000_000)
    )
    assert feishu["msg_type"] == "text"
    assert feishu["timestamp"] == "1700000000"
    assert feishu["sign"] == feishu_sign(1_700_000_000, "k")
    unsigned = json.loads(build_body(WebhookConfig(url="http://x/", kind="feishu"), event))
    assert "sign" not in unsigned


def test_notifier_delivers_to_every_matching_target(receiver):
    targets = [
        WebhookConfig(url=receiver.url, kind="generic", name="generic", headers={"X-Token": "abc"}),
        WebhookConfig(url=receiver.url, kind="slack", name="slack"),
        WebhookConfig(url=receiver.url, kind="feishu", name="feishu", secret="k", events=("*",)),
    ]
    notifier = Notifier(targets, retry_delays=())
    notifier.handle_event(_event())
    assert notifier.flush(timeout=10)
    notifier.close()

    kinds = sorted(
        "feishu" if "msg_type" in r["json"] else "slack" if set(r["json"]) == {"text"} else "generic"
        for r in receiver.requests
    )
    assert kinds == ["feishu", "generic", "slack"]
    generic = next(r for r in receiver.requests if r["json"].get("source") == "super_gpu")
    assert generic["headers"]["x-token"] == "abc"
    assert generic["headers"]["user-agent"].startswith("super-gpu/")
    assert generic["headers"]["content-type"].startswith("application/json")


def test_notifier_filters_by_event_kind_and_never_forwards_its_own_failures(receiver):
    target = WebhookConfig(url=receiver.url, name="only-plans", events=("plan_completed",))
    catch_all = WebhookConfig(url=receiver.url, name="all", events=("*",))
    notifier = Notifier([target, catch_all], retry_delays=())

    notifier.handle_event(_event(kind="job_started"))
    notifier.handle_event(_event(kind="notification_failed"))
    notifier.handle_event(_event(kind="plan_completed", payload={"job_counts": {"completed": 2}}))
    assert notifier.flush(timeout=10)
    notifier.close()

    received = [r["json"]["event"] for r in receiver.requests]
    assert received.count("plan_completed") == 2   # both targets
    assert received.count("job_started") == 1      # catch-all only
    assert "notification_failed" not in received


def test_notifier_retries_then_succeeds():
    sink = _Receiver(fail_first=2)
    try:
        failures = []
        notifier = Notifier(
            [WebhookConfig(url=sink.url, name="flaky")],
            retry_delays=(0.01, 0.01),
            on_failure=lambda t, e, err: failures.append(err),
        )
        notifier.handle_event(_event())
        assert notifier.flush(timeout=10)
        notifier.close()
        assert len(sink.requests) == 3
        assert failures == []
    finally:
        sink.close()


def test_notifier_reports_final_failure_and_feishu_rejection():
    dead = WebhookConfig(url="http://127.0.0.1:9/hook", name="dead", timeout=0.5)
    sink = _Receiver(feishu_reject=True)
    try:
        rejecting = WebhookConfig(url=sink.url, kind="feishu", name="feishu", secret="wrong")
        failures = []
        notifier = Notifier(
            [dead, rejecting],
            retry_delays=(0.01,),
            on_failure=lambda t, e, err: failures.append((t.name, err)),
        )
        notifier.handle_event(_event())
        assert notifier.flush(timeout=15)
        notifier.close()
    finally:
        sink.close()
    names = sorted(name for name, _ in failures)
    assert names == ["dead", "feishu"]
    feishu_error = next(err for name, err in failures if name == "feishu")
    assert "19021" in feishu_error


def test_disabled_target_when_url_env_missing_and_public_dict_redacts():
    config = config_from_dict(
        {
            "database": ":memory:",
            "nodes": [{"name": "n", "ssh": "local", "role": "dedicated"}],
            "notifications": [
                {"url_env": "SUPER_GPU_TEST_HOOK_UNSET_XYZ", "kind": "feishu", "secret": "k"},
                {"url": "https://hooks.slack.com/services/T000/B000/SECRETPART", "kind": "slack",
                 "headers": {"Authorization": "Bearer hidden"}},
            ],
        }
    )
    public = config.public_dict()["notifications"]
    assert public[0]["enabled"] is False
    assert public[0]["url_env"] == "SUPER_GPU_TEST_HOOK_UNSET_XYZ"
    assert public[1]["enabled"] is True
    assert public[1]["host"] == "hooks.slack.com"
    assert "SECRETPART" not in json.dumps(public)
    assert "hidden" not in json.dumps(public)
    assert public[1]["header_keys"] == ["Authorization"]
    assert public[0]["signed"] is True

    notifier = Notifier.from_config(config)
    assert [t.name for t in notifier.targets] == ["slack"]


def test_webhook_config_rejects_bad_values():
    with pytest.raises(ValueError):
        WebhookConfig.from_dict({"url": "ftp://example.com/x"})
    with pytest.raises(ValueError):
        WebhookConfig.from_dict({"url": "https://example.com/x", "kind": "pager"})
    with pytest.raises(ValueError):
        WebhookConfig.from_dict({})
    with pytest.raises(ValueError):
        WebhookConfig.from_dict({"url": "https://example.com/x", "events": []})


def test_store_event_listener_receives_committed_events_and_survives_errors(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    seen = []

    def bad_listener(event):
        raise RuntimeError("listener bug")

    store.add_event_listener(bad_listener)
    store.add_event_listener(seen.append)
    store.add_event("job_failed", "job x failed", {"node": "n"})

    assert len(seen) == 1
    assert seen[0]["kind"] == "job_failed"
    assert seen[0]["payload"] == {"node": "n"}
    assert seen[0]["id"] == store.list_events()[0]["id"]
