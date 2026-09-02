"""Outbound webhook notifications for scheduler events.

The notifier observes the store's event stream and posts matching events to
configured webhooks from a background thread, so a slow or dead endpoint can
never stall a scheduling tick. Delivery is best effort with bounded retries;
a final failure is recorded as a ``notification_failed`` scheduler event,
which is itself never forwarded (no feedback loops).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from . import __version__
from .models import SystemConfig, WebhookConfig

# Kinds that must never leave the controller through this channel.
SUPPRESSED_KINDS = {"notification_failed"}
DEFAULT_RETRY_DELAYS = (1.0, 3.0, 9.0)


def feishu_sign(timestamp: int, secret: str) -> str:
    """Feishu custom-bot signature: HMAC-SHA256 keyed by ``"<ts>\\n<secret>"``
    over an empty message, base64-encoded."""
    key = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(key, msg=b"", digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _text_summary(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "event")
    message = str(event.get("message") or "")
    payload = event.get("payload") or {}
    lines = [f"[super_gpu] {kind}", message]
    node = payload.get("node")
    if node:
        lines.append(f"node: {node}")
    error = str(payload.get("error") or "").strip()
    if error:
        lines.append(f"error: {error[:400]}")
    counts = payload.get("job_counts")
    if isinstance(counts, dict) and counts:
        lines.append("jobs: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return "\n".join(line for line in lines if line)


def build_body(target: WebhookConfig, event: dict[str, Any], *, now: float | None = None) -> bytes:
    """Render an event in the dialect the target expects."""
    if target.kind == "slack":
        body: dict[str, Any] = {"text": _text_summary(event)}
    elif target.kind == "feishu":
        body = {"msg_type": "text", "content": {"text": _text_summary(event)}}
        if target.secret:
            timestamp = int(now if now is not None else time.time())
            body["timestamp"] = str(timestamp)
            body["sign"] = feishu_sign(timestamp, target.secret)
    else:
        body = {
            "source": "super_gpu",
            "version": __version__,
            "event": event.get("kind"),
            "message": event.get("message"),
            "created_at": event.get("created_at"),
            "payload": event.get("payload") or {},
        }
    return json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")


def deliver(target: WebhookConfig, body: bytes) -> None:
    """POST one payload; raise on any transport or application-level failure."""
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": f"super-gpu/{__version__}",
    }
    headers.update(target.headers)
    request = urllib.request.Request(target.url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=target.timeout) as response:
        status = getattr(response, "status", 200)
        text = response.read(4096).decode("utf-8", errors="replace")
    if not 200 <= int(status) < 300:
        raise RuntimeError(f"HTTP {status}: {text[:200]}")
    if target.kind == "feishu":
        # Feishu answers 200 even when it rejects the message; the JSON code
        # is authoritative (0 = accepted).
        try:
            answer = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            answer = {}
        code = answer.get("code", answer.get("StatusCode", 0))
        if code not in (0, "0", None):
            raise RuntimeError(f"feishu rejected message: code={code} {answer.get('msg', '')}")


class Notifier:
    def __init__(
        self,
        targets: list[WebhookConfig],
        *,
        on_failure: Callable[[WebhookConfig, dict[str, Any], str], None] | None = None,
        retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
        max_queue: int = 1000,
        deliver_fn: Callable[[WebhookConfig, bytes], None] = deliver,
    ) -> None:
        self.targets = [target for target in targets if target.enabled]
        self.on_failure = on_failure
        self.retry_delays = tuple(retry_delays)
        self._deliver = deliver_fn
        self._queue: queue.Queue[tuple[WebhookConfig, dict[str, Any]] | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self.dropped = 0

    @classmethod
    def from_config(cls, config: SystemConfig, **kwargs: Any) -> "Notifier":
        return cls(list(config.notifications), **kwargs)

    @property
    def active(self) -> bool:
        return bool(self.targets)

    def handle_event(self, event: dict[str, Any]) -> None:
        """Store listener entry point: filter and enqueue, never block."""
        kind = str(event.get("kind") or "")
        if not kind or kind in SUPPRESSED_KINDS or self._closed:
            return
        for target in self.targets:
            if not target.accepts(kind):
                continue
            try:
                self._queue.put_nowait((target, event))
            except queue.Full:
                self.dropped += 1
                continue
            self._ensure_worker()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="super-gpu-notifier", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                target, event = item
                self._send_with_retries(target, event)
            finally:
                self._queue.task_done()

    def _send_with_retries(self, target: WebhookConfig, event: dict[str, Any]) -> None:
        attempts = len(self.retry_delays) + 1
        last_error = ""
        for attempt in range(attempts):
            try:
                self._deliver(target, build_body(target, event))
                return
            except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
            if attempt < len(self.retry_delays):
                time.sleep(self.retry_delays[attempt])
        if self.on_failure is not None:
            try:
                self.on_failure(target, event, last_error)
            except Exception:  # noqa: BLE001 - reporting must not kill the worker
                pass

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until queued deliveries have been attempted (tests, shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            thread = self._thread
            if self._queue.unfinished_tasks == 0 or thread is None or not thread.is_alive():
                return self._queue.unfinished_tasks == 0
            time.sleep(0.02)
        return False

    def close(self, timeout: float = 10.0) -> None:
        self._closed = True
        thread = self._thread
        if thread and thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            thread.join(timeout=max(0.1, timeout))
