"""Concurrent multi-server NVIDIA GPU monitoring."""
from __future__ import annotations

import math
import re
import shlex
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .models import GpuProcess, GpuSnapshot, NodeConfig, NodeSnapshot, SystemConfig
from .transport import CommandTransport


GPU_MARKER = "__SUPER_GPU_GPU__"
PROCESS_MARKER = "__SUPER_GPU_PROCESS__"
PROCESS_USER_MARKER = "__SUPER_GPU_PROCESS_USER__"
PROCESS_DETAIL_MARKER = "__SUPER_GPU_PROCESS_DETAIL__"
# Detail rows use the ASCII unit separator so full command lines containing
# commas or spaces survive the round trip.
DETAIL_SEPARATOR = "\x1f"
QUERY_SCRIPT = f"""
set +e
echo {GPU_MARKER}
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
gpu_rc=$?
echo {PROCESS_MARKER}
proc_out=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)
printf '%s\n' "$proc_out"
echo {PROCESS_DETAIL_MARKER}
printf '%s\n' "$proc_out" | awk -F, '{{gsub(/[[:space:]]/, "", $2); if ($2 ~ /^[0-9]+$/) print $2}}' | sort -u | while read -r pid; do
  user=$(ps -o user= -p "$pid" 2>/dev/null | awk '{{$1=$1; print}}')
  etimes=$(ps -o etimes= -p "$pid" 2>/dev/null | awk '{{$1=$1; print}}')
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
  args=$(ps -ww -o args= -p "$pid" 2>/dev/null)
  printf '%s\037%s\037%s\037%s\037%s\n' "$pid" "$user" "$etimes" "$cwd" "$args"
done
exit "$gpu_rc"
""".strip()

_SENSITIVE_ARG = re.compile(
    r"(api[-_]?key|access[-_]?key|secret|token|password|passwd|credential)", re.I
)


def redact_command(command: str) -> str:
    """Mask values of credential-looking CLI arguments before storage."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    redacted: list[str] = []
    hide_next = False
    for token in tokens:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        if _SENSITIVE_ARG.search(token):
            if "=" in token:
                redacted.append(f"{token.split('=', 1)[0]}=[REDACTED]")
            else:
                redacted.append(token)
                hide_next = True
            continue
        redacted.append(token)
    return " ".join(redacted)


def _number(value: str, *, integer: bool = True) -> int | float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return int(parsed) if integer else parsed


def parse_nvidia_output(stdout: str) -> list[GpuSnapshot]:
    section = ""
    gpu_rows: list[str] = []
    process_rows: list[str] = []
    process_user_rows: list[str] = []
    process_detail_rows: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if line == GPU_MARKER:
            section = "gpu"
            continue
        if line == PROCESS_MARKER:
            section = "process"
            continue
        if line == PROCESS_USER_MARKER:
            section = "process_user"
            continue
        if line == PROCESS_DETAIL_MARKER:
            section = "process_detail"
            continue
        if not line:
            continue
        if section == "gpu":
            gpu_rows.append(line)
        elif section == "process":
            process_rows.append(line)
        elif section == "process_user":
            process_user_rows.append(line)
        elif section == "process_detail":
            process_detail_rows.append(line)

    process_users: dict[int, str] = {}
    for row in process_user_rows:
        parts = [part.strip() for part in row.split(",", 1)]
        if len(parts) != 2:
            continue
        pid = _number(parts[0])
        if pid is not None:
            process_users[int(pid)] = parts[1]

    process_details: dict[int, dict[str, str | int | None]] = {}
    for row in process_detail_rows:
        parts = row.split(DETAIL_SEPARATOR, 4)
        if len(parts) != 5:
            continue
        pid = _number(parts[0])
        if pid is None:
            continue
        elapsed = _number(parts[2])
        process_details[int(pid)] = {
            "user": parts[1].strip(),
            "elapsed_seconds": int(elapsed) if elapsed is not None else None,
            "cwd": parts[3].strip(),
            "command": redact_command(parts[4].strip()),
        }

    processes: dict[str, list[GpuProcess]] = defaultdict(list)
    for row in process_rows:
        parts = [part.strip() for part in row.split(",", 3)]
        if len(parts) != 4:
            continue
        pid = _number(parts[1])
        memory = _number(parts[3])
        if pid is None or memory is None:
            continue
        detail = process_details.get(int(pid), {})
        processes[parts[0]].append(
            GpuProcess(
                pid=int(pid),
                process_name=parts[2],
                used_memory_mib=int(memory),
                user=str(detail.get("user") or process_users.get(int(pid), "")),
                command=str(detail.get("command") or ""),
                elapsed_seconds=detail.get("elapsed_seconds"),  # type: ignore[arg-type]
                cwd=str(detail.get("cwd") or ""),
            )
        )

    result: list[GpuSnapshot] = []
    for row in gpu_rows:
        parts = [part.strip() for part in row.split(",", 8)]
        if len(parts) != 9:
            continue
        index = _number(parts[0])
        util = _number(parts[3])
        used = _number(parts[4])
        free = _number(parts[5])
        total = _number(parts[6])
        if None in {index, util, used, free, total}:
            continue
        temperature = _number(parts[7])
        power = _number(parts[8], integer=False)
        result.append(
            GpuSnapshot(
                index=int(index),
                uuid=parts[1],
                name=parts[2],
                utilization=int(util),
                memory_used_mib=int(used),
                memory_free_mib=int(free),
                memory_total_mib=int(total),
                temperature_c=int(temperature) if temperature is not None else None,
                power_w=float(power) if power is not None else None,
                processes=processes.get(parts[1], []),
            )
        )
    return result


class ClusterMonitor:
    def __init__(
        self,
        config: SystemConfig,
        transport: CommandTransport | None = None,
        *,
        history_size: int = 20,
    ) -> None:
        self.config = config
        self.transport = transport or CommandTransport()
        self._history_size = max(3, history_size)
        self._history: dict[tuple[str, int], deque[GpuSnapshot]] = {}
        self._latest: dict[str, NodeSnapshot] = {}
        self._last_pressure: dict[tuple[str, int], float] = {}
        self._lock = threading.RLock()

    def collect_node(self, node: NodeConfig) -> NodeSnapshot:
        result = self.transport.run_script(node, QUERY_SCRIPT, timeout=25)
        gpus = parse_nvidia_output(result.stdout)
        reachable = result.ok and bool(gpus)
        error = ""
        if not reachable:
            error = (result.stderr or result.stdout or f"nvidia-smi rc={result.returncode}").strip()
        snapshot = NodeSnapshot(
            node=node.name,
            role=node.role,
            reachable=reachable,
            gpus=gpus,
            error=error[-1000:],
            latency_ms=result.duration_seconds * 1000,
        )
        self._record(snapshot)
        return snapshot

    def collect_all(self) -> list[NodeSnapshot]:
        nodes = [node for node in self.config.nodes if node.enabled]
        with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
            snapshots = list(pool.map(self.collect_node, nodes))
        return snapshots

    def _record(self, snapshot: NodeSnapshot) -> None:
        with self._lock:
            self._latest[snapshot.node] = snapshot
            if not snapshot.reachable:
                return
            try:
                node_cfg = self.config.node(snapshot.node)
            except KeyError:
                node_cfg = None
            for gpu in snapshot.gpus:
                key = (snapshot.node, gpu.index)
                history = self._history.setdefault(key, deque(maxlen=self._history_size))
                history.append(gpu)
                if (
                    node_cfg is not None
                    and node_cfg.role == "shared"
                    and (
                        gpu.utilization >= node_cfg.policy.max_gpu_utilization
                        or gpu.memory_used_ratio >= node_cfg.policy.max_memory_used_ratio
                    )
                ):
                    self._last_pressure[key] = time.time()

    def latest(self) -> list[NodeSnapshot]:
        with self._lock:
            return list(self._latest.values())

    def stable_for(
        self,
        node: str,
        gpu_index: int,
        samples: int,
        predicate: Callable[[GpuSnapshot], bool],
    ) -> bool:
        with self._lock:
            history = list(self._history.get((node, gpu_index), ()))
        required = max(1, int(samples))
        if len(history) < required:
            return False
        return all(predicate(item) for item in history[-required:])

    def seconds_since_pressure(self, node: str, gpu_index: int) -> float:
        with self._lock:
            timestamp = self._last_pressure.get((node, gpu_index))
        if timestamp is None:
            return float("inf")
        return max(0.0, time.time() - timestamp)
