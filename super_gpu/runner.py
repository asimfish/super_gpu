"""Durable local/SSH experiment runner.

Commands run under nohup in a per-job directory. Completion status and logs
live on the target server, allowing the controller to reconnect after restart.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Mapping

from .models import JobSpec, NodeConfig, RunnerHandle
from .transport import CommandTransport


LAUNCH_SCRIPT = r"""
set -eu
run_root="$HOME/.super_gpu/runs"
run_dir="$run_root/$SUPER_GPU_JOB_ID"
mkdir -p "$run_dir"
workspace="$SUPER_GPU_WORKSPACE"
case "$workspace" in
  "~") workspace="$HOME" ;;
  "~/"*) workspace="$HOME/${workspace#~/}" ;;
esac
printf '%s\n' "$SUPER_GPU_COMMAND" > "$run_dir/command.sh"
chmod 700 "$run_dir/command.sh"
rm -f "$run_dir/exit_code" "$run_dir/finished_at"
cat > "$run_dir/wrapper.sh" <<'SUPER_GPU_WRAPPER'
#!/usr/bin/env bash
set +e
run_dir="$SUPER_GPU_RUN_DIR"
workspace="$SUPER_GPU_WORKSPACE"
case "$workspace" in
  "~") workspace="$HOME" ;;
  "~/"*) workspace="$HOME/${workspace#~/}" ;;
esac
mkdir -p "$workspace"
cd "$workspace"
bash "$run_dir/command.sh" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
rc=$?
tmp="$run_dir/exit_code.$$"
printf '%s\n' "$rc" > "$tmp"
mv "$tmp" "$run_dir/exit_code"
date +%s > "$run_dir/finished_at"
exit "$rc"
SUPER_GPU_WRAPPER
chmod 700 "$run_dir/wrapper.sh"
export SUPER_GPU_RUN_DIR="$run_dir"
export SUPER_GPU_WORKSPACE="$workspace"
if command -v setsid >/dev/null 2>&1; then
  nohup setsid bash "$run_dir/wrapper.sh" </dev/null >/dev/null 2>&1 &
else
  nohup bash "$run_dir/wrapper.sh" </dev/null >/dev/null 2>&1 &
fi
pid=$!
printf '%s\n' "$pid" > "$run_dir/pid"
printf '%s\n' "$SUPER_GPU_STARTED_AT" > "$run_dir/started_at"
echo "OK pid=$pid run_dir=$run_dir"
"""


POLL_SCRIPT = r"""
set +e
run_dir="$SUPER_GPU_RUN_DIR"
pid="$SUPER_GPU_PID"
if [ -f "$run_dir/exit_code" ]; then
  rc=$(tr -d '[:space:]' < "$run_dir/exit_code")
  echo "__SUPER_GPU_STATE__ finished"
  echo "__SUPER_GPU_EXIT__ ${rc:-255}"
elif kill -0 "$pid" 2>/dev/null; then
  echo "__SUPER_GPU_STATE__ running"
else
  echo "__SUPER_GPU_STATE__ lost"
  echo "__SUPER_GPU_EXIT__ 255"
fi
echo "__SUPER_GPU_STDOUT__"
tail -n 100 "$run_dir/stdout.log" 2>/dev/null || true
echo "__SUPER_GPU_STDERR__"
tail -n 100 "$run_dir/stderr.log" 2>/dev/null || true
"""


CANCEL_SCRIPT = r"""
set +e
pid="$SUPER_GPU_PID"
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
fi
echo "OK"
"""


@dataclass
class RunnerStatus:
    state: str
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    error: str = ""

    @property
    def finished(self) -> bool:
        return self.state in {"finished", "lost"}


class PersistentRunner:
    def __init__(self, transport: CommandTransport | None = None) -> None:
        self.transport = transport or CommandTransport()

    def launch(
        self,
        node: NodeConfig,
        job: JobSpec,
        gpu_indices: list[int],
        *,
        env: Mapping[str, str] | None = None,
    ) -> RunnerHandle:
        started_at = time.time()
        launch_env = {str(k): str(v) for k, v in (env or {}).items()}
        launch_env.update(
            {
                "SUPER_GPU_JOB_ID": job.id,
                "SUPER_GPU_COMMAND": job.command,
                "SUPER_GPU_WORKSPACE": node.workspace,
                "SUPER_GPU_STARTED_AT": str(started_at),
                "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in gpu_indices),
            }
        )
        result = self.transport.run_script(
            node,
            LAUNCH_SCRIPT,
            timeout=30,
            env=launch_env,
        )
        match = re.search(r"OK pid=(\d+) run_dir=(\S+)", result.stdout)
        if not result.ok or match is None:
            detail = (result.stderr or result.stdout or "unknown launch failure").strip()
            raise RuntimeError(f"runner launch failed on {node.name}: {detail}")
        return RunnerHandle(
            kind="local" if node.ssh in {"local", "localhost", "127.0.0.1"} else "ssh",
            node=node.name,
            pid=int(match.group(1)),
            run_dir=match.group(2),
            started_at=started_at,
        )

    def poll(self, node: NodeConfig, handle: RunnerHandle) -> RunnerStatus:
        result = self.transport.run_script(
            node,
            POLL_SCRIPT,
            timeout=20,
            env={
                "SUPER_GPU_RUN_DIR": handle.run_dir,
                "SUPER_GPU_PID": str(handle.pid),
            },
        )
        if not result.ok and "__SUPER_GPU_STATE__" not in result.stdout:
            return RunnerStatus(
                state="unknown",
                exit_code=None,
                stdout_tail="",
                stderr_tail="",
                error=(result.stderr or result.stdout or "poll failed").strip(),
            )
        return _parse_poll(result.stdout)

    def cancel(self, node: NodeConfig, handle: RunnerHandle) -> None:
        self.transport.run_script(
            node,
            CANCEL_SCRIPT,
            timeout=20,
            env={"SUPER_GPU_PID": str(handle.pid)},
        )


def _parse_poll(stdout: str) -> RunnerStatus:
    state = "unknown"
    exit_code: int | None = None
    section = ""
    out_lines: list[str] = []
    err_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("__SUPER_GPU_STATE__ "):
            state = line.split(" ", 1)[1].strip()
            continue
        if line.startswith("__SUPER_GPU_EXIT__ "):
            try:
                exit_code = int(line.split(" ", 1)[1].strip())
            except ValueError:
                exit_code = 255
            continue
        if line == "__SUPER_GPU_STDOUT__":
            section = "stdout"
            continue
        if line == "__SUPER_GPU_STDERR__":
            section = "stderr"
            continue
        if section == "stdout":
            out_lines.append(line)
        elif section == "stderr":
            err_lines.append(line)
    return RunnerStatus(
        state=state,
        exit_code=exit_code,
        stdout_tail="\n".join(out_lines)[-12000:],
        stderr_tail="\n".join(err_lines)[-12000:],
        error="runner process disappeared before writing exit status" if state == "lost" else "",
    )
