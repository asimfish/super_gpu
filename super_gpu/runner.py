"""Durable local/SSH experiment runner.

Commands run under nohup in a per-job directory. Completion status and logs
live on the target server, allowing the controller to reconnect after restart.
"""
from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from typing import Mapping

from .models import JobSpec, NodeConfig, RunnerHandle
from .transport import CommandTransport


LAUNCH_SCRIPT = r"""
set -eu
umask 077
run_root="$HOME/.super_gpu/runs"
run_dir="$run_root/$SUPER_GPU_JOB_ID"
mkdir -p "$run_dir"
if [ -f "$run_dir/identity" ] && [ ! -f "$run_dir/exit_code" ] && [ ! -f "$run_dir/cancelled" ]; then
  existing_token=$(sed -n '1p' "$run_dir/identity" 2>/dev/null || true)
  existing_pid=$(sed -n '2p' "$run_dir/identity" 2>/dev/null || true)
  existing_ticks=$(sed -n '3p' "$run_dir/identity" 2>/dev/null || true)
  existing_started_at=$(sed -n '4p' "$run_dir/identity" 2>/dev/null || true)
  case "$existing_pid:$existing_ticks" in
    *[!0-9:]*|:*|*:) echo "existing runner identity is incomplete" >&2; exit 1 ;;
  esac
  if kill -0 "$existing_pid" 2>/dev/null; then
    current_ticks=$(awk '{print $22}' "/proc/$existing_pid/stat" 2>/dev/null || true)
    current_ticks=${current_ticks:-0}
    if ! printf '%s' "$existing_token" | grep -Eq '^[0-9a-f]{48}$' || [ "$current_ticks" != "$existing_ticks" ] || [ -z "$existing_started_at" ]; then
      echo "existing live runner identity could not be verified" >&2
      exit 1
    fi
    if [ -n "${SUPER_GPU_SNAPSHOT_DIGEST:-}" ]; then
      existing_snapshot=$(cat "$run_dir/workspace/.super_gpu_snapshot" 2>/dev/null || true)
      [ "$existing_snapshot" = "$SUPER_GPU_SNAPSHOT_DIGEST" ] || {
        echo "existing runner snapshot does not match requested digest" >&2
        exit 1
      }
    fi
    echo "OK pid=$existing_pid start_ticks=$existing_ticks token=$existing_token run_dir=$run_dir started_at=$existing_started_at adopted=1"
    exit 0
  fi
fi
workspace="$SUPER_GPU_WORKSPACE"
case "$workspace" in
  "~") workspace="$HOME" ;;
  "~/"*) workspace="$HOME/${workspace#~/}" ;;
esac
if [ -n "${SUPER_GPU_SNAPSHOT_DIGEST:-}" ]; then
  archive="$HOME/.super_gpu/source-snapshots/$SUPER_GPU_SNAPSHOT_DIGEST.tar.gz"
  workspace="$run_dir/workspace"
  staging="$run_dir/workspace.tmp.$$"
  rm -rf "$staging"
  mkdir -p "$staging"
  tar -xzf "$archive" -C "$staging"
  printf '%s\n' "$SUPER_GPU_SNAPSHOT_DIGEST" > "$staging/.super_gpu_snapshot"
  rm -rf "$workspace"
  mv "$staging" "$workspace"
fi
printf '%s\n' "$SUPER_GPU_COMMAND" > "$run_dir/command.sh"
chmod 700 "$run_dir/command.sh"
rm -f "$run_dir/exit_code" "$run_dir/finished_at" "$run_dir/identity" "$run_dir/pid" "$run_dir/cancelled" "$run_dir/result.json"
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
start_ticks=$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || true)
start_ticks=${start_ticks:-0}
identity_tmp="$run_dir/identity.$$"
printf '%s\n%s\n%s\n%s\n' "$SUPER_GPU_LAUNCH_TOKEN" "$$" "$start_ticks" "$SUPER_GPU_STARTED_AT" > "$identity_tmp"
mv "$identity_tmp" "$run_dir/identity"
export SUPER_GPU_RESULT_FILE="$run_dir/result.json"
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
launcher_pid=$!
for _ in $(seq 1 100); do
  [ -f "$run_dir/identity" ] && break
  sleep 0.02
done
[ -f "$run_dir/identity" ] || { echo "identity file was not created" >&2; exit 1; }
launch_token=$(sed -n '1p' "$run_dir/identity")
pid=$(sed -n '2p' "$run_dir/identity")
start_ticks=$(sed -n '3p' "$run_dir/identity")
[ "$launch_token" = "$SUPER_GPU_LAUNCH_TOKEN" ] || { echo "launch token mismatch" >&2; exit 1; }
printf '%s\n' "$pid" > "$run_dir/pid"
printf '%s\n' "$SUPER_GPU_STARTED_AT" > "$run_dir/started_at"
echo "OK pid=$pid start_ticks=${start_ticks:-0} token=$launch_token run_dir=$run_dir started_at=$SUPER_GPU_STARTED_AT adopted=0"
"""


POLL_SCRIPT = r"""
set +e
run_dir="$SUPER_GPU_RUN_DIR"
pid="$SUPER_GPU_PID"
if [ -n "${SUPER_GPU_LAUNCH_TOKEN:-}" ]; then
  stored_token=$(sed -n '1p' "$run_dir/identity" 2>/dev/null || true)
  stored_pid=$(sed -n '2p' "$run_dir/identity" 2>/dev/null || true)
  stored_ticks=$(sed -n '3p' "$run_dir/identity" 2>/dev/null || true)
  if [ "$stored_token" != "$SUPER_GPU_LAUNCH_TOKEN" ] || [ "$stored_pid" != "$pid" ] || [ "${stored_ticks:-0}" != "$SUPER_GPU_PROCESS_START_TICKS" ]; then
    echo "__SUPER_GPU_STATE__ lost"
    echo "__SUPER_GPU_EXIT__ 255"
    echo "__SUPER_GPU_ERROR__ process identity mismatch"
    exit 0
  fi
fi
state=""
if [ -f "$run_dir/exit_code" ]; then
  state=finished
elif kill -0 "$pid" 2>/dev/null; then
  current_ticks=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)
  # An empty read means /proc is unavailable or the process exited between the
  # liveness check and this read; neither is evidence of a different process.
  if [ -n "${SUPER_GPU_LAUNCH_TOKEN:-}" ] && [ -n "$current_ticks" ] && [ "$current_ticks" != "$SUPER_GPU_PROCESS_START_TICKS" ]; then
    state=mismatch
  else
    state=running
  fi
else
  state=lost
fi
# The wrapper may have written exit_code while the checks above were running.
# Only the wrapper (verified above) writes that file, so its presence settles
# the outcome regardless of what the probes observed.
if [ "$state" != "finished" ] && [ -f "$run_dir/exit_code" ]; then
  state=finished
fi
case "$state" in
  finished)
    rc=$(tr -d '[:space:]' < "$run_dir/exit_code")
    echo "__SUPER_GPU_STATE__ finished"
    echo "__SUPER_GPU_EXIT__ ${rc:-255}"
    ;;
  running)
    echo "__SUPER_GPU_STATE__ running"
    ;;
  mismatch)
    echo "__SUPER_GPU_STATE__ lost"
    echo "__SUPER_GPU_EXIT__ 255"
    echo "__SUPER_GPU_ERROR__ process identity start time mismatch"
    ;;
  *)
    echo "__SUPER_GPU_STATE__ lost"
    echo "__SUPER_GPU_EXIT__ 255"
    ;;
esac
echo "__SUPER_GPU_STDOUT__"
tail -n 100 "$run_dir/stdout.log" 2>/dev/null || true
echo "__SUPER_GPU_STDERR__"
tail -n 100 "$run_dir/stderr.log" 2>/dev/null || true
echo "__SUPER_GPU_RESULT__"
head -c 4096 "$run_dir/result.json" 2>/dev/null || true
"""


CANCEL_SCRIPT = r"""
set +e
run_dir="$SUPER_GPU_RUN_DIR"
pid="$SUPER_GPU_PID"
if [ -n "${SUPER_GPU_LAUNCH_TOKEN:-}" ]; then
  stored_token=$(sed -n '1p' "$run_dir/identity" 2>/dev/null || true)
  stored_pid=$(sed -n '2p' "$run_dir/identity" 2>/dev/null || true)
  stored_ticks=$(sed -n '3p' "$run_dir/identity" 2>/dev/null || true)
  current_ticks=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)
  if [ "$stored_token" != "$SUPER_GPU_LAUNCH_TOKEN" ] || [ "$stored_pid" != "$pid" ] || [ "${stored_ticks:-0}" != "$SUPER_GPU_PROCESS_START_TICKS" ]; then
    echo "__SUPER_GPU_CANCEL__ refused"
    exit 0
  fi
  if kill -0 "$pid" 2>/dev/null && [ "${current_ticks:-0}" != "$SUPER_GPU_PROCESS_START_TICKS" ]; then
    echo "__SUPER_GPU_CANCEL__ refused"
    exit 0
  fi
fi
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
cancelled_tmp="$run_dir/cancelled.$$"
printf '%s\n' "$(date +%s)" > "$cancelled_tmp"
mv "$cancelled_tmp" "$run_dir/cancelled"
echo "__SUPER_GPU_CANCEL__ ok"
"""


@dataclass
class RunnerStatus:
    state: str
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    error: str = ""
    result_raw: str = ""

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
        snapshot_digest: str = "",
        snapshot_path: str = "",
    ) -> RunnerHandle:
        started_at = time.time()
        launch_token = secrets.token_hex(24)
        if snapshot_digest:
            uploaded = self.transport.upload_file(
                node,
                snapshot_path,
                f".super_gpu/source-snapshots/{snapshot_digest}.tar.gz",
                expected_sha256=snapshot_digest,
            )
            if not uploaded.ok:
                detail = (uploaded.stderr or uploaded.stdout or "snapshot upload failed").strip()
                raise RuntimeError(f"source snapshot upload failed on {node.name}: {detail}")
        launch_env = {str(k): str(v) for k, v in (env or {}).items()}
        launch_env.update(
            {
                "SUPER_GPU_JOB_ID": job.id,
                "SUPER_GPU_COMMAND": job.command,
                "SUPER_GPU_WORKSPACE": node.workspace,
                "SUPER_GPU_STARTED_AT": str(started_at),
                "SUPER_GPU_LAUNCH_TOKEN": launch_token,
                "SUPER_GPU_SNAPSHOT_DIGEST": snapshot_digest,
                "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in gpu_indices),
            }
        )
        result = self.transport.run_script(
            node,
            LAUNCH_SCRIPT,
            timeout=30,
            env=launch_env,
        )
        match = re.search(
            r"OK pid=(\d+) start_ticks=(\d+) token=([0-9a-f]{48}) "
            r"run_dir=(\S+) started_at=([0-9.]+) adopted=([01])",
            result.stdout,
        )
        if not result.ok or match is None:
            detail = (result.stderr or result.stdout or "unknown launch failure").strip()
            raise RuntimeError(f"runner launch failed on {node.name}: {detail}")
        if match.group(6) == "0" and match.group(3) != launch_token:
            raise RuntimeError(f"runner launch failed on {node.name}: launch token mismatch")
        return RunnerHandle(
            kind="local" if node.ssh in {"local", "localhost", "127.0.0.1"} else "ssh",
            node=node.name,
            pid=int(match.group(1)),
            run_dir=match.group(4),
            started_at=float(match.group(5)),
            launch_token=match.group(3),
            process_start_ticks=int(match.group(2)),
            snapshot_digest=snapshot_digest,
        )

    def poll(self, node: NodeConfig, handle: RunnerHandle) -> RunnerStatus:
        result = self.transport.run_script(
            node,
            POLL_SCRIPT,
            timeout=20,
            env={
                "SUPER_GPU_RUN_DIR": handle.run_dir,
                "SUPER_GPU_PID": str(handle.pid),
                "SUPER_GPU_LAUNCH_TOKEN": handle.launch_token,
                "SUPER_GPU_PROCESS_START_TICKS": str(handle.process_start_ticks),
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

    def cancel(self, node: NodeConfig, handle: RunnerHandle) -> bool:
        result = self.transport.run_script(
            node,
            CANCEL_SCRIPT,
            timeout=20,
            env={
                "SUPER_GPU_RUN_DIR": handle.run_dir,
                "SUPER_GPU_PID": str(handle.pid),
                "SUPER_GPU_LAUNCH_TOKEN": handle.launch_token,
                "SUPER_GPU_PROCESS_START_TICKS": str(handle.process_start_ticks),
            },
        )
        return result.ok and "__SUPER_GPU_CANCEL__ ok" in result.stdout


def _parse_poll(stdout: str) -> RunnerStatus:
    state = "unknown"
    exit_code: int | None = None
    section = ""
    out_lines: list[str] = []
    err_lines: list[str] = []
    result_lines: list[str] = []
    error = ""
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
        if line.startswith("__SUPER_GPU_ERROR__ "):
            error = line.split(" ", 1)[1].strip()
            continue
        if line == "__SUPER_GPU_STDOUT__":
            section = "stdout"
            continue
        if line == "__SUPER_GPU_STDERR__":
            section = "stderr"
            continue
        if line == "__SUPER_GPU_RESULT__":
            section = "result"
            continue
        if section == "stdout":
            out_lines.append(line)
        elif section == "stderr":
            err_lines.append(line)
        elif section == "result":
            result_lines.append(line)
    return RunnerStatus(
        state=state,
        exit_code=exit_code,
        stdout_tail="\n".join(out_lines)[-12000:],
        stderr_tail="\n".join(err_lines)[-12000:],
        error=error or ("runner process disappeared before writing exit status" if state == "lost" else ""),
        result_raw="\n".join(result_lines).strip()[:4096],
    )
