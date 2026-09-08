"""On-demand job log retrieval from the node that ran the job.

The scheduler stores only the last 100 lines it saw at its most recent poll.
When an agent has to diagnose a failure it usually needs more than that, and
it needs it now rather than at the next tick, so ``GET /api/jobs/<id>/logs``
tails the durable ``stdout.log`` / ``stderr.log`` in the job's run directory
over the same transport the runner uses. Nothing is transferred beyond the
requested lines, and the run directory survives job completion.
"""
from __future__ import annotations

from typing import Any

from .api_manifest import LOG_LINES_DEFAULT, LOG_LINES_MAX
from .transport import CommandTransport

TAIL_SCRIPT = r"""
set +e
run_dir="$SUPER_GPU_RUN_DIR"
lines="$SUPER_GPU_LOG_LINES"
[ -d "$run_dir" ] || { echo "run directory not found: $run_dir" >&2; exit 3; }
echo "__SUPER_GPU_STDOUT__"
tail -n "$lines" "$run_dir/stdout.log" 2>/dev/null || true
echo "__SUPER_GPU_STDERR__"
tail -n "$lines" "$run_dir/stderr.log" 2>/dev/null || true
echo "__SUPER_GPU_RESULT__"
head -c 4096 "$run_dir/result.json" 2>/dev/null || true
echo
echo "__SUPER_GPU_EXIT__"
cat "$run_dir/exit_code" 2>/dev/null || true
"""


def clamp_lines(value: Any) -> int:
    try:
        lines = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("lines must be an integer") from exc
    if lines < 1 or lines > LOG_LINES_MAX:
        raise ValueError(f"lines must be between 1 and {LOG_LINES_MAX}")
    return lines


def _parse_tail(stdout: str) -> dict[str, Any]:
    sections: dict[str, list[str]] = {"stdout": [], "stderr": [], "result": [], "exit": []}
    current = ""
    for line in stdout.splitlines():
        if line == "__SUPER_GPU_STDOUT__":
            current = "stdout"
            continue
        if line == "__SUPER_GPU_STDERR__":
            current = "stderr"
            continue
        if line == "__SUPER_GPU_RESULT__":
            current = "result"
            continue
        if line == "__SUPER_GPU_EXIT__":
            current = "exit"
            continue
        if current:
            sections[current].append(line)
    exit_text = "".join(sections["exit"]).strip()
    exit_code: int | None
    try:
        exit_code = int(exit_text) if exit_text else None
    except ValueError:
        exit_code = None
    return {
        "stdout": "\n".join(sections["stdout"]),
        "stderr": "\n".join(sections["stderr"]),
        "result_raw": "\n".join(sections["result"]).strip(),
        "exit_code": exit_code,
    }


def fetch_job_logs(
    config: Any,
    store: Any,
    job_id: str,
    *,
    lines: int = LOG_LINES_DEFAULT,
    transport: CommandTransport | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Return the last ``lines`` lines of a job's stdout and stderr.

    Jobs that never started have no run directory; their (empty) stored tails
    are returned with ``source: "stored"`` so the caller can tell the two
    cases apart.
    """
    job = store.get_job(job_id)
    if not job:
        raise ValueError(f"unknown job: {job_id}")
    lines = clamp_lines(lines)
    base = {
        "job_id": job["id"],
        "job_name": job.get("name"),
        "status": job.get("status"),
        "node": job.get("node") or "",
        "attempt": job.get("attempt"),
        "lines": lines,
    }
    run_dir = str(job.get("run_dir") or "")
    node_name = str(job.get("node") or "")
    if not run_dir or not node_name:
        return {
            **base,
            "source": "stored",
            "run_dir": "",
            "stdout": job.get("stdout_tail") or "",
            "stderr": job.get("stderr_tail") or "",
            "result_raw": "",
            "exit_code": job.get("exit_code"),
            "note": "job has not started on a node yet; showing stored tails",
        }
    node = config.node(node_name)
    transport = transport or CommandTransport()
    result = transport.run_script(
        node,
        TAIL_SCRIPT,
        timeout=timeout,
        env={"SUPER_GPU_RUN_DIR": run_dir, "SUPER_GPU_LOG_LINES": str(lines)},
    )
    if not result.ok:
        detail = (result.stderr or result.stdout or "log tail failed").strip()
        raise RuntimeError(f"log retrieval failed on {node.name}: {detail}")
    parsed = _parse_tail(result.stdout)
    return {**base, "source": "node", "run_dir": run_dir, **parsed}
