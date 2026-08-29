"""Declarative artifact collection.

Jobs may declare ``outputs`` as workspace-relative glob patterns. After a job
reaches a terminal state, ``super-gpu pull`` expands those patterns on the
node that ran the job, streams a single tar.gz archive back, and unpacks it
locally. Patterns never leave the workspace: absolute paths, ``..`` segments,
and whitespace are rejected at plan validation time, and the manifest script
re-checks every match before it is archived.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from .models import NodeConfig
from .transport import CommandTransport

# Expands glob patterns (newline-separated in SUPER_GPU_PULL_PATTERNS)
# relative to the pull base and prints each matched file once. Directories
# are walked recursively so a pattern like "checkpoints" collects everything
# beneath it. Patterns travel via the environment because run_script already
# uses stdin for the script body.
MANIFEST_SCRIPT = r"""
set -u
base="$SUPER_GPU_PULL_BASE"
case "$base" in
  "~") base="$HOME" ;;
  "~/"*) base="$HOME/${base#~/}" ;;
esac
cd "$base" 2>/dev/null || { echo "workspace not found: $base" >&2; exit 3; }
shopt -s nullglob globstar
printf '%s\n' "$SUPER_GPU_PULL_PATTERNS" | while IFS= read -r pattern; do
  [ -n "$pattern" ] || continue
  for match in $pattern; do
    case "$match" in
      /*|../*|*/../*|*/..|..) continue ;;
    esac
    if [ -f "$match" ]; then
      printf '%s\n' "$match"
    elif [ -d "$match" ]; then
      find "$match" -type f -print
    fi
  done
done | LC_ALL=C sort -u
"""


def resolve_workspace(job: dict[str, Any], plan_payload: dict[str, Any], node: NodeConfig) -> str:
    """Return the directory the job's command ran in.

    With a source snapshot the runner works inside ``<run_dir>/workspace``;
    without one it works directly in the node's configured workspace.
    """
    snapshot = dict((plan_payload.get("plan") or {}).get("source_snapshot") or {})
    if str(snapshot.get("digest") or ""):
        run_dir = str(job.get("run_dir") or "")
        if not run_dir:
            raise ValueError(f"job {job.get('name')!r} has no recorded run directory")
        return f"{run_dir.rstrip('/')}/workspace"
    return node.workspace


def collect_manifest(
    transport: CommandTransport,
    node: NodeConfig,
    workspace: str,
    patterns: list[str],
    *,
    timeout: float = 60.0,
) -> list[str]:
    """Expand output patterns on the node and return matched relative paths."""
    result = transport.run_script(
        node,
        MANIFEST_SCRIPT,
        timeout=timeout,
        env={
            "SUPER_GPU_PULL_BASE": workspace,
            "SUPER_GPU_PULL_PATTERNS": "\n".join(patterns),
        },
    )
    if not result.ok:
        detail = (result.stderr or result.stdout or "manifest expansion failed").strip()
        raise RuntimeError(f"output manifest failed on {node.name}: {detail}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _job_pull_context(
    config: Any, store: Any, job_id: str
) -> tuple[dict[str, Any], NodeConfig, str, list[str]]:
    """Validate that a job's outputs are pullable and resolve where they live."""
    job = store.get_job(job_id)
    if not job:
        raise ValueError(f"unknown job: {job_id}")
    patterns = list(job.get("outputs") or [])
    if not patterns:
        raise ValueError(f"job {job.get('name')!r} declares no outputs")
    if job.get("status") not in {"completed", "failed", "cancelled"}:
        raise ValueError(
            f"job {job.get('name')!r} is {job.get('status')}; outputs are pullable "
            "once the job reaches a terminal state"
        )
    node_name = str(job.get("node") or "")
    if not node_name:
        raise ValueError(f"job {job.get('name')!r} never ran on a node")
    node = config.node(node_name)
    plan_payload = store.get_plan(job["plan_id"])
    workspace = resolve_workspace(job, plan_payload, node)
    return job, node, workspace, patterns


def describe_job_outputs(
    config: Any,
    store: Any,
    job_id: str,
    *,
    transport: CommandTransport | None = None,
) -> dict[str, Any]:
    """Expand a job's declared outputs remotely without transferring bytes."""
    transport = transport or CommandTransport()
    job, node, workspace, patterns = _job_pull_context(config, store, job_id)
    files = collect_manifest(transport, node, workspace, patterns)
    return {
        "job_id": job["id"],
        "job_name": job.get("name"),
        "node": node.name,
        "workspace": workspace,
        "patterns": patterns,
        "files": files,
    }


def pull_job_outputs(
    config: Any,
    store: Any,
    job_id: str,
    destination: str | Path,
    *,
    transport: CommandTransport | None = None,
) -> dict[str, Any]:
    """Fetch a finished job's declared outputs into ``destination``.

    Returns a summary with the resolved workspace, matched files, and the
    local directory the files were unpacked into.
    """
    transport = transport or CommandTransport()
    job, node, workspace, patterns = _job_pull_context(config, store, job_id)
    files = collect_manifest(transport, node, workspace, patterns)
    summary: dict[str, Any] = {
        "job_id": job["id"],
        "job_name": job.get("name"),
        "node": node.name,
        "workspace": workspace,
        "patterns": patterns,
        "files": files,
    }
    if not files:
        summary["pulled"] = 0
        summary["destination"] = ""
        return summary

    dest_dir = Path(destination).expanduser() / str(job.get("name") or job["id"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / ".super_gpu_outputs.tar.gz"
    result = transport.download_archive(node, workspace, files, archive_path)
    if not result.ok:
        detail = (result.stderr or result.stdout or "archive download failed").strip()
        raise RuntimeError(f"output download failed on {node.name}: {detail}")
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = []
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"archive contains unsafe member: {member.name!r}")
                if member.isreg() or member.isdir():
                    members.append(member)
            try:
                archive.extractall(dest_dir, members=members, filter="data")
            except TypeError:  # Python < 3.10.12 lacks the filter parameter
                archive.extractall(dest_dir, members=members)
    finally:
        archive_path.unlink(missing_ok=True)
    summary["pulled"] = len(files)
    summary["destination"] = str(dest_dir)
    return summary
