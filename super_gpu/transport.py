"""Local and SSH command transport.

SSH aliases intentionally come from the user's ~/.ssh/config so ProxyJump,
IdentityFile, ports, and keys stay outside the repository.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import NodeConfig


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandTransport:
    def __init__(self, connect_timeout: float = 10.0) -> None:
        self.connect_timeout = connect_timeout

    @staticmethod
    def _is_local(node: NodeConfig) -> bool:
        return node.ssh in {"local", "localhost", "127.0.0.1"}

    def run(
        self,
        node: NodeConfig,
        command: str,
        *,
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
    ) -> CommandResult:
        import time

        started = time.monotonic()
        command_env = {str(k): str(v) for k, v in (env or {}).items()}
        if self._is_local(node):
            process_env = os.environ.copy()
            process_env.update(node.env)
            process_env.update(command_env)
            argv = ["bash", "-lc", command]
        else:
            remote_env = dict(node.env)
            remote_env.update(command_env)
            env_prefix = ""
            if remote_env:
                assignments = " ".join(
                    f"{key}={shlex.quote(value)}" for key, value in remote_env.items()
                )
                env_prefix = f"env {assignments} "
            remote_command = f"{env_prefix}bash -lc {shlex.quote(command)}"
            argv = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={max(1, int(self.connect_timeout))}",
                node.ssh,
                remote_command,
            ]
            process_env = None
        try:
            completed = subprocess.run(
                argv,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=process_env,
                check=False,
            )
            return CommandResult(
                completed.returncode,
                completed.stdout,
                completed.stderr,
                time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandResult(
                124,
                stdout,
                stderr or f"command timed out after {timeout:g}s",
                time.monotonic() - started,
            )
        except OSError as exc:
            return CommandResult(
                127,
                "",
                str(exc),
                time.monotonic() - started,
            )

    def run_script(
        self,
        node: NodeConfig,
        script: str,
        *,
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        command_env = {str(k): str(v) for k, v in (env or {}).items()}
        if self._is_local(node):
            import time

            started = time.monotonic()
            process_env = os.environ.copy()
            process_env.update(node.env)
            process_env.update(command_env)
            try:
                completed = subprocess.run(
                    ["bash", "-s"],
                    input=script,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    env=process_env,
                    check=False,
                )
                return CommandResult(
                    completed.returncode,
                    completed.stdout,
                    completed.stderr,
                    time.monotonic() - started,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                return CommandResult(
                    124,
                    stdout,
                    stderr or f"script timed out after {timeout:g}s",
                    time.monotonic() - started,
                )

        remote_env = dict(node.env)
        remote_env.update(command_env)
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in remote_env.items()
        )
        remote = f"{'env ' + assignments + ' ' if assignments else ''}bash -s"
        import time

        started = time.monotonic()
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={max(1, int(self.connect_timeout))}",
                    node.ssh,
                    remote,
                ],
                input=script,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                completed.returncode,
                completed.stdout,
                completed.stderr,
                time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandResult(
                124,
                stdout,
                stderr or f"script timed out after {timeout:g}s",
                time.monotonic() - started,
            )
        except OSError as exc:
            return CommandResult(127, "", str(exc), time.monotonic() - started)

    def upload_file(
        self,
        node: NodeConfig,
        source: str | os.PathLike[str],
        destination: str,
        *,
        expected_sha256: str,
        timeout: float = 120.0,
    ) -> CommandResult:
        """Atomically upload one verified file below the remote home directory."""
        import hashlib
        import time

        if (
            destination.startswith("/")
            or ".." in Path(destination).parts
            or len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)
        ):
            raise ValueError("upload destination or digest is unsafe")
        source_path = Path(source).expanduser().resolve()
        started = time.monotonic()
        if self._is_local(node):
            target = (Path.home() / destination).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.parent.chmod(0o700)
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            shutil.copyfile(source_path, temporary)
            digest = hashlib.sha256()
            with temporary.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected_sha256:
                temporary.unlink(missing_ok=True)
                return CommandResult(1, "", "uploaded file digest mismatch", time.monotonic() - started)
            temporary.replace(target)
            return CommandResult(0, "OK\n", "", time.monotonic() - started)

        remote_script = r'''set -eu
umask 077
destination="$HOME/$SUPER_GPU_UPLOAD_DESTINATION"
mkdir -p "$(dirname "$destination")"
if [ -f "$destination" ] && [ "$(sha256sum "$destination" | awk '{print $1}')" = "$SUPER_GPU_UPLOAD_SHA256" ]; then
  cat >/dev/null
  echo OK
  exit 0
fi
temporary="$destination.tmp.$$"
trap 'rm -f "$temporary"' EXIT
cat >"$temporary"
actual=$(sha256sum "$temporary" | awk '{print $1}')
[ "$actual" = "$SUPER_GPU_UPLOAD_SHA256" ]
chmod 400 "$temporary"
mv "$temporary" "$destination"
trap - EXIT
echo OK
'''
        assignments = " ".join(
            (
                f"SUPER_GPU_UPLOAD_DESTINATION={shlex.quote(destination)}",
                f"SUPER_GPU_UPLOAD_SHA256={shlex.quote(expected_sha256)}",
            )
        )
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(self.connect_timeout))}",
            node.ssh,
            f"env {assignments} bash -c {shlex.quote(remote_script)}",
        ]
        try:
            with source_path.open("rb") as stream:
                completed = subprocess.run(
                    argv,
                    stdin=stream,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
            return CommandResult(
                completed.returncode,
                completed.stdout.decode("utf-8", errors="replace"),
                completed.stderr.decode("utf-8", errors="replace"),
                time.monotonic() - started,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
            return CommandResult(code, "", str(exc), time.monotonic() - started)
