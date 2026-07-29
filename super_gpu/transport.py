"""Local and SSH command transport.

SSH aliases intentionally come from the user's ~/.ssh/config so ProxyJump,
IdentityFile, ports, and keys stay outside the repository.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
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
