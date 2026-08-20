"""Deterministic, content-addressed experiment source snapshots."""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import os
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDES = (
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    ".super_gpu",
    ".super_gpu/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
)


class SnapshotError(ValueError):
    """A source tree cannot be represented safely as an immutable snapshot."""


@dataclass(frozen=True)
class SourceSnapshot:
    digest: str
    format: str
    file_count: int
    source_bytes: int
    archive_bytes: int
    excluded: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["excluded"] = list(self.excluded)
        return data


class SourceSnapshotStore:
    """Create and verify canonical archives under a content-addressed root."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_files: int = 100_000,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.max_files = max_files
        self.max_bytes = max_bytes

    def archive_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise SnapshotError("invalid source snapshot digest")
        return self.root / f"{digest}.tar.gz"

    def create(self, source: str | os.PathLike[str], exclude: list[str] | None = None) -> SourceSnapshot:
        source_root = Path(source).expanduser().resolve()
        if not source_root.is_dir():
            raise SnapshotError(f"snapshot source is not a directory: {source_root}")
        patterns = tuple(dict.fromkeys((*DEFAULT_EXCLUDES, *(exclude or []))))
        entries: list[tuple[Path, str]] = []
        source_bytes = 0
        for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(source_root).as_posix()
            if self._excluded(relative, patterns):
                continue
            if path.is_symlink():
                target = os.readlink(path)
                resolved = (path.parent / target).resolve()
                try:
                    resolved.relative_to(source_root)
                except ValueError as exc:
                    raise SnapshotError(f"snapshot symlink escapes source root: {relative}") from exc
            elif path.is_file():
                source_bytes += path.stat().st_size
            elif not path.is_dir():
                raise SnapshotError(f"unsupported snapshot entry: {relative}")
            entries.append((path, relative))
            if len(entries) > self.max_files:
                raise SnapshotError(f"snapshot exceeds {self.max_files} entries")
            if source_bytes > self.max_bytes:
                raise SnapshotError(f"snapshot exceeds {self.max_bytes} source bytes")

        temporary = tempfile.NamedTemporaryFile(dir=self.root, prefix=".snapshot-", delete=False)
        temp_path = Path(temporary.name)
        try:
            with temporary:
                with gzip.GzipFile(filename="", fileobj=temporary, mode="wb", mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                        for path, relative in entries:
                            info = archive.gettarinfo(str(path), arcname=relative)
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            info.mtime = 0
                            info.mode = 0o755 if path.is_dir() or (info.mode & 0o111) else 0o644
                            if path.is_file() and not path.is_symlink():
                                with path.open("rb") as stream:
                                    archive.addfile(info, stream)
                            else:
                                archive.addfile(info)
            digest = self._sha256(temp_path)
            final_path = self.archive_path(digest)
            if final_path.exists():
                if self._sha256(final_path) != digest:
                    raise SnapshotError(f"corrupt snapshot object already exists: {digest}")
                temp_path.unlink()
            else:
                temp_path.replace(final_path)
                final_path.chmod(0o400)
            return SourceSnapshot(
                digest=digest,
                format="tar.gz.v1",
                file_count=sum(1 for path, _ in entries if path.is_file() or path.is_symlink()),
                source_bytes=source_bytes,
                archive_bytes=final_path.stat().st_size,
                excluded=patterns,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def verify(self, digest: str) -> Path:
        path = self.archive_path(digest)
        if not path.is_file() or self._sha256(path) != digest:
            raise SnapshotError(f"source snapshot is missing or corrupt: {digest}")
        return path

    @staticmethod
    def _excluded(relative: str, patterns: tuple[str, ...]) -> bool:
        parts = relative.split("/")
        return any(
            fnmatch.fnmatchcase(relative, pattern)
            or fnmatch.fnmatchcase(parts[-1], pattern)
            for pattern in patterns
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(io.DEFAULT_BUFFER_SIZE * 16), b""):
                digest.update(block)
        return digest.hexdigest()
