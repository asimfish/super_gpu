"""Configuration and experiment-plan loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import ExperimentPlan, SystemConfig, NodeConfig


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload


def load_config(path: str | os.PathLike[str]) -> SystemConfig:
    data = _read_json(path)
    nodes = [NodeConfig.from_dict(dict(item)) for item in data.get("nodes", [])]
    cfg = SystemConfig(
        nodes=nodes,
        database=str(data.get("database") or "~/.super_gpu/state.sqlite3"),
        poll_interval=float(data.get("poll_interval", 5)),
        lease_ttl=float(data.get("lease_ttl", 60)),
        default_memory_mib=int(data.get("default_memory_mib", 16384)),
        default_gpu_utilization=int(data.get("default_gpu_utilization", 70)),
        max_parallel=int(data.get("max_parallel", 64)),
        api_host=str(data.get("api_host") or "127.0.0.1"),
        api_port=int(data.get("api_port", 8765)),
        api_token=str(
            os.environ.get("SUPER_GPU_API_TOKEN")
            or data.get("api_token")
            or ""
        ),
    )
    cfg.database = str(Path(cfg.database).expanduser())
    cfg.validate()
    return cfg


def load_plan(path: str | os.PathLike[str]) -> ExperimentPlan:
    return ExperimentPlan.from_dict(_read_json(path))


def config_from_dict(data: dict[str, Any]) -> SystemConfig:
    nodes = [NodeConfig.from_dict(dict(item)) for item in data.get("nodes", [])]
    cfg = SystemConfig(
        nodes=nodes,
        database=str(data.get("database") or "~/.super_gpu/state.sqlite3"),
        poll_interval=float(data.get("poll_interval", 5)),
        lease_ttl=float(data.get("lease_ttl", 60)),
        default_memory_mib=int(data.get("default_memory_mib", 16384)),
        default_gpu_utilization=int(data.get("default_gpu_utilization", 70)),
        max_parallel=int(data.get("max_parallel", 64)),
        api_host=str(data.get("api_host") or "127.0.0.1"),
        api_port=int(data.get("api_port", 8765)),
        api_token=str(data.get("api_token") or ""),
    )
    cfg.database = str(Path(cfg.database).expanduser())
    cfg.validate()
    return cfg
