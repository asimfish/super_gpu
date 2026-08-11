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


def import_gpumgr_inventory(
    path: str | os.PathLike[str],
    *,
    role: str = "shared",
    workspace: str = "~/.super_gpu/work",
    database: str = "~/.super_gpu/state.sqlite3",
) -> dict[str, Any]:
    """Convert gpumgr's private node inventory into a super_gpu config.

    Imported nodes intentionally default to the conservative shared policy.
    SSH connection details remain in ~/.ssh/config and are never copied.
    """
    if role not in {"dedicated", "shared"}:
        raise ValueError("role must be dedicated or shared")
    source = _read_json(path)
    raw_nodes = source.get("nodes", [])
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("gpumgr inventory must contain at least one node")

    nodes: list[dict[str, Any]] = []
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise ValueError("gpumgr inventory nodes must be JSON objects")
        name = str(raw.get("name") or "").strip()
        ssh_alias = str(raw.get("ssh") or name).strip()
        vendor = str(raw.get("vendor") or "nvidia").strip().lower()
        gpu_count = max(1, int(raw.get("gpus", 1)))
        labels = [vendor, f"gpu-count-{gpu_count}"]
        if bool(raw.get("can_vllm", False)):
            labels.append("vllm")
        nodes.append(
            {
                "name": name,
                "ssh": ssh_alias,
                "role": role,
                "priority": 0,
                "enabled": True,
                "workspace": workspace,
                "labels": labels,
            }
        )

    payload: dict[str, Any] = {
        "database": database,
        "poll_interval": 5,
        "lease_ttl": 60,
        "default_memory_mib": 16384,
        "default_gpu_utilization": 70,
        "max_parallel": 32,
        "api_host": "127.0.0.1",
        "api_port": 8765,
        "nodes": nodes,
    }
    config_from_dict(payload)
    return payload


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
