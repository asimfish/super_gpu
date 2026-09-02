# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Everything below has been developed on `main` ahead of the first tagged
release (planned as `v0.2.0`, matching `super_gpu.__version__`).

### Added
- Multi-server GPU monitoring over SSH with per-GPU memory, utilization,
  temperature, power, and process telemetry (including owning user, command,
  and elapsed time).
- Role-aware placement: dedicated nodes are packed with best-fit memory and a
  target utilization; shared nodes require several consecutive stable samples
  below conservative thresholds before admission.
- Continuous supervision and backfill: lease renewal, retries, OOM budget
  escalation, timeouts, cancellation, and immediate placement on stably freed
  GPUs.
- Learned resource estimation from recorded peak memory and runtime of
  identical jobs.
- Durable SSH runner with per-job PID, log, and exit-code files, so a
  controller restart resumes supervision; process identity is verified with a
  launch token, PID, and Linux start ticks before any signal is sent.
- Immutable, content-addressed source snapshots (deterministic `tar.gz`,
  SHA-256 verified on the node, private extraction per job, upload skipped
  when the node already holds the digest).
- Idempotent submission keyed by `request_id`: safe retries replay the
  original plan, conflicting content returns HTTP 409.
- Explainable scheduling: every pending job carries a live `pending_reason`.
- Typed job results via `$SUPER_GPU_RESULT_FILE` and dependency predicates
  (`after: success | complete | result`, `result_states`), with unsatisfiable
  branches marked `skipped` instead of `failed`.
- Declared outputs: workspace-relative globs collected with `super-gpu pull`
  or inspected via `GET /api/jobs/<id>/outputs` and the `experiment_outputs`
  MCP tool.
- Idle-GPU watchdog that reports low-utilization occupancy and, only when
  explicitly enabled, cancels managed exclusive jobs it launched itself.
- Interfaces: CLI, REST API, MCP server (stdio and HTTP), zero-CDN live
  dashboard, `AGENTS.md` operating contract, Agent Skill, and a JSON Schema
  for experiment plans.
- `scripts/supergpu` launchd wrapper for macOS controllers and
  `scripts/demo_dashboard.py` for a simulated cluster without GPUs.
- Docker-based SSH end-to-end test suite exercising the full production path
  in CI.
- Packaging: PEP 621 metadata in `pyproject.toml`, single-sourced version,
  `super-gpu --version`, a packaging check in CI, and a tag-driven PyPI
  release workflow using Trusted Publishing.
- Webhook notifications (`notifications` in `config.json`): Feishu (with
  optional signing), Slack, and generic JSON targets receive selected
  scheduler events from a background worker with bounded retries; URLs may
  come from environment variables and are never exposed by the API.
- Plan-level events `plan_completed`, `plan_failed`, and `plan_cancelled`,
  emitted exactly once when a plan reaches a terminal state, with per-status
  job counts.

### Fixed
- A plan's `finished_at` was rewritten on every scheduler tick; it is now set
  once when the plan becomes terminal.
- A job that finished between the liveness check and the `/proc` start-ticks
  read was reported as lost with an identity mismatch (Linux only).
- GPU process detail panels stayed collapsed across dashboard refreshes.

[Unreleased]: https://github.com/asimfish/super_gpu/commits/main
