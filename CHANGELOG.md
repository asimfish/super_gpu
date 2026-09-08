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
- Agent-facing API contract (`super_gpu/api_manifest.py`): public
  `GET /api/meta` describing every route, its access tier, the error-code
  catalog, the MCP tools, limits, and capability flags; every error body now
  carries a stable `code` (`invalid_request`, `unauthorized`, `not_found`,
  `method_not_allowed`, `idempotency_conflict`, `remote_failure`,
  `internal_error`, `not_ready`); unknown routes and wrong methods answer in
  the same JSON envelope (405 includes an `Allow` header); `/healthz` and
  `/readyz` (503 until the first scheduling tick). Tests fail when the README
  tables, `AGENTS.md`, or the registered MCP tools drift from the manifest.
- Situation brief: `GET /api/brief` (`?hours=`, `?format=text`),
  `super-gpu brief`, and MCP `experiment_brief` rank controller health,
  attention items (failed jobs with OOM budget hints, pending jobs grouped by
  cause, offline nodes, active watchdog findings, failed webhook deliveries),
  queue and active plans, a default capacity dry-run, event counts in the
  window, and recommendations. Built from a CLI process, it recognizes a
  separate controller through the database lock and telemetry freshness.
- Capacity dry-run: `GET /api/capacity`, `super-gpu capacity`, and MCP
  `capacity_query` walk the real placement engine with synthetic leases to
  report how many jobs of a given shape could start now, on which GPUs, and
  what blocks the next one; observations, never reservations.
- Plan preview: `POST /api/plans/preview`, `super-gpu preview`, and MCP
  `plan_preview` validate a plan and report per job `fits_fleet`,
  `would_start_now`, placement or `blocked_by`, the estimate, and plan-level
  warnings (shapes that fit no node, heuristic memory budgets, missing
  snapshot paths) without submitting.
- Bounded long-poll waits: `GET /api/plans/<id>/wait` and
  `GET /api/jobs/<id>/wait` (`?timeout=` up to 300 s), `super-gpu wait`
  (chains long-polls; exit 0 completed, 1 failed/cancelled, 3 timeout), and
  MCP `experiment_wait`.
- On-demand logs: `GET /api/jobs/<id>/logs?lines=N`, `super-gpu logs`, and
  MCP `experiment_logs` tail stdout/stderr (and result.json, exit code) from
  the node that ran the job; jobs that never started return their stored
  tails with `source: "stored"`.
- `super-gpu api PATH [--data JSON]` calls any controller route with the
  configured URL and token; `GET /api/events` accepts `since` and `kind`
  filters; `SuperGPUClient` raises `SuperGPUError` with `code` and `status`,
  which the CLI prints as `{"ok": false, "error", "code"}`.

### Changed
- HTTP access logs are written to stderr so stdout carries only data.

### Fixed
- A plan's `finished_at` was rewritten on every scheduler tick; it is now set
  once when the plan becomes terminal.
- A job that finished between the liveness check and the `/proc` start-ticks
  read was reported as lost with an identity mismatch (Linux only).
- GPU process detail panels stayed collapsed across dashboard refreshes.

[Unreleased]: https://github.com/asimfish/super_gpu/commits/main
