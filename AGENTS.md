# super_gpu — Agent Operating Contract

This repository is an experiment scheduler, not an experiment implementation.
Use it to validate cluster access, submit a user-approved experiment plan, and
monitor that plan until it reaches a terminal state.

## Repository-link workflow

When the user gives you this repository URL together with an experiment plan
and server list, clone or open the repository, then follow this contract
without waiting for the user to restate it. The installable Agent Skill is at
`skills/super-gpu/SKILL.md`.

The only additional inputs that may be required are working SSH access and
scientific details that cannot safely be inferred. A repository URL never
grants server credentials.

## Safe workflow

1. Read `README.md`, `examples/nodes.example.json`, and
   `schemas/experiment-plan.schema.json`.
2. Never invent SSH credentials. The user must provide working aliases in
   `~/.ssh/config`.
3. Copy the node example to a private `config.json`. Mark every node as either:
   - `dedicated`: owned by the user; aggressive packing is allowed.
   - `shared`: other users may be present; conservative admission is required.
4. Run `super-gpu --config config.json validate --plan PLAN.json`.
5. Run `super-gpu --config config.json doctor` before submitting work.
6. Prefer a long-running controller:
   `super-gpu --config config.json serve`.
7. Give every logical submission a stable `request_id`, and reuse that exact ID
   only when retrying the same plan. Prefer immutable source execution by adding
   `source: {"mode": "snapshot", "path": "/controller/path/to/project"}`.
8. Submit through MCP `experiment_submit`, the REST `POST /api/plans`, or:
   `super-gpu submit PLAN.json --request-id ID --url http://127.0.0.1:8765`.
9. Monitor `experiment_status`, `scheduler_events`, and the dashboard until the
   plan is `completed`, `failed`, or `cancelled`.
10. Report failed jobs with their exit code and stderr tail. Do not silently
   change scientific parameters to make a failed experiment pass.
11. Do not stop supervising immediately after submission unless the user asks
    for a detached handoff. Stable capacity discovered on later scans is
    automatically backfilled by the controller.

## Hard rules

- Do not bypass super_gpu placement by manually choosing `CUDA_VISIBLE_DEVICES`.
- Do not change a shared node to `dedicated` without explicit user approval.
- Do not raise shared-node utilization or memory thresholds without approval.
- Do not expose the HTTP service beyond loopback without an API token and
  network-level access control.
- Do not commit real cluster configs, tokens, experiment databases, logs, or
  SSH keys.
- Treat `source.path` as a path on the controller host. Do not snapshot secret
  material; default secret-file patterns are excluded and custom exclusions
  must be recorded in the plan.
- An `idempotency_conflict` means the request ID was reused for different
  content. Never bypass it; choose a new ID only for an intentional new run.
- Cancellation is an external side effect. Only cancel jobs the user requested
  to cancel, or jobs created by the current plan when cleanup was explicitly
  requested.

## MCP tools

- `cluster_snapshot`
- `scheduler_status`
- `experiment_submit`
- `experiment_status`
- `experiment_jobs`
- `experiment_cancel`
- `scheduler_events`
- `anomaly_report`

## Watchdog authority

- Treat unmanaged idle-GPU findings as report-only. Never turn a telemetry PID
  into an automatic signal action.
- `cancel_managed` may cancel only a job owned by the active scheduler database
  after its configured runtime and grace period.
- Enabling `SUPER_GPU_WATCHDOG_ACTION=cancel_managed` is a destructive policy
  change and requires explicit operator intent.
