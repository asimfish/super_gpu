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
7. Submit through MCP `experiment_submit`, the REST `POST /api/plans`, or:
   `super-gpu submit PLAN.json --url http://127.0.0.1:8765`.
8. Monitor `experiment_status`, `scheduler_events`, and the dashboard until the
   plan is `completed`, `failed`, or `cancelled`.
9. Report failed jobs with their exit code and stderr tail. Do not silently
   change scientific parameters to make a failed experiment pass.
10. Do not stop supervising immediately after submission unless the user asks
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
