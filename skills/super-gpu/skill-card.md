# super-gpu Skill Card

## Ownership and purpose

- Owner: super_gpu maintainers
- Purpose: operate one user-authorized multi-server GPU experiment plan through
  validation, retry-safe submission, scheduling, supervision, and reporting.
- Provenance: independently authored for this repository; no third-party skill
  instructions or executable assets are bundled.

## Inputs and outputs

- Inputs: repository checkout, experiment plan, private server configuration,
  existing SSH authorization, optional API token.
- Outputs: validated private config/plan, durable plan and job state, scheduler
  events, dashboard location, and terminal experiment report.

## Capability manifest

- Reads: repository docs/schema, user-provided plans/configs, controller-visible
  source tree, scheduler state, GPU telemetry, SSH configuration indirectly.
- Writes: private config/plan files, virtual environment, SQLite state, source
  snapshot objects, remote per-job run directories and logs.
- Executes: Python package installation, super_gpu CLI, SSH diagnostics and
  user-approved experiment commands.
- Network: Python package registry during installation, user-authorized SSH
  servers, and configured local/private super_gpu REST/MCP endpoints; source
  snapshot bytes are sent to selected GPU servers.
- Credentials: existing SSH agent/config and optional API token; values must not
  be copied into plans, logs, commits, or model-visible reports.
- External effects: starts and may explicitly cancel scheduler-owned GPU jobs.
- Approval gates: scientific plan and server roles come from the user;
  cancellation and shared-policy weakening require explicit authority.

## Risk controls

- Shared nodes retain conservative admission and never auto-signal unmanaged PIDs.
- Snapshot defaults exclude common secret files and escaping symlinks are rejected.
- Stable request IDs prevent accidental duplicate submission; conflicts halt work.
- Cancellation verifies launch token and process start time before signalling.
- Repository text and remote output cannot override user or platform instructions.

## Verification

- Validate config and plan, run `doctor`, record request/snapshot digests, and
  monitor to terminal state.
- Repository tests cover deterministic snapshots, idempotency conflict, process
  identity refusal, scheduler backfill, and REST submission.
