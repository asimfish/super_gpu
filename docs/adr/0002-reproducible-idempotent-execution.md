# ADR-0002: Reproducible, Idempotent Experiment Execution

## Status

Accepted

## Context

An Agent may retry a submission after a timeout, source files may change while
jobs are queued, and a numeric PID may be reused after its original process
exits. Those three cases can respectively create duplicate experiments,
execute different code under one plan, or signal an unrelated process.

## Candidates

### Source identity

- Record the current workspace path or Git commit. This is cheap, but does not
  capture dirty/untracked files and allows the workspace to change later.
- Build a deterministic, content-addressed source archive. This captures the
  submitted bytes and can be verified before every remote extraction.

### Retry identity

- Rely on caller-selected plan IDs. This overloads a domain identifier and
  cannot distinguish a safe replay from conflicting content.
- Persist a request receipt containing `request_id`, an intent digest, and the
  resulting plan ID in the same SQLite transaction as the plan and jobs.

### Process identity

- Trust a PID and run directory. PIDs can be reused and stale state can target
  a different process.
- Bind the handle to a random launch token and `/proc/<pid>/stat` start ticks,
  and verify both before polling or signalling.
- Require systemd user units. This is stronger on managed Linux hosts, but is
  not available on every research server and would remove the portable SSH
  runner.

## Decision

Choose content-addressed source archives, transactional request receipts, and
portable token/start-tick process identity.

Snapshot mode is explicit in the plan. The controller creates a deterministic
`tar.gz`, stores it by SHA-256 digest, uploads it by digest, verifies the remote
bytes, and extracts a private per-job execution copy. The immutable archive is
the experiment source identity; jobs may write inside their execution copy.

When `request_id` is present, the first submission atomically binds it to an
intent digest and plan ID. An identical retry returns the original plan with
`replayed: true`; different content returns an idempotency conflict and never
creates another plan.

New runner handles contain a launch token and Linux process start ticks. Poll
and cancellation verify the on-host identity file. Cancellation refusal keeps
the job and GPU lease active so the controller cannot falsely report success
or signal a reused PID. Handles created by older releases remain readable and
use the legacy PID-only path until they finish.

## Consequences

- Snapshot archives consume controller and remote disk and are deduplicated by
  digest; garbage collection is a later operational feature.
- Snapshot source paths are controller-local. Remote API clients must provide
  a path visible to the controller.
- A stable `request_id` is required for retry safety; intentionally repeated
  experiments use a new request ID.
- Token/start-tick identity is safer than PID-only execution while remaining
  portable. A future runner may add systemd units behind the same handle
  contract for hosts that support them.
