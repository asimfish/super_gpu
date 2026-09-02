# super_gpu Architecture

## System Boundary

`super_gpu` is the scheduling authority for experiments that it accepts. It
owns desired job state, GPU leases, remote runner handles, retries, and job
cancellation. `gpumgr` remains the inventory and operator-facing monitoring
system. The two systems are integrated as sidecars rather than sharing an
in-process scheduler.

```text
Agent / operator
      |
      | experiment plan JSON, REST, MCP
      v
super_gpu controller ---- SQLite desired state
      |
      | SSH aliases from the imported gpumgr inventory
      v
GPU servers
      ^
      | read-only inventory and operator actions
      |
gpumgr dashboard
```

The canonical boundary formats are JSON for node configuration and experiment
plans, HTTP JSON for service integration, and SQLite for scheduler state. SSH
credentials and connection topology remain in the operator's `~/.ssh/config`.

## Ownership Rules

- Only one active `super_gpu` controller may own a scheduler database.
- GPU placement for submitted plans must go through `super_gpu` leases.
- Imported gpumgr nodes default to `shared`. Promotion to `dedicated` requires
  an explicit operator decision.
- The watchdog may automatically cancel only jobs launched and tracked by the
  current `super_gpu` database.
- Unmanaged GPU processes are observable anomalies. They are never terminated
  automatically by `super_gpu`; an operator can investigate them in gpumgr.

## Main Components

- `ClusterMonitor`: collects node, GPU, and process telemetry.
- `PlacementEngine`: applies dedicated/shared admission and best-fit placement.
- `Scheduler`: reconciles desired job state, leases, runners, and retries.
- `IdleGpuGuardian`: identifies sustained low-utilization GPU occupancy and
  proposes managed-job cancellation according to configuration.
- `StateStore`: persists plans, jobs, leases, telemetry, and audit events.
- `SourceSnapshotStore`: builds deterministic source archives and addresses
  them by SHA-256 digest.
- `Notifier`: observes the store's committed event stream and posts selected
  events to configured webhooks from a background worker with bounded
  retries; it never blocks a tick and never forwards its own failures.
- REST/MCP adapters: expose typed scheduling and inspection operations to
  local agents and operators.

## Reliable Submission Boundary

Submission has three durable identities:

1. `source_snapshot.digest` identifies the exact submitted source bytes.
2. `request_id` plus `intent_digest` identifies one retry-safe API intent.
3. `launch_token` plus process start ticks identifies one remote process.

The snapshot is created before the database transaction; an interrupted
submission may leave an unreferenced content object, but cannot leave a plan
without jobs or an idempotency receipt without its plan. Remote extraction is
allowed only after the archive digest is verified.

## Failure Model

SSH and telemetry failures do not release live leases. Watchdog observations
must be consecutive and time-bounded and are persisted in SQLite. Automatic
cancellation uses the existing durable job handle and runner cancellation path,
never a telemetry PID supplied by a client. If process identity cannot be
proven, cancellation is refused and the lease remains active.
