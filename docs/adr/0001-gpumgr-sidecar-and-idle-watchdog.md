# ADR-0001: gpumgr Sidecar Integration and Idle GPU Watchdog

## Status

Accepted

## Context

gpumgr already provides cluster inventory, live GPU telemetry, and explicit
operator process termination. super_gpu already provides durable experiment
plans, leases, placement, retries, REST, MCP, and remote runners. Agents need a
single task-submission path that can use gpumgr's servers without creating a
second copy of the scheduling logic. The system also needs to detect jobs that
hold GPU memory while producing sustained near-zero GPU utilization.

## Driving Factors

- Preserve one placement authority and one durable scheduler database.
- Reuse existing SSH aliases without copying credentials.
- Protect other users on shared servers.
- Keep arbitrary process termination outside the autonomous agent boundary.
- Permit conservative, opt-in cleanup of jobs launched by this scheduler.

## Candidates

### Option A: Embed super_gpu in the gpumgr Dashboard Process

- Pros: one HTTP port and one visual surface; direct reuse of gpumgr telemetry.
- Cons: couples scheduler availability to Dashboard restarts, creates competing
  state models, increases the blast radius of monitoring changes, and makes
  single-controller ownership harder to enforce.

### Option B: Run super_gpu as a Sidecar and Import gpumgr Inventory

- Pros: preserves the released scheduler and its SQLite ownership model;
  failures remain isolated; REST/MCP are already Agent-ready; gpumgr remains a
  read-mostly operator system.
- Cons: two local services must be supervised and node metadata needs a small
  import adapter.

## Decision

Chosen: Option B. super_gpu imports gpumgr's JSON node inventory into a private
configuration whose nodes default to `shared`. Agents submit experiment-plan
JSON to super_gpu through REST or MCP. gpumgr continues to provide manual fleet
inspection and manual termination for unmanaged processes.

The idle watchdog has two modes:

- `report`: report sustained low-utilization occupancy for managed and
  unmanaged processes without side effects.
- `cancel_managed`: after the configured runtime and grace period, request
  cancellation only for an active job owned by the scheduler database.

No mode automatically signals an unmanaged PID. This is an authorization
boundary, not merely a default setting.

## Impact

- Add a gpumgr inventory import command and private-config workflow.
- Add config-driven watchdog policy and an in-memory observation tracker.
- Expose watchdog findings through REST/MCP and scheduler state.
- Record every automatic cancellation as an audit event.
- Controller restart resets watchdog grace timers, favoring false negatives
  over accidental termination.

