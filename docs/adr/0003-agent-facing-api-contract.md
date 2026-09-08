# ADR-0003: Agent-Facing API Contract, Situation Brief, and Dry-Run Placement

## Status

Accepted

## Context

`super_gpu` is meant to be operated by an AI agent end to end. The first
release exposed the scheduler through REST, CLI, and MCP, but the surface
still assumed a human reader in three places:

- Error responses were prose. An agent that wanted to distinguish "the plan
  ID is unknown" from "the node was unreachable" had to parse English, and
  only one error (`idempotency_conflict`) carried a code.
- Supervision required reading several endpoints (`/api/state`, `/api/jobs`,
  `/api/events`, `/api/anomalies`) and re-deriving the same judgments on every
  poll: which pending reasons are normal queueing and which are configuration
  mistakes, whether a failure was an OOM, whether GPUs sit idle while nothing
  is queued.
- The only way to learn whether a job could run on the fleet was to submit it
  and read `pending_reason` afterwards, and the only way to wait for a plan
  was to poll in a loop with a client-chosen interval.

Mocop (an SSH-based GPU monitor built around the same "AI-native" idea)
demonstrated a set of conventions that address the first two points for a
monitor: a public self-describing `/api/meta`, a stable error-code catalog, a
`/api/brief` situation summary, `/api/capacity` matching, `/healthz` and
`/readyz`, and a generic CLI passthrough. This decision adapts them to a
scheduler, where the relevant questions are about placement rather than
observation.

## Candidates

### Contract discovery

- Document the routes in the README only. Cheap, but the agent has to fetch a
  file that may not match the running version, and nothing catches drift.
- A public `GET /api/meta` generated from one manifest module that the
  handlers, the README tables, `AGENTS.md`, and the MCP registration are all
  tested against.

### Errors

- Keep free-text errors; agents pattern-match. Fragile and language-bound.
- A closed catalog of stable `code` values with fixed HTTP statuses, published
  by `/api/meta`, and one envelope for every error including unknown routes
  and wrong methods.

### Supervision

- Leave composition to the agent. Every agent re-implements the same rules and
  pays for several round trips per check.
- One `GET /api/brief` document composed from the projections the other
  endpoints already serve (so it cannot disagree with them), ordered worst
  first, with pending reasons grouped into causes and recommendations attached.
  A plain-text rendering keeps the token cost of a check low.

### Knowing before submitting

- Submit and inspect `pending_reason`. Creates a plan for a question.
- Run the real `PlacementEngine` against the latest snapshot and active leases
  with synthetic leases added after each hypothetical placement
  (`GET /api/capacity`), and apply the same walk to every job of a plan
  (`POST /api/plans/preview`) including a structural `fits_fleet` check
  against an unloaded fleet. Nothing is written; the answer is an observation.

### Waiting

- Client-side polling loops. Interval is guesswork and every poll costs a
  round trip.
- Bounded server-side long-poll (`/api/plans/<id>/wait`, `?timeout` capped at
  300 s) that returns the record with `terminal: true|false` either way; the
  CLI chains calls so proxies never see a request longer than the cap.

## Decision

Adopt the manifest-driven contract (`api_manifest.py`, `GET /api/meta`,
stable error codes, `/healthz`, `/readyz`), the situation brief
(`brief.py`, `GET /api/brief`, `super-gpu brief`, MCP `experiment_brief`),
dry-run placement (`capacity.py`, `GET /api/capacity`,
`POST /api/plans/preview`), bounded long-poll waits, on-demand log retrieval
from the node, and a generic `super-gpu api PATH` passthrough.

Two scheduler-specific rules shape the brief:

- A pending job is only an *attention* item when its cause is something the
  operator can change. Waiting for dependencies or for GPUs to free up is
  normal queueing and stays informational; being blocked by `max_parallel`
  or by an impossible constraint while the capacity probe shows free slots is
  a warning with a recommendation.
- Watchdog findings stay report-only for unmanaged processes, exactly as in
  ADR-0001; the brief points at the owner, never at a signal.

Capacity and preview never reserve GPUs. A submission that follows a preview
may still queue if another submission took the slots first; the preview's
`fits_fleet` answers the structural question (could this ever run here) and
`would_start_now` the momentary one.

## Consequences

- A route, error code, or MCP tool changes in `api_manifest.py` first; the
  README tables in both languages, `AGENTS.md`, and the MCP registration are
  checked against it by tests, so the documentation cannot silently lag.
- Agents can branch on `code`, discover limits (`wait_timeout_max_seconds`,
  `log_lines_max`) before calling, and treat `/readyz` as the gate for
  submitting work after a controller restart.
- The brief adds one composed read per supervision step instead of four to
  six; its text form is intended for tool output budgets.
- Dry-run placement reuses the production engine, so it inherits shared-node
  stabilization semantics: with no telemetry history (for example a
  controller that has not ticked), shared nodes report "waiting for quiet
  samples" rather than free capacity, which is the conservative answer.
- The brief built by a CLI process cannot see the serving process's tick
  counter; it infers controller liveness from the database lock and the age
  of the freshest telemetry sample, which every tick refreshes.
