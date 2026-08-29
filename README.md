# super_gpu

**Agent-ready, resource-aware multi-server GPU experiment scheduler.**

[![CI](https://github.com/asimfish/super_gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/asimfish/super_gpu/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [简体中文](README.zh-CN.md)

`super_gpu` continuously samples per-GPU memory, utilization, temperature,
power, and process telemetry across your servers, places experiments
automatically according to server roles and declared resource needs, and
supervises them end to end — renewing leases, retrying failures, and
backfilling freed GPUs without human intervention. It is designed to be handed
directly to an AI agent: point the agent at this repository, give it your
server list and experiment plan, and it can run the whole campaign.

## Table of Contents

- [Highlights](#highlights)
- [Hand It to an Agent](#hand-it-to-an-agent)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Node Policy](#node-policy)
- [Experiment Plans](#experiment-plans)
- [Resource Estimation](#resource-estimation)
- [MCP Server](#mcp-server)
- [Idle-GPU Watchdog](#idle-gpu-watchdog)
- [REST API](#rest-api)
- [Security](#security)
- [Current Limitations](#current-limitations)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Highlights

- **Dedicated servers** are packed aggressively: best-fit memory placement and
  a target utilization keep fragmentation low and parallel throughput high.
- **Shared servers** are protected: a job is admitted only after memory *and*
  utilization stay below conservative thresholds for several consecutive
  samples, so other people's workloads are never disturbed.
- **Continuous backfill**: the scheduler re-samples the cluster every few
  seconds and immediately places pending jobs on any GPU that has stably freed
  up.
- **Resource estimation that learns**: the first run uses declared budgets or
  conservative heuristics; afterwards the recorded peak memory and runtime of
  identical jobs drive future placement. An OOM automatically raises the next
  memory budget.
- **Reliable execution**: remote commands run under a persistent runner with
  durable PID, log, and exit-code files, so a controller restart can resume
  supervision of every job.
- **Explainable scheduling**: every pending job carries a live
  `pending_reason` — which rule excluded each node, which limit was reached,
  or which dependency it is waiting for — visible in the dashboard, API, and
  CLI, so "why is this not running yet" never needs guesswork.
- **Typed results, not just exit codes**: a job may write a JSON verdict
  (for example `scientific_reject`) to `$SUPER_GPU_RESULT_FILE`; dependency
  predicates such as `{"job": "probe", "after": "result", "result_states":
  [...]}` route downstream jobs on those verdicts, and unsatisfiable branches
  are skipped rather than failed.
- **Declared outputs**: jobs list workspace-relative globs under `outputs`;
  after the job finishes, `super-gpu pull <job-id>` expands them on the node
  that ran the job and streams one archive back — no manual scp archaeology.
- **Immutable source snapshots**: submission builds a deterministic,
  SHA-256 content-addressed archive of your source tree; remote hosts verify
  the digest before unpacking a private copy per job, and an upload is
  skipped entirely when the node already holds the same digest.
- **Idempotent submission**: a stable `request_id` and an intent digest are
  committed in one transaction. Retrying a timed-out submit returns the
  original plan; the same ID with different content returns HTTP 409.
- **Strong job identity**: the runner verifies launch token, PID, *and* Linux
  process start ticks before signaling, so a recycled PID is never killed by
  mistake.
- **Agent-first interfaces**: CLI, REST, and MCP, plus an in-repo
  [`AGENTS.md`](AGENTS.md) operating contract and a
  [JSON Schema](schemas/experiment-plan.schema.json) for plans.
- **Live dashboard**: a zero-CDN web UI showing every server, GPU, lease, job
  queue, and scheduler event in real time.

## Hand It to an Agent

Give an agent this repository URL, your experiment plan, and your server list.
Agents that honor repository instructions will pick up [`AGENTS.md`](AGENTS.md)
automatically; environments that support Agent Skills can also load
[`skills/super-gpu`](skills/super-gpu/SKILL.md).

The in-repo contract instructs the agent to:

1. Clone and install `super_gpu`.
2. Convert your server list into a private, never-committed `config.json`,
   marking every node `dedicated` (yours) or `shared` (other users present).
3. Validate the plan and check SSH, workspaces, and `nvidia-smi`.
4. Start or reuse a long-running controller and dashboard.
5. Submit the plan and supervise status and scheduler events until every job
   reaches a terminal state.
6. Let the controller backfill freshly freed, stabilized GPUs on later scans —
   no manual GPU picking.

A ready-to-send prompt:

```text
Use https://github.com/asimfish/super_gpu. Follow the repository AGENTS.md and
skills/super-gpu/SKILL.md to validate and orchestrate my experiments; keep
supervising until every job is terminal, and protect other users' workloads on
the shared servers.
```

The controller host still needs non-interactive SSH access to the servers —
the repository URL never grants credentials. For a program that has never run
before, no system can know its exact memory footprint from the command line
alone: declare a budget for the first run or accept the conservative
heuristic, and the historical peak takes over afterwards.

## Architecture

```text
Experiment plan / Agent / Dashboard
                 │
          REST · CLI · MCP
                 │
        ┌────────▼────────┐
        │ Scheduler loop  │  monitor → reconcile → renew → place/backfill
        └───┬─────────┬───┘
            │         │
        SQLite      SSH runner
     desired state   persistent PID/log/exit files
            │         │
            └──── GPU servers
```

Each scheduling cycle runs strictly in this order:

1. Query `nvidia-smi` on all nodes in parallel.
2. Recover and supervise `starting/running/cancelling` jobs.
3. Update peak memory, logs, and lease heartbeats.
4. Handle completions, failures, OOM, timeouts, cancellations, and retries.
5. Place pending jobs against the freshest snapshot.
6. Launch everything that can be placed safely right now.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component boundaries and
the failure model, and [docs/adr/](docs/adr) for the design decisions behind
snapshotting, idempotency, and the watchdog.

## Quick Start

Requirements: Python 3.10+ on the controller host; NVIDIA GPUs with
`nvidia-smi` on the target servers, reachable non-interactively through
aliases in `~/.ssh/config`.

```bash
git clone https://github.com/asimfish/super_gpu.git
cd super_gpu
python3 -m venv .venv
.venv/bin/pip install -e ".[mcp]"

cp examples/nodes.example.json config.json
# edit node aliases, role, workspace, and thresholds

# already have a gpumgr node inventory? generate an all-shared private config:
.venv/bin/super-gpu import-gpumgr \
  --source ~/.config/gpumgr/nodes.json \
  --output ~/.config/super_gpu/config.json

.venv/bin/super-gpu --config config.json validate \
  --plan examples/experiment.example.json
.venv/bin/super-gpu --config config.json doctor
.venv/bin/super-gpu --config config.json serve
```

Open <http://127.0.0.1:8765> for the live dashboard. Submit a plan from
another terminal:

```bash
.venv/bin/super-gpu submit examples/experiment.example.json \
  --request-id learning-rate-sweep-001 \
  --url http://127.0.0.1:8765

.venv/bin/super-gpu jobs --url http://127.0.0.1:8765
.venv/bin/super-gpu events --url http://127.0.0.1:8765
```

Or skip the HTTP service entirely and block until the plan finishes:

```bash
.venv/bin/super-gpu --config config.json run experiment.json
```

`scripts/supergpu` is an optional macOS launchd wrapper
(`supergpu start|stop|restart|status|logs|open`) for keeping a controller
running across logins; adapt the plist label and port to your machine.

## Node Policy

Every node must declare its role explicitly:

```json
{
  "name": "main-a100",
  "ssh": "main-a100",
  "role": "dedicated",
  "priority": 100,
  "workspace": "/data/project",
  "policy": {
    "max_gpu_utilization": 94,
    "max_memory_used_ratio": 0.94,
    "reserve_memory_mib": 2048,
    "stabilization_samples": 1,
    "allow_colocation": true,
    "max_jobs_per_gpu": 2
  }
}
```

| Field | Meaning |
|---|---|
| `role` | `dedicated`: yours, pack aggressively. `shared`: conservative admission |
| `priority` | explicit ordering among nodes of the same role |
| `max_gpu_utilization` | stop adding jobs once current utilization reaches this |
| `max_memory_used_ratio` | maximum memory watermark after placing a new job |
| `reserve_memory_mib` | headroom for the driver, fluctuation, and estimation error |
| `stabilization_samples` | consecutive samples below thresholds before admission; ≥3 recommended for shared nodes |
| `allow_colocation` | whether multiple super_gpu jobs may share one GPU |
| `max_jobs_per_gpu` | maximum managed jobs colocated on a single GPU |

Shared nodes default to 35% utilization, 35% memory, 3 consecutive stable
samples, and no colocation. Even a momentarily idle shared GPU is observed for
stability first, so another user's phase transition is never mistaken for a
free card.

## Experiment Plans

Full example: [examples/experiment.example.json](examples/experiment.example.json).
Machine-readable constraints: [schemas/experiment-plan.schema.json](schemas/experiment-plan.schema.json).

```json
{
  "request_id": "ablation-001",
  "name": "ablation",
  "source": {
    "mode": "snapshot",
    "path": "/controller/path/to/project",
    "exclude": ["outputs/**", "data/**"]
  },
  "max_parallel": 8,
  "defaults": {
    "max_retries": 1,
    "resources": {
      "gpus": 1,
      "memory_mib": "auto",
      "gpu_utilization": "auto"
    }
  },
  "jobs": [
    {
      "name": "baseline",
      "command": "python3 train.py --config base.yaml",
      "priority": 20,
      "params": {"batch_size": 16}
    }
  ]
}
```

Every job command receives:

```text
CUDA_VISIBLE_DEVICES
SUPER_GPU_PLAN_ID
SUPER_GPU_JOB_ID
SUPER_GPU_JOB_NAME
SUPER_GPU_GPU_INDICES
SUPER_GPU_PARAM_<PARAM_NAME>
PYTHONUNBUFFERED=1
```

`dependencies` refer to job names within the same plan. The plain string form
(`"dependencies": ["baseline"]`) schedules a job after the named dependency
*succeeds*. The object form adds routing on typed results:

```json
{
  "name": "scale-up",
  "command": "python3 train.py --big",
  "dependencies": [
    {"job": "probe", "after": "result", "result_states": ["success"]}
  ]
}
```

- `after: "success"` (default) — run only if the dependency succeeded.
- `after: "complete"` — run once the dependency is terminal, regardless of
  outcome.
- `after: "result"` — run only if the dependency's result state is in
  `result_states`.

A job reports its result by writing JSON to the path in
`$SUPER_GPU_RESULT_FILE` before exiting 0:

```json
{"state": "scientific_reject", "metrics": {"val_acc": 0.51}}
```

Accepted states are `success` and `scientific_reject`; anything else is
recorded as metadata but the state falls back to `success`. Infrastructure
failures (`infra_failure`), non-zero exits (`execution_failure`), and
cancellations are assigned by the scheduler itself. Jobs whose dependency
predicate can never be satisfied are marked `skipped` with result state
`dependency_skipped` instead of `failed`, and the skip cascades downstream.

### Declared Outputs

Jobs may declare the artifacts they produce as workspace-relative glob
patterns (no absolute paths, `..`, or whitespace):

```json
{
  "name": "train",
  "command": "python3 train.py",
  "outputs": ["results/*.json", "checkpoints/**/*.pt", "logs"]
}
```

Once the job is terminal, collect them from the controller host:

```bash
.venv/bin/super-gpu pull <job-id> --dest ./outputs
# files land in ./outputs/<job-name>/, preserving relative paths
```

`GET /api/jobs/<id>/outputs` (and the MCP tool `experiment_outputs`) expands
the patterns on the node without transferring bytes, so an agent can inspect
what exists before pulling. A directory pattern collects every file beneath
it; patterns that match nothing simply return an empty list.

### Snapshot and Idempotency Semantics

- `source.mode: snapshot` reads `source.path` on the controller and builds a
  deterministic `tar.gz` free of unstable metadata (mtime, uid/gid). Identical
  file content always yields the identical digest.
- `.git`, virtual environments, `.env`, and private keys are excluded by
  default; extra exclusions are recorded in the snapshot metadata.
- *Immutable* means the archive object and digest never change. Each job gets
  its own writable extraction; cross-job artifacts belong in an explicit
  shared path or object store.
- The same `request_id` with the same plan/snapshot digest is a safe retry:
  the original plan is returned with `submission.replayed=true`. The same ID
  with different content returns `idempotency_conflict`.
- Intentionally repeating an experiment requires a new request ID.
- Snapshot paths live on the controller host; when submitting remotely via
  REST/MCP, the path must be visible to the controller.

## Resource Estimation

Estimation precedence:

1. `resources.memory_mib` declared on the job or plan.
2. Historical peak of identical command, params, and GPU count, times a safety
   factor.
3. `default_memory_mib` plus a batch-size heuristic.

No system can know an arbitrary program's memory from its command line on the
first run, so declare explicit budgets for critical experiments; after one
run, the historical estimate takes over automatically. Estimates are
per-GPU budgets.

## MCP Server

Start the main service first, then the MCP server:

```bash
SUPER_GPU_URL=http://127.0.0.1:8765 \
  .venv/bin/super-gpu-mcp --transport stdio
```

Exposed tools:

| Tool | Purpose |
|---|---|
| `cluster_snapshot` | latest GPU telemetry for every node |
| `scheduler_status` | tick, queue depth, watchdog, and lease state |
| `experiment_submit` | idempotent plan submission with `request_id` |
| `experiment_status` | one plan with all of its jobs |
| `experiment_jobs` | filterable job listing |
| `experiment_outputs` | expand a finished job's declared outputs on its node |
| `experiment_cancel` | cancel a single job |
| `scheduler_events` | recent scheduling decisions and transitions |
| `anomaly_report` | idle-yet-occupied GPU findings and watchdog policy |

HTTP transport is also available:

```bash
SUPER_GPU_URL=http://127.0.0.1:8765 \
  .venv/bin/super-gpu-mcp --transport http --host 127.0.0.1 --port 8766
```

## Idle-GPU Watchdog

The controller continuously identifies GPUs that hold compute processes and
memory while utilization stays near 0% for a long time. The default policy
only reports:

```bash
export SUPER_GPU_WATCHDOG_ENABLED=true
export SUPER_GPU_WATCHDOG_LOW_UTILIZATION=3
export SUPER_GPU_WATCHDOG_MIN_MEMORY_MIB=1024
export SUPER_GPU_WATCHDOG_GRACE_SECONDS=900
export SUPER_GPU_WATCHDOG_MIN_RUNTIME_SECONDS=1800
export SUPER_GPU_WATCHDOG_ACTION=report
```

Findings are available through `GET /api/anomalies`, the MCP `anomaly_report`
tool, and the `anomalies` field of `/api/state`. External processes are only
ever reported — `super_gpu` never signals them.

Only with the explicit opt-in below will the controller request cancellation,
and only of exclusive jobs it launched itself and still holds a valid lease
for:

```bash
export SUPER_GPU_WATCHDOG_ACTION=cancel_managed
```

Automatic cancellation additionally requires the minimum runtime, a
consecutive low-utilization grace period, the memory threshold, visible GPU
processes, and an exclusive lease. Watchdog observations are persisted in
SQLite; stale samples are discarded rather than counting controller downtime
as idleness. Treat `cancel_managed` as destructive authority — see
[SECURITY.md](SECURITY.md).

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/state` | full cluster, job, lease, and event state for the dashboard |
| GET | `/api/snapshot` | latest GPU snapshot |
| GET | `/api/plans` | list plans |
| GET | `/api/plans/<id>` | one plan with all jobs |
| POST | `/api/plans` | submit a plan JSON; supports top-level `request_id`, conflict → 409 |
| GET | `/api/jobs` | query jobs (each job carries `pending_reason` and `result_state`) |
| GET | `/api/jobs/<id>/outputs` | expand a finished job's declared outputs on its node |
| POST | `/api/jobs/<id>/cancel` | cancel a job |
| POST | `/api/scan` | refresh GPU state immediately |
| GET | `/api/events` | scheduler events |
| GET | `/api/anomalies` | current low-utilization occupancy and watchdog policy |

## Security

Access to the CLI, REST API, or MCP server is equivalent to shell access on
the configured GPU servers. The service binds to loopback by default; a
non-loopback bind is rejected unless an API token is configured:

```bash
export SUPER_GPU_API_TOKEN='<long-random-token>'
super-gpu --config config.json serve --host 0.0.0.0
```

Keep the service behind a VPN, SSH tunnel, or private network regardless.
Real `config.json` files and state databases are git-ignored. Full guidance,
including how to report a vulnerability: [SECURITY.md](SECURITY.md).

## Current Limitations

- The first monitoring backend targets NVIDIA `nvidia-smi`.
- GPU utilization cannot be reliably attributed per process, so shared nodes
  gate on total utilization conservatively.
- Historical peak memory is a conservative estimate under colocation — the
  system prefers placing fewer jobs over risking OOM.
- One SQLite database allows exactly one active scheduler controller; this is
  the safety constraint that prevents double scheduling.
- New jobs verify process identity with a token plus `/proc` start ticks; jobs
  still running from before an upgrade are supervised in PID-only
  compatibility mode until they finish.
- Real multi-server SSH/GPU end-to-end validation requires live server
  credentials and is exercised outside CI.

## Documentation

| Document | Contents |
|---|---|
| [AGENTS.md](AGENTS.md) | the operating contract agents follow when given this repo |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | system boundary, components, failure model |
| [docs/adr/](docs/adr) | architecture decision records |
| [SECURITY.md](SECURITY.md) | threat model, hardening, vulnerability reporting |
| [skills/super-gpu/SKILL.md](skills/super-gpu/SKILL.md) | installable Agent Skill |
| [schemas/experiment-plan.schema.json](schemas/experiment-plan.schema.json) | plan JSON Schema |

## Development

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

CI runs the test suite on Python 3.10, 3.11, and 3.12 for every push and pull
request. Bug reports and pull requests are welcome — please include a failing
test or a reproduction where possible, and run the suite before submitting.

## License

[MIT](LICENSE)
