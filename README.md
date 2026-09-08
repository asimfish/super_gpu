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

![super_gpu dashboard — live fleet view](https://raw.githubusercontent.com/asimfish/super_gpu/main/docs/assets/dashboard-fleet.png)

<sub>Reproducible without any GPU server: `python3 scripts/demo_dashboard.py`
seeds a simulated three-node cluster and serves the real dashboard on top of
it. See [Try It First](#try-it-first-no-gpus-required).</sub>

## Table of Contents

- [Highlights](#highlights)
- [Why super_gpu?](#why-super_gpu)
- [Hand It to an Agent](#hand-it-to-an-agent)
- [Agent Workflow](#agent-workflow)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Node Policy](#node-policy)
- [Experiment Plans](#experiment-plans)
- [Resource Estimation](#resource-estimation)
- [MCP Server](#mcp-server)
- [Idle-GPU Watchdog](#idle-gpu-watchdog)
- [Notifications](#notifications)
- [REST API](#rest-api)
- [Security](#security)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
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
- **Self-describing API**: public `GET /api/meta` lists every route with its
  access tier, the stable error-code catalog, the MCP tools, and capability
  flags; every error body carries a machine-readable `code`, and `/healthz`
  versus `/readyz` tells "alive" from "has scheduled at least once" apart.
- **Situation brief**: `GET /api/brief` (or `super-gpu brief`) is the whole
  supervision check on one screen — controller health, what needs a decision
  (failures with OOM hints, blocked jobs grouped by cause, offline nodes,
  idle-yet-occupied GPUs), queue, free capacity, changes in the window, and
  concrete recommendations.
- **Know before you submit**: `GET /api/capacity` dry-runs the real placement
  engine ("how many 2-GPU 40 GiB jobs could start now, and where?") and
  `POST /api/plans/preview` explains, per job, whether it would start now,
  queue, or never fit the fleet — nothing is written.
- **Wait and logs instead of polling loops**: bounded long-poll
  `/api/plans/<id>/wait`, and `/api/jobs/<id>/logs` tails stdout/stderr from
  the node that ran the job on demand.
- **Live dashboard**: a zero-CDN web UI showing every server, GPU, lease, job
  queue, and scheduler event in real time.
- **Push notifications**: plan and job outcomes delivered to Feishu, Slack,
  or any HTTP endpoint, so nobody has to poll.

## Why super_gpu?

| Alternative | Typical fit | Where super_gpu differs |
|---|---|---|
| Slurm / K8s + Kueue | large managed clusters you administer | zero cluster infrastructure: any box you can SSH into becomes a node in minutes, including shared lab machines you do *not* administer |
| Ray and similar frameworks | code written against the framework API | jobs stay plain shell commands — nothing to import, no daemon on the nodes |
| gpustat / nvitop-style monitors | watching GPUs by eye | the same telemetry feeds an actual scheduler that places, supervises, retries, and backfills |
| tmux + `CUDA_VISIBLE_DEVICES` by hand | one machine, a handful of runs | learned resource estimation, shared-node etiquette, idempotent submission, typed results, output collection |

The design target is the gap between "my lab runs Slurm" and "I have SSH
access to a few machines, some of them shared": pack your dedicated machines
aggressively, never disturb other people's work on shared ones, and stay
operable end to end by an AI agent.

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
5. Preview the plan against live capacity, then submit it with a stable
   `request_id`.
6. Supervise with `wait` and the situation brief until every job reaches a
   terminal state, pulling logs for anything that fails.
7. Let the controller backfill freshly freed, stabilized GPUs on later scans —
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

## Agent Workflow

Every question an agent asks while running a campaign has one bounded call.
The CLI forms below talk to a running controller (`--url` or
`SUPER_GPU_URL`); the same operations exist as REST routes and MCP tools.

| Question | Call |
|---|---|
| What does this controller support? | `super-gpu api /api/meta` — routes, tiers, error codes, MCP tools, limits |
| Is it alive and scheduling? | `GET /healthz`, `GET /readyz` (503 with `not_ready` until the first tick) |
| How much could I start right now? | `super-gpu capacity --gpus 2 --memory-mib 40960` |
| Will this plan run, and where? | `super-gpu preview plan.json` |
| Submit (retry-safe) | `super-gpu submit plan.json --request-id sweep-001` |
| Block until it finishes | `super-gpu wait <plan-id> [--timeout 3600]` — exit 0 completed, 1 failed, 3 timeout |
| What needs my attention? | `super-gpu brief` (text) or `super-gpu brief --json` |
| Why did a job fail? | `super-gpu logs <job-id> --lines 300 --text` |
| Collect artifacts | `super-gpu pull <job-id> --dest ./outputs` |

A typical loop:

```bash
export SUPER_GPU_URL=http://127.0.0.1:8765
super-gpu preview plan.json            # fits_fleet / would_start_now per job, warnings
super-gpu submit plan.json --request-id sweep-001
until super-gpu wait plan-abc --timeout 600; do   # returns 3 while still running
  super-gpu brief                      # one screen: attention, capacity, recommendations
done
super-gpu logs job-def --text          # for anything the brief reports as failed
```

The brief is the same document as JSON or text; the text form is one screen,
worst first:

```text
super_gpu brief 2026-09-08T08:34:36Z | window 6h | status ATTENTION
controller: running | tick 3s ago (#1284) | poll 5s | v0.2.0
fleet: 3/3 nodes online | 16 GPUs: 6 leased, 5 idle, 5 busy (unmanaged)
queue: 4 active, 3 pending | window: 2 completed, 1 failed | plans active: 1
  llm-scaling-ablation [plan-22f28d958820]: running | 2 completed, 3 pending, 4 running

attention (2): 0 critical, 2 warning, 0 info
   ! job lr-1e-3 failed on main-a100 (exit 1, attempt 2): RuntimeError: CUDA out of memory
       -> CUDA OOM: declare resources.memory_mib >= 28672 for the retry
   ! 3 pending jobs blocked by cluster max parallel (e.g. wd-sweep-0.1: cluster max_parallel (6) reached; oldest 4m)
       -> raise max_parallel in config.json; GPUs are free

capacity now: 8 slots for 1-GPU jobs (20224 MiB heuristic) | cluster max_parallel leaves 2 | effective 2
  lab-shared-v100 (shared): 0 slots - gpu0 utilization 92% >= limit 35%; ...
  main-4090 (dedicated): 2 slots on gpu 2,3
  main-a100 (dedicated): 6 slots on gpu 4,5,6,7

changes (6h): 1 job_failed, 2 job_completed, 7 job_started

recommendations:
  - cluster max_parallel (6) is the bottleneck while 8 GPU slots are free: raise max_parallel in config.json
  - job lr-1e-3: CUDA OOM: declare resources.memory_mib >= 28672 for the retry
```

Capacity and preview are observations, not reservations: another submission
may take the GPUs first, and only submitting a plan holds them. A job whose
`fits_fleet` is false in the preview would never run on the current fleet
(for example four GPUs on nodes with two), so fix it before submitting rather
than discovering it through `pending_reason` later.

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

### Try It First (no GPUs required)

```bash
git clone https://github.com/asimfish/super_gpu.git
cd super_gpu
python3 scripts/demo_dashboard.py
```

This seeds a simulated three-node fleet (two dedicated, one shared with other
users' workloads), runs a small ablation plan through the real scheduler, and
serves the dashboard at <http://127.0.0.1:8899> — completed, running, and
pending jobs included, each pending job carrying its live `pending_reason`.
Every screenshot in this README comes from that command. The agent commands
work against it too:

```bash
export SUPER_GPU_URL=http://127.0.0.1:8899
python3 -m super_gpu.cli brief                      # one-screen situation brief
python3 -m super_gpu.cli capacity --gpus 2 --memory-mib 40960
python3 -m super_gpu.cli api /api/meta              # everything the API supports
```

### Real Cluster

Requirements: Python 3.10+ on the controller host; NVIDIA GPUs with
`nvidia-smi` on the target servers, reachable non-interactively through
aliases in `~/.ssh/config`. The core has no third-party runtime
dependencies; `[mcp]` adds the MCP server.

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

![Experiment queue with live pending reasons and scheduler events](https://raw.githubusercontent.com/asimfish/super_gpu/main/docs/assets/dashboard-queue.png)

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
| `experiment_brief` | one-screen situation brief (text by default, `format="json"` for the document) |
| `capacity_query` | dry-run placement for a GPU/memory shape: slots now, and where |
| `plan_preview` | validate a plan and preview estimates and placement without submitting |
| `experiment_submit` | idempotent plan submission with `request_id` |
| `experiment_wait` | block up to 300 s until a plan is terminal; returns `terminal` and the plan either way |
| `experiment_status` | one plan with all of its jobs |
| `experiment_jobs` | filterable job listing |
| `experiment_logs` | last N lines of a job's stdout/stderr fetched from its node |
| `experiment_outputs` | expand a finished job's declared outputs on its node |
| `experiment_cancel` | cancel a single job |
| `scheduler_events` | recent scheduling decisions and transitions (optionally `since` an epoch time) |
| `anomaly_report` | idle-yet-occupied GPU findings and watchdog policy |

The tool list is generated from the same manifest that `GET /api/meta`
serves, and the test suite fails if the two — or this table — drift apart.

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

## Notifications

Instead of polling, let the controller push scheduler events to Feishu,
Slack, or any HTTP endpoint. Webhooks are declared in `config.json`:

```json
"notifications": [
  {"name": "team-feishu", "kind": "feishu",
   "url_env": "SUPER_GPU_FEISHU_WEBHOOK", "secret_env": "SUPER_GPU_FEISHU_SECRET"},
  {"name": "lab-slack", "kind": "slack", "url_env": "SUPER_GPU_SLACK_WEBHOOK"},
  {"name": "pipeline", "kind": "generic", "url": "https://ci.example.com/hooks/super-gpu",
   "events": ["*"], "headers": {"Authorization": "Bearer ..."}}
]
```

| Field | Meaning |
|---|---|
| `kind` | `feishu` (custom bot, optional signing `secret`), `slack` (incoming webhook), or `generic` (full JSON envelope) |
| `url` / `url_env` | the endpoint, or the environment variable holding it; an unset variable disables the target instead of failing validation |
| `events` | scheduler event kinds to forward; `["*"]` for everything. Default: `plan_completed`, `plan_failed`, `plan_cancelled`, `job_failed` |
| `headers`, `timeout` | extra request headers and per-request timeout in seconds |

A chat webhook URL is a credential: prefer `url_env`, and note that
`/api/state`, `validate`, and `doctor` only ever expose the host, never the
URL or secret. Delivery runs on a background thread with bounded retries, so
an unreachable endpoint never slows a scheduling tick; a delivery that finally
fails is recorded as a `notification_failed` event (which is itself never
forwarded). The generic envelope is:

```json
{"source": "super_gpu", "version": "0.2.0", "event": "plan_completed",
 "message": "plan ablation completed: 5 completed, 1 skipped",
 "created_at": 1756800000.0,
 "payload": {"plan_id": "plan-...", "name": "ablation", "status": "completed",
             "total_jobs": 6, "job_counts": {"completed": 5, "skipped": 1}}}
```

Plan-level events (`plan_completed`, `plan_failed`, `plan_cancelled`) fire
exactly once, when the last job of a plan reaches a terminal state; every
other kind in `GET /api/events` (`job_started`, `job_failed`, `job_retry`,
`node_offline`, `watchdog_anomaly_detected`, ...) can be subscribed to the
same way.

## REST API

`GET /api/meta` is the machine-readable version of this table (plus limits
and capability flags) and never requires the token, so an agent can discover
a controller before authenticating. Routes marked *public* are the only
other ones that skip the token check.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/meta` | *public* — API manifest: routes, error codes, MCP tools, limits, capability flags |
| GET | `/healthz` | *public* — liveness: the HTTP server answers |
| GET | `/readyz` | *public* — readiness: scheduler thread alive and at least one tick completed; 503 `not_ready` otherwise |
| GET | `/api/state` | full cluster, job, lease, and event state for the dashboard |
| GET | `/api/brief` | situation brief; `?hours=6` sets the change window, `?format=text` renders one screen |
| GET | `/api/capacity` | dry-run placement; `?gpus=1&memory_mib=auto&nodes=a,b&labels=x&allow_colocation=false&max_slots=8` |
| GET | `/api/snapshot` | latest GPU snapshot and active leases |
| GET | `/api/scheduler` | scheduler heartbeat, queue counts, watchdog status |
| GET | `/api/config` | public view of the controller configuration (no secrets) |
| GET | `/api/plans` | list plans |
| POST | `/api/plans` | submit a plan JSON; supports top-level `request_id`, conflict → 409 |
| POST | `/api/plans/preview` | validate a plan and preview estimates and placement without submitting |
| GET | `/api/plans/<id>` | one plan with all jobs |
| GET | `/api/plans/<id>/wait` | long-poll until the plan is terminal; `?timeout=30` (≤ 300 s); answer carries `terminal` |
| GET | `/api/jobs` | query jobs (each job carries `pending_reason` and `result_state`) |
| GET | `/api/jobs/<id>` | one job including command, stored log tails, and result |
| GET | `/api/jobs/<id>/wait` | long-poll until the job is terminal (same parameters as the plan form) |
| GET | `/api/jobs/<id>/logs` | last `?lines=200` (≤ 5000) lines of stdout/stderr fetched from the node that ran the job |
| GET | `/api/jobs/<id>/outputs` | expand a finished job's declared outputs on its node |
| POST | `/api/jobs/<id>/cancel` | cancel a job |
| POST | `/api/scan` | refresh GPU state immediately |
| GET | `/api/events` | scheduler events; `?since=<epoch seconds>&kind=job_failed,job_retry` filter |
| GET | `/api/anomalies` | current low-utilization occupancy and watchdog policy |

Every error is `{"ok": false, "error": "<text>", "code": "<code>"}`; branch on
`code`, never on the text:

| Code | Status | Meaning |
|---|---|---|
| `invalid_request` | 400 | malformed JSON, schema violation, or an out-of-range parameter |
| `unauthorized` | 401 | API token missing or wrong |
| `not_found` | 404 | unknown route, plan, or job |
| `method_not_allowed` | 405 | the route exists but not for this method (`Allow` header lists the right ones) |
| `idempotency_conflict` | 409 | `request_id` reused with different plan content |
| `internal_error` | 500 | unexpected controller failure; see scheduler events |
| `remote_failure` | 502 | the node could not be reached or the remote command failed |
| `not_ready` | 503 | scheduler has not completed its first tick or is not running |

`super-gpu api PATH [--data JSON]` calls any route from the shell with the
configured URL and token, so every step of a runbook is one command.

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
- CI exercises the full SSH path against docker nodes with a fake
  `nvidia-smi` (`tests/test_e2e_docker.py`); validation on real NVIDIA
  hardware still requires live server credentials.

## Roadmap

Planned directions, roughly in priority order — issues and PRs welcome:

- **PyPI releases**: `pip install super-gpu`, semantic versions, a changelog.
- **Per-process utilization attribution**: NVML accounting so shared-node
  gating can distinguish managed jobs from other users' load.
- **Prometheus `/metrics`**: a first-class scrape endpoint for fleet and
  queue telemetry.
- **AMD ROCm backend**: `rocm-smi` monitoring alongside NVIDIA.
- **Cross-plan priority and preemption**: let an urgent plan preempt
  lower-priority managed jobs it is allowed to displace.

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

# full SSH end-to-end path against docker "GPU nodes" (needs a docker daemon)
SUPER_GPU_E2E=1 python3 -m pytest tests/test_e2e_docker.py -v
```

CI runs the unit suite on Python 3.10, 3.11, and 3.12, plus the docker-based
SSH end-to-end job, for every push and pull request. Bug reports and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, conventions, and how to propose
changes.

## License

[MIT](LICENSE)
