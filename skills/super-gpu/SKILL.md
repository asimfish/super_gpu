---
name: super-gpu
description: "Operate the super_gpu multi-server NVIDIA GPU experiment scheduler: turn a server list and experiment plan into validated configuration, start or reuse the controller and dashboard, submit parallel jobs, and continuously monitor, retry, and report them. Use when an Agent is given the super_gpu repository or GitHub URL, needs to schedule ML experiments across dedicated and shared GPU servers, wants to maximize safe GPU utilization, or needs live cluster and job status."
---

# Operate super_gpu

Use super_gpu as the only placement authority for the requested experiment
plan. Keep dedicated nodes busy, protect shared nodes with conservative
admission, and supervise the plan through a terminal state.

## Establish the workspace

1. Locate an existing super_gpu checkout. If none exists, clone the repository
   URL supplied by the user. Use `https://github.com/asimfish/super_gpu` only
   when the user refers to super_gpu without supplying another fork.
2. Read the checkout's `AGENTS.md`, `README.md`,
   `examples/nodes.example.json`, and
   `schemas/experiment-plan.schema.json`.
3. Treat the checkout root as `SUPER_GPU_ROOT` in prose; substitute its real
   absolute path in commands rather than relying on an unset shell variable.
4. Create or reuse an isolated virtual environment and install the checkout
   with `python3 -m pip install -e ".[mcp]"`.

## Prepare private inputs

1. Convert the user's server list into an untracked `config.json`, or use the
   private config path they provide.
2. Require each node to declare `role: "dedicated"` or `role: "shared"`.
3. Preserve conservative shared-node defaults unless the user explicitly
   approves a change. Never infer that an idle-looking shared node is
   dedicated.
4. When gpumgr already owns the node inventory, generate the private config
   with `super-gpu import-gpumgr`; imported nodes must remain `shared` until the
   operator explicitly classifies them.
5. Use SSH aliases or destinations already authorized by the user. Never
   fabricate credentials, copy private keys, or commit private host data.
6. Convert the experiment description into a plan that satisfies the JSON
   Schema. Preserve commands and scientific parameters exactly.
7. Add `source.mode: snapshot` with an absolute controller-visible project
   path unless the user explicitly requires the mutable node workspace. Review
   default and custom exclusions so secrets and large datasets are not packed.
8. Choose a stable, safe `request_id` for this logical submission. Persist it
   in the private plan or run record and reuse it only for transport retries.
9. Prefer explicit `resources.memory_mib` for a never-before-run
   memory-sensitive workload. Use `"auto"` when a conservative first-run
   estimate is acceptable; explain that later runs improve from measured
   history.

## Validate before scheduling

Run:

```bash
.venv/bin/super-gpu --config /private/path/config.json validate --plan /private/path/plan.json
.venv/bin/super-gpu --config /private/path/config.json doctor
```

Resolve schema, duplicate-name, dependency-cycle, workspace, SSH, and
`nvidia-smi` failures before submission. Do not silently drop an unreachable
server or weaken a shared-node policy.

## Run continuously

Prefer controller mode whenever the user wants dynamic backfill, a dashboard,
or detached multi-hour execution:

1. Reuse an already healthy controller for the same config and database.
2. Otherwise start
   `.venv/bin/super-gpu --config /private/path/config.json serve` in a managed
   long-running terminal or session.
3. Open or report `http://127.0.0.1:8765`.
4. Submit with MCP `experiment_submit` when connected, otherwise run:

```bash
.venv/bin/super-gpu submit /private/path/plan.json --url http://127.0.0.1:8765
```

Include `--request-id ID` when using the CLI. MCP `experiment_submit` requires
the same stable ID. If the API returns `idempotency_conflict`, stop and compare
the plan and snapshot digest; do not silently choose a new ID. Use
`.venv/bin/super-gpu --config /private/path/config.json run /private/path/plan.json`
only for a foreground one-shot run.

Do not manually select GPUs or launch jobs around the scheduler. The controller
rescans all nodes, attributes managed allocations, observes shared-node
thresholds, and backfills newly stable capacity on every cycle.

## Supervise to completion

After submission:

1. Record the returned plan ID.
2. Record `submission.intent_digest`, `submission.replayed`, and the source
   snapshot digest. A replay must refer to the same returned plan ID.
3. Poll MCP `experiment_status`, `experiment_jobs`, and `scheduler_events`, or
   the equivalent CLI endpoints.
4. Keep the controller alive and continue monitoring until every job is
   `completed`, `failed`, or `cancelled`, unless the user explicitly asks for
   detached handoff.
5. On pending jobs, inspect placement events before changing anything. Capacity
   becoming available requires no manual action; the next stable scan schedules
   eligible work.
6. On failure, report the node, GPU indices, attempt count, exit code, and
   stderr tail. Let configured retries run; never alter scientific parameters
   merely to make a job pass.
7. Cancel only when explicitly requested or when the user explicitly
   authorized cleanup of this plan.
8. Query `anomaly_report` during supervision. Unmanaged findings are
   report-only. Enable `cancel_managed` only when the operator explicitly
   authorizes automatic cleanup of scheduler-owned jobs.

## Report the outcome

Return the plan ID, controller and dashboard address, counts by status,
placements, retries, failed-job diagnostics, and result or log locations. State
clearly whether supervision is still running or the plan reached a terminal
state.
