# Contributing to super_gpu

Thanks for taking the time to contribute. This project is small enough that
one page covers everything.

## Development Setup

```bash
git clone https://github.com/asimfish/super_gpu.git
cd super_gpu
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/python -m pytest
```

The core runtime is standard-library only; keep it that way unless a
dependency clearly pays for itself. Optional features belong behind extras
(like the existing `[mcp]`).

To explore the dashboard without GPU servers:

```bash
python3 scripts/demo_dashboard.py
```

To run the SSH end-to-end suite locally (real ssh into docker containers
that carry a fake `nvidia-smi`; needs a running docker daemon):

```bash
SUPER_GPU_E2E=1 python3 -m pytest tests/test_e2e_docker.py -v
```

## Before You Open a PR

1. Run the full suite: `python -m pytest`. CI runs it on Python 3.10, 3.11,
   and 3.12.
2. Add or extend a test that fails without your change. Scheduler behavior
   changes especially need a regression test in `tests/test_super_gpu.py`
   (see `FakeMonitor` / `FakeRunner` for how to simulate a cluster).
3. Keep commits focused; one logical change per commit, message in the
   `type: summary` style used in `git log` (`feat:`, `fix:`, `docs:`,
   `perf:`, `chore:`).
4. Update `README.md` (and `README.zh-CN.md` if the section exists there)
   when you change user-facing behavior, flags, or endpoints.

## Reporting Bugs

Open a GitHub issue with:

- what you expected and what happened,
- controller logs around the event (`~/.super_gpu/controller.stderr.log` or
  your serve terminal),
- your node policy settings (redact hostnames as needed),
- a reproduction, ideally as a failing test or a minimal plan JSON.

## Security Issues

Do not open public issues for vulnerabilities; follow
[SECURITY.md](SECURITY.md) instead.

## Design Changes

For anything that changes scheduling semantics, persistence, or the security
model, open an issue first and reference the relevant ADR in `docs/adr/`.
New significant decisions should come with a new ADR in the same format.
