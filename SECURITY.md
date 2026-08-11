# Security

`super_gpu` can execute arbitrary experiment commands over SSH. Treat access to
its CLI, REST API, and MCP server as equivalent to shell access on configured
GPU servers.

- The REST service binds to `127.0.0.1` by default.
- A non-loopback bind is rejected unless `api_token` or
  `SUPER_GPU_API_TOKEN` is configured.
- An API token is only one layer. Use a firewall, VPN, SSH tunnel, or private
  network as the primary boundary.
- Keep SSH hosts, keys, passwords, tokens, and private workspace paths out of
  Git. Real `config.json` and local state databases are ignored.
- Review experiment plans before submission. A plan's `command` field is
  intentionally capable of running arbitrary shell commands.
- The idle-GPU watchdog defaults to `report`. Its automatic mode can request
  cancellation only through a job handle owned by the active scheduler
  database. Unmanaged telemetry PIDs are never automatic termination targets.
- Treat `SUPER_GPU_WATCHDOG_ACTION=cancel_managed` as destructive authority.
  Use conservative runtime and grace thresholds and audit
  `watchdog_cancel_requested` events.

To report a vulnerability, use a private security advisory in the GitHub
repository rather than a public issue.
