---
name: cairn-proxy
description: The opt-out bidirectional HTTP proxy between Claude Code and the Anthropic API: cache-safe context injection and Cairn artifact stripping.
---

## API proxy (artifact-free injection)

`cairn/proxy/` is an opt-out bidirectional HTTP proxy between Claude Code and the Anthropic API. It injects retrieved context into outbound requests **without disturbing the cacheable prefix** (Anthropic prompt cache stays byte-exact — a prompt-cache integrity guard verifies this) and strips every Cairn artifact (`<memory>`/`[cm]` blocks, `<cairn_context>`, system reminders) from inbound responses, capturing the stripped artifacts via `sidecar.py` for the hook pipeline. It is the artifact-hiding alternative to tag-stripping: capture/injection keep working even if Claude Code changes its tag rendering.

- `server.py` runs a detached daemon (`start`/`stop`/`restart`, port-specific PID file) on `127.0.0.1:8789` (`CAIRN_PROXY_PORT`). It `dup2`s fd0←/dev/null and fd1/fd2←log so it never holds an inherited stdout pipe open.
- `install.sh` enables it **by default** (opt out with `CAIRN_PROXY_ENABLED=0`), installs a `c` shell launcher (marked, idempotent rc block — `c` routes through the proxy, bare `claude` stays direct), and a `*/5` keep-alive cron (`start` is idempotent).
- Context is injected only on agentic requests.
