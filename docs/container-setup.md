# Cairn in dev containers

Cairn's hook pipeline (venv, models, SQLite DB) lives on the host. To let
Copilot Chat / Claude Code sessions running inside a dev container
participate in the same memory pool, the container ships a thin shim
that relays hook payloads to the existing cairn daemon over the mounted
`.daemon.sock`.

This avoids:
- Installing cairn (and ~500MB of models) inside every container.
- Pinning the container to a Python ABI that matches the host venv.
- Forking the memory DB per container.

The container only needs `python3` (any modern version). No `curl`,
no cairn deps, no extra services.

## 1. Make sure the host daemon is running

The cairn daemon already handles embedding/rerank/NLI traffic on
`$CAIRN_HOME/cairn/.daemon.sock`. Container hook relay is added as a new
`hook` action on the same socket — no second process.

```bash
python3 $CAIRN_HOME/cairn/daemon.py status
# if not running:
python3 $CAIRN_HOME/cairn/daemon.py start
```

## 2. Mount the socket and shim into the container

Add to your project's `.devcontainer/devcontainer.json` (or
`docker-compose-dev.yml` for compose-based setups):

```jsonc
{
  "mounts": [
    "source=${localEnv:HOME}/Projects/cairn/cairn/.daemon.sock,target=/run/cairn/.daemon.sock,type=bind",
    "source=${localEnv:HOME}/Projects/cairn/shims,target=/opt/cairn-shims,type=bind,readonly"
  ],
  "remoteEnv": {
    "CAIRN_SOCK": "/run/cairn/.daemon.sock"
  },
  "postCreateCommand": "mkdir -p ~/.github/hooks && sed 's|{{SHIM}}|/opt/cairn-shims/cairn-hook.py|g' /opt/cairn-shims/../templates/copilot-hooks-container.json > ~/.github/hooks/cairn.json"
}
```

Adjust the `source=` paths if your cairn install isn't at
`$HOME/Projects/cairn`.

## 3. Verify

Inside the container, after rebuilding:

```bash
ls -la $CAIRN_SOCK            # should exist, owned by your uid
echo '{}' | python3 /opt/cairn-shims/cairn-hook.py stop   # should return cleanly
cat ~/.github/hooks/cairn.json
```

Then ask Copilot Chat something and check the host:

```bash
$CAIRN_HOME/.venv/bin/python3 $CAIRN_HOME/cairn/query.py --recent
```

A memory tagged with the container's project should appear.

## Known limitations (MVP)

- **Project label**: comes from the container's `cwd`, which is typically
  the workspace mount target (`/workspaces/<name>` or
  `/home/rpl/workspace`), not the host workspace path. Until path
  translation is added, set `CAIRN_PROJECT` in `remoteEnv` to override.
- **`ingest.py --distill`**: still requires the `claude` CLI on whatever
  machine runs it. Out of scope for the shim path.
- **Transcript size**: bodies over `CAIRN_TRANSCRIPT_INLINE_MAX` bytes
  (default 2MB) are not inlined; host-side hooks won't see the
  transcript content for those calls. Bump the env var or share the
  transcripts dir via an additional bind mount if you hit this.
- **Fail-open**: if the host daemon is unreachable from the container,
  the shim exits 0 silently rather than blocking the user's prompt.
  Check `stderr` for `cairn-hook: daemon unreachable` if memories stop
  appearing.
