# Cairn in dev containers

Cairn's hook pipeline (venv, models, SQLite DB) lives on the host. To let
Copilot Chat / Claude Code sessions running inside a dev container
participate in the same memory pool, the container ships a thin shim that
relays hook payloads to a host-side daemon over a Unix socket.

This avoids:
- Installing cairn (and ~500MB of models) inside every container.
- Pinning the container to a Python ABI that matches the host venv.
- Forking the memory DB per container.

The container only needs `bash`, `curl`, and `python3` (any modern version).

## 1. Start the host API daemon

On the host, run:

```bash
$CAIRN_HOME/.venv/bin/python3 $CAIRN_HOME/cairn/api_server.py &
```

The socket is created at `$CAIRN_HOME/cairn/.api.sock` with mode `0600`.

To keep it running across reboots, add a user-systemd unit or wrap it in
the existing `cairn/daemon.py` supervisor (TODO — for the MVP run it
manually and confirm the path works end-to-end before automating).

## 2. Mount the socket and shim into the container

Add to your project's `.devcontainer/devcontainer.json` (or compose file):

```jsonc
{
  "mounts": [
    "source=${localEnv:HOME}/Projects/cairn/cairn/.api.sock,target=/run/cairn/.api.sock,type=bind",
    "source=${localEnv:HOME}/Projects/cairn/shims,target=/opt/cairn-shims,type=bind,readonly"
  ],
  "remoteEnv": {
    "CAIRN_SOCK": "/run/cairn/.api.sock"
  },
  "postCreateCommand": "mkdir -p ~/.github/hooks && sed 's|{{SHIM}}|/opt/cairn-shims/cairn-hook.sh|g' /opt/cairn-shims/../templates/copilot-hooks-container.json > ~/.github/hooks/cairn.json"
}
```

Adjust the `source=` paths if your cairn install isn't at
`$HOME/Projects/cairn`.

## 3. Verify

Inside the container, after rebuilding:

```bash
ls -la $CAIRN_SOCK            # should exist, owned by your uid
echo '{}' | /opt/cairn-shims/cairn-hook.sh stop   # should return cleanly
cat ~/.github/hooks/cairn.json # should have absolute paths to the shim
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
