# >>> cairn proxy launcher >>>
# `c` routes Claude Code through the local cairn proxy: it hides every Cairn
# artifact from the conversation and injects context under the hood, keeping the
# Anthropic prompt cache byte-exact. The daemon is started on demand and the
# route is health-gated — if the proxy can't be reached, `c` falls back to a
# direct connection. Plain `claude` always bypasses the proxy (escape hatch).
# Managed by cairn install.sh / uninstall.sh — edits between these markers are
# overwritten on reinstall.
unalias c 2>/dev/null || true
c() {
    local port="${CAIRN_PROXY_PORT:-{{PROXY_PORT}}}"
    local py="{{VENV_PYTHON}}"
    # Start the daemon if needed (idempotent; no-op when already running).
    CAIRN_PROXY_ENABLED=1 CAIRN_PROXY_PORT="$port" "$py" -m cairn.proxy.server start --port "$port" >/dev/null 2>&1
    # Wait briefly for the port to accept (cold start forks then binds).
    local i=0
    while [ "$i" -lt 20 ]; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then exec 3>&- 3<&-; break; fi
        sleep 0.1; i=$((i+1))
    done
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        exec 3>&- 3<&-
        # Export BOTH: ANTHROPIC_BASE_URL routes traffic through the proxy, and
        # CAIRN_PROXY_ENABLED tells the cairn hooks (spawned by claude) to stage
        # injected context to sidecars instead of emitting it visibly — so the
        # request side is hidden too, not just the response-side [cm] block.
        CAIRN_PROXY_ENABLED=1 CAIRN_PROXY_PORT="$port" \
            ANTHROPIC_BASE_URL="http://127.0.0.1:$port" \
            claude --dangerously-skip-permissions "$@"
    else
        echo "cairn-proxy unavailable on 127.0.0.1:$port — running claude directly" >&2
        claude --dangerously-skip-permissions "$@"
    fi
}
# <<< cairn proxy launcher <<<
