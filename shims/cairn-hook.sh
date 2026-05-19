#!/usr/bin/env bash
#
# Generic cairn hook shim for use inside a dev container.
#
# Reads the hook payload JSON from stdin, optionally inlines the contents
# of `transcript_path` (so the host-side hook can read transcripts that
# live in the container filesystem), POSTs to the cairn host daemon over
# the mounted Unix socket, then replays stdout/stderr/exit-code from the
# response envelope so Copilot Chat / Claude Code see a transparent hook.
#
# Invoke as: cairn-hook.sh <route>
#   route: userpromptsubmit | stop | pretool | posttool
#
# Required mounts inside the container:
#   $CAIRN_SOCK   path to host's cairn .api.sock (read-write bind)
# Optional:
#   $CAIRN_TRANSCRIPT_INLINE_MAX  max bytes to inline (default 2_000_000)

set -euo pipefail

ROUTE="${1:?route required: userpromptsubmit|stop|pretool|posttool}"
SOCK="${CAIRN_SOCK:-/run/cairn/.api.sock}"
MAX_INLINE="${CAIRN_TRANSCRIPT_INLINE_MAX:-2000000}"

payload="$(cat)"

# Inline transcript body if the payload references one that exists.
transcript_path="$(printf '%s' "$payload" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d.get("transcript_path",""))')"
if [ -n "$transcript_path" ] && [ -r "$transcript_path" ]; then
    size=$(stat -c '%s' "$transcript_path" 2>/dev/null || echo 0)
    if [ "$size" -le "$MAX_INLINE" ]; then
        payload="$(printf '%s' "$payload" | python3 -c '
import json, sys
d = json.load(sys.stdin)
with open(d["transcript_path"]) as f:
    d["_transcript_body"] = f.read()
json.dump(d, sys.stdout)
')"
    fi
fi

response="$(printf '%s' "$payload" | curl -sS --unix-socket "$SOCK" \
    -H 'Content-Type: application/json' --data-binary @- \
    "http://localhost/${ROUTE}")"

printf '%s' "$response" | python3 -c '
import json, sys
env = json.loads(sys.stdin.read())
sys.stdout.write(env.get("stdout", ""))
sys.stderr.write(env.get("stderr", ""))
sys.exit(env.get("exit_code", 0))
'
