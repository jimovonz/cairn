#!/bin/bash
set -eo pipefail

# Cairn Installer
# Sets up persistent AI memory for Claude Code

# Default: auto-detect a CUDA GPU (nvidia-smi present + a GPU listed). Override
# with --gpu (force CUDA wheel) or --cpu (force the portable CPU-only wheel).
USE_GPU=auto
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) USE_GPU=true; shift ;;
        --cpu) USE_GPU=false; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: ./install.sh [--gpu|--cpu]"; exit 1 ;;
    esac
done
if [ "$USE_GPU" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
        USE_GPU=true
    else
        USE_GPU=false
    fi
fi

CAIRN_HOME="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$CAIRN_HOME/.venv"
VENV_PYTHON="$VENV_PATH/bin/python3"
CLAUDE_DIR="$HOME/.claude"

echo "=== Cairn Installer ==="
echo "Install location: $CAIRN_HOME"
if [ "$USE_GPU" = true ]; then
    echo "Mode: GPU (CUDA — auto-detected; --cpu to force CPU)"
else
    echo "Mode: CPU (no CUDA GPU detected; --gpu to force CUDA)"
fi
echo ""

# --- Python venv ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_PATH"
    if [ ! -f "$VENV_PATH/bin/pip" ]; then
        echo "Installing pip..."
        "$VENV_PYTHON" -m ensurepip 2>/dev/null || {
            echo "ERROR: python3-venv not installed. Run: sudo apt install python3-venv"
            exit 1
        }
    fi
else
    echo "Virtual environment exists."
fi

# --- Dependencies ---
echo "Installing dependencies (this may take a few minutes on first install)..."

# Install PyTorch before other deps to control CPU vs GPU wheel selection
if ! "$VENV_PYTHON" -c "import torch" 2>/dev/null; then
    if [ "$USE_GPU" = true ]; then
        echo "Installing PyTorch (CUDA)..."
        # Use PIPESTATUS so pip exit code (not grep's) determines success.
        # grep -E '...' || true keeps the pipeline alive if no output line matches the filter.
        "$VENV_PATH/bin/pip" install torch 2>&1 \
            | { grep -E "^(Collecting|Downloading|Installing|Successfully)" || true; }
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: PyTorch install failed."; exit 1; }
    else
        echo "Installing PyTorch (CPU-only, ~200MB)..."
        "$VENV_PATH/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 \
            | { grep -E "^(Collecting|Downloading|Installing|Successfully)" || true; }
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: PyTorch install failed."; exit 1; }
    fi
fi

"$VENV_PATH/bin/pip" install --progress-bar on -e "$CAIRN_HOME[test,ast,graph]" 2>&1 \
    | { grep -E "^(Collecting|Downloading|Installing|Successfully)" || true; }
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: Dependency install failed. Run manually:"; \
         echo "  $VENV_PATH/bin/pip install -e \"$CAIRN_HOME[test,ast,graph]\""; exit 1; }

# Verify critical imports before proceeding. Do NOT swallow stderr — the pysqlite3
# guard raises a specific, actionable ImportError if pysqlite3-binary is missing.
"$VENV_PYTHON" -c "from cairn import embeddings; from hooks.hook_helpers import log" \
    || { echo "ERROR: Post-install import check failed. Dependencies may be incomplete."; exit 1; }

# Verify the SINGLE sqlite library. Mixed stdlib(3.45)-vs-pysqlite3(3.51) writers on
# a WAL-mode cairn DB cause corruption (be91366), so every cairn writer must resolve
# sqlite3 -> pysqlite3. Fail loud if stdlib leaked (missing pysqlite3-binary, or a
# stray CAIRN_ALLOW_STDLIB_SQLITE=1 in the environment).
"$VENV_PYTHON" - <<'PYSQLITE_CHECK' \
    || { echo "ERROR: cairn is NOT running on pysqlite3 (mixed-sqlite WAL corruption risk)."; \
         echo "  Ensure pysqlite3-binary is installed and CAIRN_ALLOW_STDLIB_SQLITE is unset."; exit 1; }
import sys
import pysqlite3
import hooks.hook_helpers as h
assert pysqlite3.sqlite_version_info >= (3, 45), "pysqlite3 too old: %s" % pysqlite3.sqlite_version
assert h.sqlite3.__name__.startswith("pysqlite3"), "cairn resolved stdlib sqlite3: %s" % h.sqlite3.__name__
print("  sqlite library OK: pysqlite3 %s (single-lib)" % pysqlite3.sqlite_version)
PYSQLITE_CHECK

# --- Database ---
# Stop any running daemon BEFORE touching the DB — a live daemon (esp. with sync
# enabled: discovery + periodic-pull writes) holds the WAL and can lock init_db's
# schema migrations. It is restarted further down once the DB is ready.
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" stop >/dev/null 2>&1 || true
echo "Initializing database..."
"$VENV_PYTHON" "$CAIRN_HOME/cairn/init_db.py"

# --- Schema upgrade backfills (idempotent — only fills rows missing data) ---
# v7 calibration_qf_embeddings sidecar — embeds existing calibration row qf
# strings into the per-qf retrieval index. Cheap local embedder, no LLM cost.
# Distinguish "daemon unavailable" (acceptable; --check-daemon flag returns nonzero
# from backfill scripts) from real errors. Capture stderr so failures are visible.
if [ -f "$CAIRN_HOME/cairn/calibration_qf_backfill.py" ]; then
    echo "Backfilling per-qf calibration embeddings (if needed)..."
    qf_err=$("$VENV_PYTHON" "$CAIRN_HOME/cairn/calibration_qf_backfill.py" 2>&1 >/dev/null) && qf_rc=0 || qf_rc=$?
    if [ "$qf_rc" -ne 0 ]; then
        echo "  WARNING: calibration qf backfill exited with code $qf_rc"
        echo "  stderr: $qf_err"
        echo "  (continuing; rerun manually after fixing daemon)"
    fi
fi
# v8 memories.topic_embedding — embeds each memory's topic separately for
# dual-embedding retrieval. Backfill is idempotent (only fills NULL rows) so
# safe to repeat on every install/upgrade.
if [ -f "$CAIRN_HOME/cairn/memory_topic_embedding_backfill.py" ]; then
    echo "Backfilling memory topic embeddings (if needed)..."
    te_err=$("$VENV_PYTHON" "$CAIRN_HOME/cairn/memory_topic_embedding_backfill.py" 2>&1 >/dev/null) && te_rc=0 || te_rc=$?
    if [ "$te_rc" -ne 0 ]; then
        echo "  WARNING: topic embedding backfill exited with code $te_rc"
        echo "  stderr: $te_err"
        echo "  (continuing; rerun manually after fixing daemon)"
    fi
fi

# --- Directories ---
mkdir -p "$CLAUDE_DIR" "$CLAUDE_DIR/rules" "$CLAUDE_DIR/commands"

# --- Clean up legacy CLAUDE.md if we installed it previously ---
if [ -f "$CLAUDE_DIR/CLAUDE.md" ] && grep -q "Cairn — Global Memory" "$CLAUDE_DIR/CLAUDE.md" 2>/dev/null; then
    rm "$CLAUDE_DIR/CLAUDE.md"
    echo "Removed legacy global CLAUDE.md (instructions moved to rules file)."
fi

# --- Global rules ---
sed "s|{{CAIRN_HOME}}|$CAIRN_HOME|g" "$CAIRN_HOME/.claude/rules/memory-system.md" | \
    sed "s|\\\$CAIRN_HOME|$CAIRN_HOME|g" > "$CLAUDE_DIR/rules/memory-system.md"
cp "$CAIRN_HOME/.claude/rules/code-graph-navigation.md" "$CLAUDE_DIR/rules/code-graph-navigation.md"
echo "Installed global rules."

# --- Global settings (hooks) ---
if [ -f "$CLAUDE_DIR/settings.json" ]; then
    echo "Syncing Cairn hooks into existing settings.json..."
    python3 -c "
import json, sys

cairn_hooks = json.loads('''$(sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" "$CAIRN_HOME/templates/global-settings.json")''')

with open('$CLAUDE_DIR/settings.json') as f:
    settings = json.load(f)

hooks = settings.setdefault('hooks', {})
changed = False
for event, groups in cairn_hooks.get('hooks', {}).items():
    if event not in hooks:
        hooks[event] = groups
        changed = True
    else:
        # Check if each cairn hook command is already present
        existing_cmds = set()
        for g in hooks[event]:
            for h in g.get('hooks', []):
                existing_cmds.add(h.get('command', ''))
        for g in groups:
            for h in g.get('hooks', []):
                if h.get('command', '') not in existing_cmds:
                    hooks[event].append(g)
                    changed = True
                    break

if changed:
    with open('$CLAUDE_DIR/settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\n')
    print('Updated hooks in settings.json.')
else:
    print('All hooks already configured.')
" || {
        echo ""
        echo "WARNING: Could not auto-merge hooks. Add them manually from:"
        echo "  $CAIRN_HOME/templates/global-settings.json"
        echo ""
    }
else
    sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" \
        "$CAIRN_HOME/templates/global-settings.json" > "$CLAUDE_DIR/settings.json"
    echo "Installed global hooks."
fi

# --- Copilot hooks ---
COPILOT_HOOKS_DIR="$HOME/.github/hooks"
mkdir -p "$COPILOT_HOOKS_DIR"
sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" \
    "$CAIRN_HOME/templates/copilot-hooks.json" > "$COPILOT_HOOKS_DIR/cairn.json"
echo "Installed Copilot hooks."

# --- Slash command ---
sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" \
    "$CAIRN_HOME/templates/cairn-command.md" > "$CLAUDE_DIR/commands/cairn.md"
echo "Installed /cairn slash command."

# --- Pre-download models ---
echo ""
echo "Pre-downloading models (one-time)..."

# Corp TLS-inspection (Zscaler/Netskope/etc.) injects a non-public CA that the
# venv's certifi bundle doesn't trust. If SSL_CERT_FILE isn't already set and
# a system CA bundle exists, point HF/requests/httpx at it.
SYSTEM_CA_BUNDLE=""
for ca in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    if [ -f "$ca" ]; then SYSTEM_CA_BUNDLE="$ca"; break; fi
done
CA_ENV=""
if [ -n "$SYSTEM_CA_BUNDLE" ] && [ -z "$SSL_CERT_FILE" ]; then
    CA_ENV="SSL_CERT_FILE=$SYSTEM_CA_BUNDLE REQUESTS_CA_BUNDLE=$SYSTEM_CA_BUNDLE CURL_CA_BUNDLE=$SYSTEM_CA_BUNDLE"
    echo "Using system CA bundle: $SYSTEM_CA_BUNDLE"
fi

# httpx without the [socks] extra crashes when ALL_PROXY=socks5://... is set.
# Strip proxy env vars for the model download — HF hub talks to public CDNs
# that corp proxies typically allow direct anyway.
PROXY_STRIP=""
if [ -n "$ALL_PROXY$HTTPS_PROXY$HTTP_PROXY$all_proxy$https_proxy$http_proxy" ]; then
    PROXY_STRIP="-u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY -u all_proxy -u https_proxy -u http_proxy"
    echo "Unsetting proxy env vars for model download (httpx[socks] not required)."
fi

env $PROXY_STRIP $CA_ENV HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" -c "
from sentence_transformers import SentenceTransformer, CrossEncoder

# Bi-encoder for embeddings
m = SentenceTransformer('all-MiniLM-L6-v2')
v = m.encode(['test'])
print(f'Embedding model ready ({v.shape[1]} dimensions).')

# Cross-encoder for retrieval re-ranking
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
ce.predict([['test', 'test']])
print('Cross-encoder model ready.')

# NLI model for consolidation/contradiction detection
nli = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
nli.predict([['test', 'test']])
print('NLI model ready.')
" 2>&1 | grep -v "^$\|Warning:\|Loading\|REPORT\|UNEXPECTED\|Notes:\|Key.*|.*Status\|---" \
    || {
        echo ""
        echo "ERROR: Model download/load failed."
        echo ""
        echo "Common causes behind corporate networks:"
        echo "  - SSL CERTIFICATE_VERIFY_FAILED: corp TLS inspection (Zscaler/Netskope) injects a CA"
        echo "    the venv certifi bundle doesn't trust. Fix:"
        echo "      $VENV_PATH/bin/pip install -U certifi"
        echo "      export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
        echo "      export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE CURL_CA_BUNDLE=\$SSL_CERT_FILE"
        echo "    If still failing, drop the corp root CA PEM in /usr/local/share/ca-certificates/"
        echo "    and run: sudo update-ca-certificates"
        echo "  - httpx 'Cannot send a request': proxy env vars set without httpx[socks]. Try:"
        echo "      env -u ALL_PROXY -u HTTPS_PROXY -u HTTP_PROXY ./install.sh"
        echo "  - Air-gapped: pre-stage models in ~/.cache/huggingface and set HF_HUB_OFFLINE=1"
        exit 1
    }

# --- Logs directory ---
mkdir -p "$CAIRN_HOME/logs"

# --- Container shim staging dir (VSIXes the daemon auto-installs into dev containers) ---
VSIX_DIR="$HOME/.local/share/cairn-vsix"
mkdir -p "$VSIX_DIR"
echo "Container VSIX stage: $VSIX_DIR (drop .vsix files here for auto-install)."

# --- Docker preflight (non-fatal — container injector only needs docker if used) ---
if ! command -v docker >/dev/null 2>&1; then
    echo "Note: docker not on PATH — container injector will be inert until docker is installed."
elif ! docker info >/dev/null 2>&1; then
    echo "Note: docker present but not accessible (add user to 'docker' group?) — container injector will log errors on container start events."
fi

# --- Start (or restart) daemon ---
# On upgrade installs the daemon may already be running with stale model_version
# or schema knowledge — stop it first so the fresh `start` picks up current state.
# `daemon.py stop` returns 0 if not running, so this is safe on fresh installs too.
# --- Multi-node sync (default OFF; opt in: CAIRN_SYNC_ENABLED=1) ---
# Persisted to cairn/.env, which cairn.config loads on import — so the daemon
# (however started: install, cron, or Stop-hook respawn), every hook, and the CLI
# all see the same settings. Discovery + pairing require explicit per-peer
# approval, so advertising presence is low-risk; raw-session sharing stays OFF
# by default (opt in per node via CAIRN_SYNC_SHARE_SESSIONS=1).
ENV_FILE="$CAIRN_HOME/cairn/.env"
set_env_kv() {  # idempotent KEY=VALUE upsert into cairn/.env
    touch "$ENV_FILE"
    grep -v "^$1=" "$ENV_FILE" 2>/dev/null > "$ENV_FILE.tmp" || true
    echo "$1=$2" >> "$ENV_FILE.tmp"
    mv "$ENV_FILE.tmp" "$ENV_FILE"
}
SYNC_ENABLED="${CAIRN_SYNC_ENABLED:-0}"
set_env_kv CAIRN_SYNC_ENABLED "$SYNC_ENABLED"
set_env_kv CAIRN_SYNC_SHARE_SESSIONS "${CAIRN_SYNC_SHARE_SESSIONS:-0}"
if [ "$SYNC_ENABLED" = "1" ]; then
    echo "Multi-node sync ON (opt out: CAIRN_SYNC_ENABLED=0 ./install.sh) — server :${CAIRN_SYNC_PORT:-8787}, discovery udp :${CAIRN_SYNC_DISCOVERY_PORT:-47391}."
    echo "  (LAN peers need TCP ${CAIRN_SYNC_PORT:-8787} + UDP ${CAIRN_SYNC_DISCOVERY_PORT:-47391} reachable through any host firewall.)"
else
    echo "Multi-node sync OFF (enable: CAIRN_SYNC_ENABLED=1 ./install.sh)."
fi

echo "Restarting embedding daemon..."
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" stop >/dev/null 2>&1 || true
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" start

# --- API proxy (default ON; opt out: CAIRN_PROXY_ENABLED=0) ---
# Routes Claude Code <-> Anthropic through a local proxy that hides all Cairn
# artifacts and injects context under the hood. ON by default. Activation is
# scoped to a `c` shell launcher (NOT global): `c` ensures the daemon is up,
# health-gates ANTHROPIC_BASE_URL onto it, and falls back to a direct connection
# if the proxy is unreachable; plain `claude` always bypasses the proxy. So a
# down daemon never blocks traffic. Fully reversed by uninstall.sh.
PROXY_PORT="${CAIRN_PROXY_PORT:-8789}"
SHELL_RC="${CAIRN_SHELL_RC:-$HOME/.bashrc}"
# Default ON: proxy is enabled unless explicitly opted out with CAIRN_PROXY_ENABLED=0.
CAIRN_PROXY_ENABLED="${CAIRN_PROXY_ENABLED:-1}"
if [ "${CAIRN_PROXY_ENABLED:-}" = "1" ]; then
    echo "Enabling cairn API proxy on 127.0.0.1:$PROXY_PORT (opt out with CAIRN_PROXY_ENABLED=0)..."
    "$VENV_PATH/bin/pip" install --progress-bar off -e "$CAIRN_HOME[proxy]" >/dev/null 2>&1 \
        || { echo "ERROR: aiohttp (proxy extra) install failed."; exit 1; }
    # Install/refresh the `c` launcher in the shell rc (marked block, idempotent).
    # Overrides any existing `c` alias; preserves --dangerously-skip-permissions.
    LAUNCHER_TMP="$(mktemp)"
    sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{PROXY_PORT}}|$PROXY_PORT|g" \
        "$CAIRN_HOME/templates/cairn-launcher.sh" > "$LAUNCHER_TMP"
    touch "$SHELL_RC"
    "$VENV_PYTHON" - "$SHELL_RC" "$LAUNCHER_TMP" <<'PYEOF'
import re, sys
rc, lf = sys.argv[1], sys.argv[2]
block = open(lf).read()
with open(rc) as f: txt = f.read()
txt = re.sub(r"\n?# >>> cairn proxy launcher >>>.*?# <<< cairn proxy launcher <<<\n?",
             "\n", txt, flags=re.DOTALL)
if not txt.endswith("\n"): txt += "\n"
txt += block
if not txt.endswith("\n"): txt += "\n"
with open(rc, "w") as f: f.write(txt)
print(f"  installed 'c' launcher in {rc} (run: source {rc})")
PYEOF
    rm -f "$LAUNCHER_TMP"
    CAIRN_PROXY_ENABLED=1 CAIRN_PROXY_PORT="$PROXY_PORT" "$VENV_PYTHON" -m cairn.proxy.server restart --port "$PROXY_PORT" || true
    echo "  cairn-proxy started. Use 'c' to launch through it; 'claude' stays direct."
else
    # Not opting in this run — make sure no stale daemon keeps intercepting traffic.
    "$VENV_PYTHON" -m cairn.proxy.server stop >/dev/null 2>&1 || true
fi

# --- Cron jobs (memory consolidation + contradiction detection) ---
echo "Configuring cron jobs..."
CRON_MARKER="# cairn-maintenance"
# Detect nvm node bin (claude CLI requires modern node — cron's default
# PATH /usr/bin:/bin can find a stale node that breaks claude -p).
NVM_NODE_BIN=""
if [ -d "$HOME/.nvm/versions/node" ]; then
    NVM_NODE_BIN="$(ls -1d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1)"
fi
CRON_PATH_PREFIX=""
if [ -n "$NVM_NODE_BIN" ]; then
    CRON_PATH_PREFIX="PATH=$NVM_NODE_BIN:/usr/local/bin:/usr/bin:/bin "
fi
CRON_CONSOLIDATION="0 3 * * * ${CRON_PATH_PREFIX}$VENV_PYTHON $CAIRN_HOME/cairn/daemon.py start >/dev/null 2>&1; $VENV_PYTHON $CAIRN_HOME/cairn/consolidate.py --execute >> $CAIRN_HOME/logs/consolidation.log 2>&1 $CRON_MARKER"
CRON_CONTRADICTION="30 3 * * * ${CRON_PATH_PREFIX}$VENV_PYTHON $CAIRN_HOME/cairn/consolidate.py --contradictions --execute >> $CAIRN_HOME/logs/contradiction.log 2>&1 $CRON_MARKER"
# Calibration analyser — distills idle sessions into calibration_rows + memories. Runs at midnight.
CRON_ANALYSER="0 0 * * * ${CRON_PATH_PREFIX}$VENV_PYTHON -m cairn.analyser cron --limit 20 >> $CAIRN_HOME/logs/calibration-analyser.log 2>&1 $CRON_MARKER"
# Calibration self-modification — Tier 1 auto-archive/promote/decay + Tier 2 surfacing. Runs 30 minutes after analyser so today's writes are evaluated.
CRON_SELFMOD="30 0 * * * ${CRON_PATH_PREFIX}$VENV_PYTHON -m cairn.calibration_selfmod >> $CAIRN_HOME/logs/calibration-selfmod.log 2>&1 $CRON_MARKER"
# Code-graph fleet sweep — discover new repos, build missing graphs, register with
# the watch daemon, and self-heal the daemon if it died. Hourly so new repos become
# graph-ready quickly without waiting for a session. The daemon keeps existing repos
# current in real time between sweeps. Only runs if code-review-graph is installed.
CRON_GRAPH_FLEET="17 * * * * $VENV_PYTHON -m cairn.graph_fleet >> $CAIRN_HOME/logs/graph-fleet.log 2>&1 $CRON_MARKER"
# Proxy keep-alive — `start-fresh` is idempotent (no-op if running AND current) but
# ALSO restarts a running-but-stale daemon (proxy code newer than the live process),
# so a pulled proxy fix reaches the daemon within 5 min without a manual restart.
# Critical because ANTHROPIC_BASE_URL points Claude Code at the proxy. Empty unless opted in.
CRON_PROXY=""
if [ "${CAIRN_PROXY_ENABLED:-}" = "1" ]; then
    CRON_PROXY="*/5 * * * * CAIRN_PROXY_ENABLED=1 CAIRN_PROXY_PORT=$PROXY_PORT $VENV_PYTHON -m cairn.proxy.server start-fresh --port $PROXY_PORT >/dev/null 2>&1 $CRON_MARKER"
fi
# Daemon rerank health — restarts a responsive-but-poisoned daemon (e.g. a
# GPU-resident cross-encoder that throws cudaErrorLaunchFailure on every predict
# after a GPU fault). Ping-liveness misses this: the model is cached so the
# process stays up while the relevance gate is silently off. Every 15 min; the
# probe is a one-pair rerank and a no-op on the happy path.
CRON_DAEMON_HEALTH="*/15 * * * * ${CRON_PATH_PREFIX}$VENV_PYTHON $CAIRN_HOME/cairn/daemon.py healthcheck >> $CAIRN_HOME/logs/daemon-health.log 2>&1 $CRON_MARKER"

# Capture watchdog — a held cairn.db write lock (e.g. a sync writer stuck in a
# transaction) silently stalls memory capture into pending_writes. This alerts on
# a held lock / queue backlog / capture staleness so it surfaces in minutes, not
# days. Hourly; non-mutating probe, no-op on the happy path.
CRON_CAPTURE_WATCHDOG="41 * * * * ${CRON_PATH_PREFIX}$VENV_PYTHON $CAIRN_HOME/cairn/capture_watchdog.py >> $CAIRN_HOME/logs/capture-watchdog.log 2>&1 $CRON_MARKER"

# Remove any existing cairn cron entries (including legacy contradiction_scan.py and calibration variants)
EXISTING_CRON=$(crontab -l 2>/dev/null | grep -v "cairn-maintenance\|cairn/consolidate\|cairn/contradiction_scan\|cairn.analyser\|cairn.calibration_selfmod\|cairn.graph_fleet\|cairn.proxy.server\|cairn/capture_watchdog" || true)

# Install fresh entries
echo "$EXISTING_CRON
$CRON_CONSOLIDATION
$CRON_CONTRADICTION
$CRON_ANALYSER
$CRON_SELFMOD
$CRON_GRAPH_FLEET
$CRON_PROXY
$CRON_DAEMON_HEALTH
$CRON_CAPTURE_WATCHDOG" | sed '/^$/d' | crontab -
echo "Installed cron: consolidation (3:00 AM), contradiction scan (3:30 AM), calibration analyser (00:00), calibration selfmod (00:30), graph fleet sweep (hourly :17), daemon rerank health (every 15 min), capture watchdog (hourly :41)."

# --- Code-graph fleet bootstrap ---
# Build graphs for all local repos so every repo is graph-ready for first contact.
# Backgrounded — the initial all-repo build can take a while. The hourly cron sweep
# keeps them current thereafter (set CAIRN_GRAPH_WATCH=1 for the real-time daemon on
# top). Skips if code-review-graph isn't installed.
if [ -x "$VENV_PATH/bin/code-review-graph" ]; then
    echo "Bootstrapping code-graph fleet in background (building graphs for all repos)..."
    nohup "$VENV_PYTHON" -m cairn.graph_fleet >> "$CAIRN_HOME/logs/graph-fleet.log" 2>&1 &
fi

# --- Git post-commit hook: auto-refresh code-review-graph ---
# code-review-graph lives inside cairn's venv (usually not on PATH), so bake the
# absolute venv binary path into the hook. Skips silently if absent or not a git checkout.
CRG_BIN="$VENV_PATH/bin/code-review-graph"
if [ -d "$CAIRN_HOME/.git" ] && [ -x "$CRG_BIN" ]; then
    HOOK_DIR="$(git -C "$CAIRN_HOME" rev-parse --git-path hooks 2>/dev/null || echo "$CAIRN_HOME/.git/hooks")"
    case "$HOOK_DIR" in /*) ;; *) HOOK_DIR="$CAIRN_HOME/$HOOK_DIR" ;; esac
    mkdir -p "$HOOK_DIR"
    HOOK_PATH="$HOOK_DIR/post-commit"
    # Unquoted heredoc: $CRG_BIN expands now (baked absolute path); runtime vars are escaped.
    cat > "$HOOK_PATH" <<POST_COMMIT_HOOK
#!/bin/sh
# Auto-refresh code-review-graph after every commit (cairn-managed).
# Backgrounded so commit returns immediately. Incremental update, full-build fallback.
CRG="$CRG_BIN"
[ -x "\$CRG" ] || exit 0
ROOT="\$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
{ "\$CRG" update --repo "\$ROOT" >/dev/null 2>&1 || "\$CRG" build --repo "\$ROOT" >/dev/null 2>&1; } &
POST_COMMIT_HOOK
    chmod +x "$HOOK_PATH"
    echo "Installed git post-commit hook: code-review-graph auto-refresh ($CRG_BIN)."
fi

# --- cairn-graph on PATH ---
# Symlink cairn-graph into ~/.local/bin so it is available without activating the venv.
CAIRN_GRAPH_BIN="$VENV_PATH/bin/cairn-graph"
if [ -x "$CAIRN_GRAPH_BIN" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$CAIRN_GRAPH_BIN" "$HOME/.local/bin/cairn-graph"
    echo "Symlinked cairn-graph to ~/.local/bin/cairn-graph."
fi

# --- cairn-sync on PATH (multi-node sync CLI) ---
CAIRN_SYNC_BIN="$VENV_PATH/bin/cairn-sync"
if [ -f "$CAIRN_SYNC_BIN" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$CAIRN_SYNC_BIN" "$HOME/.local/bin/cairn-sync"
    echo "Symlinked cairn-sync to ~/.local/bin/cairn-sync."
fi

# --- cairn-bench tools on PATH ---
BENCH_B_BIN="$CAIRN_HOME/bin/cairn-bench-b"
if [ -x "$BENCH_B_BIN" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$BENCH_B_BIN" "$HOME/.local/bin/cairn-bench-b"
    echo "Symlinked cairn-bench-b to ~/.local/bin/cairn-bench-b."
fi
BENCH_REPORT_BIN="$VENV_PATH/bin/cairn-bench-report"
if [ -x "$BENCH_REPORT_BIN" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$BENCH_REPORT_BIN" "$HOME/.local/bin/cairn-bench-report"
    echo "Symlinked cairn-bench-report to ~/.local/bin/cairn-bench-report."
fi

# --- Health check ---
echo ""
if ! "$VENV_PYTHON" "$CAIRN_HOME/cairn/query.py" --check; then
    echo ""
    echo "Health check detected issues."
    # Check if it's DB corruption specifically
    if "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '$CAIRN_HOME')
from hooks.hook_helpers import get_conn
conn = get_conn()
r = conn.execute('PRAGMA quick_check').fetchone()
conn.close()
sys.exit(0 if r and r[0] == 'ok' else 1)
" 2>/dev/null; then
        echo "Database integrity OK — other issues detected (see above)."
    else
        echo ""
        echo "DATABASE CORRUPTION DETECTED."
        echo "Run the recovery script to repair:"
        echo ""
        echo "  $VENV_PYTHON $CAIRN_HOME/cairn/recover.py"
        echo ""
        read -p "Run recovery now? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            "$VENV_PYTHON" "$CAIRN_HOME/cairn/recover.py"
        fi
    fi
fi

echo ""
echo "=== Cairn installed successfully ==="
echo ""
echo "Restart Claude Code / VS Code Copilot to activate hooks."
echo ""
echo "Commands:"
echo "  /cairn          — memory stats"
echo "  /cairn recent   — recent memories"
echo "  /cairn search X — search memories"
echo "  /cairn review   — low-confidence memories"
echo "  /cairn projects — list projects"
echo ""
echo "Maintenance (cron, daily at 3:00 AM):"
echo "  Consolidation — merge duplicate memories"
echo "  Contradiction — detect and archive superseded memories"
echo "  Logs: $CAIRN_HOME/logs/"
echo ""
echo "Dev container support:"
echo "  TCP listener — port 47390 (container shims dial host daemon)"
echo "  VSIX stage   — $VSIX_DIR (drop .vsix files for auto-install on container start)"
echo "  See docs/container-setup.md for shim deploy and devcontainer.json wiring."
echo ""
