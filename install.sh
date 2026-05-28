#!/bin/bash
set -eo pipefail

# Cairn Installer
# Sets up persistent AI memory for Claude Code

USE_GPU=false
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) USE_GPU=true; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: ./install.sh [--gpu]"; exit 1 ;;
    esac
done

CAIRN_HOME="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$CAIRN_HOME/.venv"
VENV_PYTHON="$VENV_PATH/bin/python3"
CLAUDE_DIR="$HOME/.claude"

echo "=== Cairn Installer ==="
echo "Install location: $CAIRN_HOME"
if [ "$USE_GPU" = true ]; then
    echo "Mode: GPU (CUDA)"
else
    echo "Mode: CPU (use --gpu for CUDA support)"
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

"$VENV_PATH/bin/pip" install --progress-bar on -e "$CAIRN_HOME[test,ast]" 2>&1 \
    | { grep -E "^(Collecting|Downloading|Installing|Successfully)" || true; }
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: Dependency install failed. Run manually:"; \
         echo "  $VENV_PATH/bin/pip install -e \"$CAIRN_HOME[test,ast]\""; exit 1; }

# Verify critical imports before proceeding
"$VENV_PYTHON" -c "from cairn import embeddings; from hooks.hook_helpers import log" 2>/dev/null \
    || { echo "ERROR: Post-install import check failed. Dependencies may be incomplete."; exit 1; }

# --- Database ---
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
echo "Restarting embedding daemon..."
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" stop >/dev/null 2>&1 || true
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" start

# --- Cron jobs (memory consolidation + contradiction detection) ---
echo "Configuring cron jobs..."
CRON_MARKER="# cairn-maintenance"
CRON_CONSOLIDATION="0 3 * * * $VENV_PYTHON $CAIRN_HOME/cairn/daemon.py start >/dev/null 2>&1; $VENV_PYTHON $CAIRN_HOME/cairn/consolidate.py --execute >> $CAIRN_HOME/logs/consolidation.log 2>&1 $CRON_MARKER"
CRON_CONTRADICTION="30 3 * * * $VENV_PYTHON $CAIRN_HOME/cairn/consolidate.py --contradictions --execute >> $CAIRN_HOME/logs/contradiction.log 2>&1 $CRON_MARKER"
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
# Calibration analyser — distills idle sessions into calibration_rows + memories. Runs at midnight.
CRON_ANALYSER="0 0 * * * ${CRON_PATH_PREFIX}$VENV_PYTHON -m cairn.analyser cron --limit 20 >> $CAIRN_HOME/logs/calibration-analyser.log 2>&1 $CRON_MARKER"
# Calibration self-modification — Tier 1 auto-archive/promote/decay + Tier 2 surfacing. Runs 30 minutes after analyser so today's writes are evaluated.
CRON_SELFMOD="30 0 * * * ${CRON_PATH_PREFIX}$VENV_PYTHON -m cairn.calibration_selfmod >> $CAIRN_HOME/logs/calibration-selfmod.log 2>&1 $CRON_MARKER"

# Remove any existing cairn cron entries (including legacy contradiction_scan.py and calibration variants)
EXISTING_CRON=$(crontab -l 2>/dev/null | grep -v "cairn-maintenance\|cairn/consolidate\|cairn/contradiction_scan\|cairn.analyser\|cairn.calibration_selfmod" || true)

# Install fresh entries
echo "$EXISTING_CRON
$CRON_CONSOLIDATION
$CRON_CONTRADICTION
$CRON_ANALYSER
$CRON_SELFMOD" | sed '/^$/d' | crontab -
echo "Installed cron: consolidation (3:00 AM), contradiction scan (3:30 AM), calibration analyser (00:00), calibration selfmod (00:30) daily."

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
