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
        "$VENV_PATH/bin/pip" install torch 2>&1 \
            | grep -E "^(Collecting|Downloading|Installing|Successfully)" \
            || { echo "ERROR: PyTorch install failed."; exit 1; }
    else
        echo "Installing PyTorch (CPU-only, ~200MB)..."
        "$VENV_PATH/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 \
            | grep -E "^(Collecting|Downloading|Installing|Successfully)" \
            || { echo "ERROR: PyTorch install failed."; exit 1; }
    fi
fi

"$VENV_PATH/bin/pip" install --progress-bar on -e "$CAIRN_HOME[test,ast]" 2>&1 \
    | grep -E "^(Collecting|Downloading|Installing|Successfully)" \
    || { echo "ERROR: Dependency install failed. Run manually:"; \
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
if [ -f "$CAIRN_HOME/cairn/calibration_qf_backfill.py" ]; then
    echo "Backfilling per-qf calibration embeddings (if needed)..."
    "$VENV_PYTHON" "$CAIRN_HOME/cairn/calibration_qf_backfill.py" >/dev/null 2>&1 || \
        echo "  (calibration qf backfill skipped — daemon may be unavailable)"
fi
# v8 memories.topic_embedding — embeds each memory's topic separately for
# dual-embedding retrieval. Backfill is idempotent (only fills NULL rows) so
# safe to repeat on every install/upgrade.
if [ -f "$CAIRN_HOME/cairn/memory_topic_embedding_backfill.py" ]; then
    echo "Backfilling memory topic embeddings (if needed)..."
    "$VENV_PYTHON" "$CAIRN_HOME/cairn/memory_topic_embedding_backfill.py" >/dev/null 2>&1 || \
        echo "  (topic embedding backfill skipped — daemon may be unavailable)"
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

# --- Slash command ---
sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" \
    "$CAIRN_HOME/templates/cairn-command.md" > "$CLAUDE_DIR/commands/cairn.md"
echo "Installed /cairn slash command."

# --- Pre-download models ---
echo ""
echo "Pre-downloading models (one-time)..."
HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" -c "
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
    || { echo "ERROR: Model download/load failed."; exit 1; }

# --- Logs directory ---
mkdir -p "$CAIRN_HOME/logs"

# --- Start daemon ---
echo "Starting embedding daemon..."
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
# Skips silently if code-review-graph not installed or this is not a git checkout.
if [ -d "$CAIRN_HOME/.git" ] && command -v code-review-graph >/dev/null 2>&1; then
    HOOK_PATH="$CAIRN_HOME/.git/hooks/post-commit"
    cat > "$HOOK_PATH" <<'POST_COMMIT_HOOK'
#!/bin/sh
# Auto-refresh code-review-graph after every commit.
# Backgrounded so commit returns immediately. ~2s for cairn-sized repo.
code-review-graph build --repo "$(git rev-parse --show-toplevel)" >/dev/null 2>&1 &
POST_COMMIT_HOOK
    chmod +x "$HOOK_PATH"
    echo "Installed git post-commit hook: code-review-graph auto-refresh."
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
echo "Restart Claude Code to activate hooks."
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
