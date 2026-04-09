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

"$VENV_PATH/bin/pip" install --progress-bar on -e "$CAIRN_HOME[test]" 2>&1 \
    | grep -E "^(Collecting|Downloading|Installing|Successfully)" \
    || { echo "ERROR: Dependency install failed. Run manually:"; \
         echo "  $VENV_PATH/bin/pip install -e \"$CAIRN_HOME[test]\""; exit 1; }

# Verify critical imports before proceeding
"$VENV_PYTHON" -c "from cairn import embeddings; from hooks.hook_helpers import log" 2>/dev/null \
    || { echo "ERROR: Post-install import check failed. Dependencies may be incomplete."; exit 1; }

# --- Database ---
echo "Initializing database..."
"$VENV_PYTHON" "$CAIRN_HOME/cairn/init_db.py"

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
    if grep -q "stop_hook.py" "$CLAUDE_DIR/settings.json" 2>/dev/null; then
        echo "Global hooks already configured."
    else
        echo "Merging Cairn hooks into existing settings.json..."
        python3 -c "
import json, sys

cairn_hooks = json.loads('''$(sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" "$CAIRN_HOME/templates/global-settings.json")''')

with open('$CLAUDE_DIR/settings.json') as f:
    settings = json.load(f)

hooks = settings.setdefault('hooks', {})
for event, groups in cairn_hooks.get('hooks', {}).items():
    if event not in hooks:
        hooks[event] = []
    hooks[event].extend(groups)

with open('$CLAUDE_DIR/settings.json', 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
" && echo "Merged hooks into settings.json." || {
            echo ""
            echo "WARNING: Could not auto-merge hooks. Add them manually from:"
            echo "  $CAIRN_HOME/templates/global-settings.json"
            echo ""
        }
    fi
else
    sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" \
        "$CAIRN_HOME/templates/global-settings.json" > "$CLAUDE_DIR/settings.json"
    echo "Installed global hooks."
fi

# --- Slash command ---
sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{CAIRN_HOME}}|$CAIRN_HOME|g" \
    "$CAIRN_HOME/templates/cairn-command.md" > "$CLAUDE_DIR/commands/cairn.md"
echo "Installed /cairn slash command."

# --- Pre-download model ---
echo ""
echo "Pre-downloading embedding model (one-time)..."
HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
v = m.encode(['test'])
print(f'Model ready ({v.shape[1]} dimensions).')
" 2>&1 | grep -v "^$\|Warning:\|Loading\|REPORT\|UNEXPECTED\|Notes:" \
    || { echo "ERROR: Model download/load failed."; exit 1; }

# --- Start daemon ---
echo "Starting embedding daemon..."
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" start

# --- Health check ---
echo ""
"$VENV_PYTHON" "$CAIRN_HOME/cairn/query.py" --check \
    || { echo "Health check failed — see above for details."; exit 1; }

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
