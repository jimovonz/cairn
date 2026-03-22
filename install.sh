#!/bin/bash
set -e

# Cairn Installer
# Sets up persistent AI memory for Claude Code

CAIRN_HOME="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$CAIRN_HOME/.venv"
VENV_PYTHON="$VENV_PATH/bin/python3"
CLAUDE_DIR="$HOME/.claude"

echo "=== Cairn Installer ==="
echo "Install location: $CAIRN_HOME"
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
echo "Installing dependencies..."
"$VENV_PATH/bin/pip" install -q -r "$CAIRN_HOME/requirements.txt" 2>&1 | tail -1

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
"$VENV_PYTHON" -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('Model ready.')
" 2>&1 | grep -v "^$\|Warning:\|Loading"

# --- Start daemon ---
echo "Starting embedding daemon..."
"$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" start

# --- Health check (informational, don't fail install) ---
echo ""
"$VENV_PYTHON" "$CAIRN_HOME/cairn/query.py" --check || true

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
